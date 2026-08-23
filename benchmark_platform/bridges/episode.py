from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from benchmark_platform.harnesses.api import CompletionClient, completion_client_from_env
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolSpec
from benchmark_platform.harnesses.methods import run_profile


SEND_MESSAGE_TOOL = "send_message_to_user"

_REASONING_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def visible_text(content: str | None) -> str:
    """The part of a completion a benchmark participant is allowed to see.

    The relay in use returns reasoning summaries inline, wrapped in <think>...</think>,
    instead of in a separate field. That is fine for the agent -- it is the baseline's own
    output -- but the hidden user simulator's deliberation reached the agent verbatim as if
    the user had said it ("<think>**Determining suitable hotel location**</think> 我想住..."),
    and the same text then landed inside the transcript the rubric evaluator grades. Both
    are hidden state escaping into a measurement.

    A completion that is nothing but a reasoning block is returned unchanged: an empty user
    turn would break the episode, and a visible artefact beats a silent one.
    """
    stripped = _REASONING_BLOCK.sub("", content or "").strip()
    return stripped or (content or "")


class UsageMeter:
    """Tokens spent by calls the broker's trace never sees.

    The hidden user simulator and the native evaluator run inside the benchmark, not inside
    a baseline, so nothing writes them to harness_trace.jsonl and the reported totals were
    the actor's alone. They are the same cost for every arm and so do not belong in the
    comparison total, but a run that cannot say what it spent is not auditable.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input = 0
        self.output = 0

    def add(self, completion: Any) -> None:
        self.calls += 1
        self.input += int(getattr(completion, "prompt_tokens", 0) or 0)
        self.output += int(getattr(completion, "completion_tokens", 0) or 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input": self.input,
            "output": self.output,
            "total": self.input + self.output,
        }

# Bound on either side of the native handshake. The adapter is allowed to stop asking for
# the next wave with requests still pending -- its step budget runs out, its episode ends,
# it raises -- and the coroutine awaiting that reply then waits forever while the adapter
# waits for the message only that coroutine can produce. Neither side is on a socket, so
# nothing ever times out: an observed arm sat for an hour with 64 threads in
# futex_wait_queue, and the sweep sat behind it.
#
# It has to clear the client's worst case, or a legitimate retry chain would trip it:
# HARNESS_API_TIMEOUT_S x (HARNESS_API_RETRIES + 1) plus backoff is ~730s at the defaults.
# It also has to stay under HARNESS_ARM_TIMEOUT_S so the arm fails with a diagnosis rather
# than being killed from outside with none.
HANDSHAKE_TIMEOUT_S = float(os.environ.get("HARNESS_EPISODE_HANDSHAKE_S", "1800"))


@dataclass(frozen=True)
class NativeTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    read_only: bool = False
    parallel: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            command=("/bin/false",),
            read_only=self.read_only,
            parallel=self.parallel,
        )


@dataclass
class ActionRequest:
    id: str
    name: str
    arguments: dict[str, Any]
    # An asyncio.Future resolved from the native thread via call_soon_threadsafe, not a
    # queue.Queue awaited through asyncio.to_thread. The queue version pinned a default
    # executor worker for as long as a request went unanswered, and the native adapter
    # legitimately stops calling next_wave with requests still pending (step budget
    # exhausted, episode ended, adapter raised). concurrent.futures' atexit hook joins
    # executor workers *before* daemon threads are killed, so one unanswered request
    # hung the interpreter at exit forever: the episode was complete but no result was
    # ever written. Observed on vitabench A0812003 actor-only, 2h at 0% CPU.
    reply: asyncio.Future[dict[str, Any]] | None = None

    @property
    def is_message(self) -> bool:
        return self.name == SEND_MESSAGE_TOOL

    def resolve(self, payload: dict[str, Any]) -> None:
        """Hand a result to the awaiting coroutine. Safe from any thread; a no-op if
        the episode loop is already gone or the request was answered twice."""
        future = self.reply
        if future is None:
            return
        loop = future.get_loop()
        def set_result() -> None:
            if not future.done():
                future.set_result(payload)

        try:
            loop.call_soon_threadsafe(set_result)
        except RuntimeError:
            # The native adapter may deliver a late result while the episode loop is
            # already closing. There is no waiter left to receive it.
            return


@dataclass(frozen=True)
class FinalResponse:
    answer: str


@dataclass(frozen=True)
class EpisodeFailure:
    error: str


class EpisodeBroker:
    """Run a complete async baseline while a native orchestrator owns actions."""

    def __init__(
        self,
        *,
        profile: str,
        prompt: str,
        tools: list[NativeTool],
        trace_path: Path,
        policy: dict[str, Any],
        client: CompletionClient | None = None,
    ):
        if any(tool.name == SEND_MESSAGE_TOOL for tool in tools):
            raise ValueError(f"Native benchmark already defines reserved tool {SEND_MESSAGE_TOOL}")
        self.profile = profile
        self.prompt = prompt
        self.native_tools = tools
        self.trace = JsonlTrace(trace_path)
        self.policy = policy
        self.client = client or completion_client_from_env()
        self._events: queue.Queue[ActionRequest | FinalResponse | EpisodeFailure] = queue.Queue()
        self._ready = threading.Event()
        self._pending: dict[str, ActionRequest] = {}
        self._broken: str | None = None
        self._thread = threading.Thread(target=self._thread_main, name=f"episode-{profile}", daemon=True)
        self.context: RunContext | None = None

    def start(self) -> None:
        self._thread.start()

    async def _request(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # Once the handshake is known broken every later call fails at once. Waiting the
        # full timeout again per tool call would drag one wedged episode out by hours.
        if self._broken:
            raise RuntimeError(self._broken)
        request = ActionRequest(uuid.uuid4().hex, name, arguments)
        request.reply = asyncio.get_running_loop().create_future()
        self._events.put(request)
        # Every coroutine in an asyncio.gather wave enqueues before the first
        # one resumes here, so the native adapter can emit one multi-tool turn.
        await asyncio.sleep(0)
        self._ready.set()
        try:
            return await asyncio.wait_for(request.reply, HANDSHAKE_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._broken = (
                f"Native adapter never answered tool request {name!r} within "
                f"{HANDSHAKE_TIMEOUT_S:.0f}s"
            )
            raise RuntimeError(self._broken) from None

    async def _run(self) -> None:
        from benchmark_platform.harnesses.core import ToolEnvironment

        communication = NativeTool(
            name=SEND_MESSAGE_TOOL,
            description=(
                "Send one complete assistant message to the benchmark's hidden user simulator and "
                "receive its next visible reply."
            ),
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
                "additionalProperties": False,
            },
            parallel=False,
        )
        declared = [*self.native_tools, communication]
        handlers = {
            tool.name: (lambda arguments, name=tool.name: self._request(name, arguments))
            for tool in declared
        }
        environment = ToolEnvironment([tool.spec() for tool in declared], self.trace, handlers)
        self.context = RunContext(
            self.profile,
            self.prompt,
            self.client,
            environment,
            self.trace,
            self.policy,
        )
        answer = await run_profile(self.context)
        self._events.put(FinalResponse(answer))
        self._ready.set()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._events.put(EpisodeFailure(f"{type(exc).__name__}: {exc}"))
            self._ready.set()

    @staticmethod
    def _decode_native_result(content: Any, error: bool) -> dict[str, Any]:
        if error:
            return {"ok": False, "error": "native_tool_failed", "detail": content}
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
        return {"ok": True, "result": content}

    def next_wave(
        self,
        *,
        tool_results: Mapping[str, tuple[Any, bool]] | None = None,
        user_message: str | None = None,
    ) -> list[ActionRequest] | FinalResponse:
        if self._pending:
            message_requests = [item for item in self._pending.values() if item.is_message]
            tool_requests = [item for item in self._pending.values() if not item.is_message]
            if message_requests:
                if len(message_requests) != 1 or tool_requests:
                    raise RuntimeError("A native wave cannot mix user communication and tool calls")
                if user_message is None:
                    raise RuntimeError("Native user reply is missing")
                message_requests[0].resolve({"ok": True, "result": {"user_message": user_message}})
            else:
                supplied = dict(tool_results or {})
                missing = sorted(set(self._pending) - set(supplied))
                extra = sorted(set(supplied) - set(self._pending))
                if missing or extra:
                    raise RuntimeError(f"Native tool result mismatch; missing={missing}, extra={extra}")
                for request_id, request in self._pending.items():
                    content, error = supplied[request_id]
                    request.resolve(self._decode_native_result(content, error))
            self._pending = {}

        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
        while True:
            while not self._ready.wait(1.0):
                # A baseline is free to declare a final answer whenever it likes, and several
                # do it before the benchmark's own conversation has ended -- dylan stops early
                # by design. Its thread is then gone, so no wave will ever arrive, and waiting
                # out the full timeout to learn that costs half an hour of a concurrency slot
                # per case. The producer being dead is knowable immediately.
                if not self._thread.is_alive():
                    raise RuntimeError(
                        "Baseline finished without the native episode reaching its own end; "
                        "no further wave can arrive"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Baseline produced no action or answer within {HANDSHAKE_TIMEOUT_S:.0f}s"
                    )
            self._ready.clear()
            events: list[ActionRequest | FinalResponse | EpisodeFailure] = []
            while True:
                try:
                    events.append(self._events.get_nowait())
                except queue.Empty:
                    break
            if not events:
                continue
            failures = [item for item in events if isinstance(item, EpisodeFailure)]
            if failures:
                raise RuntimeError(failures[0].error)
            finals = [item for item in events if isinstance(item, FinalResponse)]
            actions = [item for item in events if isinstance(item, ActionRequest)]
            if finals and actions:
                raise RuntimeError("Baseline completed while native actions were still pending")
            if finals:
                return finals[0]
            self._pending = {item.id: item for item in actions}
            return actions

    def metrics(self) -> dict[str, Any]:
        if self.context is None:
            return {}
        calls = self.context.environment.calls
        return {
            "llm_calls": self.context.llm_calls,
            "prompt_tokens": self.context.prompt_tokens,
            "completion_tokens": self.context.completion_tokens,
            # Native user communication is represented as a bridge tool only so an async
            # paper method can yield to the benchmark-owned simulator. It is not a benchmark
            # tool call, and the product arm records the same transition as a new turn rather
            # than in committed_calls. Keep the comparison column symmetric and expose the
            # communication count separately.
            "tool_calls": sum(item.get("name") != SEND_MESSAGE_TOOL for item in calls),
            "user_messages": sum(item.get("name") == SEND_MESSAGE_TOOL for item in calls),
        }
