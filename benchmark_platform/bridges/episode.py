from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from benchmark_platform.harnesses.api import ApiConfig, OpenAICompatibleClient
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolSpec
from benchmark_platform.harnesses.methods import run_profile


SEND_MESSAGE_TOOL = "send_message_to_user"


@dataclass(frozen=True)
class NativeTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    read_only: bool = False
    parallel: bool = True

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
    reply: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)

    @property
    def is_message(self) -> bool:
        return self.name == SEND_MESSAGE_TOOL


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
        client: OpenAICompatibleClient | None = None,
    ):
        if any(tool.name == SEND_MESSAGE_TOOL for tool in tools):
            raise ValueError(f"Native benchmark already defines reserved tool {SEND_MESSAGE_TOOL}")
        self.profile = profile
        self.prompt = prompt
        self.native_tools = tools
        self.trace = JsonlTrace(trace_path)
        self.policy = policy
        self.client = client or OpenAICompatibleClient(ApiConfig.from_env())
        self._events: queue.Queue[ActionRequest | FinalResponse | EpisodeFailure] = queue.Queue()
        self._ready = threading.Event()
        self._pending: dict[str, ActionRequest] = {}
        self._thread = threading.Thread(target=self._thread_main, name=f"episode-{profile}", daemon=True)
        self.context: RunContext | None = None

    def start(self) -> None:
        self._thread.start()

    async def _request(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        request = ActionRequest(uuid.uuid4().hex, name, arguments)
        self._events.put(request)
        # Every coroutine in an asyncio.gather wave enqueues before the first
        # one resumes here, so the native adapter can emit one multi-tool turn.
        await asyncio.sleep(0)
        self._ready.set()
        return await asyncio.to_thread(request.reply.get)

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
                message_requests[0].reply.put({"ok": True, "result": {"user_message": user_message}})
            else:
                supplied = dict(tool_results or {})
                missing = sorted(set(self._pending) - set(supplied))
                extra = sorted(set(supplied) - set(self._pending))
                if missing or extra:
                    raise RuntimeError(f"Native tool result mismatch; missing={missing}, extra={extra}")
                for request_id, request in self._pending.items():
                    content, error = supplied[request_id]
                    request.reply.put(self._decode_native_result(content, error))
            self._pending = {}

        while True:
            self._ready.wait()
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
        return {
            "llm_calls": self.context.llm_calls,
            "prompt_tokens": self.context.prompt_tokens,
            "completion_tokens": self.context.completion_tokens,
            "tool_calls": len(self.context.environment.calls),
        }
