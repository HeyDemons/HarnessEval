from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from benchmark_platform.util import atomic_json

from .episode import NativeTool, SEND_MESSAGE_TOOL


@dataclass
class PendingProductAction:
    id: str
    name: str
    arguments: dict[str, Any]
    kind: str = "tool"
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    replay_response: dict[str, Any] | None = None


class ProductEpisodeBridge:
    """Thread-safe rendezvous between an external product and a native episode."""

    def __init__(self, benchmark: str, case_id: str, job: Path):
        self.benchmark = benchmark
        self.case_id = case_id
        self.job = job
        self.started = time.perf_counter()
        self._manifest: dict[str, Any] | None = None
        self._manifest_ready = threading.Event()
        self._actions: queue.Queue[PendingProductAction] = queue.Queue()
        self._dispatch_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._result: dict[str, Any] | None = None
        self._failure: BaseException | None = None
        self._inflight: dict[str, PendingProductAction] = {}
        self._speculations: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self.speculative_calls: list[dict[str, Any]] = []
        self.speculative_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None

    def publish_manifest(
        self,
        *,
        prompt: str,
        tools: list[NativeTool],
        metadata: dict[str, Any],
        speculative_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        allow_speculation: bool = True,
    ) -> None:
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
        )
        declared = [*tools, communication]
        self.speculative_executor = speculative_executor
        self._manifest = {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "case_id": self.case_id,
            "prompt": prompt,
            "tools": [tool.spec().prompt_schema() for tool in declared],
            "safe_tools": (
                [tool.name for tool in tools if tool.read_only and tool.parallel]
                if allow_speculation
                else []
            ),
            "metadata": metadata,
        }
        atomic_json(self.job / "product_bridge_ready.json", self._manifest)
        self._manifest_ready.set()

    def manifest(self) -> dict[str, Any]:
        if not self._manifest_ready.is_set() or self._manifest is None:
            raise RuntimeError("native_episode_not_ready")
        return self._manifest

    def status(self) -> dict[str, Any]:
        with self._result_lock:
            if self._failure is not None:
                return {
                    "ok": False,
                    "state": "failed",
                    "error": f"{type(self._failure).__name__}: {self._failure}",
                }
            if self._result is not None:
                return {"ok": True, "state": "completed"}
            if self._manifest_ready.is_set():
                return {"ok": True, "state": "ready"}
            return {"ok": True, "state": "starting"}

    def next_action(self) -> PendingProductAction:
        while True:
            pending = self._actions.get()
            if pending.kind != "speculative":
                return pending
            if self.speculative_executor is None:
                self.resolve(pending, {"ok": False, "error": "speculative_sandbox_unavailable"})
                continue
            started = time.perf_counter()
            try:
                response = self.speculative_executor(pending.name, pending.arguments)
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            record = {
                "speculation_id": pending.id,
                "name": pending.name,
                "arguments": pending.arguments,
                "result": response,
                "execution_seconds": time.perf_counter() - started,
            }
            with self._result_lock:
                self._speculations[pending.id] = record
                self.speculative_calls.append(record)
            self.resolve(pending, {**response, "_harnesseval_speculation_id": pending.id})

    def resolve(self, pending: PendingProductAction, response: dict[str, Any]) -> None:
        pending.response = response
        pending.event.set()

    def replay_response(self, action_id: str) -> dict[str, Any] | None:
        """Return a pre-executed result after its action is authoritatively committed."""
        with self._result_lock:
            pending = self._inflight.get(action_id)
            return None if pending is None else pending.replay_response

    def _dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        replay_response: dict[str, Any] | None = None,
        speculation_id: str | None = None,
    ) -> dict[str, Any]:
        with self._dispatch_lock:
            pending = PendingProductAction(
                uuid.uuid4().hex,
                name,
                arguments,
                replay_response=replay_response,
            )
            with self._result_lock:
                if self._failure is not None:
                    return {
                        "ok": False,
                        "error": f"native_episode_failed: {type(self._failure).__name__}: {self._failure}",
                    }
                if self._result is not None:
                    return {"ok": False, "error": "native_episode_already_finished"}
                self._inflight[pending.id] = pending
            started = time.perf_counter()
            self._actions.put(pending)
            pending.event.wait()
            with self._result_lock:
                self._inflight.pop(pending.id, None)
            response = pending.response or {"ok": False, "error": "native_episode_missing_response"}
            self.calls.append(
                {
                    "id": pending.id,
                    "name": name,
                    "arguments": arguments,
                    "result": response,
                    "execution_seconds": time.perf_counter() - started,
                    "replayed_speculation": replay_response is not None,
                    "speculation_id": speculation_id,
                }
            )
            return response

    def execute(self, name: str, arguments: dict[str, Any], *, speculative: bool) -> dict[str, Any]:
        if speculative:
            if self._failure is not None:
                return {"ok": False, "error": f"native_episode_failed: {type(self._failure).__name__}: {self._failure}"}
            if self._result is not None:
                return {"ok": False, "error": "native_episode_already_finished"}
            if self.speculative_executor is None:
                return {"ok": False, "error": "speculative_sandbox_unavailable"}
            pending = PendingProductAction(uuid.uuid4().hex, name, arguments, kind="speculative")
            with self._result_lock:
                self._inflight[pending.id] = pending
            self._actions.put(pending)
            pending.event.wait()
            with self._result_lock:
                self._inflight.pop(pending.id, None)
            return pending.response or {"ok": False, "error": "native_episode_missing_speculative_response"}
        return self._dispatch(name, arguments)

    def continue_conversation(self, content: str) -> dict[str, Any]:
        """Submit a plain assistant message to the native user without inventing a tool call."""
        with self._dispatch_lock:
            pending = PendingProductAction(
                uuid.uuid4().hex,
                SEND_MESSAGE_TOOL,
                {"content": content},
                kind="message",
            )
            with self._result_lock:
                if self._failure is not None:
                    return {
                        "ok": False,
                        "error": f"native_episode_failed: {type(self._failure).__name__}: {self._failure}",
                    }
                if self._result is not None:
                    return {"ok": True, "episode_complete": True}
                self._inflight[pending.id] = pending
            self._actions.put(pending)
            pending.event.wait()
            with self._result_lock:
                self._inflight.pop(pending.id, None)
            return pending.response or {"ok": False, "error": "native_episode_missing_user_response"}

    def commit(self, speculation_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._result_lock:
            record = self._speculations.pop(speculation_id, None)
        if record is None:
            return {"ok": False, "error": "unknown_or_already_committed_speculation"}
        if record["name"] != name or record["arguments"] != arguments:
            return {"ok": False, "error": "speculation_commit_mismatch"}
        return self._dispatch(
            name,
            arguments,
            replay_response=record["result"],
            speculation_id=speculation_id,
        )

    def finalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._dispatch_lock:
            if self._result is None and self._failure is None:
                pending = PendingProductAction(
                    uuid.uuid4().hex,
                    "final_answer",
                    {"answer": str(payload.get("answer") or "")},
                    kind="final",
                )
                self._actions.put(pending)
                pending.event.wait()
            if self._failure is not None:
                raise RuntimeError(f"Native episode failed: {type(self._failure).__name__}: {self._failure}")
            result = dict(self._result or {})
            result.update(
                {
                    "profile": payload.get("profile"),
                    "final_answer": str(payload.get("answer") or ""),
                    "tool_calls": len(payload.get("committed_calls") or []),
                    "calls": list(payload.get("committed_calls") or []),
                    "environment_tool_calls": len(self.calls),
                    "environment_calls": list(self.calls),
                    "speculative_environment_tool_calls": len(self.speculative_calls),
                    "speculative_environment_calls": list(self.speculative_calls),
                    "execution_seconds": time.perf_counter() - self.started,
                }
            )
            atomic_json(self.job / "harness_result.json", result)
            return result

    def complete(self, result: dict[str, Any], final: PendingProductAction | None) -> None:
        with self._result_lock:
            self._result = result
            inflight = list(self._inflight.values())
        if final is not None:
            self.resolve(final, result)
        for pending in inflight:
            if not pending.event.is_set():
                response = (
                    {"ok": True, "episode_complete": True}
                    if pending.kind == "message"
                    else {"ok": False, "error": "native_episode_already_finished"}
                )
                self.resolve(pending, response)

    def fail(self, error: BaseException, final: PendingProductAction | None) -> None:
        with self._result_lock:
            self._failure = error
            inflight = list(self._inflight.values())
        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if final is not None:
            self.resolve(final, response)
        for pending in inflight:
            if not pending.event.is_set():
                self.resolve(pending, response)


def handler_for(bridge: ProductEpisodeBridge):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HarnessEvalProductEpisode/1.0"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            value = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(value, dict):
                raise ValueError("Request body must be a JSON object")
            return value

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            try:
                if self.path == "/manifest":
                    self._send(200, bridge.manifest())
                elif self.path == "/status":
                    self._send(200, bridge.status())
                elif self.path == "/health":
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"ok": False, "error": "not_found"})
            except Exception as exc:
                self._send(503, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            try:
                payload = self._body()
                if self.path == "/execute":
                    arguments = payload.get("arguments")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    result = bridge.execute(
                        str(payload.get("tool") or ""),
                        arguments,
                        speculative=payload.get("speculative") is True,
                    )
                elif self.path == "/commit":
                    arguments = payload.get("arguments")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    result = bridge.commit(
                        str(payload.get("speculation_id") or ""),
                        str(payload.get("tool") or ""),
                        arguments,
                    )
                elif self.path == "/turn":
                    result = bridge.continue_conversation(str(payload.get("content") or ""))
                elif self.path == "/final":
                    result = bridge.finalize(payload)
                else:
                    self._send(404, {"ok": False, "error": "not_found"})
                    return
                self._send(200, result)
            except Exception as exc:
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(bridge: ProductEpisodeBridge, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), handler_for(bridge))
    try:
        server.serve_forever()
    finally:
        server.server_close()
