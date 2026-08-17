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

    def next_action(self) -> PendingProductAction:
        return self._actions.get()

    def resolve(self, pending: PendingProductAction, response: dict[str, Any]) -> None:
        pending.response = response
        pending.event.set()

    def execute(self, name: str, arguments: dict[str, Any], *, speculative: bool) -> dict[str, Any]:
        if self._failure is not None:
            return {"ok": False, "error": f"native_episode_failed: {type(self._failure).__name__}: {self._failure}"}
        if self._result is not None:
            return {"ok": False, "error": "native_episode_already_finished"}
        if speculative:
            if self.speculative_executor is None:
                return {"ok": False, "error": "speculative_sandbox_unavailable"}
            started = time.perf_counter()
            response = self.speculative_executor(name, arguments)
            self.speculative_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": response,
                    "execution_seconds": time.perf_counter() - started,
                }
            )
            return response
        with self._dispatch_lock:
            pending = PendingProductAction(uuid.uuid4().hex, name, arguments)
            started = time.perf_counter()
            self._actions.put(pending)
            pending.event.wait()
            response = pending.response or {"ok": False, "error": "native_episode_missing_response"}
            self.calls.append(
                {
                    "id": pending.id,
                    "name": name,
                    "arguments": arguments,
                    "result": response,
                    "execution_seconds": time.perf_counter() - started,
                }
            )
            return response

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
        if final is not None:
            self.resolve(final, result)

    def fail(self, error: BaseException, final: PendingProductAction | None) -> None:
        self._failure = error
        if final is not None:
            self.resolve(final, {"ok": False, "error": f"{type(error).__name__}: {error}"})


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
