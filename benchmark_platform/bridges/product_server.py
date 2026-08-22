from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.core import JsonlTrace, ToolEnvironment
from benchmark_platform.harnesses.content import wire_tool_result
from benchmark_platform.util import atomic_json

from .adapters import load_case


PRODUCT_WORKSPACE_ROOT = "/job/benchmark_server/case_workspace/workspace"
_WORKSPACE_PATH_FIELDS = {
    "list_files": ("path",),
    "read_file": ("path",),
    "run_command": ("cwd",),
    "write_file": ("path",),
    "edit_file": ("path",),
}


def translate_product_workspace_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Translate the product container's bind-mount path into bridge-relative paths."""
    translated = dict(arguments)
    prefix = PRODUCT_WORKSPACE_ROOT + "/"
    for field in _WORKSPACE_PATH_FIELDS.get(name, ()):
        raw = translated.get(field)
        if raw == PRODUCT_WORKSPACE_ROOT:
            translated[field] = "."
        elif isinstance(raw, str) and raw.startswith(prefix):
            translated[field] = raw[len(prefix) :]
    return translated


class ProductBridge:
    def __init__(self, benchmark: str, case_id: str, source: Path, job: Path):
        self.benchmark = benchmark
        self.case_id = case_id
        self.job = job
        self.started = time.perf_counter()
        self.loop = asyncio.new_event_loop()
        self._finalize_lock = threading.Lock()
        self._finalized: dict[str, Any] | None = None
        self.thread = threading.Thread(target=self.loop.run_forever, name="product-tool-loop", daemon=True)
        self.thread.start()
        workspace = job / "case_workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(source, workspace)
        # The matched product container starts in this shared directory for every static
        # benchmark. Workspace benchmarks already contain it; declaration-only benchmarks
        # get an empty cwd so Docker never falls back to /tmp or fails on a missing -w path.
        (workspace / "workspace").mkdir(exist_ok=True)
        bridge = load_case(benchmark, case_id, workspace)
        self.prompt = bridge.prompt
        self.metadata = bridge.metadata
        self.trace = JsonlTrace(job / "tool_trace.jsonl")
        self.environment = ToolEnvironment(bridge.tools, self.trace, bridge.handlers)
        self.tools = [tool.prompt_schema() for tool in bridge.tools]
        # BFCL tools are answer declarations, not executable reads. Advertising them as
        # safe for speculative prelaunch made PERSEUS call /execute before the actor had
        # even committed a function call, which both violates BFCL's single-turn protocol
        # and turns a bridge/network problem into model feedback. The agent-side extension
        # acknowledges the actor's declared calls locally and terminates that first batch.
        declaration_only = self.metadata.get("lifecycle") == "single_turn_declaration_only"
        explicit_safe = self.metadata.get("safe_for_prelaunch")
        if declaration_only:
            self.safe_tools = []
        elif isinstance(explicit_safe, list):
            declared_names = {str(name) for name in explicit_safe}
            self.safe_tools = [
                tool.name
                for tool in bridge.tools
                if tool.name in declared_names and tool.read_only and tool.parallel
            ]
        else:
            self.safe_tools = [
                tool.name
                for tool in bridge.tools
                if tool.read_only and tool.parallel
            ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.benchmark in {"gaia", "gdpval"}:
            arguments = translate_product_workspace_arguments(name, arguments)
        future = asyncio.run_coroutine_threadsafe(
            self.environment.call(name, arguments),
            self.loop,
        )
        return future.result()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "case_id": self.case_id,
            "prompt": self.prompt,
            "tools": self.tools,
            "safe_tools": self.safe_tools,
            "metadata": self.metadata,
        }

    def finalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._finalize_lock:
            if self._finalized is None:
                self._finalized = {
                    "schema_version": 1,
                    "status": "completed",
                    "benchmark": self.benchmark,
                    "case_id": self.case_id,
                    "profile": payload.get("profile"),
                    "final_answer": str(payload.get("answer") or ""),
                    "execution_seconds": time.perf_counter() - self.started,
                    "tool_calls": len(payload.get("committed_calls") or []),
                    "calls": list(payload.get("committed_calls") or []),
                    "environment_tool_calls": len(self.environment.calls),
                    "environment_calls": list(self.environment.calls),
                    "bridge": self.metadata,
                }
                atomic_json(self.job / "harness_result.json", self._finalized)
            return self._finalized

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


def handler_for(bridge: ProductBridge):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HarnessEvalProductBridge/1.0"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _json_body(self) -> dict[str, Any]:
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
            if self.path == "/manifest":
                self._send(200, bridge.manifest())
            elif self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            try:
                payload = self._json_body()
                if self.path == "/execute":
                    arguments = payload.get("arguments")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    execute = getattr(bridge, "execute", None)
                    # Image bytes exist only on this response hop. Environment traces and
                    # final artifacts retain ToolImage's small metadata dictionary.
                    result = wire_tool_result(
                        execute(
                            str(payload.get("tool") or ""),
                            arguments,
                            speculative=payload.get("speculative") is True,
                        )
                        if callable(execute)
                        else bridge.call(str(payload.get("tool") or ""), arguments)
                    )
                elif self.path == "/commit":
                    commit = getattr(bridge, "commit", None)
                    arguments = payload.get("arguments")
                    if not callable(commit):
                        raise ValueError("speculative_commit_unsupported")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    result = wire_tool_result(
                        commit(
                            str(payload.get("speculation_id") or ""),
                            str(payload.get("tool") or ""),
                            arguments,
                        )
                    )
                elif self.path == "/final":
                    result = bridge.finalize(payload)
                elif self.path == "/cancel":
                    cancel = getattr(bridge, "cancel_active", None)
                    result = cancel() if callable(cancel) else {"cancelled": 0}
                else:
                    self._send(404, {"ok": False, "error": "not_found"})
                    return
                self._send(200, result)
            except Exception as exc:
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--input", type=Path, default=Path("/bridge"))
    parser.add_argument("--job", type=Path, default=Path("/job"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    bridge = ProductBridge(args.benchmark, args.case, args.input, args.job)
    atomic_json(args.job / "product_bridge_ready.json", bridge.manifest())
    server = ThreadingHTTPServer((args.host, args.port), handler_for(bridge))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        bridge.close()


if __name__ == "__main__":
    main()
