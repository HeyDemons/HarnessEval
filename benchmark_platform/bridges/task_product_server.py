from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.core import JsonlTrace, ToolEnvironment
from benchmark_platform.util import atomic_json

from .product_server import handler_for
from .terminal_episode import _handlers, _tool_specs, _workspace_path


class TaskProductBridge:
    """Expose one live task container to an external product harness."""

    def __init__(
        self,
        *,
        benchmark: str,
        case_id: str,
        prompt: str,
        container: str,
        workspace_root: str,
        default_workdir: str | None = None,
        agent_timeout_sec: float = 3600.0,
        job: Path,
    ):
        default_workdir = default_workdir or workspace_root
        self.benchmark = benchmark
        self.case_id = case_id
        self.prompt = prompt
        self.container = container
        self.workspace_root = workspace_root
        self.default_workdir = default_workdir
        self.agent_timeout_sec = agent_timeout_sec
        self.job = job
        self.started = time.perf_counter()
        self.metadata = {
            "bridge": "live_task_container",
            "workspace_root": workspace_root,
            "default_workdir": default_workdir,
            "speculation_policy": "read_only_tools_only",
        }
        self.loop = asyncio.new_event_loop()
        self._finalize_lock = threading.Lock()
        self._speculation_lock = threading.Lock()
        self._finalized: dict[str, Any] | None = None
        self._speculations: dict[str, dict[str, Any]] = {}
        self.speculative_calls: list[dict[str, Any]] = []
        self.committed_speculations: list[dict[str, Any]] = []
        self.snapshot_run_command = (
            os.environ.get("HARNESSEVAL_ENABLE_SNAPSHOT_RUN_COMMAND_SPECULATION")
            == "1"
        )
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="task-product-tool-loop", daemon=True
        )
        self.thread.start()
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self.trace = JsonlTrace(job / "tool_trace.jsonl")
        tools = _tool_specs()
        self.environment = ToolEnvironment(
            tools,
            self.trace,
            _handlers(
                container,
                workspace_root,
                default_workdir,
                # The outer product runner cancels active calls at this same official
                # task deadline; the model cannot override it per command.
                command_timeout_sec=agent_timeout_sec,
            ),
        )
        self.tools = [tool.prompt_schema() for tool in tools]
        self.safe_tools = [
            tool.name for tool in tools if tool.read_only and tool.parallel
        ]
        if self.snapshot_run_command:
            self.safe_tools.append("run_command")
            self.metadata["run_command_speculation"] = "docker_snapshot_diff_v1"

    async def _tracked_call(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        task = asyncio.current_task()
        assert task is not None
        self._active_tasks.add(task)
        try:
            return await self.environment.call(name, arguments)
        finally:
            self._active_tasks.discard(task)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self._tracked_call(name, arguments), self.loop
        )
        return future.result()

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        speculative: bool,
    ) -> dict[str, Any]:
        if not speculative:
            return self.call(name, arguments)
        if not self.snapshot_run_command or name != "run_command":
            return {
                "ok": False,
                "error": "speculative_snapshot_unsupported",
                "_harnesseval_speculation_rejected": True,
            }
        started = time.perf_counter()
        result = self._snapshot_run_command(arguments)
        record = {
            "name": name,
            "arguments": arguments,
            "result": result,
            "execution_seconds": time.perf_counter() - started,
        }
        self.speculative_calls.append(record)
        if result.get("_harnesseval_speculation_rejected") is True:
            return result
        speculation_id = uuid.uuid4().hex
        with self._speculation_lock:
            self._speculations[speculation_id] = record
        return {**result, "_harnesseval_speculation_id": speculation_id}

    def commit(
        self,
        speculation_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        with self._speculation_lock:
            record = self._speculations.pop(speculation_id, None)
        if record is None:
            return {"ok": False, "error": "speculation_not_found"}
        if record["name"] != name or record["arguments"] != arguments:
            return {"ok": False, "error": "speculation_commit_mismatch"}
        committed = {
            **record,
            "speculation_id": speculation_id,
            "replayed_speculation": True,
        }
        self.committed_speculations.append(committed)
        return record["result"]

    def _snapshot_run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        argv = arguments.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            return self._rejected_snapshot("invalid_argv")
        try:
            cwd = _workspace_path(
                arguments.get("cwd", self.default_workdir),
                self.workspace_root,
                self.default_workdir,
            )
        except Exception as exc:
            return self._rejected_snapshot(f"invalid_cwd:{type(exc).__name__}")

        token = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        image = f"harnesseval-spec-snapshot:{token}"
        container = f"harnesseval-spec-snapshot-{token}"
        timeout = min(
            self.agent_timeout_sec,
            float(os.environ.get("TERMINAL_BENCH_SNAPSHOT_SPECULATION_TIMEOUT_S", "120")),
        )
        try:
            committed = subprocess.run(
                ["docker", "commit", self.container, image],
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
            if committed.returncode != 0:
                return self._rejected_snapshot("snapshot_commit_failed")
            created = subprocess.run(
                [
                    "docker",
                    "create",
                    "--name",
                    container,
                    "--network",
                    "none",
                    "-e",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "-e",
                    "GIT_OPTIONAL_LOCKS=0",
                    "-w",
                    cwd,
                    "--entrypoint",
                    argv[0],
                    image,
                    *argv[1:],
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
            if created.returncode != 0:
                return self._rejected_snapshot("snapshot_container_create_failed")
            executed = subprocess.run(
                ["docker", "start", "-a", container],
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
            if executed.returncode != 0:
                return self._rejected_snapshot(
                    "snapshot_command_failed",
                    returncode=executed.returncode,
                )
            diff = subprocess.run(
                ["docker", "diff", container],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if diff.returncode != 0:
                return self._rejected_snapshot("snapshot_diff_failed")
            changed = [line for line in diff.stdout.splitlines() if line.strip()]
            if changed:
                return self._rejected_snapshot(
                    "snapshot_filesystem_changed",
                    changed_paths=len(changed),
                )
            return {
                "ok": True,
                "result": {
                    "returncode": 0,
                    "stdout": executed.stdout,
                    "stderr": executed.stderr,
                },
            }
        except subprocess.TimeoutExpired:
            return self._rejected_snapshot("snapshot_timeout")
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container],
                text=True,
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["docker", "image", "rm", image],
                text=True,
                capture_output=True,
                check=False,
            )

    @staticmethod
    def _rejected_snapshot(reason: str, **details: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "error": reason,
            "details": details,
            "_harnesseval_speculation_rejected": True,
        }

    async def _cancel_active(self) -> int:
        active = list(self._active_tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        return len(active)

    def cancel_active(self) -> dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(self._cancel_active(), self.loop)
        return {"cancelled": future.result(timeout=30)}

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
                    "environment_tool_calls": (
                        len(self.environment.calls)
                        + len(self.committed_speculations)
                    ),
                    "environment_calls": [
                        *self.environment.calls,
                        *self.committed_speculations,
                    ],
                    "speculative_calls": list(self.speculative_calls),
                    "bridge": self.metadata,
                }
                atomic_json(self.job / "harness_result.json", self._finalized)
            return self._finalized

    def close(self) -> None:
        if self.loop.is_running():
            try:
                self.cancel_active()
            except Exception:
                # Container cleanup is the final backstop; shutdown must not strand the
                # server thread merely because an already-disconnected tool call raced it.
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--default-workdir", required=True)
    parser.add_argument("--agent-timeout-sec", type=float, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    bridge = TaskProductBridge(
        benchmark=args.benchmark,
        case_id=args.case,
        prompt=args.prompt_file.read_text(encoding="utf-8"),
        container=args.container,
        workspace_root=args.workspace_root,
        default_workdir=args.default_workdir,
        agent_timeout_sec=args.agent_timeout_sec,
        job=args.job,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_for(bridge))
    atomic_json(
        args.job / "task_product_server.json",
        {"host": args.host, "port": server.server_address[1]},
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        bridge.close()


if __name__ == "__main__":
    main()
