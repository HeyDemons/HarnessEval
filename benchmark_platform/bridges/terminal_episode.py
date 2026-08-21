from __future__ import annotations

import asyncio
import subprocess
import time
import uuid
from pathlib import PurePosixPath
from typing import Any

from benchmark_platform.harnesses.api import ProviderError, completion_client_from_env
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import get_profile

from .base import clip_tool_output


def _workspace_path(
    value: str,
    workspace_root: str = "/app",
    default_workdir: str | None = None,
) -> str:
    root = PurePosixPath(workspace_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError(f"Invalid workspace root: {workspace_root}")
    relative_root = PurePosixPath(default_workdir or workspace_root)
    if not relative_root.is_absolute() or ".." in relative_root.parts:
        raise ValueError(f"Invalid default workdir: {default_workdir}")
    if relative_root != root and root not in relative_root.parents:
        raise ValueError(
            f"Default workdir is outside the accessible root: {default_workdir}"
        )
    path = PurePosixPath(value)
    if not path.is_absolute():
        path = relative_root / path
    if path != root and root not in path.parents:
        raise ValueError(f"Path is outside the task workspace: {value}")
    if ".." in path.parts:
        raise ValueError(f"Parent traversal is not allowed: {value}")
    return str(path)


# A task container answers `cat` with whatever the task ships. On terminal-bench-2 that reached
# 1,760,345 characters for gcode-to-text and 6,900,728 for mcmc-sampling-stan -- roughly 440k
# and 1.7M tokens in one observation -- and the relay rejected the next request with
# context_length_exceeded, taking the case from every arm that read the file.
#
def _completed(
    command: list[str],
    *,
    input_text: str | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout or ""
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr or ""
        )
        return {
            "ok": False,
            "error": "task_command_timeout",
            "detail": {
                "returncode": 124,
                "stdout": clip_tool_output(stdout, "stdout"),
                "stderr": clip_tool_output(stderr, "stderr"),
                "timeout_sec": timeout_sec,
            },
        }
    detail = {
        "returncode": result.returncode,
        "stdout": clip_tool_output(result.stdout, "stdout"),
        "stderr": clip_tool_output(result.stderr, "stderr"),
    }
    if result.returncode == 0:
        return {"ok": True, "result": detail}
    return {"ok": False, "error": "task_command_failed", "detail": detail}


def _tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="read_file",
            description="Read one UTF-8 file from the task container; long content is middle-truncated.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            command=("/bin/false",),
            read_only=True,
            parallel=True,
        ),
        ToolSpec(
            name="list_files",
            description="List files beneath a directory in the task container.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            command=("/bin/false",),
            read_only=True,
            parallel=True,
        ),
        ToolSpec(
            name="write_file",
            description="Write complete UTF-8 content to a file in the task container.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            command=("/bin/false",),
        ),
        ToolSpec(
            name="run_command",
            description=(
                "Run one argv command in the task container. Relative paths start at the image's native working "
                "directory; long output is middle-truncated and the command receives no harness API credentials."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
            command=("/bin/false",),
        ),
    ]


_EXEC_WRAPPER = r"""
token=$1
limit=$2
shift 2
pidfile=/tmp/.harnesseval-exec-${token}.pid
child=
terminate_child() {
    [ -n "$child" ] || return 0
    kill -TERM -"$child" 2>/dev/null || true
    command -v pkill >/dev/null 2>&1 && pkill -TERM -P "$child" 2>/dev/null || true
    kill -TERM "$child" 2>/dev/null || true
}
cleanup() {
    terminate_child
    rm -f "$pidfile"
}
trap cleanup HUP INT TERM
exec 3<&0
timeout --signal=TERM --kill-after=5s "${limit}s" "$@" <&3 &
child=$!
printf '%s\n' "$child" > "$pidfile"
wait "$child"
status=$?
rm -f "$pidfile"
exit "$status"
"""


_STOP_EXEC = r"""
pidfile=/tmp/.harnesseval-exec-$1.pid
[ -f "$pidfile" ] || exit 0
pid=$(cat "$pidfile")
kill -TERM -"$pid" 2>/dev/null || true
command -v pkill >/dev/null 2>&1 && pkill -TERM -P "$pid" 2>/dev/null || true
kill -TERM "$pid" 2>/dev/null || true
sleep 1
kill -KILL -"$pid" 2>/dev/null || true
command -v pkill >/dev/null 2>&1 && pkill -KILL -P "$pid" 2>/dev/null || true
kill -KILL "$pid" 2>/dev/null || true
rm -f "$pidfile"
"""


def _docker_exec(
    container: str,
    argv: list[str],
    *,
    timeout_sec: float,
    cwd: str | None = None,
    interactive: bool = False,
) -> tuple[list[str], str]:
    token = uuid.uuid4().hex
    command = ["docker", "exec"]
    if interactive:
        command.append("-i")
    if cwd is not None:
        command.extend(["-w", cwd])
    command.extend(
        [
            container,
            "sh",
            "-c",
            _EXEC_WRAPPER,
            "harnesseval-exec",
            token,
            f"{timeout_sec:.3f}",
            *argv,
        ]
    )
    return command, token


def _stop_container_exec(container: str, token: str) -> None:
    try:
        subprocess.run(
            [
                "docker",
                "exec",
                container,
                "sh",
                "-c",
                _STOP_EXEC,
                "harnesseval-stop",
                token,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        # The caller still terminates its host-side docker client below. This fallback is
        # diagnostic protection for a wedged daemon, not a reason to destroy the task
        # container and lose the state the shared verifier must inspect.
        return


async def _async_completed(
    command: list[str],
    *,
    container: str,
    token: str,
    timeout_sec: float,
    input_text: str | None = None,
) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(
                input_text.encode("utf-8") if input_text is not None else None
            ),
            timeout=timeout_sec + 10,
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        # Cancellation of asyncio.to_thread() cannot stop its subprocess. This explicit
        # marker lets an agent phase timeout terminate only the in-flight docker exec process
        # group while the task container (and services such as nginx) stays alive for grading.
        await asyncio.shield(asyncio.to_thread(_stop_container_exec, container, token))
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return {
            "ok": False,
            "error": "task_command_timeout",
            "detail": {
                "returncode": 124,
                "stdout": "",
                "stderr": f"Command exceeded {timeout_sec:.1f}s and was terminated",
                "timeout_sec": timeout_sec,
            },
        }
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    detail = {
        "returncode": process.returncode,
        "stdout": clip_tool_output(stdout_text, "stdout"),
        "stderr": clip_tool_output(stderr_text, "stderr"),
    }
    if process.returncode == 0:
        return {"ok": True, "result": detail}
    return {
        "ok": False,
        "error": "task_command_timeout"
        if process.returncode == 124
        else "task_command_failed",
        "detail": detail,
    }


def _handlers(
    container: str,
    workspace_root: str = "/app",
    default_workdir: str | None = None,
    *,
    command_timeout_sec: float = 600.0,
):
    default_workdir = default_workdir or workspace_root
    if command_timeout_sec <= 0:
        raise ValueError("command_timeout_sec must be positive")

    async def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _workspace_path(arguments["path"], workspace_root, default_workdir)
        command, token = _docker_exec(
            container, ["cat", path], timeout_sec=command_timeout_sec
        )
        return await _async_completed(
            command,
            container=container,
            token=token,
            timeout_sec=command_timeout_sec,
        )

    async def list_files(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _workspace_path(
            arguments.get("path", default_workdir), workspace_root, default_workdir
        )
        command, token = _docker_exec(
            container,
            ["find", path, "-type", "f", "-print"],
            timeout_sec=command_timeout_sec,
        )
        return await _async_completed(
            command,
            container=container,
            token=token,
            timeout_sec=command_timeout_sec,
        )

    async def write_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _workspace_path(arguments["path"], workspace_root, default_workdir)
        parent = str(PurePosixPath(path).parent)
        command, token = _docker_exec(
            container,
            [
                "sh",
                "-c",
                'mkdir -p "$1" && cat > "$2"',
                "harnesseval-write",
                parent,
                path,
            ],
            timeout_sec=command_timeout_sec,
            interactive=True,
        )
        return await _async_completed(
            command,
            container=container,
            token=token,
            timeout_sec=command_timeout_sec,
            input_text=arguments["content"],
        )

    async def run_command(arguments: dict[str, Any]) -> dict[str, Any]:
        argv = arguments["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise ValueError("run_command.argv must be a non-empty string array")
        cwd = _workspace_path(
            arguments.get("cwd", default_workdir), workspace_root, default_workdir
        )
        command, token = _docker_exec(
            container, argv, timeout_sec=command_timeout_sec, cwd=cwd
        )
        return await _async_completed(
            command,
            container=container,
            token=token,
            timeout_sec=command_timeout_sec,
        )

    return {
        "read_file": read_file,
        "list_files": list_files,
        "write_file": write_file,
        "run_command": run_command,
    }


async def execute(
    *,
    profile_id: str,
    prompt: str,
    policy: dict[str, Any],
    container: str,
    trace_path,
    workspace_root: str = "/app",
    default_workdir: str | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    trace = JsonlTrace(trace_path)
    tools = _tool_specs()
    environment = ToolEnvironment(
        tools,
        trace,
        _handlers(
            container,
            workspace_root,
            default_workdir,
            # This is the task's official agent wall-clock ceiling, not a
            # model-selectable per-command timeout.
            command_timeout_sec=timeout_sec or 600.0,
        ),
    )
    context = RunContext(
        profile_id,
        prompt,
        completion_client_from_env(),
        environment,
        trace,
        policy,
    )
    started = time.perf_counter()
    try:
        if timeout_sec is None:
            answer = await run_profile(context)
        else:
            async with asyncio.timeout(timeout_sec):
                answer = await run_profile(context)
        return {
            "schema_version": 1,
            "status": "completed",
            "profile": profile.id,
            "provenance": profile.provenance,
            "topology": profile.topology,
            "final_answer": answer,
            "execution_seconds": time.perf_counter() - started,
            "llm_calls": context.llm_calls,
            "tool_calls": len(environment.calls),
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": context.completion_tokens,
            "policy": policy,
            "returncode": 0,
        }
    except TimeoutError:
        timeout_error = (
            f"Agent exceeded its {timeout_sec:.1f}s wall-clock timeout"
            if timeout_sec is not None
            else "Agent execution timed out"
        )
        result = {
            "schema_version": 1,
            "status": "failed",
            "failure_kind": "agent_timeout",
            "profile": profile.id,
            "provenance": profile.provenance,
            "topology": profile.topology,
            "execution_seconds": time.perf_counter() - started,
            "llm_calls": context.llm_calls,
            "tool_calls": len(environment.calls),
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": context.completion_tokens,
            "returncode": 124,
            "error": timeout_error,
        }
        await trace.emit("harness_error", error=result["error"])
        return result
    except ProviderError as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "failure_kind": "provider_error",
            "provider_error_kind": exc.kind,
            "provider_status_code": exc.status_code,
            "profile": profile.id,
            "provenance": profile.provenance,
            "topology": profile.topology,
            "execution_seconds": time.perf_counter() - started,
            "llm_calls": context.llm_calls,
            "tool_calls": len(environment.calls),
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": context.completion_tokens,
            "returncode": 1,
            "error": str(exc),
        }
        await trace.emit("harness_error", error=result["error"], failure_kind="provider_error")
        return result
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "profile": profile.id,
            "provenance": profile.provenance,
            "topology": profile.topology,
            "execution_seconds": time.perf_counter() - started,
            "llm_calls": context.llm_calls,
            "tool_calls": len(environment.calls),
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": context.completion_tokens,
            "returncode": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
        await trace.emit("harness_error", error=result["error"])
        return result


def run(**kwargs) -> dict[str, Any]:
    return asyncio.run(execute(**kwargs))
