from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import PurePosixPath
from typing import Any

from benchmark_platform.harnesses.api import completion_client_from_env
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import get_profile


def _workspace_path(value: str, workspace_root: str = "/app") -> str:
    root = PurePosixPath(workspace_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError(f"Invalid workspace root: {workspace_root}")
    path = PurePosixPath(value)
    if not path.is_absolute():
        path = root / path
    if path != root and root not in path.parents:
        raise ValueError(f"Path is outside the task workspace: {value}")
    if ".." in path.parts:
        raise ValueError(f"Parent traversal is not allowed: {value}")
    return str(path)


def _completed(command: list[str], *, input_text: str | None = None) -> dict[str, Any]:
    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    detail = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode == 0:
        return {"ok": True, "result": detail}
    return {"ok": False, "error": "task_command_failed", "detail": detail}


def _tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="read_file",
            description="Read one complete UTF-8 file from the task workspace.",
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
            description="List files beneath a directory in the task workspace.",
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
            description="Write complete UTF-8 content to a file in the task workspace.",
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
                "Run one argv command in the task container. Output is returned completely; the command receives "
                "no harness API credentials."
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


def _handlers(container: str, workspace_root: str = "/app"):
    async def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _workspace_path(arguments["path"], workspace_root)
        return await asyncio.to_thread(_completed, ["docker", "exec", container, "cat", path])

    async def list_files(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _workspace_path(arguments.get("path", workspace_root), workspace_root)
        return await asyncio.to_thread(
            _completed,
            ["docker", "exec", container, "find", path, "-type", "f", "-print"],
        )

    async def write_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = _workspace_path(arguments["path"], workspace_root)
        parent = str(PurePosixPath(path).parent)
        command = [
            "docker",
            "exec",
            "-i",
            container,
            "sh",
            "-c",
            'mkdir -p "$1" && cat > "$2"',
            "harnesseval-write",
            parent,
            path,
        ]
        return await asyncio.to_thread(
            _completed,
            command,
            input_text=arguments["content"],
        )

    async def run_command(arguments: dict[str, Any]) -> dict[str, Any]:
        argv = arguments["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise ValueError("run_command.argv must be a non-empty string array")
        cwd = _workspace_path(arguments.get("cwd", workspace_root), workspace_root)
        return await asyncio.to_thread(
            _completed,
            ["docker", "exec", "-w", cwd, container, *argv],
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
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    trace = JsonlTrace(trace_path)
    tools = _tool_specs()
    environment = ToolEnvironment(tools, trace, _handlers(container, workspace_root))
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
        }
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
            "error": f"{type(exc).__name__}: {exc}",
        }
        await trace.emit("harness_error", error=result["error"])
        return result


def run(**kwargs) -> dict[str, Any]:
    return asyncio.run(execute(**kwargs))
