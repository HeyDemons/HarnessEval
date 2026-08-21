from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .api import completion_client_from_env
from .core import JsonlTrace, RunContext, ToolEnvironment, ToolSpec
from .methods import run_profile
from .profiles import get_profile


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _validate_request(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 1:
        raise ValueError("Harness request schema_version must be 1")
    task = value.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
        raise ValueError("Harness request requires task.prompt")
    tools = value.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Harness request tools must be a list")


def _run_finalizer(spec: dict[str, Any], result_path: Path) -> dict[str, Any]:
    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError("finalizer.command must be a non-empty argv list")
    pass_env = spec.get("pass_env", [])
    if not isinstance(pass_env, list) or not all(isinstance(item, str) for item in pass_env):
        raise ValueError("finalizer.pass_env must be a list of names")
    allowed = {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", *pass_env}
    environment = {name: value for name, value in os.environ.items() if name in allowed}
    environment["HARNESS_RESULT_PATH"] = str(result_path)
    completed = subprocess.run(
        command,
        cwd=spec.get("cwd"),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


async def execute(profile_id: str, request_path: Path, output_path: Path, trace_path: Path) -> dict[str, Any]:
    profile = get_profile(profile_id)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _validate_request(request)
    _write_json(output_path.parent / "harness_request.json", request)
    trace = JsonlTrace(trace_path)
    tools = [ToolSpec.from_dict(item) for item in request.get("tools", [])]
    environment = ToolEnvironment(tools, trace)
    context = RunContext(
        profile_id,
        request["task"]["prompt"],
        completion_client_from_env(),
        environment,
        trace,
        request.get("policy") or {},
    )
    started = time.perf_counter()
    result: dict[str, Any]
    try:
        answer = await run_profile(context)
        result = {
            "schema_version": 1,
            "status": "completed",
            "profile": profile.id,
            "provenance": profile.provenance,
            "topology": profile.topology,
            "task_id": request["task"].get("id"),
            "final_answer": answer,
            "execution_seconds": time.perf_counter() - started,
            "llm_calls": context.llm_calls,
            "tool_calls": len(environment.calls),
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": context.completion_tokens,
            "policy": request.get("policy") or {},
        }
        _write_json(output_path, result)
        if finalizer := request.get("finalizer"):
            finalizer_result = _run_finalizer(finalizer, output_path)
            result["finalizer"] = finalizer_result
            if finalizer_result["returncode"] != 0:
                result["status"] = "failed"
                result["error"] = "Finalizer command failed"
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "profile": profile.id,
            "provenance": profile.provenance,
            "topology": profile.topology,
            "task_id": request.get("task", {}).get("id"),
            "execution_seconds": time.perf_counter() - started,
            "llm_calls": context.llm_calls,
            "tool_calls": len(environment.calls),
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": context.completion_tokens,
            "error": f"{type(exc).__name__}: {exc}",
        }
        await trace.emit("harness_error", error=result["error"])
    _write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one built-in HarnessEval profile")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/job/harness_result.json"))
    parser.add_argument("--trace", type=Path, default=Path("/job/harness_trace.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(execute(args.profile, args.request, args.output, args.trace))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
