from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import completion_client_from_env
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import get_profile

from .adapters import load_case


def _write(path: Path, value: Any) -> None:
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


async def execute(benchmark: str, profile_id: str, case_id: str, root: Path, job: Path, policy: dict[str, Any]) -> dict[str, Any]:
    profile = get_profile(profile_id)
    trace = JsonlTrace(job / "harness_trace.jsonl")
    case_root = job / "case_workspace"
    if case_root.exists():
        shutil.rmtree(case_root)
    shutil.copytree(root, case_root)
    if os.name == "posix" and os.getuid() == 0:
        for path in [case_root, *case_root.rglob("*")]:
            path.chmod(path.stat().st_mode | (0o222 if path.is_file() else 0o333))
    bridge = load_case(benchmark, case_id, case_root)
    environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
    effective_policy = dict(policy)
    if benchmark == "bfcl":
        effective_policy["declaration_only_tools"] = True
    context = RunContext(
        profile_id,
        bridge.prompt,
        completion_client_from_env(),
        environment,
        trace,
        effective_policy,
    )
    _write(job / "bridge_manifest.json", {"benchmark": benchmark, "case_id": case_id, "profile": profile_id, "tool_schemas": [tool.prompt_schema() for tool in bridge.tools], "metadata": bridge.metadata})
    started = time.perf_counter()
    try:
        answer = await run_profile(context)
        result = {
            "schema_version": 1,
            "status": "completed",
            "benchmark": benchmark,
            "case_id": case_id,
            "profile": profile.id,
            "provenance": profile.provenance,
            "topology": profile.topology,
            "final_answer": answer,
            "execution_seconds": time.perf_counter() - started,
            "llm_calls": context.llm_calls,
            "tool_calls": len(environment.calls),
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": context.completion_tokens,
            "bridge": bridge.metadata,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if benchmark == "bfcl" and environment.calls:
            # BFCL scores the declared calls themselves. Some multi-turn profiles have a
            # stricter terminal protocol (for example MemGPT's heartbeat and Magentic-One's
            # final synthesis) than BFCL's one-response lifecycle. Once a call exists, a
            # later profile termination error must not erase that measurable prediction.
            result = {
                "schema_version": 1,
                "status": "completed",
                "benchmark": benchmark,
                "case_id": case_id,
                "profile": profile.id,
                "provenance": profile.provenance,
                "topology": profile.topology,
                "final_answer": None,
                "execution_seconds": time.perf_counter() - started,
                "llm_calls": context.llm_calls,
                "tool_calls": len(environment.calls),
                "prompt_tokens": context.prompt_tokens,
                "completion_tokens": context.completion_tokens,
                "bridge": bridge.metadata,
                "termination": {
                    "kind": "profile_error_after_calls",
                    "error": error,
                },
            }
            await trace.emit(
                "bridge_warning",
                kind="profile_error_after_calls",
                error=error,
                tool_calls=len(environment.calls),
            )
        else:
            result = {
                "schema_version": 1,
                "status": "failed",
                "benchmark": benchmark,
                "case_id": case_id,
                "profile": profile.id,
                "execution_seconds": time.perf_counter() - started,
                "error": error,
                "llm_calls": context.llm_calls,
                "tool_calls": len(environment.calls),
            }
            await trace.emit("bridge_error", error=result["error"])
    _write(job / "harness_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--input", type=Path, default=Path("/bridge"))
    parser.add_argument("--job", type=Path, default=Path("/job"))
    parser.add_argument("--policy", default="{}")
    args = parser.parse_args()
    result = asyncio.run(execute(args.benchmark, args.profile, args.case, args.input, args.job, json.loads(args.policy)))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
