from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import (
    ProviderError,
    completion_client_from_env,
    sa_speculator_client_from_env,
)
from benchmark_platform.harnesses.core import (
    JsonlTrace,
    RunContext,
    ToolEnvironment,
)
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import get_profile
from benchmark_platform.harnesses.declaration import (
    PUBLISHER_PROTOCOL,
    publish_method_declaration,
)
from benchmark_platform.budgets import positive_int

from .adapters import load_case
from .bfcl import render_bfcl_method_prompt


def _write(path: Path, value: Any) -> None:
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


def _published_calls(environment: ToolEnvironment) -> list[dict[str, Any]]:
    return [
        {"name": str(record["name"]), "arguments": dict(record.get("arguments") or {})}
        for record in environment.calls
    ]


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
    environment = ToolEnvironment(
        bridge.tools,
        trace,
        bridge.handlers,
        proposal_only=benchmark == "bfcl",
        expose_execution_metadata=benchmark != "bfcl",
    )
    effective_policy = dict(policy)
    if benchmark == "bfcl":
        effective_policy["bfcl_declaration_mode"] = True
        effective_policy["bfcl_native_tools"] = True
        effective_policy["react_protocol"] = "native"
        # The runtime harness is the measured system. BFCL constrains its one outward
        # response, never the number of internal model calls its own algorithm performs.
        effective_policy.pop("model_response_limit", None)
        effective_policy["max_turns"] = positive_int(
            os.environ.get("HARNESS_BFCL_AGENT_TURNS", 6), "HARNESS_BFCL_AGENT_TURNS"
        )
        # Magentic-One owns an orchestrator loop above its participant turns. A
        # proposal acknowledgement cannot satisfy its environment-state ledger,
        # so bound that outer loop independently from each participant's guard.
        effective_policy["magentic_max_rounds"] = positive_int(
            os.environ.get("HARNESS_BFCL_MAGENTIC_ROUNDS", 3),
            "HARNESS_BFCL_MAGENTIC_ROUNDS",
        )
    if benchmark == "trajectory-bench":
        safe = list(bridge.metadata.get("safe_for_prelaunch") or [])
        effective_policy["speculation_safe_tools"] = safe
        effective_policy["branch_safe_tools"] = safe
    tool_capable = profile.tool_contract != "no-external-tools"
    task_messages = list(bridge.metadata.get("messages") or []) if benchmark == "bfcl" else []
    client = completion_client_from_env()
    context = RunContext(
        profile_id,
        render_bfcl_method_prompt(task_messages) if benchmark == "bfcl" else bridge.prompt,
        client,
        environment,
        trace,
        effective_policy,
        speculator_client=(
            sa_speculator_client_from_env(client) if profile_id == "sa" else None
        ),
        task_messages=task_messages or None,
    )
    _write(job / "bridge_manifest.json", {
        "benchmark": benchmark,
        "case_id": case_id,
        "profile": profile_id,
        "tool_schemas": [
            tool.native_schema() if benchmark == "bfcl" else tool.prompt_schema()
            for tool in bridge.tools
        ],
        "metadata": bridge.metadata,
    })
    started = time.perf_counter()
    method_output: str | None = None
    try:
        if benchmark == "bfcl":
            # Every profile produces its own final response. Publication below is a
            # deterministic parse/commit step and never calls another model.
            method_output = await run_profile(context)
            proposal_calls = list(environment.proposal_calls)
            answer = await publish_method_declaration(
                context,
                method_output=method_output,
                proposal_calls=proposal_calls,
                tool_capable=tool_capable,
            )
        else:
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
            "tool_calls": len(environment.calls),
            **context.usage_metrics(),
            "bridge": bridge.metadata,
            "policy": effective_policy,
        }
        if benchmark == "bfcl":
            result["method_output"] = method_output
            result["committed_calls"] = environment.committed_calls
            result["tool_calls"] = len(result["committed_calls"])
        elif benchmark == "trajectory-bench":
            result["trajectory_calls"] = _published_calls(environment)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = {
            "schema_version": 1,
            "status": "failed",
            "benchmark": benchmark,
            "case_id": case_id,
            "profile": profile.id,
            "execution_seconds": time.perf_counter() - started,
            "error": error,
            "tool_calls": len(environment.committed_calls) if benchmark == "bfcl" else len(environment.calls),
            **context.usage_metrics(),
            "failure_kind": (
                "provider_error" if isinstance(exc, ProviderError) else "agent_runtime"
            ),
        }
        if isinstance(exc, ProviderError):
            result["provider_error_kind"] = exc.kind
            result["provider_status_code"] = exc.status_code
        if benchmark == "trajectory-bench":
            result["trajectory_calls"] = _published_calls(environment)
        await trace.emit("bridge_error", error=result["error"])
    if benchmark == "bfcl":
        proposals = list(environment.proposal_calls)
        result["method_output"] = method_output
        result["committed_calls"] = environment.committed_calls
        result["declaration_protocol"] = PUBLISHER_PROTOCOL
        result["committed_response_id"] = environment.declaration_response_id
        result["external_assistant_responses"] = int(environment.declaration_committed)
        result["publisher_llm_calls"] = 0
        result["internal_llm_calls"] = int(result.get("llm_calls") or 0)
        result["source_response_ids"] = (
            list(context.declaration_output.get("source_response_ids") or [])
            if context.declaration_output is not None
            else ([environment.declaration_response_id] if environment.declaration_response_id is not None else [])
        )
        result["declaration_output_protocol"] = (
            context.declaration_output.get("protocol")
            if context.declaration_output is not None
            else "bfcl-text-only-empty-v1"
        )
        result["publication_tool_capable"] = tool_capable
        result["proposal_calls"] = [
            {
                "name": str(item.get("name") or ""),
                "arguments": item.get("arguments") or {},
                "assistant_response_id": item.get("assistant_response_id"),
            }
            for item in proposals
        ]
        result["proposal_tool_calls"] = len(proposals)
        result["proposal_response_ids"] = sorted({
            item["assistant_response_id"] for item in proposals
            if item.get("assistant_response_id") is not None
        })
        result["tool_calls"] = len(result["committed_calls"])
        result["environment_calls"] = 0
    result["policy"] = effective_policy
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
