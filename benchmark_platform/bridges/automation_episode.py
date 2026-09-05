"""AutomationBench 1.0.6 API-tool episodes with post-agent native assertion scoring.

The same official world, service restrictions and API toolset used by the
standalone product integration are exposed to the baseline tool transport.
Assertions and initial state stay in the controller, never in the agent prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import ProviderError, completion_client_from_env, sa_speculator_client_from_env
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import get_profile


SOURCE_REVISION = "4a8e1061254004d9dac807054eed33fad7d1ff14"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    pending.replace(path)


def public_prompt(row: dict) -> str:
    return "\n\n".join(
        f"[{str(message.get('role') or 'message').upper()}]\n{message.get('content') or ''}"
        for message in row.get("prompt", [])
    )


def api_specs(functions: dict) -> list[ToolSpec]:
    parameters = {
        "api_search": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1}}, "required": ["query"]},
        "api_fetch": {"type": "object", "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
            "url": {"type": "string"}, "params": {"type": ["string", "null"]}, "body": {"type": ["string", "null"]},
        }, "required": ["method", "url"]},
        "base64_encode": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }
    return [ToolSpec(name, inspect.getdoc(functions[name]) or name, schema, (),
                     parallel=name != "api_fetch", read_only=name != "api_fetch")
            for name, schema in parameters.items()]


class AutomationEpisode:
    def __init__(self, case_id: str):
        from automationbench.domains import PUBLIC_DOMAINS, get_domain_dataset
        from automationbench.runner import compute_allowed_services, strip_none_values
        from automationbench.schema.world import WorldState
        from automationbench.task_contract import task_contract_sha256
        from automationbench.tools.api import api_fetch, api_search, base64_encode

        domain, separator, example_id = case_id.partition(":")
        if not separator or domain not in PUBLIC_DOMAINS:
            raise ValueError("AutomationBench case must be PUBLIC_DOMAIN:EXAMPLE_ID")
        row = next((dict(item) for item in get_domain_dataset(domain) if str(item.get("example_id")) == example_id), None)
        if row is None:
            raise ValueError(f"Unknown AutomationBench case: {case_id}")
        raw_info = row.get("info") or {}
        if isinstance(raw_info, str):
            raw_info = json.loads(raw_info)
        self.contract = task_contract_sha256(example_id=row["example_id"], prompt=row["prompt"], info=raw_info)
        self.info = copy.deepcopy(raw_info)
        self.initial = strip_none_values(self.info.get("initial_state", {}))
        self.info["assertions"] = [strip_none_values(item) for item in self.info.get("assertions", [])]
        self.world = WorldState(**self.initial)
        self.world.meta.allowed_services = compute_allowed_services(self.initial, self.info["assertions"], self.info.get("zapier_tools", []))
        self.prompt = public_prompt(row)
        self.functions = {"api_search": api_search, "api_fetch": api_fetch, "base64_encode": base64_encode}
        self.tools = api_specs(self.functions)
        self.metadata = {"source_revision": SOURCE_REVISION, "toolset": "api", "task_contract_sha256": self.contract,
                         "domain": domain, "task_name": self.info.get("task_name"),
                         "safe_for_prelaunch": ["api_search", "base64_encode"]}

    def handlers(self):
        def bind(name):
            async def invoke(arguments):
                # Match upstream update_tool_args' empty-object sentinel. No
                # caller may provide the private WorldState argument.
                args = {key: value for key, value in arguments.items() if not (isinstance(value, dict) and not value)}
                if "world" in args:
                    raise ValueError("WorldState is controller-owned")
                if name == "api_fetch":
                    args["world"] = self.world
                return self.functions[name](**args)
            return invoke
        return {name: bind(name) for name in self.functions}

    def finalize(self):
        from automationbench.rubric import partial_credit, task_completed_correctly
        state = {"info": self.info, "world": self.world, "initial_state": copy.deepcopy(self.initial)}
        partial = partial_credit(state)
        strict = task_completed_correctly(state)
        return {"authority": "automationbench_official_assertions", "partial_credit": partial,
                "task_completed_correctly": strict, "assertion_results": state.get("_assertion_results", []),
                "end_state": state.get("_end_state")}


async def run_episode(profile_id: str, case_id: str, policy: dict, job: Path, *, episode=None, client=None):
    arm_started = time.monotonic()
    profile = get_profile(profile_id)
    job.mkdir(parents=True, exist_ok=True)
    episode = episode or AutomationEpisode(case_id)
    trace = JsonlTrace(job / "harness_trace.jsonl")
    environment = ToolEnvironment(episode.tools, trace, episode.handlers())
    client = client or completion_client_from_env()
    context = RunContext(profile_id, episode.prompt, client, environment, trace, policy,
                         speculator_client=sa_speculator_client_from_env(client) if profile_id == "sa" else None)
    write_json(job / "bridge_manifest.json", {"benchmark": "automationbench", "case_id": case_id,
               "prompt": episode.prompt, "tools": [tool.prompt_schema() for tool in episode.tools], "metadata": episode.metadata})
    result = {"schema_version": 1, "benchmark": "automationbench", "profile": profile.id, "case_id": case_id,
              "status": "completed", "native_score_status": "not_requested", "native_score": None,
              "bridge": episode.metadata, "policy": policy}
    started = time.monotonic()
    try:
        deadline = float(policy.get("automationbench_arm_timeout_s", os.environ.get("HARNESS_ARM_TIMEOUT_S", "890")))
        reserve = float(policy.get("automationbench_finalize_grace_s", 10))
        # Dataset materialization can be expensive. It must consume the same
        # outer arm budget, leaving time to persist the native assertion score
        # before the host watchdog removes the container.
        remaining = deadline - (time.monotonic() - arm_started) - reserve
        if remaining <= 0:
            raise TimeoutError
        result["final_answer"] = await asyncio.wait_for(run_profile(context), remaining)
    except ProviderError as error:
        result.update(status="failed", failure_kind="provider_error", error=str(error))
    except TimeoutError:
        result.update(status="failed", failure_kind="agent_timeout", error="AutomationBench arm deadline exceeded")
    except Exception as error:
        result.update(status="failed", failure_kind="agent_runtime", error=f"{type(error).__name__}: {error}")
    # Only the native post-agent scorer can read assertions. A lost provider
    # measurement retains evidence, but must not acquire a numeric run score.
    try:
        evaluation = episode.finalize()
        write_json(job / "official_score.json", evaluation)
        result["verifier_details"] = evaluation
        result["native_score_status"] = "completed"
        if result.get("failure_kind") != "provider_error":
            result["native_score"] = evaluation["task_completed_correctly"]
            result["native_partial_credit"] = evaluation["partial_credit"]
    except Exception as error:
        result.update(status="failed", failure_kind="verifier_error", native_score_status="infra_failed",
                      error=f"Native assertion scorer failed: {type(error).__name__}: {error}")
    result.update(execution_seconds=time.monotonic() - started, tool_calls=len(environment.calls), **context.usage_metrics())
    result.update(setup_seconds=started - arm_started, total_episode_seconds=time.monotonic() - arm_started)
    write_json(job / "harness_result.json", result)
    write_json(job / "payload.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--job", type=Path, default=Path("/job"))
    args = parser.parse_args()
    try:
        result = asyncio.run(run_episode(args.profile, args.case, json.loads(args.policy), args.job))
    except Exception as error:
        result = {"status": "failed", "benchmark": "automationbench", "case_id": args.case, "profile": args.profile,
                  "failure_kind": "bridge_error", "error": f"{type(error).__name__}: {error}", "native_score": None}
        write_json(args.job / "harness_result.json", result)
        write_json(args.job / "payload.json", result)
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
