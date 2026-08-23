from __future__ import annotations

from typing import Any, Iterable

from .catalog import Benchmark
from .harnesses.profiles import HarnessProfile


BRIDGE_CAPABILITIES = {
    "gaia": ("single-turn-workspace", "implemented"),
    "gdpval": ("single-turn-artifact-workspace", "implemented"),
    "trajectory-bench": ("single-turn-remote-tools", "implemented_external_tool_service_and_host_scorer"),
    "bfcl": ("single-turn-function-declarations", "implemented_single_turn_scored_stateful_categories_pending"),
    "vitabench": ("native-conversation", "implemented_native_episode"),
    "tau2": ("native-conversation", "implemented_native_episode"),
    "terminal-bench-2": ("task-container", "implemented_task_container_bridge"),
    "swe-bench-verified": ("task-container", "implemented_task_container_bridge_configured_case"),
    "osworld": ("external-vm", "blocked_external_runtime"),
}


def compatibility_rows(
    profiles: Iterable[HarnessProfile], benchmarks: Iterable[Benchmark]
) -> list[dict[str, Any]]:
    rows = []
    for profile in profiles:
        for benchmark in benchmarks:
            lifecycle, bridge_status = BRIDGE_CAPABILITIES.get(
                benchmark.id, (benchmark.adapter["kind"], "blocked_no_baseline_bridge")
            )
            if profile.id == "aflow-custom-init":
                baseline_requirement = "disclosed_unoptimized_custom_initialization_control"
            elif profile.id == "dmas":
                baseline_requirement = "agentnet_aligned_cold_start_without_cross_case_memory"
            elif profile.id == "lats":
                baseline_requirement = "branch_snapshot_or_all_tools_read_only"
            elif profile.tool_contract == "no-external-tools":
                baseline_requirement = "published_method_has_no_external_tool_loop"
            else:
                baseline_requirement = "dynamic_tool_schema"
            runnable = bridge_status.startswith("implemented")
            if profile.id == "lats":
                runnable = runnable and benchmark.id == "bfcl"
            if profile.tool_contract == "no-external-tools" and benchmark.id == "gdpval":
                # GDPVal grades files created in the writable attempt workspace. The
                # published DyLAN and Multi-Persona profiles only exchange text between
                # LLM roles, so they cannot submit a deliverable. Structural zeros would
                # mislabel an inapplicable lifecycle as weak task performance.
                runnable = False
                baseline_requirement = "gdpval_requires_workspace_artifact_tools"
            rows.append(
                {
                    "baseline": profile.id,
                    "benchmark": benchmark.id,
                    "benchmark_lifecycle": lifecycle,
                    "bridge_status": bridge_status,
                    "tool_contract": profile.tool_contract,
                    "baseline_requirement": baseline_requirement,
                    "runnable": runnable,
                    "publishable_score": False,
                }
            )
    return rows
