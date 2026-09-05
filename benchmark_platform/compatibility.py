from __future__ import annotations

from typing import Any, Iterable

from .catalog import Benchmark
from .harnesses.profiles import HarnessProfile
from .harnesses.declaration import SINGLE_TURN_PROFILES


BRIDGE_CAPABILITIES = {
    "automationbench": ("native-stateful-workflow", "implemented_native_assertion_episode"),
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
            if profile.id == "aflow":
                baseline_requirement = "frozen_workflow_from_disjoint_optimization_split"
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
            if benchmark.id == "bfcl" and profile.id not in SINGLE_TURN_PROFILES:
                runnable = False
                baseline_requirement = "requires_multi_response_agent_protocol"
            elif profile.id == "magentic-one" and lifecycle not in {
                "single-turn-workspace", "single-turn-artifact-workspace", "task-container"
            }:
                runnable = False
                baseline_requirement = "magentic_requires_workspace_code_execution"
            if profile.tool_contract == "no-external-tools" and benchmark.id == "gdpval":
                # GDPVal grades files created in the writable attempt workspace. The
                # published DyLAN and Multi-Persona profiles only exchange text between
                # LLM roles, so they cannot submit a deliverable. Structural zeros would
                # mislabel an inapplicable lifecycle as weak task performance.
                runnable = False
                baseline_requirement = "gdpval_requires_workspace_artifact_tools"
            if profile.tool_contract == "no-external-tools" and benchmark.id == "vitabench":
                # Checked against the suite, not assumed: all 60 light cases carry evaluation
                # criteria, and every one of them requires at least one order to be created --
                # none is scoreable by talking. A published text-only method has no tool loop,
                # so it cannot score at all here, and 60 structural zeros read as a weak method
                # rather than an inapplicable one.
                #
                # Deliberately not extended to the other native conversation, tau2: its light
                # suite contains at least one task with no evaluation criteria at all
                # (airline:9), where evaluate_simulation returns a flat 1.0 and a method that
                # called nothing scored full marks. Until that rate is measured there, the same
                # reasoning is not established for it.
                runnable = False
                baseline_requirement = "vitabench_rubrics_all_require_a_tool_mediated_order"
            if profile.id == "llmcompiler" and lifecycle == "native-conversation":
                # LLMCompiler's premise is planning one parallel DAG of calls up front, which a
                # conversation cannot supply: every user turn invalidates the plan, so each turn
                # costs a new compiled plan. Raising the source's total planning-pass limit to
                # cover a 14-32 turn episode turns the method into a very expensive
                # ReAct -- planner, scheduler and joiner calls every turn -- which is no longer
                # the published method. Inapplicable lifecycle, not a weak method.
                runnable = False
                baseline_requirement = "dag_planner_cannot_replan_per_conversation_turn"
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
