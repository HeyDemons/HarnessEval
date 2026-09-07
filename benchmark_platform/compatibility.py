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

# Native baseline support does not imply support in the Product HTTP server.
# Candidate PERSEUS runners list AutomationBench, but the shared product_server
# and adapters.load_case do not implement its controller-owned world yet.
PRODUCT_BRIDGE_BLOCKERS = {
    "automationbench": "automationbench_product_http_bridge_not_implemented",
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
            if profile.id in {"aflow", "aflow-tools"}:
                baseline_requirement = "frozen_workflow_from_disjoint_optimization_split"
            elif profile.id == "dylan":
                baseline_requirement = "published_optimized_teams_with_dynamic_tool_policy"
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
            if profile.tool_contract == "no-external-tools" and benchmark.id in {
                "automationbench", "terminal-bench-2"
            }:
                # Both grade world/container state after the agent exits and neither has a
                # conversational channel a text-only method could earn credit through.
                # AutomationBench: the official scorer was run over all 36 initialized light
                # worlds with no model and no environment action -- strict 0/36 and partial
                # 0/36 (reports/automationbench-baseline-audit-20260906/noop-suite.json).
                # Terminal-Bench-2: every task's reward comes from tests/test.sh executed
                # inside the container the agent never touched. Marking these eligible lets a
                # structurally inapplicable lifecycle be read as a weak method, and it was
                # only being avoided by hand-editing the campaign plan.
                runnable = False
                baseline_requirement = "benchmark_scores_only_post_agent_world_state"
            if profile.tool_contract == "no-external-tools" and benchmark.id == "vitabench":
                # Checked against the suite, not assumed: all 60 light cases carry evaluation
                # criteria, and every one of them requires at least one order to be created --
                # none is scoreable by talking. A published text-only method has no tool loop,
                # so it cannot score at all here, and 60 structural zeros read as a weak method
                # rather than an inapplicable one.
                #
                # Still not extended to the other native conversation, tau2, and the rate is
                # now measured rather than merely unknown: 7 of its 60 light cases require
                # zero actions (airline 10/31/46/34/0, retail 57/24) and are graded on
                # communication alone. All nine methods in the 2026-09-06 sweep scored 1.0 on
                # every one of them. tau2 also gives a text-only profile a real channel -- its
                # reply becomes an assistant turn and the hidden user answers -- so such a
                # method can earn credit there. It is a weak configuration on tau2, not an
                # inapplicable one, and gating it would hide a measurable result.
                runnable = False
                baseline_requirement = "vitabench_rubrics_all_require_a_tool_mediated_order"
            if profile.id == "llmcompiler" and lifecycle == "native-conversation":
                if benchmark.id == "tau2":
                    # tau_episode now treats a profile return as one assistant
                    # reply and starts a fresh broker on the next user message.
                    # Keep the original one-pass planner/scheduler/joiner inside
                    # each invocation; do not expand a replan budget to span an
                    # episode or reveal future user messages to its planner.
                    baseline_requirement = "turn_local_dag_with_visible_conversation"
                else:
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
                    "participation_status": ("not_participating" if profile.id == "lats" and not runnable
                                             else "eligible" if runnable else "incompatible"),
                    "publishable_score": False,
                }
            )
    return rows
