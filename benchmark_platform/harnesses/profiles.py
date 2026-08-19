from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessProfile:
    id: str
    name: str
    topology: str
    provenance: str
    source: str | None
    revision: str | None
    tool_contract: str
    notes: str


PROFILES = (
    HarnessProfile(
        id="actor-only",
        name="Actor-only control",
        topology="single authoritative JSON tool loop",
        provenance="local-control",
        source=None,
        revision=None,
        tool_contract="dynamic",
        notes="A non-speculative single-agent control with the same dynamic tool transport.",
    ),
    HarnessProfile(
        id="react",
        name="ReAct",
        topology="Thought -> Action -> Observation serial loop",
        provenance="protocol-reproduction",
        source="https://github.com/ysymyth/ReAct",
        revision="6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9",
        tool_contract="dynamic",
        notes="Reproduces the published interleaved reasoning/action protocol with runtime tool schemas.",
    ),
    HarnessProfile(
        id="plan-execute",
        name="Plan-and-Execute",
        topology="planner -> sequential executor -> final response",
        provenance="protocol-reproduction",
        source="https://blog.langchain.com/plan-and-execute-agents/",
        revision=None,
        tool_contract="dynamic",
        notes="The planner emits textual steps; a full tool-using executor resolves each step in order.",
    ),
    HarnessProfile(
        id="cmws",
        name="Central manager-worker swarm",
        topology="manager -> parallel worker wave -> manager synthesis",
        provenance="protocol-reproduction",
        source=None,
        revision=None,
        tool_contract="dynamic",
        notes="The manager emits textual assignments; independent tool-using workers execute concurrently.",
    ),
    HarnessProfile(
        id="lats",
        name="Language Agent Tree Search",
        topology="MCTS selection -> sampled action expansion -> value/rollout -> reflection/backpropagation",
        provenance="protocol-reproduction",
        source="https://github.com/lapisrocks/LanguageAgentTreeSearch",
        revision="853d81614607dd27433faf17c7b0a7d660f95d22",
        tool_contract="dynamic-branch-isolated",
        notes="Faithful branching requires read-only tools or benchmark-provided environment snapshots.",
    ),
    HarnessProfile(
        id="memgpt",
        name="MemGPT",
        topology="LLM processor -> memory/function executor -> heartbeat queue",
        provenance="protocol-reproduction",
        source="https://github.com/cpacker/MemGPT",
        revision="4f93faff35e9ac1f7d6050a498a7e9a11e66296c",
        tool_contract="dynamic-virtual-memory",
        notes="Preserves core, recall, and archival memory functions plus chained heartbeat execution.",
    ),
    HarnessProfile(
        id="aflow",
        name="AFlow frozen workflow",
        topology="frozen optimized operator graph -> execution",
        provenance="protocol-reproduction",
        source="https://github.com/FoundationAgents/AFlow",
        revision="3f457218fc716093fe53f6df8a5d5e6379d66346",
        tool_contract="dynamic-frozen-workflow",
        notes="Evaluation executes a frozen operator graph; workflow optimization must use a separate training split.",
    ),
    HarnessProfile(
        id="dylan",
        name="DyLAN",
        topology="dynamic text-agent network -> consensus/pruning",
        provenance="protocol-reproduction",
        source="https://github.com/SALT-NLP/DyLAN",
        revision="006e440a519f7cf21e2826f3b8033d84ae9bf07c",
        tool_contract="no-external-tools",
        notes="The published topology has no external tool loop; tool-dependent failures are part of the baseline.",
    ),
    HarnessProfile(
        id="magentic-one",
        name="Magentic-One",
        topology="orchestrator ledgers -> specialist turns -> synthesis",
        provenance="protocol-reproduction",
        source="https://github.com/microsoft/autogen",
        revision="bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0",
        tool_contract="dynamic",
        notes="Preserves the ledger, selected-speaker, stall, and replan topology with dynamic benchmark tools.",
    ),
    HarnessProfile(
        id="multi-persona",
        name="Multi-Persona self-collaboration",
        topology="single-model dynamically selected personas",
        provenance="protocol-reproduction",
        source="https://github.com/MikeWangWZHL/Solo-Performance-Prompting",
        revision="619c8a0ff4205bfd39e33f0867647b40e1703b94",
        tool_contract="no-external-tools",
        notes="The published topology has no external tool loop; tool-dependent failures are part of the baseline.",
    ),
    HarnessProfile(
        id="llmcompiler",
        name="LLMCompiler",
        topology="planner DAG -> parallel scheduler -> joiner",
        provenance="protocol-reproduction",
        source="https://github.com/SqueezeAILab/LLMCompiler",
        revision="a00c9d35507507da70e8c637eee64efc8c1857ae",
        tool_contract="dynamic",
        notes="Executes dependency-ready benchmark tool calls concurrently.",
    ),
    HarnessProfile(
        id="rewoo",
        name="ReWOO",
        topology="planner -> evidence workers -> solver",
        provenance="protocol-reproduction",
        source="https://github.com/billxbf/ReWOO",
        revision="9cd0283043ff4be0c9d614fda2789d143ca6ffd1",
        tool_contract="dynamic",
        notes="Plans evidence calls before executing the benchmark toolset.",
    ),
    HarnessProfile(
        id="sa",
        name="Speculative Actions",
        topology="response predictor -> safe pre-actions -> authoritative Actor",
        provenance="protocol-reproduction",
        source="https://github.com/naimengye/speculative-action",
        revision="dc938b9ef7474caf07fe4ad16549c1fa8c7d268c",
        tool_contract="dynamic-read-only-speculation",
        notes="Only benchmark-declared parallel read-only tools may execute speculatively.",
    ),
)

_BY_ID = {profile.id: profile for profile in PROFILES}


def get_profile(profile_id: str) -> HarnessProfile:
    try:
        return _BY_ID[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown harness profile: {profile_id}") from exc
