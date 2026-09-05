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
        topology="planner -> sequential executor steps -> last step response",
        provenance="protocol-reproduction",
        source=(
            "https://github.com/langchain-ai/langchain/tree/"
            "0207dc1431c29379b724f51c09fa49e6b0333639/libs/experimental/"
            "langchain_experimental/plan_and_execute"
        ),
        revision="0207dc1431c29379b724f51c09fa49e6b0333639",
        tool_contract="dynamic",
        notes=(
            "The planner emits minimal textual steps; each tool-using executor receives previous steps and its "
            "current objective, and the final step response is returned."
        ),
    ),
    HarnessProfile(
        id="cmas",
        name="Centralized multi-agent system",
        topology="manager -> parallel worker wave -> manager synthesis",
        provenance="local-control",
        source=None,
        revision=None,
        tool_contract="dynamic",
        notes=(
            "A local centralized control: the manager emits textual assignments, and independent tool-using "
            "workers receive only their assigned subtask before manager synthesis."
        ),
    ),
    HarnessProfile(
        id="dmas",
        name="Decentralized multi-agent system",
        topology="capability match -> local router DAG -> local executor/split -> peer handoff",
        provenance="protocol-reproduction",
        source="https://github.com/zoe-yyx/AgentNet",
        revision="325d39f2a940be5fa903d28c411bd3426b8007f5",
        tool_contract="dynamic-decentralized-dag",
        notes=(
            "Reproduces AgentNet's evaluation-time router/executor, forward/split/execute, result-only handoff, "
            "and acyclic forwarding protocol. The default is a disclosed cold start without cross-case RAG memory."
        ),
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
        revision="134df8f7ea68d4dd07a9f9d6cdac6b0c46c12ff3",
        tool_contract="dynamic-virtual-memory",
        notes="Preserves core, recall, and archival memory functions plus chained heartbeat execution.",
    ),
    HarnessProfile(
        id="aflow-custom-init",
        name="AFlow Custom initialization control",
        topology="unoptimized AFlow round-1 Custom operator -> one text response",
        provenance="local-control",
        source="https://github.com/FoundationAgents/AFlow",
        revision="3f457218fc716093fe53f6df8a5d5e6379d66346",
        tool_contract="no-external-tools",
        notes=(
            "Executes AFlow's documented round-1 Custom as one plain-text generation without external tools. It is not an "
            "optimized AFlow workflow; a canonical AFlow arm requires a frozen graph produced "
            "on a disjoint optimization split."
        ),
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
        topology="official task/progress ledgers -> one selected participant response -> synthesis",
        provenance="protocol-reproduction",
        source="https://github.com/microsoft/autogen",
        revision="bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0",
        tool_contract="dynamic",
        notes=(
            "Pins AutoGen's task/progress prompts, validated selected-speaker, stall/replan, "
            "one-response participant boundary and max-round finalization. Every participant "
            "receives the complete dynamic toolset selected by the benchmark bridge; these "
            "tools replace the upstream browser/file/terminal implementations."
        ),
    ),
    HarnessProfile(
        id="multi-persona",
        name="Multi-Persona self-collaboration",
        topology="single-model dynamically selected personas",
        provenance="protocol-reproduction",
        source="https://github.com/MikeWangWZHL/Solo-Performance-Prompting",
        revision="619c8a0ff4205bfd39e33f0867647b40e1703b94",
        tool_contract="no-external-tools",
        notes=(
            "Uses the source SPP profile protocol with two complete benchmark-neutral demonstrations, "
            "dynamic participant profiles, multi-round criticism, revision, and a delimited final answer. "
            "The published topology has no external tool loop."
        ),
    ),
    HarnessProfile(
        id="llmcompiler",
        name="LLMCompiler",
        topology="planner DAG -> parallel scheduler -> joiner",
        provenance="protocol-reproduction",
        source="https://github.com/SqueezeAILab/LLMCompiler",
        revision="a00c9d35507507da70e8c637eee64efc8c1857ae",
        tool_contract="dynamic",
        notes=(
            "Executes dependency-ready benchmark tool calls concurrently. The upstream max_replans "
            "parameter counts total planning passes (default one), and its final Joiner cannot request "
            "another pass."
        ),
    ),
    HarnessProfile(
        id="rewoo",
        name="ReWOO",
        topology="planner -> evidence workers -> solver",
        provenance="protocol-reproduction",
        source="https://github.com/billxbf/ReWOO",
        revision="9cd0283043ff4be0c9d614fda2789d143ca6ffd1",
        tool_contract="dynamic",
        notes=(
            "Uses the source Plan/#E protocol: the Planner fixes all calls first, explicit sequential Evidence "
            "Workers execute dynamic benchmark tools or the LLM worker with prior-evidence substitution, and the "
            "Solver receives the complete worker log."
        ),
    ),
    HarnessProfile(
        id="sa",
        name="Speculative Actions",
        topology="independent fast action predictor -> per-turn top-k safe pre-actions -> exact-match Actor commit",
        provenance="protocol-reproduction",
        source="https://github.com/naimengye/speculative-action",
        revision="dc938b9ef7474caf07fe4ad16549c1fa8c7d268c",
        tool_contract="dynamic-read-only-speculation",
        notes=(
            "Only benchmark-declared parallel read-only tools may execute speculatively. "
            "HARNESS_SA_MODEL explicitly selects the independent Speculator; endpoint and transport "
            "settings inherit from the Actor unless HARNESS_SA_* overrides them."
        ),
    ),
)

_BY_ID = {profile.id: profile for profile in PROFILES}


def get_profile(profile_id: str) -> HarnessProfile:
    try:
        return _BY_ID[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown harness profile: {profile_id}") from exc
