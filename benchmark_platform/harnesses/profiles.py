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
    notes: str


PROFILES = (
    HarnessProfile(
        id="actor-only",
        name="Actor-only control",
        topology="single authoritative JSON tool loop",
        provenance="local-control",
        source=None,
        revision=None,
        notes="A non-speculative single-agent control with the same dynamic tool transport.",
    ),
    HarnessProfile(
        id="react",
        name="ReAct",
        topology="Thought -> Action -> Observation serial loop",
        provenance="protocol-reproduction",
        source="https://github.com/ysymyth/ReAct",
        revision="6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9",
        notes="Reproduces the published interleaved reasoning/action protocol with runtime tool schemas.",
    ),
    HarnessProfile(
        id="plan-execute",
        name="Plan-and-Execute",
        topology="planner -> sequential executor -> final response",
        provenance="protocol-reproduction",
        source="https://blog.langchain.com/plan-and-execute-agents/",
        revision=None,
        notes="Separates planning from deterministic execution of the emitted tool plan.",
    ),
    HarnessProfile(
        id="cmws",
        name="Central manager-worker swarm",
        topology="manager -> parallel worker wave -> manager synthesis",
        provenance="protocol-reproduction",
        source=None,
        revision=None,
        notes="A conventional centralized MAS control; independent assignments execute concurrently.",
    ),
)

_BY_ID = {profile.id: profile for profile in PROFILES}


def get_profile(profile_id: str) -> HarnessProfile:
    try:
        return _BY_ID[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown harness profile: {profile_id}") from exc
