from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from .core import RunContext


ABILITY_NAMES = (
    "reasoning",
    "mathematical",
    "language",
    "knowledge",
    "sequence",
    "spatial",
    "inference",
    "coding",
    "retrieval",
    "tool_use",
    "synthesis",
)


@dataclass(frozen=True)
class DmasAgent:
    id: str
    abilities: dict[str, float]


def _bounded_score(value: Any, *, field: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return score


def _default_agents(count: int) -> list[DmasAgent]:
    if count < 2:
        raise ValueError("policy.dmas_agent_count must be at least 2")
    agents: list[DmasAgent] = []
    for index in range(count):
        abilities = {name: 0.6 for name in ABILITY_NAMES}
        if index == 2:
            abilities["coding"] = 1.0
        if index in {4, 7}:
            abilities["mathematical"] = 1.0
        agents.append(DmasAgent(str(index), abilities))
    return agents


def _agents_from_policy(policy: dict[str, Any]) -> list[DmasAgent]:
    configured = policy.get("dmas_agents")
    if configured is None:
        return _default_agents(int(policy.get("dmas_agent_count", 10)))
    if not isinstance(configured, list) or len(configured) < 2:
        raise ValueError("policy.dmas_agents must contain at least two agents")
    agents: list[DmasAgent] = []
    seen: set[str] = set()
    for index, item in enumerate(configured):
        if not isinstance(item, dict):
            raise ValueError(f"policy.dmas_agents[{index}] must be an object")
        agent_id = str(item.get("id", index))
        if agent_id in seen:
            raise ValueError(f"Duplicate DMAS agent id: {agent_id}")
        raw_abilities = item.get("abilities")
        if not isinstance(raw_abilities, dict):
            raise ValueError(f"policy.dmas_agents[{index}].abilities must be an object")
        abilities = {
            name: _bounded_score(raw_abilities.get(name, 0.6), field=f"agent {agent_id} ability {name}")
            for name in ABILITY_NAMES
        }
        agents.append(DmasAgent(agent_id, abilities))
        seen.add(agent_id)
    return agents


def _capability_score(agent: DmasAgent, requirements: dict[str, float]) -> float:
    total = sum(requirements.values())
    if total == 0:
        return sum(agent.abilities.values()) / len(agent.abilities)
    return sum(agent.abilities[name] * weight for name, weight in requirements.items()) / total


def _select_agent(
    agents: list[DmasAgent],
    requirements: dict[str, float],
    *,
    excluded: set[str],
    rng: random.Random,
) -> DmasAgent | None:
    candidates = [agent for agent in agents if agent.id not in excluded]
    if not candidates:
        return None
    best_score = max(_capability_score(agent, requirements) for agent in candidates)
    best = [agent for agent in candidates if _capability_score(agent, requirements) == best_score]
    return rng.choice(best)


def _agent_summary(agents: list[DmasAgent], *, exclude: str | None = None) -> list[dict[str, Any]]:
    return [
        {"id": agent.id, "abilities": agent.abilities, "current_load": 0, "success_rate": None}
        for agent in agents
        if agent.id != exclude
    ]


async def _task_requirements(ctx: RunContext) -> dict[str, float]:
    mapped = await ctx.complete_json(
        "dmas_capability_mapper",
        [
            {
                "role": "user",
                "content": (
                    "Map the task to capability requirements without solving it. Each score must be between 0 and 1.\n"
                    f"Capabilities: {json.dumps(ABILITY_NAMES)}\n"
                    f'Task: {ctx.prompt}\nReturn JSON only: {{"requirements":{{"reasoning":0.0}}}}'
                ),
            }
        ],
    )
    raw = mapped.get("requirements")
    if not isinstance(raw, dict):
        raise ValueError("DMAS capability mapper omitted requirements")
    requirements = {
        name: _bounded_score(raw.get(name, 0.0), field=f"DMAS requirement {name}")
        for name in ABILITY_NAMES
    }
    if not any(requirements.values()):
        raise ValueError("DMAS capability mapper returned no positive requirement")
    return requirements


async def _route(
    ctx: RunContext,
    agent: DmasAgent,
    agents: list[DmasAgent],
    *,
    current_task: str,
    progress: list[dict[str, str]],
    allow_forward: bool,
) -> dict[str, Any]:
    decisions = "split/forward/execute" if allow_forward else "split/execute"
    return await ctx.complete_json(
        f"dmas_router_{agent.id}",
        [
            {
                "role": "system",
                "content": (
                    "You are the independent router inside one agent in a decentralized multi-agent system. "
                    "There is no central coordinator. Decide from local capability, completed results, and connected "
                    "neighbors whether to forward the unchanged task, split and execute a suitable part locally, or "
                    "execute the remaining task entirely."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Major task: {ctx.prompt}\nCurrent task: {current_task}\n"
                    f"Completed subtasks and results: {json.dumps(progress, ensure_ascii=False)}\n"
                    f"Current agent: {json.dumps({'id': agent.id, 'abilities': agent.abilities}, ensure_ascii=False)}\n"
                    f"Connected agents: {json.dumps(_agent_summary(agents, exclude=agent.id), ensure_ascii=False)}\n"
                    "Forwarding must not modify the task. Splitting must identify both the locally executable part and "
                    "the uncompleted remainder. Only completed results, never decomposition reasoning, will be passed.\n"
                    f'Return JSON only: {{"decision":"{decisions}","reason":"...",'
                    '"next_agent_id":"id or null","executable":"local subtask or null",'
                    '"remaining":"remaining subtask or null",'
                    '"description":"local execution guidance or null"}}'
                ),
            },
        ],
    )


async def _reason(
    ctx: RunContext,
    agent: DmasAgent,
    *,
    current_task: str,
    progress: list[dict[str, str]],
    guidance: str,
) -> str:
    return await ctx.complete(
        f"dmas_executor_reason_{agent.id}",
        [
            {
                "role": "user",
                "content": (
                    "Reason concisely about the current executor task without claiming any tool result or returning "
                    "the final answer.\n"
                    f"Major task: {ctx.prompt}\nCurrent executor task: {current_task}\n"
                    f"Completed subtasks and results: {json.dumps(progress, ensure_ascii=False)}\n"
                    f"Local router guidance: {guidance or 'None'}\n"
                    f"Agent abilities: {json.dumps(agent.abilities, ensure_ascii=False)}"
                ),
            }
        ],
    )


async def _execute(
    ctx: RunContext,
    agent: DmasAgent,
    *,
    current_task: str,
    progress: list[dict[str, str]],
    guidance: str = "",
) -> str:
    from .methods import _json_tool_loop

    thought = await _reason(
        ctx,
        agent,
        current_task=current_task,
        progress=progress,
        guidance=guidance,
    )
    return await _json_tool_loop(
        ctx,
        f"dmas_executor_{agent.id}",
        prompt=(
            f"Major task: {ctx.prompt}\nCurrent executor task: {current_task}\n"
            f"Completed subtasks and results: {json.dumps(progress, ensure_ascii=False)}\n"
            f"Local router guidance: {guidance or 'None'}\n"
            f"Executor reasoning: {thought}\n"
            "Complete the current executor task using available tools as needed. Return its complete result as final."
        ),
    )


async def _route_after_split(
    ctx: RunContext,
    agent: DmasAgent,
    agents: list[DmasAgent],
    *,
    remaining: str,
    progress: list[dict[str, str]],
) -> dict[str, Any]:
    return await ctx.complete_json(
        f"dmas_router_next_{agent.id}",
        [
            {
                "role": "user",
                "content": (
                    "Decide locally whether the major task is complete after your subtask, or route the remaining task "
                    "to one connected agent. There is no central coordinator.\n"
                    f"Major task: {ctx.prompt}\nRemaining task: {remaining}\n"
                    f"Completed subtasks and results: {json.dumps(progress, ensure_ascii=False)}\n"
                    f"Connected agents: {json.dumps(_agent_summary(agents, exclude=agent.id), ensure_ascii=False)}\n"
                    'Return JSON only: {"status":"completed/incompleted","reason":"...",'
                    '"next_agent_id":"id or null","remaining":"remaining task or null"}'
                ),
            }
        ],
    )


async def run_dmas(ctx: RunContext) -> str:
    """AgentNet-aligned cold-start decentralized router/executor task chain."""
    agents = _agents_from_policy(ctx.policy)
    by_id = {agent.id: agent for agent in agents}
    requirements = await _task_requirements(ctx)
    rng = random.Random(int(ctx.policy.get("dmas_seed", 0)))
    current = _select_agent(agents, requirements, excluded=set(), rng=rng)
    if current is None:
        raise RuntimeError("DMAS has no entry agent")

    max_forward = int(ctx.policy.get("dmas_forward_path_max_length", 3))
    max_executions = int(ctx.policy.get("dmas_max_execution_times", 30))
    if max_forward < 0:
        raise ValueError("policy.dmas_forward_path_max_length must be non-negative")
    if max_executions < 1:
        raise ValueError("policy.dmas_max_execution_times must be positive")

    current_task = ctx.prompt
    progress: list[dict[str, str]] = []
    visited: set[str] = set()
    forward_count = 0
    execution_count = 0
    await ctx.trace.emit(
        "dmas_start",
        entry_agent=current.id,
        agent_count=len(agents),
        graph="complete_directed",
        requirements=requirements,
        mode="cold_start_no_cross_case_memory",
    )

    while execution_count < max_executions:
        visited.add(current.id)
        available = [agent for agent in agents if agent.id not in visited]
        allow_forward = forward_count < max_forward and bool(available)
        route = await _route(
            ctx,
            current,
            agents,
            current_task=current_task,
            progress=progress,
            allow_forward=allow_forward,
        )
        decision = str(route.get("decision", "")).lower().strip()
        await ctx.trace.emit(
            "dmas_route",
            agent_id=current.id,
            decision=decision,
            current_task=current_task,
            visited=sorted(visited),
        )

        if decision == "forward":
            if not allow_forward:
                raise ValueError("DMAS router selected forward after the DAG forwarding boundary")
            requested = str(route.get("next_agent_id", ""))
            next_agent = by_id.get(requested)
            if next_agent is None or next_agent.id in visited:
                next_agent = _select_agent(agents, requirements, excluded=visited, rng=rng)
            if next_agent is None:
                raise RuntimeError("DMAS forwarding path has no unvisited neighbor")
            current = next_agent
            forward_count += 1
            continue

        if decision == "execute":
            execution_count += 1
            guidance = str(route.get("description") or "")
            return await _execute(
                ctx,
                current,
                current_task=current_task,
                progress=progress,
                guidance=guidance,
            )

        if decision != "split":
            raise ValueError(f"DMAS router returned unsupported decision: {decision or '<empty>'}")
        executable = route.get("executable")
        remaining = route.get("remaining")
        if not isinstance(executable, str) or not executable.strip():
            raise ValueError("DMAS split decision omitted executable subtask")
        if not isinstance(remaining, str) or not remaining.strip():
            raise ValueError("DMAS split decision omitted remaining subtask")

        execution_count += 1
        result = await _execute(ctx, current, current_task=executable.strip(), progress=progress)
        progress.append({"agent_id": current.id, "subtask": executable.strip(), "result": result})
        await ctx.trace.emit("dmas_progress", agent_id=current.id, subtask=executable.strip(), result=result)

        handoff = await _route_after_split(
            ctx,
            current,
            agents,
            remaining=remaining.strip(),
            progress=progress,
        )
        status = str(handoff.get("status", "")).lower().strip()
        if status == "completed":
            return result
        if status != "incompleted":
            raise ValueError("DMAS post-split router must return completed or incompleted")
        next_remaining = handoff.get("remaining")
        if not isinstance(next_remaining, str) or not next_remaining.strip():
            raise ValueError("DMAS post-split router omitted remaining task")
        requested = str(handoff.get("next_agent_id", ""))
        next_agent = by_id.get(requested)
        if next_agent is None or next_agent.id == current.id:
            next_agent = _select_agent(agents, requirements, excluded={current.id}, rng=rng)
        if next_agent is None:
            raise RuntimeError("DMAS post-split handoff has no connected agent")
        current = next_agent
        current_task = next_remaining.strip()
        visited = set()
        forward_count = 0

    raise RuntimeError("DMAS execution budget exhausted")
