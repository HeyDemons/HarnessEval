from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Any

from .core import RunContext, extract_json


class _LatsBudgetExhausted(RuntimeError):
    pass


class _LatsBudget:
    """Bound LATS' own HTTP fan-out independently from case-level concurrency."""

    def __init__(self, ctx: RunContext, *, max_calls: int, max_parallel: int):
        self.ctx = ctx
        self.max_calls = max_calls
        self.used = 0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_parallel)

    async def reserve(self, count: int) -> None:
        """Atomically reserve a complete sampling wave before dispatching any request."""

        async with self._lock:
            if self.used + count > self.max_calls:
                raise _LatsBudgetExhausted(
                    f"LATS LLM-call budget exhausted ({self.used}/{self.max_calls}); "
                    f"next wave requires {count}"
                )
            self.used += count

    async def complete(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        reserved: bool = False,
    ) -> str:
        if not reserved:
            await self.reserve(1)
        async with self._semaphore:
            return await self.ctx.complete(
                role,
                messages,
                json_mode=json_mode,
                temperature=temperature,
            )

    async def complete_json(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        reserved: bool = False,
    ) -> dict[str, Any]:
        """Run the normal JSON repair protocol while charging every HTTP request."""

        conversation = list(messages)
        protocol_repairs = int(self.ctx.policy.get("protocol_repairs", 1))
        for attempt in range(protocol_repairs + 1):
            raw = await self.complete(
                role,
                conversation,
                json_mode=True,
                temperature=temperature,
                reserved=reserved and attempt == 0,
            )
            try:
                return extract_json(raw, expected_type=dict)
            except ValueError:
                if attempt >= protocol_repairs:
                    raise
                conversation.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Return one complete JSON object matching the requested schema. "
                                "Preserve every field and argument."
                            ),
                        },
                    ]
                )
        raise AssertionError("unreachable")


async def _gather_wave(*coroutines: Any) -> list[Any]:
    """Cancel and drain sibling requests if any member of a sampling wave fails."""

    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass
class _Node:
    parent: _Node | None
    thought: str = ""
    action: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    call_record: dict[str, Any] | None = None
    children: list[_Node] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    terminal: bool = False
    reward: float = 0.0
    answer: str | None = None
    exhausted: bool = False

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def uct(self) -> float:
        if self.visits == 0:
            return self.value
        parent_visits = self.parent.visits if self.parent is not None else 0
        if parent_visits <= 1:
            return self.value / self.visits
        return self.value / self.visits + math.sqrt(2 * math.log(parent_visits) / self.visits)

    def path(self) -> list[_Node]:
        nodes: list[_Node] = []
        current: _Node | None = self
        while current is not None:
            nodes.append(current)
            current = current.parent
        return list(reversed(nodes))

    def trajectory(self, task: str) -> str:
        records: list[dict[str, Any]] = []
        for node in self.path()[1:]:
            records.append(
                {
                    "depth": node.depth,
                    "thought": node.thought,
                    "action": node.action,
                    "observation": node.observation,
                }
            )
        return f"Task: {task}\nTrajectory: {json.dumps(records, ensure_ascii=False)}"


def _select_node(root: _Node) -> _Node | None:
    node: _Node | None = root
    while node is not None and node.children:
        candidates = [child for child in node.children if not child.terminal and not child.exhausted]
        if not candidates:
            node.exhausted = True
            node = node.parent
            continue
        node = max(candidates, key=lambda child: child.uct())
    if node is not None and (node.terminal or node.exhausted):
        return None
    return node


def _backpropagate(node: _Node, reward: float) -> None:
    current: _Node | None = node
    while current is not None:
        current.visits += 1
        if current.terminal and current.reward == 0:
            update = -1.0
        else:
            update = reward
        current.value = (current.value * (current.visits - 1) + update) / current.visits
        current = current.parent


def _candidate_key(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def _reflect(
    ctx: RunContext,
    budget: _LatsBudget,
    failures: list[str],
    temperature: float,
) -> str:
    return await budget.complete(
        "lats_reflection",
        [
            {
                "role": "user",
                "content": (
                    "Reflect on the failed trajectories and identify a concrete correction for the next search.\n"
                    f"Task: {ctx.prompt}\nFailed trajectories: {json.dumps(failures, ensure_ascii=False)}"
                ),
            }
        ],
        temperature=temperature,
    )


async def _propose(
    ctx: RunContext,
    budget: _LatsBudget,
    node: _Node,
    count: int,
    failures: list[str],
    reflections: list[str],
    temperature: float,
) -> list[dict[str, Any]]:
    prompt = (
        "Generate one next LATS thought/action candidate for this trajectory. Do not invent an observation.\n"
        f"Available tools: {ctx.environment.schema}\n"
        'Return JSON only: {"thought":"...","tool":"name","arguments":{}} or '
        '{"thought":"...","final":"answer"}.\n'
        f"{node.trajectory(ctx.prompt)}\n"
        f"Failed trajectories: {json.dumps(failures, ensure_ascii=False)}\n"
        f"Reflections: {json.dumps(reflections, ensure_ascii=False)}"
    )
    await budget.reserve(count)
    candidates = await _gather_wave(
        *(
            budget.complete_json(
                "lats_proposal",
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                reserved=True,
            )
            for _ in range(count)
        )
    )
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        thought = candidate.get("thought")
        if not isinstance(thought, str):
            raise ValueError("LATS proposal omitted thought")
        if "final" in candidate:
            normalized = {"thought": thought, "final": str(candidate["final"])}
        else:
            arguments = candidate.get("arguments")
            if not isinstance(arguments, dict) or candidate.get("tool") not in ctx.environment.names:
                raise ValueError("LATS proposal must contain a declared tool and object arguments")
            normalized = {
                "thought": thought,
                "tool": str(candidate["tool"]),
                "arguments": arguments,
            }
        unique.setdefault(_candidate_key(normalized), normalized)
    return list(unique.values())


async def _evaluate(
    ctx: RunContext,
    budget: _LatsBudget,
    node: _Node,
    samples: int,
    temperature: float,
    *,
    reserved: bool = False,
) -> tuple[float, bool, str]:
    prompt = (
        "Evaluate this LATS trajectory. Judge whether a final answer, if present, fully solves the task and score "
        "the trajectory's progress from 0 to 1.\n"
        'Return JSON only: {"score":0.0,"success":false,"feedback":"..."}.\n'
        f"{node.trajectory(ctx.prompt)}"
    )
    if not reserved:
        await budget.reserve(samples)
    evaluations = await _gather_wave(
        *(
            budget.complete_json(
                "lats_value",
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                reserved=True,
            )
            for _ in range(samples)
        )
    )
    scores: list[float] = []
    successes = 0
    feedback: list[str] = []
    for evaluation in evaluations:
        score = float(evaluation.get("score"))
        if not 0.0 <= score <= 1.0:
            raise ValueError("LATS value score must be between 0 and 1")
        scores.append(score)
        successes += evaluation.get("success") is True
        feedback.append(str(evaluation.get("feedback", "")))
    return sum(scores) / len(scores), successes > len(evaluations) / 2, "\n".join(feedback)


async def _expand(
    ctx: RunContext,
    budget: _LatsBudget,
    node: _Node,
    count: int,
    value_samples: int,
    failures: list[str],
    reflections: list[str],
    temperature: float,
) -> list[_Node]:
    candidates = await _propose(
        ctx, budget, node, count, failures, reflections, temperature
    )
    # Reserve the whole value-estimation wave before executing even the first
    # candidate.  Otherwise the budget could run out halfway through a sibling set,
    # leaving the tree biased toward whichever proposal happened to be listed first.
    await budget.reserve(len(candidates) * value_samples)
    children: list[_Node] = []
    for candidate in candidates:
        if "final" in candidate:
            child = _Node(
                parent=node,
                thought=candidate["thought"],
                action={"final": candidate["final"]},
                terminal=True,
                answer=candidate["final"],
            )
        else:
            action = {"tool": candidate["tool"], "arguments": candidate["arguments"]}
            observation, call_record = await ctx.environment.call_isolated(
                action["tool"], action["arguments"]
            )
            child = _Node(
                parent=node,
                thought=candidate["thought"],
                action=action,
                observation=observation,
                call_record=call_record,
            )
        child.value, success, feedback = await _evaluate(
            ctx,
            budget,
            child,
            value_samples,
            temperature,
            reserved=True,
        )
        if child.terminal:
            child.reward = 1.0 if success else 0.0
            if not success:
                failures.append(child.trajectory(ctx.prompt) + "\nEvaluator feedback: " + feedback)
        node.children.append(child)
        children.append(child)
        await ctx.trace.emit(
            "lats_node",
            depth=child.depth,
            action=child.action,
            observation=child.observation,
            value=child.value,
            reward=child.reward,
            terminal=child.terminal,
        )
    return children


async def _commit_path(ctx: RunContext, node: _Node) -> None:
    records = [item.call_record for item in node.path() if item.call_record is not None]
    await ctx.environment.commit_isolated_calls(records)
    await ctx.trace.emit(
        "lats_path_committed",
        depth=node.depth,
        tool_calls=len(records),
    )


async def run_lats(ctx: RunContext) -> str:
    """LATS MCTS with proposal, value, reflection, rollout, and backpropagation."""
    branch_safe = ctx.policy.get("branch_safe_tools")
    if isinstance(branch_safe, list):
        verified = {str(name) for name in branch_safe}
        unverified = sorted(set(ctx.environment.names) - verified)
        if unverified:
            raise ValueError(
                "LATS requires a verified branch-safe tool contract; unverified tools: "
                f"{unverified}"
            )
    mutable = [tool.name for tool in ctx.environment.tools.values() if not tool.read_only]
    if mutable:
        raise ValueError(
            "LATS requires branch-isolated environment snapshots; this environment exposes "
            f"non-snapshotable mutating tools: {mutable}"
        )

    iterations = int(ctx.policy.get("lats_iterations", 30))
    generate_samples = int(ctx.policy.get("lats_generate_samples", 5))
    value_samples = int(ctx.policy.get("lats_value_samples", 1))
    rollout_width = int(ctx.policy.get("lats_rollout_width", 5))
    tree_depth = int(ctx.policy.get("lats_tree_depth", 7))
    rollout_depth = int(ctx.policy.get("lats_rollout_depth", 4))
    failure_memory = int(ctx.policy.get("lats_failure_memory", 5))
    reflection_limit = int(ctx.policy.get("lats_reflection_limit", 3))
    temperature = float(ctx.policy.get("lats_temperature", 1.0))
    max_parallel = int(ctx.policy.get("lats_max_parallel", 1))
    max_llm_calls = int(ctx.policy.get("lats_max_llm_calls", 16))
    if min(
        iterations,
        generate_samples,
        value_samples,
        rollout_width,
        tree_depth,
        rollout_depth,
        failure_memory,
        reflection_limit,
        max_parallel,
        max_llm_calls,
    ) < 1:
        raise ValueError("LATS policy values must be positive")

    budget = _LatsBudget(
        ctx,
        max_calls=max_llm_calls,
        max_parallel=max_parallel,
    )
    root = _Node(parent=None)
    failures: list[str] = []
    reflections: list[str] = []
    terminal_nodes: list[_Node] = []
    try:
        for iteration in range(iterations):
            unique_failures = list(dict.fromkeys(failures))
            remembered_failures = unique_failures[:failure_memory]
            if len(unique_failures) > len(reflections) and len(reflections) < reflection_limit:
                reflections.append(
                    await _reflect(ctx, budget, remembered_failures, temperature)
                )

            selected = _select_node(root)
            if selected is None:
                break
            if selected.depth >= tree_depth:
                selected.terminal = True
                selected.reward = 0.0
                _backpropagate(selected, 0.0)
                continue

            children = await _expand(
                ctx,
                budget,
                selected,
                generate_samples,
                value_samples,
                remembered_failures,
                reflections,
                temperature,
            )
            terminal_nodes.extend(child for child in children if child.terminal)
            success = next(
                (child for child in children if child.terminal and child.reward == 1),
                None,
            )
            if success is not None:
                _backpropagate(success, success.reward)
                await _commit_path(ctx, success)
                return str(success.answer)
            if not children:
                selected.exhausted = True
                continue

            rollout = max(children, key=lambda child: child.value)
            while (
                not rollout.terminal
                and rollout.depth < rollout_depth
                and rollout.depth < tree_depth
            ):
                rollout_children = await _expand(
                    ctx,
                    budget,
                    rollout,
                    rollout_width,
                    value_samples,
                    list(dict.fromkeys(failures))[:failure_memory],
                    reflections,
                    temperature,
                )
                terminal_nodes.extend(
                    child for child in rollout_children if child.terminal
                )
                if not rollout_children:
                    rollout.exhausted = True
                    break
                rollout = max(rollout_children, key=lambda child: child.value)
                if rollout.terminal and rollout.reward == 1:
                    _backpropagate(rollout, rollout.reward)
                    await _commit_path(ctx, rollout)
                    return str(rollout.answer)
            _backpropagate(
                rollout, rollout.reward if rollout.terminal else rollout.value
            )
            await ctx.trace.emit(
                "lats_iteration",
                iteration=iteration + 1,
                selected_depth=selected.depth,
            )
    except _LatsBudgetExhausted as exc:
        await ctx.trace.emit(
            "lats_budget_exhausted",
            used=budget.used,
            limit=budget.max_calls,
            detail=str(exc),
        )

    if not terminal_nodes:
        raise RuntimeError(
            f"LATS search produced no terminal answer within {budget.used}/{budget.max_calls} "
            "LLM calls"
        )
    best = max(terminal_nodes, key=lambda node: (node.reward, node.value))
    if best.answer is None:
        raise RuntimeError("LATS terminal node omitted an answer")
    await _commit_path(ctx, best)
    return best.answer
