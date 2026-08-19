from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Any

from .core import RunContext


@dataclass
class _Node:
    parent: _Node | None
    thought: str = ""
    action: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
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


async def _reflect(ctx: RunContext, failures: list[str], temperature: float) -> str:
    return await ctx.complete(
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
    candidates = await asyncio.gather(
        *(
            ctx.complete_json(
                "lats_proposal",
                [{"role": "user", "content": prompt}],
                temperature=temperature,
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
    node: _Node,
    samples: int,
    temperature: float,
) -> tuple[float, bool, str]:
    prompt = (
        "Evaluate this LATS trajectory. Judge whether a final answer, if present, fully solves the task and score "
        "the trajectory's progress from 0 to 1.\n"
        'Return JSON only: {"score":0.0,"success":false,"feedback":"..."}.\n'
        f"{node.trajectory(ctx.prompt)}"
    )
    evaluations = await asyncio.gather(
        *(
            ctx.complete_json(
                "lats_value",
                [{"role": "user", "content": prompt}],
                temperature=temperature,
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
    node: _Node,
    count: int,
    value_samples: int,
    failures: list[str],
    reflections: list[str],
    temperature: float,
) -> list[_Node]:
    candidates = await _propose(ctx, node, count, failures, reflections, temperature)
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
            observation = await ctx.environment.call(action["tool"], action["arguments"])
            child = _Node(
                parent=node,
                thought=candidate["thought"],
                action=action,
                observation=observation,
            )
        child.value, success, feedback = await _evaluate(ctx, child, value_samples, temperature)
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


async def run_lats(ctx: RunContext) -> str:
    """LATS MCTS with proposal, value, reflection, rollout, and backpropagation."""
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
    if min(
        iterations,
        generate_samples,
        value_samples,
        rollout_width,
        tree_depth,
        rollout_depth,
        failure_memory,
        reflection_limit,
    ) < 1:
        raise ValueError("LATS policy values must be positive")

    root = _Node(parent=None)
    failures: list[str] = []
    reflections: list[str] = []
    terminal_nodes: list[_Node] = []
    for iteration in range(iterations):
        unique_failures = list(dict.fromkeys(failures))
        remembered_failures = unique_failures[:failure_memory]
        if len(unique_failures) > len(reflections) and len(reflections) < reflection_limit:
            reflections.append(await _reflect(ctx, remembered_failures, temperature))

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
            selected,
            generate_samples,
            value_samples,
            remembered_failures,
            reflections,
            temperature,
        )
        terminal_nodes.extend(child for child in children if child.terminal)
        success = next((child for child in children if child.terminal and child.reward == 1), None)
        if success is not None:
            _backpropagate(success, success.reward)
            return str(success.answer)
        if not children:
            selected.exhausted = True
            continue

        rollout = max(children, key=lambda child: child.value)
        while not rollout.terminal and rollout.depth < rollout_depth and rollout.depth < tree_depth:
            rollout_children = await _expand(
                ctx,
                rollout,
                rollout_width,
                value_samples,
                list(dict.fromkeys(failures))[:failure_memory],
                reflections,
                temperature,
            )
            terminal_nodes.extend(child for child in rollout_children if child.terminal)
            if not rollout_children:
                rollout.exhausted = True
                break
            rollout = max(rollout_children, key=lambda child: child.value)
            if rollout.terminal and rollout.reward == 1:
                _backpropagate(rollout, rollout.reward)
                return str(rollout.answer)
        _backpropagate(rollout, rollout.reward if rollout.terminal else rollout.value)
        await ctx.trace.emit("lats_iteration", iteration=iteration + 1, selected_depth=selected.depth)

    if not terminal_nodes:
        raise RuntimeError("LATS search produced no terminal answer")
    best = max(terminal_nodes, key=lambda node: (node.reward, node.value))
    if best.answer is None:
        raise RuntimeError("LATS terminal node omitted an answer")
    return best.answer
