from __future__ import annotations

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
    call_records: list[dict[str, Any]] = field(default_factory=list)
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
        # Follow the HotPotQA source tree policy instead of substituting a
        # generic MCTS implementation.
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


@dataclass
class _SearchMemory:
    failures: list[str] = field(default_factory=list)
    failure_answers: set[str] = field(default_factory=set)
    reflections: list[dict[str, str]] = field(default_factory=list)
    value_cache: dict[str, tuple[float, bool, str]] = field(default_factory=dict)


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
        update = -1.0 if current.terminal and current.reward == 0 else reward
        current.value = (current.value * (current.visits - 1) + update) / current.visits
        current = current.parent


def _candidate_key(candidate: dict[str, Any]) -> str:
    action = {key: value for key, value in candidate.items() if key != "assistant_response_id"}
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_failures(memory: _SearchMemory, limit: int) -> list[str]:
    return list(dict.fromkeys(memory.failures))[:limit]


async def _reflect(ctx: RunContext, failure: str, temperature: float) -> dict[str, str]:
    reflection = await ctx.complete(
        "lats_reflection",
        [
            {
                "role": "user",
                "content": (
                    "Reflect on this failed trajectory and identify a concrete correction for the next search.\n"
                    f"Task: {ctx.prompt}\nFailed trajectory: {failure}"
                ),
            }
        ],
        temperature=temperature,
    )
    return {"trajectory": failure, "reflection": reflection}


async def _maybe_reflect(
    ctx: RunContext,
    memory: _SearchMemory,
    failure_memory: int,
    reflection_limit: int,
    temperature: float,
) -> None:
    failures = _unique_failures(memory, failure_memory)
    reflected_failures = failures[:reflection_limit]
    # The source refreshes reflection memory for one to three distinct failed
    # trajectories. It reflects on each trajectory separately and replaces the
    # complete map; four or more do not trigger another refresh.
    if len(reflected_failures) > len(memory.reflections) and len(failures) < 4:
        refreshed: list[dict[str, str]] = []
        for failure in reflected_failures:
            refreshed.append(await _reflect(ctx, failure, temperature))
        memory.reflections = refreshed


async def _propose(
    ctx: RunContext,
    node: _Node,
    count: int,
    memory: _SearchMemory,
    failure_memory: int,
    reflection_limit: int,
    temperature: float,
) -> list[dict[str, Any]]:
    await _maybe_reflect(ctx, memory, failure_memory, reflection_limit, temperature)
    prompt = (
        "Generate one next LATS thought/action candidate for this trajectory. Do not invent an observation.\n"
        f"Available tools: {ctx.environment.schema}\n"
        'Return JSON only: {"thought":"...","tool":"name","arguments":{}} or '
        '{"thought":"...","final":"answer"}.\n'
        f"{node.trajectory(ctx.prompt)}\n"
        f"Failed trajectories: {json.dumps(_unique_failures(memory, failure_memory), ensure_ascii=False)}\n"
        f"Reflections: {json.dumps(memory.reflections, ensure_ascii=False)}"
    )

    # The source samples n choices in one request. The portable one-choice API
    # samples them sequentially so tree width never becomes HTTP concurrency.
    candidates: list[dict[str, Any]] = []
    for sample_index in range(count):
        candidate = await ctx.complete_json(
            "lats_proposal",
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )
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
        normalized["assistant_response_id"] = ctx.last_actor_response_id
        candidates.append(normalized)
        await ctx.trace.emit(
            "lats_sample",
            depth=node.depth,
            sample_index=sample_index + 1,
            sample_count=count,
            candidate=normalized,
        )

    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(_candidate_key(candidate), candidate)
    return list(unique.values())


async def _value(
    ctx: RunContext,
    node: _Node,
    samples: int,
    memory: _SearchMemory,
    temperature: float,
) -> tuple[float, bool, str]:
    trajectory = node.trajectory(ctx.prompt)
    if trajectory in memory.value_cache:
        return memory.value_cache[trajectory]
    prompt = (
        "Evaluate this LATS trajectory. Judge whether a final answer, if present, fully solves the task and score "
        "the trajectory's progress from 0 to 1.\n"
        'Return JSON only: {"score":0.0,"success":false,"feedback":"..."}.\n'
        f"{trajectory}"
    )
    scores: list[float] = []
    successes = 0
    feedback: list[str] = []
    for _ in range(samples):
        evaluation = await ctx.complete_json(
            "lats_value",
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        score = float(evaluation.get("score"))
        if not 0.0 <= score <= 1.0:
            raise ValueError("LATS value score must be between 0 and 1")
        scores.append(score)
        successes += evaluation.get("success") is True
        feedback.append(str(evaluation.get("feedback", "")))
    mean_score = sum(scores) / len(scores)
    joined_feedback = "\n".join(feedback)
    result = (mean_score, successes > len(scores) / 2, joined_feedback)
    memory.value_cache[trajectory] = result
    return result


def _remember_failure(ctx: RunContext, node: _Node, feedback: str, memory: _SearchMemory) -> None:
    answer = node.answer or ""
    if answer in memory.failure_answers:
        return
    memory.failure_answers.add(answer)
    memory.failures.append(node.trajectory(ctx.prompt) + "\nEvaluator feedback: " + feedback)


async def _expand(
    ctx: RunContext,
    node: _Node,
    count: int,
    value_samples: int,
    memory: _SearchMemory,
    failure_memory: int,
    reflection_limit: int,
    tree_depth: int,
    temperature: float,
) -> list[_Node]:
    if node.depth >= tree_depth:
        node.exhausted = True
        await ctx.trace.emit("lats_depth_limit", depth=node.depth, tree_depth=tree_depth)
        return []

    candidates = await _propose(
        ctx,
        node,
        count,
        memory,
        failure_memory,
        reflection_limit,
        temperature,
    )
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
            terminal_reward = await ctx.evaluate_terminal(child.answer)
            if terminal_reward is None:
                _, success, feedback = await _value(ctx, child, value_samples, memory, temperature)
                child.reward = 1.0 if success else 0.0
                reward_source = "value_model_fallback"
            else:
                child.reward = terminal_reward
                success = terminal_reward == 1.0
                feedback = f"Candidate scorer reward: {terminal_reward}"
                reward_source = "candidate_scorer"
            child.value = child.reward
            await ctx.trace.emit(
                "lats_terminal_reward",
                depth=child.depth,
                reward=child.reward,
                source=reward_source,
            )
            if not success:
                _remember_failure(ctx, child, feedback, memory)
        else:
            action = {"tool": candidate["tool"], "arguments": candidate["arguments"]}
            # Exploration must not publish. A benchmark that scores the agent's tool calls
            # (BFCL) would otherwise score every candidate the search went on to reject.
            # Only the returned trajectory is published, by _commit_path.
            observation, call_record = await ctx.environment.call_isolated(
                action["tool"], action["arguments"],
                assistant_response_id=candidate["assistant_response_id"],
            )
            child = _Node(
                parent=node,
                thought=candidate["thought"],
                action=action,
                observation=observation,
                call_records=[call_record],
            )
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
    """Publish the returned trajectory's already-executed calls as the profile's answer."""

    records = [record for item in node.path() for record in item.call_records]
    await ctx.environment.commit_isolated_calls(records)
    await ctx.trace.emit("lats_path_committed", depth=node.depth, tool_calls=len(records))


async def _evaluate_children(
    ctx: RunContext,
    node: _Node,
    value_samples: int,
    memory: _SearchMemory,
    temperature: float,
) -> None:
    # Source LATS evaluates children one at a time after expansion. Terminal
    # children already carry their environment-equivalent reward.
    for child in node.children:
        if child.terminal:
            continue
        child.value, _, _ = await _value(ctx, child, value_samples, memory, temperature)


async def _rollout(
    ctx: RunContext,
    node: _Node,
    rollout_width: int,
    value_samples: int,
    memory: _SearchMemory,
    failure_memory: int,
    reflection_limit: int,
    tree_depth: int,
    rollout_depth: int,
    temperature: float,
) -> tuple[float, _Node, list[_Node]]:
    depth = node.depth
    rewards = [0.0]
    terminal_nodes: list[_Node] = []
    while not node.terminal and not node.exhausted and depth < rollout_depth:
        children = await _expand(
            ctx,
            node,
            rollout_width,
            value_samples,
            memory,
            failure_memory,
            reflection_limit,
            tree_depth,
            temperature,
        )
        terminal_nodes.extend(child for child in children if child.terminal)
        if not children:
            node.exhausted = True
            return -1.0, node, terminal_nodes

        # Official rollout stops as soon as expansion reaches a terminal state,
        # before scoring the remaining non-terminal children.
        terminal = next((child for child in children if child.terminal), None)
        if terminal is not None:
            return terminal.reward, terminal, terminal_nodes

        await _evaluate_children(ctx, node, value_samples, memory, temperature)
        node = max(children, key=lambda child: child.value)
        rewards.append(node.value)
        depth += 1
        if depth == rollout_depth:
            return -1.0, node, terminal_nodes
    return sum(rewards) / len(rewards), node, terminal_nodes


async def run_lats(ctx: RunContext) -> str:
    """Source-aligned LATS MCTS for branch-safe benchmark tools."""
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
    memory = _SearchMemory()
    terminal_nodes: list[_Node] = []
    await ctx.trace.emit(
        "lats_search_started",
        iterations=iterations,
        generate_samples=generate_samples,
        value_samples=value_samples,
        rollout_width=rollout_width,
        tree_depth=tree_depth,
        rollout_depth=rollout_depth,
        request_concurrency=1,
    )

    for iteration in range(iterations):
        selected = _select_node(root)
        while selected is not None and selected.depth >= tree_depth:
            selected.exhausted = True
            selected = _select_node(root)
        if selected is None:
            break

        children = await _expand(
            ctx,
            selected,
            generate_samples,
            value_samples,
            memory,
            failure_memory,
            reflection_limit,
            tree_depth,
            temperature,
        )
        terminal_nodes.extend(child for child in children if child.terminal)
        success = next((child for child in children if child.terminal and child.reward == 1), None)
        if success is not None:
            _backpropagate(success, success.reward)
            await _commit_path(ctx, success)
            return str(success.answer)
        if not children:
            selected.exhausted = True
            continue

        await _evaluate_children(ctx, selected, value_samples, memory, temperature)
        rollout_start = max(children, key=lambda child: child.value)
        rollout_reward, rollout_end, rollout_terminals = await _rollout(
            ctx,
            rollout_start,
            rollout_width,
            value_samples,
            memory,
            failure_memory,
            reflection_limit,
            tree_depth,
            rollout_depth,
            temperature,
        )
        terminal_nodes.extend(rollout_terminals)
        if rollout_end.terminal and rollout_end.reward == 1:
            _backpropagate(rollout_end, rollout_end.reward)
            await _commit_path(ctx, rollout_end)
            return str(rollout_end.answer)

        _backpropagate(rollout_end, rollout_reward)
        await ctx.trace.emit(
            "lats_iteration",
            iteration=iteration + 1,
            selected_depth=selected.depth,
            rollout_end_depth=rollout_end.depth,
            rollout_reward=rollout_reward,
            failures=len(memory.failure_answers),
            reflections=len(memory.reflections),
        )

    if not terminal_nodes:
        raise RuntimeError("LATS search produced no terminal answer")
    best = max(terminal_nodes, key=lambda node: (node.reward, node.value))
    if best.answer is None:
        raise RuntimeError("LATS terminal node omitted an answer")
    await _commit_path(ctx, best)
    return best.answer
