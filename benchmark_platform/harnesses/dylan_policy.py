"""DyLAN decision policy with the published optimized WebShop teams.

COLM paper 2310.02170v2, Algorithm 1 and Appendix B.1. The generic visible-
observation router and tool-action encoding are benchmark adaptations. This
transfers the published team-selection results; it does not claim to optimize
new teams on benchmark evaluation cases.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .core import RunContext, extract_json, tool_result_content
from .dylan import DM_ROLE_PROMPTS, parse_ranks

IMPLEMENTATION = "published-teams-tool-policy-v2"
ROLES = {**DM_ROLE_PROMPTS,
         "ProductExplorer": DM_ROLE_PROMPTS["StateExplorer"],
         "DescriptionReader": DM_ROLE_PROMPTS["DetailReader"],
         "ResultEstimator": DM_ROLE_PROMPTS["OutcomeEstimator"]}
PAPER_TEAMS = {
    "searching": ("SearchOptimizer", "BudgetAnalyst", "InstructionAnalyst", "DecisionReflector"),
    "exploring": ("DecisionMaker", "BudgetAnalyst", "ProductExplorer", "InstructionAnalyst"),
    "item": ("BudgetAnalyst", "DescriptionReader", "DecisionMaker", "ResultEstimator"),
}


def visible_state_kind(calls: list[dict]) -> str:
    """Declared generic router: initial/new user request, collection, or detail.

    Only committed, publicly observed tools are inspected. No task category,
    expected answer, assertion, hidden state or scorer may select a team.
    """
    if not calls or calls[-1]["name"] == "send_message_to_user":
        return "searching"
    call = calls[-1]
    result = call.get("result") or {}
    if not result.get("ok", True):
        return "searching"
    value = result.get("result", result)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    if "search" in call["name"].lower() or "list" in call["name"].lower():
        return "exploring"
    if isinstance(value, list) or (isinstance(value, dict) and any(
            isinstance(value.get(key), list) for key in ("results", "items", "files", "records", "entries"))):
        return "exploring"
    return "item"


def parse_action(reply: str, names: list[str], *, finalizing: bool = False) -> tuple[str | None, list]:
    try:
        payload = extract_json(reply, expected_type=dict)
    except ValueError:
        return None, []
    action = payload.get("action", payload)
    ratings = payload.get("ratings") if isinstance(payload.get("ratings"), list) else []
    if not isinstance(action, dict):
        return None, ratings
    if set(action) == {"final"} and isinstance(action["final"], str):
        accepted = action
    elif (not finalizing and set(action) == {"tool", "arguments"}
          and action["tool"] in names and isinstance(action["arguments"], dict)):
        accepted = action
    else:
        return None, ratings
    return json.dumps(accepted, ensure_ascii=False, sort_keys=True, separators=(",", ":")), ratings


@dataclass
class PolicyNode:
    agent: str
    reply: str
    action: str | None
    weights: dict[str, float] = field(default_factory=dict)
    importance: float = 0.0


def winner(layer: list[PolicyNode]) -> tuple[str | None, int]:
    valid = [node.action for node in layer if node.action is not None]
    if not valid:
        return None, 0
    answer = max(valid, key=valid.count)  # Stable first occurrence for ties.
    return answer, valid.count(answer)


def importance(layers: list[list[PolicyNode]], answer: str | None) -> dict[str, float]:
    """Equations 10–12, using exact parsed decisions, including identity copy edges."""
    scores = {node.agent: 0.0 for layer in layers for node in layer}
    if answer is None:
        return scores
    supporters = [node for node in layers[-1] if node.action == answer]
    for node in layers[-1]:
        node.importance = 1 / len(supporters) if node.action == answer else 0.0
    for previous, following in zip(reversed(layers[:-1]), reversed(layers[1:])):
        for node in previous:
            node.importance = sum(child.importance * child.weights.get(node.agent, 0) for child in following)
    for layer in layers:
        for node in layer:
            scores[node.agent] += node.importance
    return scores


async def deliberate(ctx: RunContext, query: str, team: tuple[str, ...], rng: random.Random,
                     step: int, *, finalizing: bool) -> str | None:
    layers: list[list[PolicyNode]] = []
    active = list(team)
    for layer_index in range(1, 5):
        previous = layers[-1] if layers else []
        if layer_index == 3 and len(active) > 2:
            candidates = previous[:]
            rng.shuffle(candidates)
            raw = await ctx.complete("dylan_ranker", [
                {"role": "system", "content": "Rank the proposed next decisions. Return only the two best distinct candidate indices as [1,2]. Do not execute or propose a new action."},
                {"role": "user", "content": query + "\nCandidates:\n" + "\n".join(
                    f"{i}: {node.reply}" for i, node in enumerate(candidates, 1))}], temperature=0.0)
            indices, fallback = parse_ranks(raw, len(candidates), rng)
            # Algorithm 1's reformation layer copies the chosen messages. It is
            # not an extra generation by each chosen agent in the same layer.
            layer = [PolicyNode(candidates[i].agent, candidates[i].reply, candidates[i].action,
                                {candidates[i].agent: 1.0}) for i in indices]
            active = [node.agent for node in layer]
            await ctx.trace.emit("dylan_reformation", step=step, layer=layer_index,
                                 active_agents=active, fallback=fallback, message_copy=True)
        else:
            order = active[:]
            rng.shuffle(order)
            layer = []
            for agent in order:
                predecessors = previous[:]
                rng.shuffle(predecessors)
                instructions = (
                    ROLES[agent] + "\nPropose the next decision based on the task and actual observations. "
                    "These are proposals only; the controller executes the network's selected action once. "
                    "Return one JSON object: {\"action\":{\"tool\":\"name\",\"arguments\":{}},"
                    "\"ratings\":[1,5],\"reasoning\":\"brief explanation\"}. "
                    "To finish, use {\"action\":{\"final\":\"answer\"},\"ratings\":[...]}. "
                    "Give one 1–5 rating for each predecessor in the shown order; use [] when there are none. "
                    "Consider their proposals critically; never invent observations.\n"
                    f"Available tools: {ctx.environment.schema}"
                )
                if finalizing:
                    instructions += "\nThe remaining budget is for the final answer only. Tool actions are prohibited."
                request = query
                if predecessors:
                    request += "\nPrevious layer proposals:\n" + "\n".join(
                        f"{i}: {node.reply}" for i, node in enumerate(predecessors, 1))
                raw = await ctx.complete(f"dylan_s{step}_l{layer_index}_{agent}", [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": request}], temperature=0.0)
                action, ratings = parse_action(raw, ctx.environment.names, finalizing=finalizing)
                valid_ratings = (len(ratings) == len(predecessors) and all(
                    type(rating) in {int, float} and 1 <= rating <= 5 for rating in ratings))
                values = ratings if valid_ratings else [1] * len(predecessors)
                total = sum(values)
                weights = {node.agent: rating / total for node, rating in zip(predecessors, values)}
                layer.append(PolicyNode(agent, raw, action, weights))
                await ctx.trace.emit("dylan_node", step=step, layer=layer_index, agent=agent,
                                     valid_action=action is not None, predecessor_weights=weights,
                                     ratings_fallback=bool(predecessors) and not valid_ratings)
        layers.append(layer)
        answer, count = winner(layer)
        # Test consensus after the whole layer, as Algorithm 1 specifies.
        if answer is not None and count * 3 > len(active) * 2:
            await ctx.trace.emit("dylan_early_stop", step=step, layer=layer_index,
                                 supporters=count, active_agents=len(active))
            break
    answer, _ = winner(layers[-1])
    await ctx.trace.emit("dylan_importance", step=step, scores=importance(layers, answer),
                         selection_source="published-optimized-teams", online_optimization=False)
    return answer


async def run_policy(ctx: RunContext) -> str:
    obsolete = set(ctx.policy) & {"dylan_team_artifact", "dylan_agents", "dylan_roles", "dylan_rounds",
                                 "dylan_team_size", "dylan_team_optimization", "dylan_temperature"}
    if obsolete:
        raise ValueError("Canonical DyLAN uses the published optimized teams and T=4; remove legacy overrides: " + ", ".join(sorted(obsolete)))
    await ctx.trace.emit("dylan_config", implementation=IMPLEMENTATION, agents=4, rounds=4,
                         temperature=0.0, reformation_layer=3, reformation_k=2,
                         consensus="exact-canonical-action", teams=PAPER_TEAMS,
                         team_source="paper-appendix-B.1", state_router="visible-tool-observation-v1",
                         benchmark_adapter=True, online_optimization=False)
    rng = random.Random(int(ctx.policy.get("dylan_seed", ctx.policy.get("seed", 0))))
    history: list[str] = []
    for step in range(ctx.max_turns):
        state = visible_state_kind(ctx.environment.calls)
        team = PAPER_TEAMS[state]
        await ctx.trace.emit("dylan_team_selected", step=step + 1, state_kind=state,
                             roles=team, source="published-optimized-teams")
        remaining = None if ctx.model_budget.limit is None else ctx.model_budget.limit - ctx.model_budget.used
        finalizing = ctx.should_finalize(step) or (remaining is not None and remaining <= len(team))
        query = "\n\n".join([ctx.prompt, *history])
        selected = await deliberate(ctx, query, team, rng, step + 1, finalizing=finalizing)
        if selected is None:
            history.append("Protocol error: no valid decision was produced. Use one of the available tools or a final answer.")
            continue
        action = json.loads(selected)
        if "final" in action:
            await ctx.trace.emit("dylan_final", step=step + 1)
            return action["final"]
        if finalizing or (ctx.model_budget.limit is not None and ctx.model_budget.used >= ctx.model_budget.limit):
            raise RuntimeError("DyLAN model response budget exhausted before committing a tool action")
        await ctx.trace.emit("dylan_action", step=step + 1, action=action)
        result = await ctx.environment.call(action["tool"], action["arguments"])
        history.append(f"Action: {selected}\nObservation: {tool_result_content(result)}")
    raise RuntimeError("DyLAN decision loop exhausted without a final answer")
