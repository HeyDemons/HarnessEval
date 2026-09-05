"""DyLAN text network, including query-local team optimization.

Algorithm: COLM 2024, sections 3.3/3.4, equations 7 and 10--12.
Protocol reference: SALT-NLP/DyLAN 006e440, code/demo. We fix its shuffled
position/agent-id confusion and invalid duplicate rank selections. The text
profile does not implement the separate WebShop/code-interpreter experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
import re

from .core import RunContext


ROLE_PROMPTS = {
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
    "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
    "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
    "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
    "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
    "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
    "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
    "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations.",
}


def equivalent(candidate: str, other: str) -> bool:
    from sacrebleu import sentence_bleu
    return sentence_bleu(candidate, [other], lowercase=True).score >= 90


def most_frequent(candidates: list[str]) -> tuple[str, int]:
    if not candidates:
        raise ValueError("DyLAN requires at least one answer")
    answer, count = candidates[0], 0
    for candidate in candidates:
        frequency = sum(equivalent(candidate, other) for other in candidates)
        if frequency > count:
            answer, count = candidate, frequency
    return answer, count


def response_prompt(question: str, responses: list[str]) -> str:
    if not responses:
        return question
    text = question + "\n\nThese are the responses from other agents: "
    for index, response in enumerate(responses, 1):
        text += f"\n\nAgent response {index}: ```{response}```"
    return text + (
        "\n\nUsing the answer from other agents as additional advice with critical thinking, can you give an updated answer? "
        "Examine your solution and that other agents step by step. Notice that their answers might be all wrong. "
        "Please answer the question in detail. Along with the answer, give a score ranged from 1 to 5 to the solutions "
        f"of other agents. Put all {len(responses)} scores in the form like [[1, 5, 2, ...]]."
    )


def edge_weights(reply: str, count: int) -> list[float]:
    if count == 0:
        return []
    matches = re.findall(r"\[\[(.*?)\]\]", reply, re.DOTALL)
    values = []
    if matches:
        for value in matches[-1].split(","):
            try:
                values.append(min(5, max(0, int(value.strip()))))
            except ValueError:
                values.append(0)
    if len(values) != count or not sum(values):
        return [1 / count] * count
    total = sum(values)
    return [value / total for value in values]


def parse_ranks(reply: str, count: int, rng: random.Random) -> tuple[list[int], bool]:
    # The source uses the final pair and random fallback. Reject duplicates and
    # out-of-range ids instead of clamping them into duplicate activations.
    matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", reply)
    if matches:
        ranks = [int(value) - 1 for value in matches[-1]]
        if len(set(ranks)) == 2 and all(0 <= value < count for value in ranks):
            return ranks, False
    return rng.sample(range(count), min(2, count)), True


@dataclass
class Node:
    agent: int
    reply: str
    weights: dict[int, float] = field(default_factory=dict)
    importance: float = 0.0


def backward(layers: list[list[Node]], answer: str, population: int) -> list[float]:
    """Normalize terminal supporters, propagate weights, sum by stable agent id."""
    supporters = [node for node in layers[-1] if equivalent(answer, node.reply)]
    # Identity is a valid vote even for an empty response (BLEU may be zero).
    if not supporters:
        supporters = [node for node in layers[-1] if node.reply == answer]
    for node in layers[-1]:
        node.importance = 1 / len(supporters) if node in supporters else 0.0
    for previous, following in zip(reversed(layers[:-1]), reversed(layers[1:])):
        for node in previous:
            node.importance = sum(child.weights.get(node.agent, 0) * child.importance for child in following)
    scores = [0.0] * population
    for layer in layers:
        for node in layer:
            scores[node.agent] += node.importance
    return scores


async def forward(ctx: RunContext, roles: list[str], active: list[int], rounds: int,
                  rng: random.Random, phase: str) -> tuple[str, list[list[Node]]]:
    layers: list[list[Node]] = []
    for round_id in range(rounds):
        previous = layers[-1] if layers else []
        if round_id >= 2 and len(active) > 2:
            candidates = previous[:]
            rng.shuffle(candidates)
            prompt = ctx.prompt + "\n\nThese are the responses from other agents: "
            for index, node in enumerate(candidates, 1):
                prompt += f"\n\nAgent response {index}: ```{node.reply}```"
            prompt += ("\n\nPlease choose the best 2 answers and think step by step. "
                       "Put your answer in the form like [1,2] or [3,4] at the end of your response.")
            raw = await ctx.complete(f"dylan_{phase}_rank_r{round_id}",
                                     [{"role": "user", "content": prompt}], temperature=1.0)
            selected, fallback = parse_ranks(raw, len(candidates), rng)
            active = [candidates[index].agent for index in selected]
            await ctx.trace.emit("dylan_activation", phase=phase, round=round_id,
                                 active_agents=active, fallback=fallback,
                                 candidate_agents=[node.agent for node in candidates])
        order = active[:]
        rng.shuffle(order)
        layer: list[Node] = []
        layers.append(layer)
        for agent in order:
            predecessors = previous[:]
            rng.shuffle(predecessors)
            reply = await ctx.complete(
                f"dylan_{phase}_r{round_id + 1}_a{agent + 1}",
                [{"role": "system", "content": roles[agent] + "\n"},
                 {"role": "user", "content": response_prompt(ctx.prompt, [node.reply for node in predecessors])}],
                temperature=1.0,
            )
            weights = dict(zip((node.agent for node in predecessors), edge_weights(reply, len(predecessors))))
            layer.append(Node(agent, reply, weights))
            await ctx.trace.emit("dylan_node", phase=phase, round=round_id + 1,
                                 agent=agent, predecessor_weights=weights)
            answer, count = most_frequent([node.reply for node in layer])
            if count > (2 * len(active)) // 3:
                await ctx.trace.emit("dylan_early_stop", phase=phase, round=round_id + 1, answer=answer)
                return answer, layers
    return most_frequent([node.reply for node in layers[-1]])[0], layers


async def run_dylan(ctx: RunContext) -> str:
    population = int(ctx.policy.get("dylan_agents", 4))
    rounds = int(ctx.policy.get("dylan_rounds", 3))
    if population < 1 or rounds < 1:
        raise ValueError("DyLAN population and rounds must be positive")
    role_names = ctx.policy.get("dylan_roles", ["Assistant"] * population)
    if not isinstance(role_names, list) or len(role_names) != population or any(
        not isinstance(role, str) or role not in ROLE_PROMPTS for role in role_names
    ):
        raise ValueError("dylan_roles must contain one supported role per agent")
    roles = [ROLE_PROMPTS[role] for role in role_names]
    optimize = ctx.policy.get("dylan_team_optimization", True)
    if not isinstance(optimize, bool):
        raise ValueError("dylan_team_optimization must be a boolean")
    team_size = int(ctx.policy.get("dylan_team_size", min(2, population)))
    if not 1 <= team_size <= population:
        raise ValueError("dylan_team_size must be between 1 and population")
    rng = random.Random(int(ctx.policy.get("dylan_seed", ctx.policy.get("seed", 0))))
    active = list(range(population))
    await ctx.trace.emit("dylan_config", implementation="text-team-optimization-v1", roles=role_names,
                         rounds=rounds, team_size=team_size, team_optimization=optimize)
    if optimize:
        answer, layers = await forward(ctx, roles, active, rounds, rng, "trial")
        scores = backward(layers, answer, population)
        # Stable ties are deliberate and recorded, rather than accidental dict order.
        active = sorted(active, key=lambda agent: (-scores[agent], agent))[:team_size]
        await ctx.trace.emit("dylan_team_selected", scores=scores, active_agents=active,
                             layer_importance=[[{"agent": n.agent, "importance": n.importance} for n in layer]
                                               for layer in layers])
    # A new graph, containing no trial replies, solves with the selected team.
    answer, _ = await forward(ctx, roles, active, rounds, rng, "solve")
    return answer
