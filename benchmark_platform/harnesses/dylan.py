"""Shared/historical network utilities and the single canonical DyLAN entry.

Algorithm: COLM 2024, sections 3.3/3.4, equations 7 and 10--12.
Protocol reference: SALT-NLP/DyLAN 006e440, code/demo. We fix its shuffled
position/agent-id confusion and invalid duplicate rank selections.

The registered `dylan` runtime delegates to dylan_policy's published optimized
teams and generic tool adapter. Legacy text/DM helpers below remain only for
historical artifact and component tests; none is a separately runnable profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import random
import re
from typing import Callable

from .core import RunContext, extract_json, tool_result_content


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


def identical(candidate: str, other: str) -> bool:
    """Consistency for classification and decision-making answers (paper B.1)."""
    return candidate == other


def most_frequent(candidates: list[str], same: Callable[[str, str], bool] = equivalent) -> tuple[str, int]:
    if not candidates:
        raise ValueError("DyLAN requires at least one answer")
    answer, count = candidates[0], 0
    for candidate in candidates:
        frequency = sum(same(candidate, other) for other in candidates)
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
    # Upstream LLMNeuron.answer = ans_parser(reply). check_consensus and the
    # backward pass compare parsed answers, not raw replies; the open-ended
    # parser is the identity, so the text profiles are unaffected.
    answer: str | None = None


def backward(layers: list[list[Node]], answer: str, population: int) -> list[float]:
    """Normalize terminal supporters, propagate weights, sum by stable agent id."""
    # The pinned backward pass treats each node's reply as the BLEU hypothesis.
    # Unlike equality, reversing hypothesis/reference can cross the threshold.
    # Only the text profiles reach this, where the open-ended parser is the
    # identity; a decision-making frozen team would have to compare node.answer.
    supporters = [node for node in layers[-1] if equivalent(node.reply, answer)]
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
                  rng: random.Random, phase: str, *, query: str | None = None, system: str = "",
                  parse: Callable[[str], str | None] | None = None,
                  same: Callable[[str, str], bool] = equivalent,
                  temperature: float = 1.0) -> tuple[str | None, list[list[Node]]]:
    query = ctx.prompt if query is None else query
    parse = (lambda reply: reply) if parse is None else parse
    layers: list[list[Node]] = []
    for round_id in range(rounds):
        previous = layers[-1] if layers else []
        if round_id >= 2 and len(active) > 2:
            candidates = previous[:]
            rng.shuffle(candidates)
            prompt = query + "\n\nThese are the responses from other agents: "
            for index, node in enumerate(candidates, 1):
                prompt += f"\n\nAgent response {index}: ```{node.reply}```"
            prompt += ("\n\nPlease choose the best 2 answers and think step by step. "
                       "Put your answer in the form like [1,2] or [3,4] at the end of your response.")
            raw = await ctx.complete(f"dylan_{phase}_rank_r{round_id}",
                                     [{"role": "user", "content": prompt}], temperature=temperature)
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
                [{"role": "system", "content": roles[agent] + "\n" + system},
                 {"role": "user", "content": response_prompt(query, [node.reply for node in predecessors])}],
                temperature=temperature,
            )
            weights = dict(zip((node.agent for node in predecessors), edge_weights(reply, len(predecessors))))
            layer.append(Node(agent, reply, weights, answer=parse(reply)))
            await ctx.trace.emit("dylan_node", phase=phase, round=round_id + 1,
                                 agent=agent, predecessor_weights=weights)
            # postProcess skips invalid outputs before maxCount (paper Algorithm 1
            # and Appendix B.3); an unparsed reply is not a vote, not an empty one.
            votes = [node.answer for node in layer if node.answer is not None]
            if votes:
                answer, count = most_frequent(votes, same)
                if count > (2 * len(active)) // 3:
                    await ctx.trace.emit("dylan_early_stop", phase=phase, round=round_id + 1, answer=answer)
                    return answer, layers
    votes = [node.answer for node in layers[-1] if node.answer is not None]
    return (most_frequent(votes, same)[0] if votes else None), layers


# The paper's eight decision-making candidates, whose published prompts name
# literal WebShop actions ("click[Buy Now]"). The WebShop code was never
# released, so these keep each candidate's deliberation stance over a generic
# tool space instead of transcribing an environment this harness does not have.
DM_ROLE_PROMPTS = {
    "SearchOptimizer": "You are a Search Optimizer. Turn a vague or broad requirement into a precise lookup. Prefer read-only search or query actions, and spend your effort on making their arguments specific and accurate.",
    "InstructionAnalyst": "You are an Instruction Analyst. Re-read what the user actually asked for and which of their requirements are still unmet. Prefer the action that resolves the least certain part of the request.",
    "BudgetAnalyst": "You are a Budget Analyst. Weigh the cost, risk and policy constraints of each candidate action. Prefer an action that is permitted and reversible over one that is cheap but binding.",
    "StateExplorer": "You are a State Explorer. Compare the available options and enumerate what has not been inspected yet. Prefer actions that reveal more of the environment before anything is committed.",
    "DetailReader": "You are a Detail Reader. Interpret the details of one specific record and match them against the stated requirements, identifying key features and drawbacks. Prefer actions that read a single record in full.",
    "DecisionMaker": "You are a Decision Maker. You are confident about committing without further refinement. If the evidence already supports an action that completes the request, propose it and persuade the other agents to take it.",
    "DecisionReflector": "You are a Decision Reflector. Critically evaluate whether the proposed action truly meets the original requirement. Avoid repeating an action that has already been taken. Either endorse committing or give the most reasonable next step.",
    "OutcomeEstimator": "You are an Outcome Estimator. Predict what each candidate action will actually return or change, and say which prediction the observations so far support.",
}
DM_DEFAULT_ROLES = ("InstructionAnalyst", "StateExplorer", "DecisionMaker", "DecisionReflector")


def canonical_action(reply: str, names: list[str]) -> str | None:
    """Parse one decision into comparable canonical text.

    Returns None for an output the post-processing step skips, so an invalid
    reply is not a vote rather than an empty one that other invalid replies
    would agree with.
    """
    # Deferred: methods imports this module through paper_methods.
    from .methods import _normalize_action

    try:
        action = _normalize_action(extract_json(reply, expected_type=dict), names)
    except ValueError:
        return None
    if "final" in action:
        action = {"final": str(action["final"])}
    elif action.get("tool") in names and isinstance(action.get("arguments"), dict):
        action = {"tool": action["tool"], "arguments": action["arguments"]}
    else:
        return None
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def run_dylan_dm(ctx: RunContext) -> str:
    """Decision-making DyLAN: the T-FFN is the policy, not the actor.

    Paper Appendix B.3: agents deliberate for at most T layers over the task plus
    the concatenated observations of previous actions, one consensus action is
    executed outside the network, and the observation is appended for the next
    decision. No node calls a tool -- the paper equips DyLAN with tools only in
    the code-generation task -- so parallel activation cannot interleave writes
    on a stateful environment, and unlike Reflexion/LATS the network never
    re-reads the environment while deliberating.

    The WebShop implementation was never released in 006e440, which ships only
    HumanEval/MATH/MMLU. This follows the published specification, not a pinned
    reference implementation.
    """
    # Deferred: methods imports this module through paper_methods.
    from .methods import ACTION_SYSTEM

    population = int(ctx.policy.get("dylan_agents", 4))
    rounds = int(ctx.policy.get("dylan_rounds", 4))
    if population < 1 or rounds < 1:
        raise ValueError("DyLAN population and rounds must be positive")
    role_names = ctx.policy.get("dylan_roles", list(DM_DEFAULT_ROLES[:population]))
    if not isinstance(role_names, list) or len(role_names) != population or any(
        not isinstance(role, str) or role not in DM_ROLE_PROMPTS for role in role_names
    ):
        raise ValueError("dylan_roles must contain one supported decision-making role per agent")
    roles = [DM_ROLE_PROMPTS[role] for role in role_names]
    temperature = float(ctx.policy.get("dylan_temperature", 0.0))
    rng = random.Random(int(ctx.policy.get("dylan_seed", ctx.policy.get("seed", 0))))
    system = "\n" + ACTION_SYSTEM.format(tools=ctx.environment.schema)

    def parse(reply: str) -> str | None:
        return canonical_action(reply, ctx.environment.names)

    await ctx.trace.emit("dylan_config", implementation="paper-decision-making-v1", roles=role_names,
                         rounds=rounds, agents=population, temperature=temperature,
                         consensus="identical-action", team_optimization=False)
    history: list[str] = []
    for step in range(ctx.max_turns):
        query = "\n\n".join([ctx.prompt, *history])
        if ctx.should_finalize(step):
            query += ('\n\nThe action budget is exhausted. Return only '
                      '{"final":"best answer supported by existing observations"}. Do not call another tool.')
            await ctx.trace.emit("budget_finalization", scope="dylan_dm", model_requests=ctx.model_budget.used)
        selected, _ = await forward(ctx, roles, list(range(population)), rounds, rng, f"dm_s{step + 1}",
                                    query=query, system=system, parse=parse, same=identical,
                                    temperature=temperature)
        if selected is None:
            # Every output in the final layer was invalid. Record it and let the
            # next deliberation see that, rather than repeating an identical query.
            await ctx.trace.emit("dylan_dm_invalid_layer", step=step + 1)
            history.append("Protocol error: no agent returned one complete action object.")
            continue
        action = json.loads(selected)
        await ctx.trace.emit("dylan_dm_action", step=step + 1, action=action)
        if "final" in action:
            return str(action["final"])
        result = await ctx.environment.call(action["tool"], action["arguments"])
        history.append(f"Action: {selected}\nObservation: {tool_result_content(result)}")
    raise RuntimeError("DyLAN decision loop exhausted without a final answer")


async def run_dylan(ctx: RunContext) -> str:
    if ctx.profile != "dylan":
        raise ValueError("Retired DyLAN profile; use the single canonical 'dylan' tool policy")
    from .dylan_policy import run_policy
    return await run_policy(ctx)


async def run_legacy_text(ctx: RunContext) -> str:
    """Historical implementation for offline artifact validation, not a runnable profile."""
    if ctx.profile == "dylan-dm":
        return await run_dylan_dm(ctx)
    if ctx.profile == "dylan":
        from .dylan_team import validate_team
        benchmark, case_id = ctx.policy.get("dylan_benchmark"), ctx.policy.get("dylan_case_id")
        if not isinstance(benchmark, str) or not benchmark or not isinstance(case_id, str) or not case_id:
            raise ValueError("Frozen DyLAN requires dylan_benchmark and dylan_case_id")
        artifact = validate_team(ctx.policy.get("dylan_team_artifact"), benchmark=benchmark, case_id=case_id)
        overrides = set(ctx.policy) & {"dylan_agents", "dylan_roles", "dylan_rounds", "dylan_seed", "dylan_team_size",
                                      "dylan_team_optimization"}
        if overrides:
            raise ValueError("Frozen DyLAN configuration comes from the artifact; remove policy overrides")
        roles = [ROLE_PROMPTS[role] for role in artifact["candidate_roles"]]
        await ctx.trace.emit("dylan_config", implementation="text-frozen-team-v1",
                             artifact_sha256=artifact["artifact_sha256"], artifact=artifact,
                             active_agents=artifact["selected_agents"], team_optimization=False)
        answer, _ = await forward(ctx, roles, artifact["selected_agents"][:], artifact["rounds"],
                                  random.Random(artifact["seed"]), "solve")
        return answer
    if ctx.profile not in {"dylan-query-local", "dylan-inference"}:
        raise ValueError("DyLAN requires an explicit frozen, query-local, or inference-only profile")
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
    inference_only = ctx.profile == "dylan-inference"
    optimize = ctx.policy.get("dylan_team_optimization", not inference_only)
    if not isinstance(optimize, bool):
        raise ValueError("dylan_team_optimization must be a boolean")
    if inference_only and optimize:
        raise ValueError("dylan-inference cannot enable team optimization")
    team_size = int(ctx.policy.get("dylan_team_size", population if inference_only else min(2, population)))
    if not 1 <= team_size <= population:
        raise ValueError("dylan_team_size must be between 1 and population")
    rng = random.Random(int(ctx.policy.get("dylan_seed", ctx.policy.get("seed", 0))))
    active = list(range(population))
    await ctx.trace.emit("dylan_config", implementation="text-inference-v1" if inference_only else "text-team-optimization-v2", roles=role_names,
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
