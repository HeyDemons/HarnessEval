from __future__ import annotations

import asyncio
import contextlib
import json
import random
import re
from collections import Counter
from typing import Any

from .core import RunContext, extract_json


async def run_aflow(ctx: RunContext) -> str:
    """Execute a frozen AFlow operator graph supplied as an evaluation artifact."""
    workflow = ctx.policy.get("aflow_workflow")
    if workflow is None:
        raise ValueError(
            "AFlow evaluation requires policy.aflow_workflow from a disjoint optimization split"
        )
    if not isinstance(workflow, list) or not workflow:
        raise ValueError("policy.aflow_workflow must be a non-empty operator list")
    candidates: list[str] = []
    for index, operator in enumerate(workflow):
        if operator in {"Custom", "AnswerGenerate"}:
            from .methods import _json_tool_loop

            candidates.append(await _json_tool_loop(ctx, f"aflow_{operator}_{index + 1}"))
        elif operator == "ScEnsemble" and candidates:
            candidates = [
                await ctx.complete(
                    "aflow_ensemble",
                    [
                        {
                            "role": "user",
                            "content": (
                                "Select the most reliable complete answer for the task. Return the answer directly.\n"
                                f"Task: {ctx.prompt}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}"
                            ),
                        }
                    ],
                )
            ]
        else:
            raise ValueError(f"Unsupported frozen AFlow operator: {operator}")
    if not candidates:
        raise RuntimeError("Frozen AFlow workflow produced no answer")
    return candidates[-1]


def _numeric_answer(text: str) -> str:
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else text.strip()


async def run_dylan(ctx: RunContext) -> str:
    """DyLAN's published text-agent network; it intentionally has no tool loop."""
    random.seed(0)
    population = int(ctx.policy.get("dylan_agents", 4))
    rounds = int(ctx.policy.get("dylan_rounds", 3))
    active = list(range(population))
    previous: list[str] = []
    replies: dict[int, str] = {}
    for round_id in range(rounds):
        order = active[:]
        random.shuffle(order)
        replies = {}
        for agent_id in order:
            context = "\n\n".join(f"Prior agent response: {item}" for item in previous)
            replies[agent_id] = await ctx.complete(
                f"dylan_r{round_id + 1}_a{agent_id + 1}",
                [
                    {
                        "role": "user",
                        "content": (
                            "You are an Assistant neuron in a dynamic LLM-agent network. Solve the task, use prior "
                            f"responses critically, and end with a clear answer.\nTask: {ctx.prompt}\n{context}"
                        ),
                    }
                ],
                temperature=1.0,
            )
            votes = Counter(_numeric_answer(item) for item in replies.values())
            answer, count = votes.most_common(1)[0]
            if count > (2 * len(active)) // 3:
                await ctx.trace.emit("dylan_early_stop", round=round_id + 1, answer=answer)
                return answer
        previous = list(replies.values())
        if round_id == 1 and len(active) > 2:
            ranked = await ctx.complete_json(
                "dylan_listwise_activation",
                [
                    {
                        "role": "user",
                        "content": (
                            'Rank the candidates and return JSON only: {"top":[1,2]}.\n'
                            f"Task: {ctx.prompt}\nCandidates: {json.dumps(previous, ensure_ascii=False)}"
                        ),
                    }
                ],
            )
            selected = [max(0, min(len(active) - 1, int(item) - 1)) for item in ranked.get("top", [])[:2]]
            active = [active[index] for index in selected] or active[:2]
    votes = Counter(_numeric_answer(item) for item in replies.values())
    return votes.most_common(1)[0][0]


async def run_multi_persona(ctx: RunContext) -> str:
    return await ctx.complete(
        "solo_performance_prompting",
        [
            {
                "role": "user",
                "content": (
                    "Identify relevant participants, conduct a multi-round collaboration among their perspectives, "
                    "critique when needed, and let the AI Assistant integrate the discussion. End with `Final answer: "
                    f"<answer>`.\nTask: {ctx.prompt}"
                ),
            }
        ],
    )


async def run_magentic_one(ctx: RunContext) -> str:
    workers = {
        "web_surfer": "gather external information with retrieval tools",
        "file_surfer": "inspect local files and structured records",
        "coder": "derive and verify results with computation tools",
        "executor": "perform other permitted task actions",
    }
    facts = await ctx.complete("orchestrator_facts", [{"role": "user", "content": f"Create a facts ledger.\n{ctx.prompt}"}])
    plan = await ctx.complete(
        "orchestrator_plan",
        [{"role": "user", "content": f"Create a team plan.\nTask: {ctx.prompt}\nTeam: {workers}\nFacts: {facts}"}],
    )
    history: list[dict[str, Any]] = []
    stalls = 0
    replans = 0
    for _ in range(int(ctx.policy.get("magentic_max_rounds", 20))):
        ledger = await ctx.complete_json(
            "orchestrator_ledger",
            [
                {
                    "role": "user",
                    "content": (
                        'Return JSON: {"satisfied":false,"in_loop":false,"progress":true,'
                        '"next_speaker":"web_surfer","instruction":"..."}.\n'
                        f"Task: {ctx.prompt}\nTeam: {workers}\nFacts: {facts}\nPlan: {plan}\n"
                        f"History: {json.dumps(history, ensure_ascii=False)}"
                    ),
                }
            ],
        )
        if ledger.get("satisfied") is True:
            return await ctx.complete(
                "orchestrator_final",
                [{"role": "user", "content": f"Return the final answer.\nTask: {ctx.prompt}\nHistory: {json.dumps(history, ensure_ascii=False)}"}],
            )
        if ledger.get("in_loop") is True or ledger.get("progress") is False:
            stalls += 1
            if stalls > 3:
                replans += 1
                if replans > 3:
                    raise RuntimeError("Magentic-One replan budget exhausted")
                plan = await ctx.complete(
                    "orchestrator_replan",
                    [{"role": "user", "content": f"Replan after stalled progress.\nTask: {ctx.prompt}\nHistory: {json.dumps(history, ensure_ascii=False)}"}],
                )
                history = []
                stalls = 0
                continue
        speaker = str(ledger.get("next_speaker", "executor"))
        if speaker not in workers:
            raise ValueError(f"Magentic-One selected unknown worker: {speaker}")
        action = await ctx.complete_json(
            speaker,
            [
                {
                    "role": "user",
                    "content": (
                        f"You are the {speaker}; {workers[speaker]}.\nAvailable tools: {ctx.environment.schema}\n"
                        'Return JSON: {"tool":"name","arguments":{}} or {"report":"..."}.\n'
                        f"Instruction: {ledger.get('instruction', '')}\nTask: {ctx.prompt}"
                    ),
                }
            ],
        )
        if "tool" in action:
            result = await ctx.environment.call(str(action["tool"]), action.get("arguments") or {})
            history.append({"speaker": speaker, "action": action, "result": result})
        else:
            history.append({"speaker": speaker, "report": action.get("report", "")})
    raise RuntimeError("Magentic-One round budget exhausted")


def _resolve_reference(value: Any, results: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        step, _, remainder = value[1:].partition(".")
        selected = results[step]
        for part in remainder.split(".") if remainder else []:
            selected = selected[int(part)] if isinstance(selected, list) else selected[part]
        return selected
    if isinstance(value, list):
        return [_resolve_reference(item, results) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_reference(item, results) for key, item in value.items()}
    return value


async def run_llmcompiler(ctx: RunContext) -> str:
    plan = await ctx.complete_json(
        "compiler_planner",
        [
            {
                "role": "user",
                "content": (
                    "Compile a dependency DAG maximizing parallel execution.\n"
                    f"Available tools: {ctx.environment.schema}\n"
                    'Return JSON: {"tasks":[{"id":"1","tool":"name","arguments":{},"dependencies":[]}]}.\n'
                    f"Task: {ctx.prompt}"
                ),
            }
        ],
    )
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("LLMCompiler planner omitted tasks")
    pending = {str(item["id"]): item for item in tasks}
    results: dict[str, Any] = {}
    while pending:
        ready = [item for item in pending.values() if all(str(dep) in results for dep in item.get("dependencies", []))]
        if not ready:
            raise RuntimeError("LLMCompiler dependency deadlock")

        async def execute(item: dict[str, Any]) -> tuple[str, Any]:
            arguments = _resolve_reference(item.get("arguments") or {}, results)
            return str(item["id"]), await ctx.environment.call(str(item["tool"]), arguments)

        for task_id, result in await asyncio.gather(*(execute(item) for item in ready)):
            results[task_id] = result
            pending.pop(task_id)
    return await ctx.complete(
        "compiler_joiner",
        [{"role": "user", "content": f"Return the final answer.\nTask: {ctx.prompt}\nDAG: {json.dumps(plan, ensure_ascii=False)}\nResults: {json.dumps(results, ensure_ascii=False)}"}],
    )


async def run_rewoo(ctx: RunContext) -> str:
    plan = await ctx.complete_json(
        "rewoo_planner",
        [
            {
                "role": "user",
                "content": (
                    "Plan all evidence calls before execution.\n"
                    f"Available tools: {ctx.environment.schema}\n"
                    'Return JSON: {"steps":[{"id":"E1","tool":"name","arguments":{}}]}. '
                    "References may use $E1.result.field.\n"
                    f"Task: {ctx.prompt}"
                ),
            }
        ],
    )
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise ValueError("ReWOO planner omitted evidence steps")
    evidence: dict[str, Any] = {}
    for step in steps:
        arguments = _resolve_reference(step.get("arguments") or {}, evidence)
        evidence[str(step["id"])] = await ctx.environment.call(str(step["tool"]), arguments)
    return await ctx.complete(
        "rewoo_solver",
        [{"role": "user", "content": f"Solve from complete evidence.\nTask: {ctx.prompt}\nPlan: {json.dumps(plan, ensure_ascii=False)}\nEvidence: {json.dumps(evidence, ensure_ascii=False)}"}],
    )


def _action_key(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"


async def run_sa(ctx: RunContext) -> str:
    """Benchmark-neutral response-first Speculative Actions protocol reproduction."""
    safe_names = [name for name, tool in ctx.environment.tools.items() if tool.read_only and tool.parallel]
    top_k = int(ctx.policy.get("sa_top_k", 3))

    async def predict_and_execute() -> dict[str, tuple[int, dict[str, Any]]]:
        if not safe_names:
            return {}
        draft = await ctx.complete_json(
            "sa_response_predictor",
            [
                {
                    "role": "user",
                    "content": (
                        "Predict likely immediate read-only tool calls without claiming they happened.\n"
                        f"Safe tools: {json.dumps([ctx.environment.tools[name].prompt_schema() for name in safe_names], ensure_ascii=False)}\n"
                        f'Return JSON: {{"actions":[{{"tool":"name","arguments":{{}}}}]}} with at most {top_k} actions.\n'
                        f"Task: {ctx.prompt}"
                    ),
                }
            ],
        )
        actions = draft.get("actions") if isinstance(draft, dict) else []
        speculation_epoch = ctx.environment.state_version
        cache: dict[str, tuple[int, dict[str, Any]]] = {}
        valid = [item for item in actions[:top_k] if isinstance(item, dict) and item.get("tool") in safe_names and isinstance(item.get("arguments"), dict)]

        async def execute(item: dict[str, Any]) -> None:
            name = str(item["tool"])
            arguments = item["arguments"]
            result = await ctx.environment.call(name, arguments)
            cache[_action_key(name, arguments)] = (speculation_epoch, result)

        await asyncio.gather(*(execute(item) for item in valid))
        if ctx.environment.state_version != speculation_epoch:
            cache.clear()
            await ctx.trace.emit(
                "sa_cache_invalidated",
                reason="state_changed_during_speculation",
                speculation_epoch=speculation_epoch,
                current_epoch=ctx.environment.state_version,
            )
        await ctx.trace.emit(
            "sa_draft_ready",
            actions=valid,
            cached=list(cache),
            state_version=speculation_epoch,
        )
        return cache

    draft_task = asyncio.create_task(predict_and_execute())
    messages = [
        {"role": "system", "content": "Use complete observations only. Available tools: " + ctx.environment.schema + '\nReturn JSON {"tool":"name","arguments":{}} or {"final":"answer"}.'},
        {"role": "user", "content": ctx.prompt},
    ]
    cache: dict[str, tuple[int, dict[str, Any]]] | None = None
    # Protocol handling must match _json_tool_loop (methods.py), which actor-only uses:
    # normalize a single-key {tool_name: {...}} action, and on a malformed response feed
    # the error back and retry within the turn budget instead of aborting. run_sa
    # previously raised on both, so one model protocol slip cost sa the entire episode
    # while actor-only recovered from the identical slip. That biases every sa-vs-
    # actor-only comparison by the probability of a slip — observed on GAIA L2, where the
    # model returned the correct answer under {"answer": ...} instead of {"final": ...}.
    from .methods import _normalize_action  # imported here: methods imports this module

    for _ in range(ctx.max_turns):
        raw = await ctx.complete("sa_actor", messages, json_mode=True)
        try:
            action = _normalize_action(extract_json(raw, expected_type=dict), ctx.environment.names)
        except ValueError as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Protocol error: {exc}. Return one complete action object."},
                ]
            )
            continue
        if "final" in action:
            if not draft_task.done():
                draft_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await draft_task
            return str(action["final"])
        name = str(action.get("tool", ""))
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Protocol error: arguments must be one JSON object."},
                ]
            )
            continue
        if cache is None:
            cache = await draft_task
        key = _action_key(name, arguments)
        cached = cache.pop(key, None)
        if cached is not None and cached[0] == ctx.environment.state_version:
            result = cached[1]
            await ctx.trace.emit(
                "sa_cache_hit",
                name=name,
                arguments=arguments,
                state_version=ctx.environment.state_version,
            )
        else:
            if cached is not None:
                await ctx.trace.emit(
                    "sa_cache_stale",
                    name=name,
                    arguments=arguments,
                    cached_state_version=cached[0],
                    current_state_version=ctx.environment.state_version,
                )
            state_before = ctx.environment.state_version
            result = await ctx.environment.call(name, arguments)
            await ctx.trace.emit(
                "sa_cache_miss",
                name=name,
                arguments=arguments,
                state_version=ctx.environment.state_version,
            )
            if ctx.environment.state_version != state_before and cache:
                invalidated = len(cache)
                cache.clear()
                await ctx.trace.emit(
                    "sa_cache_invalidated",
                    reason="state_transition",
                    name=name,
                    previous_state_version=state_before,
                    current_state_version=ctx.environment.state_version,
                    entries=invalidated,
                )
        canonical_action = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        messages.extend(
            [
                {"role": "assistant", "content": canonical_action},
                {"role": "user", "content": "Observation: " + json.dumps(result, ensure_ascii=False)},
            ]
        )
    raise RuntimeError("Speculative Actions turn budget exhausted")
