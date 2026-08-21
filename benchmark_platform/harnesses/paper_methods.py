from __future__ import annotations

import asyncio
import contextlib
import json
import random
import re
from collections import Counter
from typing import Any

from .core import RunContext, extract_json, json_safe, tool_result_content
from .dmas import run_dmas
from .lats import run_lats
from .memgpt import run_memgpt


async def run_aflow_custom_init(ctx: RunContext) -> str:
    """Execute only AFlow's disclosed, unoptimized round-1 Custom control."""
    workflow = ctx.policy.get("aflow_workflow")
    if workflow != ["Custom"]:
        raise ValueError(
            "aflow-custom-init requires policy.aflow_workflow == ['Custom']; "
            "optimized AFlow graphs need a separate canonical profile"
        )
    from .methods import _json_tool_loop

    return await _json_tool_loop(ctx, "aflow_custom_initialization")


async def run_dylan(ctx: RunContext) -> str:
    """DyLAN's published text-agent network; it intentionally has no tool loop."""
    random.seed(0)
    population = int(ctx.policy.get("dylan_agents", 4))
    rounds = int(ctx.policy.get("dylan_rounds", 3))
    base_roles = (
        "Logical Solver: construct a complete solution from first principles",
        "Critical Reviewer: find unsupported steps and counterexamples",
        "Alternative Solver: explore a genuinely different approach",
        "Verification Specialist: check constraints, calculations, and final format",
    )
    roles = [
        base_roles[index] if index < len(base_roles) else f"Independent Solver {index + 1}"
        for index in range(population)
    ]
    active = list(range(population))
    previous: list[tuple[int, str]] = []
    replies: dict[int, str] = {}
    for round_id in range(rounds):
        order = active[:]
        random.shuffle(order)
        replies = {}
        for agent_id in order:
            context = "\n\n".join(
                f"Prior {roles[prior_id]} response: {item}"
                for prior_id, item in previous
            )
            replies[agent_id] = await ctx.complete(
                f"dylan_r{round_id + 1}_a{agent_id + 1}",
                [
                    {
                        "role": "user",
                        "content": (
                            f"You are {roles[agent_id]} in a dynamic LLM-agent network. Solve the task, use prior "
                            f"responses critically, and end with a clear answer.\nTask: {ctx.prompt}\n{context}"
                        ),
                    }
                ],
                temperature=1.0,
            )
            # DyLAN's open-ended evaluator keeps each complete response as its candidate
            # (``ans_parser = lambda x: x``). Projecting it to the last number corrupts
            # compound GAIA answers such as ``7, 9`` and ``White;5876``.
            votes = Counter(item.strip() for item in replies.values())
            answer, count = votes.most_common(1)[0]
            if count > (2 * len(active)) // 3:
                await ctx.trace.emit("dylan_early_stop", round=round_id + 1, answer=answer)
                return answer
        previous = list(replies.items())
        if round_id == 1 and len(active) > 2:
            candidate_agents = list(replies)
            candidates = [
                {"agent": index + 1, "role": roles[agent_id], "response": replies[agent_id]}
                for index, agent_id in enumerate(candidate_agents)
            ]
            ranked = await ctx.complete_json(
                "dylan_listwise_activation",
                [
                    {
                        "role": "user",
                        "content": (
                            'Rank the candidates and return JSON only: {"top":[1,2]}.\n'
                            f"Task: {ctx.prompt}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}"
                        ),
                    }
                ],
            )
            selected = [
                max(0, min(len(candidate_agents) - 1, int(item) - 1))
                for item in ranked.get("top", [])[:2]
            ]
            active = [candidate_agents[index] for index in selected] or candidate_agents[:2]
            await ctx.trace.emit(
                "dylan_activation",
                round=round_id + 1,
                active_agents=active,
                active_roles=[roles[index] for index in active],
            )
    votes = Counter(item.strip() for item in replies.values())
    return votes.most_common(1)[0][0]


async def run_multi_persona(ctx: RunContext) -> str:
    return await ctx.complete(
        "solo_performance_prompting",
        [
            {
                "role": "user",
                "content": (
                    "Use Solo Performance Prompting. First identify task-specific participants and state each "
                    "participant's expertise or evaluation need. Then simulate a multi-round collaboration: an "
                    "initial proposal, critical checks from the participants, explicit correction of every discovered "
                    "error, and a final verification by the AI Assistant.\n\n"
                    "Structural example (do not solve the real task from this example):\n"
                    "Participants: AI Assistant; Domain Expert; Skeptical Verifier\n"
                    "Domain Expert: proposes a constraint-aware approach.\n"
                    "AI Assistant: produces a draft.\n"
                    "Skeptical Verifier: identifies a concrete violated constraint.\n"
                    "AI Assistant: revises the draft and rechecks the constraint.\n"
                    "Finish collaboration!\nFinal answer: <verified answer>\n\n"
                    "For the real task, continue collaboration until objections are resolved and end exactly with "
                    f"`Final answer: <answer>`.\nTask: {ctx.prompt}"
                ),
            }
        ],
    )


WORKER_ROLES = (
    ("web_surfer", "gather external information with retrieval tools", ("web_search",)),
    ("file_surfer", "inspect local files and structured records", ("read_file", "list_files")),
    ("coder", "derive and verify results with computation tools", ("run_command",)),
    # The catch-all is always on the team: every environment permits some action, and a team
    # with no members would leave the orchestrator nothing to select.
    ("executor", "perform other permitted task actions", ()),
)


def _magentic_team(names: set[str]) -> dict[str, str]:
    """Offer only the roles this environment can actually staff.

    Upstream fills the orchestrator's {team} and {names} from the participants the caller
    assembled -- "there is no requirement to involve all team members" is about a team that
    exists, not about roles with nothing behind them. This profile instead hardcoded all four
    and never told the ledger prompt which tools exist, so on terminal-bench, where no
    retrieval tool is declared at all, 19 of 32 dispatches went to web_surfer: two LLM calls
    each, at reasoning_effort=high, for a worker holding the same toolset as everyone else and
    a brief it could not act on. The tool names below are this platform's own; a benchmark
    whose tools are all domain actions (tau2, bfcl) correctly staffs only the executor.
    """
    return {
        role: description
        for role, description, required in WORKER_ROLES
        if not required or names.intersection(required)
    }


def _magentic_worker_tools(role: str, names: set[str]) -> list[str]:
    """Approximate the official specialist capability boundaries with dynamic tools."""

    if role == "web_surfer":
        preferred = {"web_search"}
    elif role == "file_surfer":
        preferred = {"read_file", "list_files"}
    elif role == "coder":
        preferred = {"read_file", "list_files", "write_file", "run_command"}
    else:
        preferred = names
    selected = sorted(names.intersection(preferred))
    return selected or sorted(names)


async def run_magentic_one(ctx: RunContext) -> str:
    workers = _magentic_team(set(ctx.environment.names))
    facts = await ctx.complete("orchestrator_facts", [{"role": "user", "content": f"Create a facts ledger.\n{ctx.prompt}"}])
    plan = await ctx.complete(
        "orchestrator_plan",
        [{"role": "user", "content": f"Create a team plan.\nTask: {ctx.prompt}\nTeam: {workers}\nFacts: {facts}"}],
    )
    history: list[dict[str, Any]] = []
    worker_histories: dict[str, list[dict[str, Any]]] = {
        worker: [] for worker in workers
    }
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
                        f"History: {json.dumps(json_safe(history), ensure_ascii=False)}"
                    ),
                }
            ],
        )
        if ledger.get("satisfied") is True:
            return await ctx.complete(
                "orchestrator_final",
                [{"role": "user", "content": f"Return the final answer.\nTask: {ctx.prompt}\nHistory: {json.dumps(json_safe(history), ensure_ascii=False)}"}],
            )
        if ledger.get("in_loop") is True or ledger.get("progress") is False:
            stalls += 1
            if stalls > 3:
                replans += 1
                if replans > 3:
                    raise RuntimeError("Magentic-One replan budget exhausted")
                plan = await ctx.complete(
                    "orchestrator_replan",
                    [{"role": "user", "content": f"Replan after stalled progress.\nTask: {ctx.prompt}\nHistory: {json.dumps(json_safe(history), ensure_ascii=False)}"}],
                )
                history.append({"speaker": "orchestrator", "replan": plan})
                stalls = 0
                continue
        speaker = str(ledger.get("next_speaker", "executor"))
        if speaker not in workers:
            # Naming someone off the roster used to end the arm outright, and a smaller roster
            # makes it likelier: the model keeps reaching for the canonical Magentic-One team.
            # The catch-all holds the same toolset as every other role, so the instruction can
            # still be carried out; the substitution is traced rather than silently applied.
            await ctx.trace.emit("magentic_unknown_worker", requested=speaker, substituted="executor")
            speaker = "executor"
        allowed_names = _magentic_worker_tools(speaker, set(ctx.environment.names))
        allowed_schema = json.dumps(
            [ctx.environment.tools[name].prompt_schema() for name in allowed_names],
            ensure_ascii=False,
        )
        instruction = str(ledger.get("instruction", ""))
        private = worker_histories.setdefault(speaker, [])
        for worker_turn in range(int(ctx.policy.get("magentic_worker_max_turns", 8))):
            action = await ctx.complete_json(
                speaker,
                [
                    {
                        "role": "user",
                        "content": (
                            f"You are the persistent {speaker}; {workers[speaker]}.\n"
                            f"Your available tools: {allowed_schema}\n"
                            'Return one JSON object: {"tool":"name","arguments":{}} to act, '
                            'or {"report":"complete result for the team"} when your assignment is done.\n'
                            f"Task: {ctx.prompt}\nInstruction: {instruction}\n"
                            f"Group message thread: {json.dumps(json_safe(history), ensure_ascii=False)}\n"
                            f"Your private history: {json.dumps(json_safe(private), ensure_ascii=False)}"
                        ),
                    }
                ],
            )
            if "tool" in action:
                name = str(action["tool"])
                arguments = action.get("arguments") or {}
                if name not in allowed_names:
                    result = {
                        "ok": False,
                        "error": "specialist_tool_not_available",
                        "available_tools": allowed_names,
                    }
                else:
                    result = await ctx.environment.call(name, arguments)
                item = {
                    "speaker": speaker,
                    "worker_turn": worker_turn + 1,
                    "action": action,
                    "result": result,
                }
                private.append(item)
                history.append(item)
                continue
            if "report" not in action:
                raise ValueError("Magentic-One specialist omitted tool or report")
            report = str(action["report"])
            item = {"speaker": speaker, "report": report}
            private.append(item)
            history.append(item)
            break
        else:
            raise RuntimeError(
                f"Magentic-One specialist {speaker} exhausted its worker turn budget"
            )
    raise RuntimeError("Magentic-One round budget exhausted")


_PLAN_REFERENCE = re.compile(r"\$([A-Za-z0-9_-]+)((?:\.[A-Za-z0-9_-]+|\[\d+\])*)")


def _reference_value(token: str, results: dict[str, Any]) -> Any:
    match = _PLAN_REFERENCE.fullmatch(token)
    if match is None:
        return {"unresolved_reference": token}
    step, remainder = match.groups()
    if step not in results:
        return {"unresolved_reference": token}
    selected = results[step]
    for part in re.findall(r"[^.\[\]]+", remainder):
        try:
            selected = (
                selected[int(part)] if isinstance(selected, list) else selected[part]
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return {"unresolved_reference": token}
    return selected


def _reference_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _resolve_reference(
    value: Any,
    results: dict[str, Any],
    *,
    interpolate_strings: bool = False,
) -> Any:
    """Resolve a "$step.field" plan reference against the results already collected.

    A reference that does not resolve yields a marker rather than raising. The planner
    prompts state only the "$E1.result.field" form, so a model naturally writes
    "$E1.result[0]" or names a step it never planned, and every one of those used to raise
    straight out of the arm: status=failed, score=None, an entire baseline lost to a
    protocol slip that ReAct hands back to the model as an ordinary tool error. Bracket
    subscripts resolve for the same reason -- the stated syntax invites them. The marker
    travels into the tool arguments and the synthesis prompt, so a reference that could not
    be resolved stays visible instead of being silently dropped.
    """
    if isinstance(value, str):
        if _PLAN_REFERENCE.fullmatch(value):
            return _reference_value(value, results)
        if interpolate_strings:

            def replace(match: re.Match[str]) -> str:
                token = match.group(0)
                resolved = _reference_value(token, results)
                if (
                    isinstance(resolved, dict)
                    and resolved.get("unresolved_reference") == token
                    and not (
                        match.group(1).isdigit()
                        or re.fullmatch(r"E\d+", match.group(1))
                    )
                ):
                    # Preserve shell variables such as $PATH. ReWOO plan references use
                    # numeric ids or the published #E<n> form adapted here as $E<n>.
                    return token
                return _reference_text(resolved)

            return _PLAN_REFERENCE.sub(replace, value)
    if isinstance(value, list):
        return [
            _resolve_reference(item, results, interpolate_strings=interpolate_strings)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _resolve_reference(
                item,
                results,
                interpolate_strings=interpolate_strings,
            )
            for key, item in value.items()
        }
    return value


async def run_llmcompiler(ctx: RunContext) -> str:
    """LLMCompiler planner/scheduler/joiner loop with bounded replanning."""

    max_replans = int(ctx.policy.get("llmcompiler_max_replans", 2))
    if max_replans < 0:
        raise ValueError("llmcompiler_max_replans must be non-negative")
    prior_runs: list[dict[str, Any]] = []
    feedback = ""

    for cycle in range(max_replans + 1):
        planner_context = ""
        if prior_runs:
            planner_context = (
                "\nThis is a replan. Correct the prior DAG using the joiner's feedback "
                "and observations.\n"
                f"Prior runs: {json.dumps(json_safe(prior_runs), ensure_ascii=False)}\n"
                f"Joiner feedback: {feedback}\n"
            )
        plan = await ctx.complete_json(
            "compiler_planner" if cycle == 0 else "compiler_replanner",
            [
                {
                    "role": "user",
                    "content": (
                        "Compile a dependency DAG maximizing dependency-ready execution.\n"
                        f"Available tools: {ctx.environment.schema}\n"
                        'Return JSON: {"tasks":[{"id":"1","tool":"name","arguments":{},'
                        '"dependencies":[]}]}.\n'
                        f"Task: {ctx.prompt}{planner_context}"
                    ),
                }
            ],
            required_root_key="tasks",
        )
        tasks = plan.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("LLMCompiler planner omitted tasks")
        pending = {
            str(item.get("id", index)): item
            for index, item in enumerate(tasks, start=1)
            if isinstance(item, dict)
        }
        results: dict[str, Any] = {}
        while pending:
            ready = [
                (task_id, item)
                for task_id, item in pending.items()
                if all(str(dep) in results for dep in item.get("dependencies", []))
            ]
            if not ready:
                results["_scheduler"] = {
                    "ok": False,
                    "error": "dependency_deadlock",
                    "pending": sorted(pending),
                }
                break

            async def execute(task_id: str, item: dict[str, Any]) -> tuple[str, Any]:
                name = item.get("tool")
                if not isinstance(name, str) or not name:
                    return task_id, {
                        "ok": False,
                        "error": "malformed_task",
                        "detail": "task omitted a tool name",
                    }
                arguments = _resolve_reference(item.get("arguments") or {}, results)
                return task_id, await ctx.environment.call(name, arguments)

            for task_id, result in await asyncio.gather(
                *(execute(tid, item) for tid, item in ready)
            ):
                results[task_id] = result
                pending.pop(task_id)

        await ctx.trace.emit(
            "llmcompiler_dag_complete",
            cycle=cycle,
            plan=plan,
            results=json_safe(results),
        )
        decision = await ctx.complete_json(
            "compiler_joiner",
            [
                {
                    "role": "user",
                    "content": (
                        "Judge whether the DAG results fully solve the task. Finish only when "
                        "the answer is supported by observations; otherwise request a replan.\n"
                        'Return exactly {"action":"finish","answer":"..."} or '
                        '{"action":"replan","feedback":"specific correction"}.\n'
                        f"Task: {ctx.prompt}\nDAG: {json.dumps(plan, ensure_ascii=False)}\n"
                        f"Results: {json.dumps(json_safe(results), ensure_ascii=False)}"
                    ),
                }
            ],
            required_root_key="action",
            strict_single_object=True,
        )
        action = str(decision.get("action", "")).strip().casefold()
        await ctx.trace.emit("llmcompiler_join", cycle=cycle, decision=decision)
        if action == "finish":
            if "answer" not in decision:
                raise ValueError("LLMCompiler Finish omitted answer")
            return str(decision["answer"])
        if action != "replan":
            raise ValueError("LLMCompiler joiner action must be finish or replan")
        prior_runs.append({"plan": plan, "results": json_safe(results)})
        feedback = str(decision.get("feedback", "")).strip()
        if cycle >= max_replans:
            raise RuntimeError(
                f"LLMCompiler replan budget exhausted after {max_replans} replans"
            )

    raise AssertionError("unreachable")


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
        # A predictor that returns no actions key, or a non-list, is a prediction miss and
        # nothing more: speculation is best effort. Slicing None raised out of the arm.
        actions = draft.get("actions") if isinstance(draft, dict) else []
        if not isinstance(actions, list):
            actions = []
        speculation_epoch = ctx.environment.state_version
        cache: dict[str, tuple[int, dict[str, Any]]] = {}
        valid = [item for item in actions[:top_k] if isinstance(item, dict) and item.get("tool") in safe_names and isinstance(item.get("arguments"), dict)]

        async def execute(item: dict[str, Any]) -> None:
            name = str(item["tool"])
            arguments = item["arguments"]
            _result, record = await ctx.environment.call_isolated(
                name,
                arguments,
                event_prefix="sa_speculative",
            )
            cache[_action_key(name, arguments)] = (speculation_epoch, record)

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
    from .methods import ACTION_SYSTEM, _normalize_action  # methods imports this module

    messages = [
        {"role": "system", "content": ACTION_SYSTEM.format(tools=ctx.environment.schema)},
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
            record = cached[1]
            result = record["result"]
            # Speculative reads stay out of authoritative traces and counters until the
            # Actor selects the exact action. Publish the cached result now in the same
            # position actor-only would have produced, without executing it twice.
            await ctx.environment.commit_isolated_calls([record])
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
                {"role": "user", "content": tool_result_content(result)},
            ]
        )
    raise RuntimeError("Speculative Actions turn budget exhausted")
