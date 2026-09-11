from __future__ import annotations

import asyncio
import contextlib
import json
import random
import re
from typing import Any

from .core import RunContext, extract_json, json_safe, tool_result_content
from .dmas import run_dmas
from .lats import run_lats
from .magentic_one import run_magentic_one
from .memgpt import run_memgpt


# Solo Performance Prompting is itself a one-call prompting method.  The official
# repository uses two complete demonstrations (rather than a schematic placeholder),
# dynamic participants, explicit profiles, iterative criticism, and a delimited final
# answer.  These benchmark-neutral demonstrations preserve that protocol without
# importing a task-specific answer format from one of the three source benchmarks.
SPP_PROFILE_PROMPT = """When solving a task, first identify the participants whose expertise or evaluation needs will help. Provide a short profile for each participant. Then simulate a multi-round collaboration until the participants agree that every requirement has been checked. Participants should give concrete criticism and the AI Assistant should revise incorrect work.

Here are two complete demonstrations.
---
Example Task 1: Use each of the numbers 2, 3, and 5 exactly once with basic arithmetic to make 10.

Participants: AI Assistant (you); Arithmetic Expert
Profiles:
- AI Assistant (you): drafts and revises a candidate solution.
- Arithmetic Expert: checks the calculation and verifies that every supplied number is used exactly once.

Start collaboration!
AI Assistant (you): My first attempt is 5 * (3 - 2) = 10.
Arithmetic Expert: The arithmetic is wrong: 3 - 2 is 1, so the expression equals 5. Please try again while retaining all three numbers exactly once.
AI Assistant (you): Revised solution: 5 + 3 + 2 = 10.
Arithmetic Expert: Verified. The sum is 10 and the numbers 2, 3, and 5 each appear exactly once.
Finish collaboration!
Final answer: 5 + 3 + 2 = 10

---
Example Task 2: Explain evaporation to an eight-year-old in exactly two sentences, and include the word sunlight.

Participants: AI Assistant (you); Science Teacher; Eight-year-old Reader
Profiles:
- AI Assistant (you): writes the explanation and applies feedback.
- Science Teacher: checks scientific correctness and the two-sentence constraint.
- Eight-year-old Reader: flags words that are hard for a child to understand.

Start collaboration!
AI Assistant (you): Sunlight gives water molecules kinetic energy. They escape into the atmosphere. This process is evaporation.
Science Teacher: The idea is broadly correct, but the response has three sentences instead of two.
Eight-year-old Reader: "Kinetic energy" and "atmosphere" are difficult words for me.
AI Assistant (you): Revised explanation: Sunlight warms liquid water and helps some of it rise into the air as an invisible gas. That change from liquid water into gas is called evaporation.
Science Teacher: Verified: it is accurate, contains sunlight, and has exactly two sentences.
Eight-year-old Reader: The revised version is easy to understand.
Finish collaboration!
Final answer: Sunlight warms liquid water and helps some of it rise into the air as an invisible gas. That change from liquid water into gas is called evaporation.

---
Now identify the participants, provide their profiles, and collaboratively solve the following task step by step. End the collaboration with "Finish collaboration!" and present the final solution with the prefix "Final answer:".

Task: {task}"""


from .aflow import run_aflow


from .dylan import most_frequent as _dylan_most_frequent, run_dylan


def spp_recommendation(collaboration: str) -> str:
    """Apply the pinned SPP final-answer marker precedence to a collaboration."""

    for marker in ("Final answer:", "final answer:"):
        if marker in collaboration:
            return collaboration.split(marker)[1].strip()
    return collaboration


async def run_multi_persona(ctx: RunContext) -> str:
    collaboration = await ctx.complete(
        "multi_persona_collaboration",
        [{"role": "user", "content": SPP_PROFILE_PROMPT.format(task=ctx.prompt)}],
    )
    # Preserve the pinned SPP prompt_unwrap (619c8a0) for the collaboration's
    # recommended plan, including marker precedence and repeated-marker behavior.
    recommendation = spp_recommendation(collaboration)

    # Declared benchmark adapter: SPP remains the complete first planning stage;
    # a standard dynamic Actor then carries that collaboration through real tools and
    # observations. This mirrors AFlow's reasoning-operator/ToolDecision split instead
    # of pretending the original text-only prompt can mutate an environment by itself.
    from .methods import _json_tool_loop
    answer = await _json_tool_loop(
        ctx,
        "multi_persona_actor",
        prompt=(
            "Execute the original task using the completed multi-persona collaboration as "
            "advisory planning. Select and use benchmark tools yourself as needed; only real "
            "controller observations are evidence. Finish only when the original task is complete.\n\n"
            f"Original task: {ctx.prompt}\n\n"
            f"Collaboration transcript: {collaboration}\n\n"
            f"Recommended plan or answer: {recommendation}"
        ),
    )
    if ctx.policy.get("bfcl_declaration_mode") is True:
        from .declaration import stage_selected_tool_records
        await stage_selected_tool_records(
            ctx,
            list(ctx.environment.proposal_calls),
            content=answer,
        )
    return answer


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
    """Legacy JSON-field dialect, selected only by explicit compatibility policy.

    This is not a superset of the pinned text substitution language: `.field`
    suffixes and whole-reference JSON types have different meanings here.

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
        # The pinned compiler accepts both $1 and ${1}. Keep shell environment
        # variables intact unless their name is an actual task-result id.
        value = re.sub(
            r"\$\{([A-Za-z0-9_-]+)\}",
            lambda match: "$" + match.group(1) if match.group(1) in results else match.group(0),
            value,
        )
        if match := _PLAN_REFERENCE.fullmatch(value):
            if match.group(1) in results or match.group(1).isdigit() or re.fullmatch(r"E\d+", match.group(1)):
                return _reference_value(value, results)
            return value
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


def _validate_compiler_plan(plan: dict) -> None:
    """Keep JSON-adapter ID errors inside the existing bounded repair loop."""
    def numeric_id(value: Any, path: str) -> int:
        if not (type(value) is int or isinstance(value, str) and re.fullmatch(r"[0-9]+", value)):
            raise ValueError(f"{path} must be a positive integer task ID")
        result = int(value)
        if result < 1:
            raise ValueError(f"{path} must be a positive integer task ID")
        return result

    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("LLMCompiler tasks must be an array")
    seen: set[int] = set()
    for index, item in enumerate(tasks, 1):
        if not isinstance(item, dict):
            raise ValueError(f"tasks[{index - 1}] must be an object")
        task_id = numeric_id(item.get("id", index), f"tasks[{index - 1}].id")
        if task_id in seen:
            raise ValueError(f"duplicate task ID {task_id}")
        seen.add(task_id)
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError(f"task {task_id} dependencies must be an array")
        for dependency in dependencies:
            numeric_id(dependency, f"task {task_id} dependency")


async def run_llmcompiler(ctx: RunContext) -> str:
    """LLMCompiler planner/scheduler/joiner loop with bounded replanning."""

    # Upstream's max_replans is the total number of planning passes, including the
    # initial one (its loop is range(max_replans)); every published config pins it to 1.
    max_planning_passes = int(ctx.policy.get("llmcompiler_max_replans", 1))
    if max_planning_passes < 1:
        raise ValueError("llmcompiler_max_replans must be positive")
    reference_mode = ctx.policy.get("llmcompiler_reference_mode", "upstream")
    if reference_mode not in {"upstream", "legacy-json-fields"}:
        raise ValueError("llmcompiler_reference_mode must be upstream or legacy-json-fields")
    await ctx.trace.emit("llmcompiler_config", implementation="validated-planner-ids-v5",
                         dependency_mode="arguments-plus-declared" if reference_mode == "upstream" else "declared-only",
                         reference_mode=reference_mode, max_planning_passes=max_planning_passes)
    reference_instructions = (
        "Reference an earlier task's complete observation with $1 or ${1}. "
        "Earlier numeric references automatically create dependencies; use dependencies for additional ordering constraints. "
        "References insert str(observation), even when they occupy the whole argument. "
        "Suffixes are literal text: $1.txt appends .txt; field/index selection is not supported. "
        if reference_mode == "upstream" else
        "List all prerequisite task IDs in dependencies. "
        "Legacy JSON-field references: use $1 or ${1} for a complete observation, "
        "$1.result.field or $1.result[0] for a JSON field. Whole references preserve JSON types; "
        "embedded references insert text. Dot suffixes are field paths, not literal filenames. "
    )
    prior_runs: list[dict[str, Any]] = []
    feedback = ""

    for cycle in range(max_planning_passes):
        final_pass = cycle == max_planning_passes - 1
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
                        "Use unique positive integer task ids. "
                        + reference_instructions
                        + "Never guess an observation that a prerequisite tool must supply.\n"
                        f"Task: {ctx.prompt}{planner_context}"
                    ),
                }
            ],
            required_root_key="tasks",
            validator=_validate_compiler_plan,
        )
        tasks = plan.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("LLMCompiler planner omitted tasks")
        pending = {
            str(int(item.get("id", index))): item
            for index, item in enumerate(tasks, start=1)
            if isinstance(item, dict)
        }
        dependency_graph = {}
        for task_id, item in pending.items():
            declared = item.get("dependencies", [])
            inferred = []
            effective = [str(int(dep)) for dep in declared]
            if reference_mode == "upstream":
                from .compiler_references import infer_dependencies
                inferred = infer_dependencies(task_id, item.get("arguments") or {})
                # Keep explicit ordering constraints from the dynamic JSON planner.
                # Compute once, before any tool can start; substitution and readiness
                # must consume exactly the same effective dependency list.
                effective = [str(dep) for dep in sorted({int(dep) for dep in [*declared, *inferred]})]
            pending[task_id] = {**item, "dependencies": effective}
            dependency_graph[task_id] = {"declared": declared, "inferred": inferred, "effective": effective}
        await ctx.trace.emit("llmcompiler_dependencies", cycle=cycle, tasks=dependency_graph)
        results: dict[str, Any] = {}
        async def execute(task_id: str, item: dict[str, Any]) -> tuple[str, Any]:
            name = item.get("tool")
            if not isinstance(name, str) or not name:
                return task_id, {
                    "ok": False,
                    "error": "malformed_task",
                    "detail": "task omitted a tool name",
                }
            if reference_mode == "upstream":
                from .compiler_references import resolve_arguments
                arguments = resolve_arguments(item.get("arguments") or {}, item.get("dependencies", []), results)
            else:
                arguments = _resolve_reference(item.get("arguments") or {}, results, interpolate_strings=True)
            result = await ctx.environment.call(name, arguments)
            # Upstream Task.observation is the tool's return value. The controller's
            # ok/result envelope belongs in the trace, not in $1 substitutions.
            # Legacy field paths explicitly address that envelope and retain it.
            observation = result.get("result") if reference_mode == "upstream" and result.get("ok") is True else result
            return task_id, observation

        # Upstream TaskFetchingUnit.schedule releases successors as each
        # prerequisite completes, even while unrelated tasks are still running.
        running: set[asyncio.Task] = set()
        try:
            while pending or running:
                ready = [
                    (task_id, item)
                    for task_id, item in pending.items()
                    if all(str(dep) in results for dep in item.get("dependencies", []))
                ]
                for task_id, item in ready:
                    running.add(asyncio.create_task(execute(task_id, item)))
                    pending.pop(task_id)
                if not running:
                    results["_scheduler"] = {
                        "ok": False,
                        "error": "dependency_deadlock",
                        "pending": sorted(pending),
                    }
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task_id, result = task.result()
                    results[task_id] = result
                running.difference_update(done)
        finally:
            # Do not leave tool operations behind when the arm is cancelled or
            # an unexpected exception escapes. Retrieve completed exceptions too.
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

        await ctx.trace.emit(
            "llmcompiler_dag_complete",
            cycle=cycle,
            plan=plan,
            results=json_safe(results),
        )
        join_context = (
            "Judge whether the DAG results fully solve the task. Finish only when "
            "the answer is supported by observations.\n"
            + (
                'This is the final planning pass, so return JSON only as '
                '{"action":"finish","answer":"best supported answer"}. Do not request another replan.\n'
                if final_pass
                else 'Return JSON only, exactly {"action":"finish","answer":"..."} or '
                '{"action":"replan","feedback":"specific correction"}.\n'
            )
            + f"Task: {ctx.prompt}\nDAG: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Results: {json.dumps(json_safe(results), ensure_ascii=False)}"
        )
        if ctx.policy.get("bfcl_declaration_mode") is True and final_pass:
            from .declaration import (
                MULTI_MODEL_PROTOCOL,
                complete_native_declaration,
                declaration_messages,
            )
            return await complete_native_declaration(
                ctx,
                role="compiler_joiner",
                messages=declaration_messages(
                    ctx,
                    method_instruction=(
                        "You are LLMCompiler's existing final joiner. Use the planner DAG and its "
                        "internal proposal records to publish the complete BFCL native call batch in "
                        "this response. Proposals were not executed and are not observations."
                    ),
                    internal_context=join_context,
                ),
                protocol=MULTI_MODEL_PROTOCOL,
            )
        decision = await ctx.complete_json(
            "compiler_joiner",
            [{"role": "user", "content": join_context}],
            required_root_key="action",
            strict_single_object=True,
        )
        action = str(decision.get("action", "")).strip().casefold()
        await ctx.trace.emit(
            "llmcompiler_join",
            cycle=cycle,
            final_pass=final_pass,
            decision=decision,
        )
        if action == "finish":
            if "answer" not in decision:
                raise ValueError("LLMCompiler Finish omitted answer")
            return str(decision["answer"])
        if action != "replan":
            raise ValueError("LLMCompiler joiner action must be finish or replan")
        if final_pass:
            # The pinned implementation unconditionally disables replanning on its final
            # pass. Preserve the model's payload as the best available answer rather than
            # turning a completed measurement into a harness error.
            forced_answer = decision.get("answer") or decision.get("feedback")
            if forced_answer is None:
                raise ValueError("LLMCompiler final Replan omitted an answer or feedback")
            await ctx.trace.emit(
                "llmcompiler_final_replan_forced_finish",
                cycle=cycle,
                answer=str(forced_answer),
            )
            return str(forced_answer)
        prior_runs.append({"plan": plan, "results": json_safe(results)})
        feedback = str(decision.get("feedback", "")).strip()

    raise AssertionError("unreachable")


def _action_key(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"


async def run_sa(ctx: RunContext) -> str:
    """Lossless top-k Speculative Actions with an independent fast model each turn."""
    from .reply_contracts import action_schema, object_schema
    safe_names = [name for name, tool in ctx.environment.tools.items() if tool.read_only and tool.parallel]
    policy_safe = ctx.policy.get("speculation_safe_tools")
    if isinstance(policy_safe, list):
        allowed = {str(name) for name in policy_safe}
        safe_names = [name for name in safe_names if name in allowed]
    if safe_names and not ctx.environment.isolated_calls_supported:
        raise RuntimeError("SA requires an isolated tool execution/commit channel; native dialogue requests are not isolated")
    top_k = int(ctx.policy.get("sa_top_k", 3))
    if top_k < 1:
        raise ValueError("sa_top_k must be positive")
    if safe_names and ctx.speculator_client is None:
        raise RuntimeError("sa requires an independent Speculator client for safe pre-actions")

    async def predict_and_execute(
        turn: int,
        actor_messages: list[dict[str, Any]],
    ) -> dict[str, tuple[int, dict[str, Any]]]:
        if not safe_names:
            return {}
        visible_safe_tools = [
            (
                ctx.environment.tools[name].native_schema()
                if ctx.policy.get("bfcl_declaration_mode") is True
                else ctx.environment.tools[name].prompt_schema()
            )
            for name in safe_names
        ]
        predictor_messages = [
            *actor_messages,
            {
                "role": "user",
                "content": (
                    "You are the fast Speculator, not the authoritative Actor. Predict the Actor's "
                    "next immediate tool action from the conversation above. Predictions are best-effort "
                    "and must never claim an observation occurred.\n"
                    f"Only these lossless prelaunch tools are allowed: "
                    f"{json.dumps(visible_safe_tools, ensure_ascii=False)}\n"
                    f'Return one JSON object {{"actions":[{{"tool":"name","arguments":{{}}}}]}} '
                    f"with at most {top_k} distinct actions. Return an empty actions list when the Actor "
                    "is likely to answer or select a mutating tool."
                ),
            },
        ]
        try:
            raw = await ctx.complete_speculator(
                "sa_speculator",
                predictor_messages,
                json_mode=True,
                temperature=float(ctx.policy.get("sa_temperature", 0.1)),
                response_schema=object_schema({"actions": {"type": "array", "items": object_schema({
                    "tool": {"type": "string", "enum": safe_names}, "arguments": {"type": "object"}})}}),
            )
            draft = extract_json(raw, expected_type=dict)
        except Exception as exc:
            await ctx.trace.emit(
                "sa_prediction_failed",
                turn=turn,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {}
        actions = draft.get("actions")
        if not isinstance(actions, list):
            actions = []
        speculation_epoch = ctx.environment.state_version
        cache: dict[str, tuple[int, dict[str, Any]]] = {}
        valid_by_key: dict[str, dict[str, Any]] = {}
        for item in actions[:top_k]:
            if (
                isinstance(item, dict)
                and item.get("tool") in safe_names
                and isinstance(item.get("arguments"), dict)
            ):
                valid_by_key.setdefault(
                    _action_key(str(item["tool"]), item["arguments"]),
                    item,
                )
        valid = list(valid_by_key.values())

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
            turn=turn,
            actions=valid,
            cached=list(cache),
            state_version=speculation_epoch,
        )
        return cache

    from .methods import (
        ACTION_SYSTEM,
        FINAL_ACTION_INSTRUCTION,
        _native_assistant_message,
        _native_tool_schemas,
        action_protocol_error,
        parse_action_reply,
    )  # methods imports this module

    native_actor = ctx.policy.get("bfcl_native_tools") is True
    if native_actor and ctx.task_messages:
        task_instructions = [
            message for message in ctx.task_messages
            if message.get("role") in {"system", "developer"}
        ]
        task_conversation = [
            message for message in ctx.task_messages
            if message.get("role") not in {"system", "developer"}
        ]
        messages = [
            *task_instructions,
            {
                "role": "system",
                "content": (
                    "You are the authoritative Speculative Actions Actor. Use the native tools to complete "
                    "the task; each returned result is a harness observation. You may issue a complete "
                    "parallel batch. Finish with a plain-text answer and never invent observations."
                ),
            },
            *task_conversation,
        ]
    else:
        messages = [
            {"role": "system", "content": ACTION_SYSTEM.format(tools=ctx.environment.schema)},
            {"role": "user", "content": ctx.prompt},
        ]
    native_tools = _native_tool_schemas(ctx) if native_actor else []

    async def adopt_or_execute(
        name: str,
        arguments: dict[str, Any],
        cache: dict[str, tuple[int, dict[str, Any]]],
    ) -> dict[str, Any]:
        key = _action_key(name, arguments)
        cached = cache.pop(key, None)
        if cached is not None and cached[0] == ctx.environment.state_version:
            record = cached[1]
            result = record["result"]
            await ctx.environment.commit_isolated_calls(
                [record],
                assistant_response_id=ctx.last_actor_response_id,
            )
            await ctx.trace.emit(
                "sa_cache_hit",
                name=name,
                arguments=arguments,
                state_version=ctx.environment.state_version,
            )
            return result
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
        return result

    for turn in range(1, ctx.max_turns + 1):
        finalizing = ctx.should_finalize(turn - 1)
        if finalizing:
            messages.append({
                "role": "user",
                "content": (
                    "The action budget is exhausted. Return the final answer without another tool call."
                    if native_actor else FINAL_ACTION_INSTRUCTION
                ),
            })
            await ctx.trace.emit("budget_finalization", scope="sa", model_requests=ctx.model_budget.used)
        # Repeat the speculative window after every observation.  Starting only once at the
        # beginning is an initial prefetch control, not Speculative Actions.
        draft_task = (
            asyncio.create_task(predict_and_execute(turn, list(messages)))
            if safe_names and not finalizing
            else None
        )
        if native_actor:
            try:
                completion = await ctx.complete_native(
                    "sa_actor",
                    messages,
                    tools=[] if finalizing else native_tools,
                    tool_choice="auto",
                )
            except asyncio.CancelledError:
                if draft_task is not None:
                    draft_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await draft_task
                raise
            except Exception:
                if draft_task is not None:
                    await draft_task
                raise
            message = _native_assistant_message(completion)
            calls = message.get("tool_calls") or []
            if not calls:
                discarded = await draft_task if draft_task is not None else {}
                if discarded:
                    await ctx.trace.emit(
                        "sa_predictions_discarded",
                        turn=turn,
                        reason="actor_finished",
                        entries=len(discarded),
                    )
                answer = str(completion.content or "").strip()
                if not answer:
                    if finalizing:
                        raise RuntimeError("Speculative Actions final response was empty")
                    messages.extend([
                        message,
                        {"role": "user", "content": "Continue the harness or provide the final answer."},
                    ])
                    continue
                from .declaration import stage_selected_tool_records
                await stage_selected_tool_records(
                    ctx,
                    list(ctx.environment.proposal_calls),
                    content=answer,
                )
                return answer
            if finalizing:
                if draft_task is not None:
                    await draft_task
                raise RuntimeError("Speculative Actions requested a tool in its final response")
            cache = await draft_task if draft_task is not None else {}
            messages.append(message)
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    raise ValueError("SA native tool call omitted its function payload")
                name = str(function.get("name") or "")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError(f"SA native tool call {name!r} arguments must be an object")
                result = await adopt_or_execute(name, arguments, cache)
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": tool_result_content(result),
                })
            continue
        try:
            raw = await ctx.complete("sa_actor", messages, json_mode=True,
                                     response_schema=action_schema(ctx.environment.names, finalizing=finalizing))
        except asyncio.CancelledError:
            if draft_task is not None:
                draft_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await draft_task
            raise
        except Exception:
            # Do not leave a provider request or speculative tool read running after the
            # authoritative arm has failed. Awaiting also preserves completed Speculator
            # usage in the failed measurement.
            if draft_task is not None:
                await draft_task
            raise
        try:
            action = parse_action_reply(raw, ctx.environment.names, finalizing=finalizing)
        except ValueError as exc:
            if draft_task is not None:
                await draft_task
            if finalizing or ctx.last_response_used_final_slot:
                raise RuntimeError("Agent-loop turn budget exhausted: invalid final response") from exc
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": action_protocol_error(str(exc))},
                ]
            )
            continue
        if "final" in action:
            discarded = await draft_task if draft_task is not None else {}
            if discarded:
                await ctx.trace.emit(
                    "sa_predictions_discarded",
                    turn=turn,
                    reason="actor_finished",
                    entries=len(discarded),
                )
            answer = str(action["final"])
            if ctx.policy.get("bfcl_declaration_mode") is True:
                from .declaration import stage_selected_tool_records
                await stage_selected_tool_records(
                    ctx,
                    list(ctx.environment.proposal_calls),
                    content=answer,
                )
            return answer
        if finalizing or ctx.last_response_used_final_slot:
            raise RuntimeError("Speculative Actions turn budget exhausted: final response requested another tool")
        name = str(action.get("tool", ""))
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            if draft_task is not None:
                await draft_task
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": action_protocol_error("arguments must be one JSON object")},
                ]
            )
            continue
        cache = await draft_task if draft_task is not None else {}
        result = await adopt_or_execute(name, arguments, cache)
        canonical_action = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        messages.extend(
            [
                {"role": "assistant", "content": canonical_action},
                {"role": "user", "content": tool_result_content(result)},
            ]
        )
    raise RuntimeError("Speculative Actions turn budget exhausted")
