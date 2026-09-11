from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .core import RunContext, extract_json, json_safe, tool_result_content
from .reply_contracts import action_schema, instruction_list_schema, validate_reply


ACTION_SYSTEM = """You are a tool-using agent. Work only from the task and complete tool observations.
Available tools: {tools}
These tools are executed by the external benchmark controller when you return a tool JSON object.
They do not require provider-side function tools. Request an available tool by returning its JSON action;
the controller will execute it and supply the observation on the next turn. Do not claim a listed tool
is unavailable merely because this model request has no provider-side tool definitions.
Return exactly one JSON object per turn, either:
{{"tool":"tool_name","arguments":{{...}}}}
or {{"final":"answer"}}.
Do not invent a tool result. All task requirements and constraints still apply to these actions."""

# Benchmark-owned instructions can describe a native tool interface, so a rejected reply has to be
# told what the envelope is, not only that it was wrong. AutomationBench replies drifted into
# {"status": ...} progress objects and never recovered, exhausting the response budget on retries.
ACTION_CONTRACT_REMINDER = (
    'Return exactly one JSON object and nothing else, either {"tool":"tool_name","arguments":{...}} '
    'or {"final":"answer"}. A status, progress, plan or summary object is not an action.'
)
FINAL_ACTION_INSTRUCTION = 'The action budget is exhausted. Return only {"final":"best answer supported by existing observations"}. Do not call another tool.'


def action_protocol_error(detail: str) -> str:
    return f"Protocol error: {detail}. {ACTION_CONTRACT_REMINDER}"


def _normalize_action(action: dict[str, Any], names: list[str]) -> dict[str, Any]:
    # The function transport puts the action inside {"response": ...}, so a model that has
    # seen that wire shape sometimes reproduces it even on the text transport, where the
    # schema asks for the bare action. Measured on the 2026-09-08 AutomationBench sweep:
    # 17 AFlow tool-workflow replies arrived wrapped this way and 24 of 36 arms died on
    # "reply.tool is required", while the payload inside the envelope was a valid call.
    # Unwrap only a single-key envelope holding an object, so a real action named
    # "response" is untouched.
    if set(action) == {"response"} and isinstance(action["response"], dict):
        action = action["response"]
    # A reserved key carrying a falsy value is the model annotating its own action
    # ("final": false next to a real tool call), not claiming both shapes at once. The
    # contract keeps tool and final mutually exclusive, but exclusivity is about the
    # action taken, not about a bare key: measured on Tau2, 59 of 60 AFlow arms
    # died because `{"tool": ..., "arguments": {...}, "final": false}` was rejected
    # outright, and the reply itself was a valid send_message_to_user call.
    if action.get("tool") and not action.get("final"):
        action = {key: value for key, value in action.items() if key != "final"}
    elif action.get("final") and not action.get("tool"):
        action = {key: value for key, value in action.items() if key not in ("tool", "arguments")}
    if "tool" in action or "final" in action:
        return action
    if len(action) == 1:
        name, arguments = next(iter(action.items()))
        if name in names and isinstance(arguments, dict):
            return {"tool": name, "arguments": arguments}
    return action


def parse_action_reply(raw: str, names: list[str], *, finalizing: bool = False) -> dict:
    action = _normalize_action(extract_json(raw, expected_type=dict), names)
    validate_reply(action, action_schema(names, finalizing=finalizing))
    return action


def _validate_instructions(value: dict, key: str) -> None:
    for index, item in enumerate(value[key], 1):
        _instruction(item, kind=key, index=index)


async def _json_tool_loop(ctx: RunContext, role: str, *, prompt: str | None = None) -> str:
    messages = [
        {"role": "system", "content": ACTION_SYSTEM.format(tools=ctx.environment.schema)},
        {"role": "user", "content": ctx.prompt if prompt is None else prompt},
    ]
    for turn in range(ctx.max_turns):
        finalizing = ctx.should_finalize(turn)
        if finalizing:
            messages.append({"role": "user", "content": FINAL_ACTION_INSTRUCTION})
            await ctx.trace.emit("budget_finalization", scope=role, model_requests=ctx.model_budget.used)
        raw = await ctx.complete(role, messages, json_mode=True,
                                 response_schema=action_schema(ctx.environment.names, finalizing=finalizing))
        try:
            action = parse_action_reply(raw, ctx.environment.names, finalizing=finalizing)
        except ValueError as exc:
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
            return str(action["final"])
        if finalizing or ctx.last_response_used_final_slot:
            raise RuntimeError("Agent-loop turn budget exhausted: final response requested another tool")
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": action_protocol_error("arguments must be one JSON object")},
                ]
            )
            continue
        result = await ctx.environment.call(str(action.get("tool", "")), arguments)
        canonical_action = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        messages.extend(
            [
                {"role": "assistant", "content": canonical_action},
                {"role": "user", "content": tool_result_content(result)},
            ]
        )
    raise RuntimeError("Agent-loop turn budget exhausted without a final answer")


# Upstream anchors the tool name on the literal "Action Input" label
# (langchain_classic/agents/output_parsers/react_single_input.py), so it reads
# "Action: web_searchAction Input: {...}" -- a step the model emitted with no newline between
# the two labels -- as the tool "web_search". This profile's own pattern captured a bare word
# and got "web_searchAction", an unknown tool, five times across one GAIA sweep. Prefer
# upstream's shape and keep the bare-word pattern as the fallback this profile added for
# models that omit the label entirely (see the comment in _parse_react).
LABELLED_ACTION = re.compile(
    r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:", flags=re.IGNORECASE | re.DOTALL
)
BARE_ACTION = re.compile(r"Action\s*:\s*([\w.-]+)", flags=re.IGNORECASE)
OBSERVATION_STOP = re.compile(r"(?m)^[ \t]*Observation(?:[ \t]+\d+)?[ \t]*:")


def _stop_react_observation(text: str) -> str:
    """Do not consume model-authored observations as environment evidence.

    A local stop also works with reasoning/Responses providers that do not
    support stop sequences. Raw provider output and all usage stay in the trace.
    """
    match = OBSERVATION_STOP.search(text)
    return text[:match.start()].rstrip() if match else text


def _parse_react(text: str) -> dict[str, Any]:
    text = _stop_react_observation(text)
    final = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    action = LABELLED_ACTION.search(text) or BARE_ACTION.search(text)
    if final and action:
        # LangChain's ReActSingleInputOutputParser rejects an output containing both
        # signals. Executing the action or accepting the answer would each silently choose
        # one half of an ambiguous turn the published parser sends back for repair.
        raise ValueError("Parsing LLM output produced both a final answer and a parse-able action")
    if final:
        return {"final": final.group(1).strip()}
    if not action:
        # Upstream's wording for this exact case, MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE.
        raise ValueError("Invalid Format: Missing 'Action:' after 'Thought:'")
    # The arguments are the first JSON object after the action name. The literal
    # "Action Input:" label the parser used to demand is only described in prose by the
    # system prompt, so a model that names the tool and then emits its JSON has followed
    # the protocol as stated; requiring the label turned every such turn into a retry
    # until the budget was gone. Every other profile parses arguments with extract_json,
    # which never demanded a label either, so ReAct was the only stricter contract here.
    decoder = json.JSONDecoder()
    tail = text[action.end() :]
    for start, character in enumerate(tail):
        if character != "{":
            continue
        try:
            arguments, _ = decoder.raw_decode(tail[start:])
        except json.JSONDecodeError:
            continue
        # Upstream strips the captured name the same way; the labelled pattern can span a
        # newline between the two labels.
        return {"tool": action.group(1).strip(), "arguments": arguments}
    raise ValueError("ReAct action names a tool but supplies no JSON action input")


async def run_react(ctx: RunContext) -> str:
    protocol = ctx.policy.get("react_protocol", "text")
    if protocol == "native":
        from .react_native import run_react_native
        return await run_react_native(ctx)
    if protocol != "text":
        raise ValueError("react_protocol must be text or native")
    messages = [
        {
            "role": "system",
            "content": (
                # The opening line follows the ReAct paper's own instruction (ysymyth/ReAct,
                # hotpotqa.ipynb): "Solve a question answering task with interleaving Thought,
                # Action, Observation steps." The format block is LangChain's ReAct
                # FORMAT_INSTRUCTIONS (langchain_classic/agents/mrkl/prompt.py), which is where
                # this profile's Action / Action Input / Final Answer labels come from -- the
                # paper itself ends with the action Finish[answer] and has no such labels.
                #
                # Both sources enumerate the legal actions: the paper as "Action can be three
                # types: (1) Search[entity] ... (3) Finish[answer]", LangChain as "should be one
                # of [{tool_names}]". This profile was the only place that dropped that clause
                # and described the protocol in prose instead, and gpt-5.6-terra read the prose
                # as licence to narrate ("Action: Search the exact title.") and to state answers
                # with no Final Answer label -- on GAIA it emitted the exact gold string bare for
                # 15 consecutive turns and lost the episode. Restoring the upstream wording
                # changes no parser behaviour: a turn spent off-protocol is still a turn.
                "Solve the task by interleaving Thought, Action, and Observation, as in ReAct.\n"
                f"Available tools: {ctx.environment.schema}\n"
                "Use the following format:\n"
                "Question: the input question you must answer\n"
                "Thought: you should always think about what to do\n"
                f"Action: the action to take, should be one of [{', '.join(ctx.environment.names)}]\n"
                "Action Input: the input to the action, as one JSON object\n"
                "Observation: the result of the action\n"
                "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: the final answer to the original input question\n"
                "Never invent an observation."
            ),
        },
        {"role": "user", "content": ctx.prompt},
    ]
    for turn in range(ctx.max_turns):
        finalizing = ctx.should_finalize(turn)
        if finalizing:
            messages.append({"role": "user", "content": "The action budget is exhausted. Provide Final Answer using only existing observations. Do not select another Action."})
            await ctx.trace.emit("budget_finalization", scope="react-text", model_requests=ctx.model_budget.used)
        raw = await ctx.complete("react", messages)
        consumed = _stop_react_observation(raw)
        if consumed != raw:
            await ctx.trace.emit("react_observation_stop", implementation="local-output-stop-v1",
                                 generated_characters=len(raw), consumed_characters=len(consumed))
        raw = consumed
        try:
            action = _parse_react(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    # Upstream sends the parse error straight back as the next observation and
                    # relies on the format block, which is in the system message every turn, to
                    # say what the shape should have been. Restating it here would be this
                    # profile inventing coaching the reproduction does not have.
                    {"role": "user", "content": str(exc)},
                ]
            )
            continue
        if "final" in action:
            answer = str(action["final"])
            if ctx.policy.get("bfcl_declaration_mode") is True:
                from .declaration import stage_selected_tool_records
                await stage_selected_tool_records(
                    ctx,
                    list(ctx.environment.proposal_calls),
                    content=answer,
                )
            return answer
        if finalizing:
            raise RuntimeError("ReAct turn budget exhausted: final response requested another action")
        result = await ctx.environment.call(action["tool"], action["arguments"])
        messages.extend(
            [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": tool_result_content(result)},
            ]
        )
    raise RuntimeError("ReAct turn budget exhausted without a final answer")


def _instruction(item: Any, *, kind: str, index: int) -> tuple[str, str]:
    if isinstance(item, str) and item.strip():
        return str(index), item.strip()
    if isinstance(item, dict):
        instruction = item.get("instruction") or item.get("step") or item.get("task")
        if isinstance(instruction, str) and instruction.strip():
            return str(item.get("id", index)), instruction.strip()
    raise ValueError(f"{kind} item {index} must contain a non-empty textual instruction")


async def run_plan_execute(ctx: RunContext) -> str:
    plan = await ctx.complete_json(
        "planner",
        [
            {
                "role": "system",
                "content": (
                    "Let's first understand the problem and devise a plan to solve the problem. Please make the "
                    "plan the minimum number of steps required to accurately complete the task. If the task is a "
                    "question, the final step should almost always be 'Given the above steps taken, please respond "
                    "to the users original question'. Do not execute the steps."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Return the plan as JSON only: {"steps":[{"id":"s1","instruction":"step"}]}.\n'
                    f"{ctx.prompt}"
                ),
            }
        ],
        required_root_key="steps",
        strict_single_object=True,
        response_schema=instruction_list_schema("steps", nonempty=True),
        validator=lambda value: _validate_instructions(value, "steps"),
    )
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan-and-Execute planner omitted non-empty steps")
    completed: list[dict[str, str]] = []
    for index, step in enumerate(steps, start=1):
        step_id, instruction = _instruction(step, kind="Plan-and-Execute", index=index)
        if ctx.policy.get("bfcl_declaration_mode") is True and index == len(steps):
            from .declaration import (
                MULTI_MODEL_PROTOCOL,
                complete_native_declaration,
                declaration_messages,
            )
            return await complete_native_declaration(
                ctx,
                role=f"executor_{step_id}",
                messages=declaration_messages(
                    ctx,
                    method_instruction=(
                        "You are the existing final executor in Plan-and-Execute. Use the plan and prior "
                        "step reports to publish the complete BFCL native call batch in this response. "
                        "Functions are declarations only and yield no observations."
                    ),
                    internal_context=(
                        f"Plan: {json.dumps(steps, ensure_ascii=False)}\n"
                        f"Previous steps: {json.dumps(completed, ensure_ascii=False)}\n"
                        f"Final executor objective: {instruction}"
                    ),
                ),
                protocol=MULTI_MODEL_PROTOCOL,
            )
        result = await _json_tool_loop(
            ctx,
            f"executor_{step_id}",
            prompt=(
                f"Original objective: {ctx.prompt}\n\n"
                f"Previous steps: {json.dumps(completed, ensure_ascii=False)}\n\n"
                f"Current objective: {instruction}"
            ),
        )
        completed.append({"id": step_id, "instruction": instruction, "result": result})
    return completed[-1]["result"]


async def run_cmas(ctx: RunContext) -> str:
    plan = await ctx.complete_json(
        "manager",
        [
            {
                "role": "user",
                "content": (
                    "Decompose the task into independent worker assignments. Do not execute or answer.\n"
                    'Return JSON only: {"assignments":[{"id":"w1","instruction":"a self-contained worker task"}]}.\n'
                    f"Task: {ctx.prompt}"
                ),
            }
        ],
        required_root_key="assignments",
        strict_single_object=True,
        response_schema=instruction_list_schema("assignments", nonempty=False),
        validator=lambda value: _validate_instructions(value, "assignments"),
    )
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("CMAS manager omitted assignments")
    semaphore = asyncio.Semaphore(ctx.max_parallel) if ctx.max_parallel is not None else None

    async def worker(index: int, assignment: Any) -> dict[str, Any]:
        assignment_id, instruction = _instruction(assignment, kind="CMAS", index=index)

        async def execute() -> dict[str, Any]:
            result = await _json_tool_loop(
                ctx,
                f"worker_{assignment_id}",
                prompt=(
                    "Work independently on the assigned subtask. Select and use tools yourself as needed, then "
                    "return a concise but complete report as `final`.\n"
                    f"Assignment: {instruction}"
                ),
            )
            return {
                "id": assignment_id,
                "instruction": instruction,
                "result": result,
            }

        if semaphore is None:
            return await execute()
        async with semaphore:
            return await execute()

    # A bare gather returns the first exception while its siblings keep running: the bridge
    # then finalizes, scores the world and writes the token counters with worker requests
    # still in flight. Cancel and drain them first, the way the LLMCompiler scheduler does,
    # so the measurement boundary is the same one the result claims.
    running = [
        asyncio.ensure_future(worker(index, assignment))
        for index, assignment in enumerate(assignments, start=1)
    ]
    try:
        reports = await asyncio.gather(*running)
    except BaseException:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise
    # A conversational benchmark restarts this profile on every user turn, so synthesis
    # regularly lands mid-task where the honest next step is a tool call. A toolless
    # synthesis step still received the domain policy telling it to call tools, emitted
    # the action schema instead of `final` in 24 of 60 tau2 cases, and the raise scored
    # each of those episodes 0. The manager now gets the loop its workers already use:
    # it can act while work remains, and still returns `final`.
    synthesis = (
        "Synthesize the independent worker reports into the answer. The workers have "
        "already acted; take further actions yourself only if the task is unfinished.\n"
        f"Task: {ctx.prompt}\nReports: {json.dumps(json_safe(reports), ensure_ascii=False)}"
    )
    if ctx.policy.get("bfcl_declaration_mode") is True:
        from .declaration import (
            MULTI_MODEL_PROTOCOL,
            complete_native_declaration,
            declaration_messages,
        )
        return await complete_native_declaration(
            ctx,
            role="manager_synthesis",
            messages=declaration_messages(
                ctx,
                method_instruction=(
                    "You are CMAS's existing manager-synthesis node. Consolidate the worker reports "
                    "and publish the complete BFCL native call batch in this response. Internal "
                    "proposals were not executed and are not observations."
                ),
                internal_context=synthesis,
            ),
            protocol=MULTI_MODEL_PROTOCOL,
        )
    return await _json_tool_loop(ctx, "manager_synthesis", prompt=synthesis)


async def run_profile(ctx: RunContext) -> str:
    if ctx.profile == "lats" and ctx.environment.declaration_only:
        raise ValueError("LATS cannot run after the BFCL declaration publisher boundary")
    if ctx.profile == "actor-only":
        answer = await _json_tool_loop(ctx, "actor")
        if ctx.policy.get("bfcl_declaration_mode") is True:
            from .declaration import stage_selected_tool_records
            await stage_selected_tool_records(
                ctx,
                list(ctx.environment.proposal_calls),
                content=answer,
            )
        return answer
    if ctx.profile == "react":
        return await run_react(ctx)
    if ctx.profile == "plan-execute":
        return await run_plan_execute(ctx)
    if ctx.profile == "cmas":
        return await run_cmas(ctx)
    from .paper_methods import (
        run_aflow,
        run_dylan,
        run_dmas,
        run_llmcompiler,
        run_lats,
        run_magentic_one,
        run_memgpt,
        run_multi_persona,
        run_sa,
    )
    from .rewoo import run_rewoo

    extended = {
        "aflow": run_aflow,
        "dylan": run_dylan,
        "dmas": run_dmas,
        "magentic-one": run_magentic_one,
        "multi-persona": run_multi_persona,
        "llmcompiler": run_llmcompiler,
        "lats": run_lats,
        "memgpt": run_memgpt,
        "rewoo": run_rewoo,
        "sa": run_sa,
    }
    if runner := extended.get(ctx.profile):
        return await runner(ctx)
    raise ValueError(f"Unknown harness profile: {ctx.profile}")
