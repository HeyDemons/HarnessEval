from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .core import RunContext, extract_json, json_safe, tool_result_content


ACTION_SYSTEM = """You are a tool-using agent. Work only from the task and complete tool observations.
Available tools: {tools}
Return exactly one JSON object per turn, either:
{{"tool":"tool_name","arguments":{{...}}}}
or {{"final":"answer"}}.
Do not invent a tool result."""


def _normalize_action(action: dict[str, Any], names: list[str]) -> dict[str, Any]:
    if "tool" in action or "final" in action:
        return action
    if len(action) == 1:
        name, arguments = next(iter(action.items()))
        if name in names and isinstance(arguments, dict):
            return {"tool": name, "arguments": arguments}
    return action


async def _json_tool_loop(ctx: RunContext, role: str, *, prompt: str | None = None) -> str:
    messages = [
        {"role": "system", "content": ACTION_SYSTEM.format(tools=ctx.environment.schema)},
        {"role": "user", "content": ctx.prompt if prompt is None else prompt},
    ]
    for _ in range(ctx.max_turns):
        raw = await ctx.complete(role, messages, json_mode=True)
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
            return str(action["final"])
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Protocol error: arguments must be one JSON object."},
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
    for _ in range(ctx.max_turns):
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
            return str(action["final"])
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
    )
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan-and-Execute planner omitted non-empty steps")
    completed: list[dict[str, str]] = []
    for index, step in enumerate(steps, start=1):
        step_id, instruction = _instruction(step, kind="Plan-and-Execute", index=index)
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

    reports = await asyncio.gather(
        *(worker(index, assignment) for index, assignment in enumerate(assignments, start=1))
    )
    decision = await ctx.complete_json(
        "manager_synthesis",
        [
            {
                "role": "user",
                "content": (
                    "Synthesize the independent worker reports into the answer.\n"
                    f"Task: {ctx.prompt}\nReports: {json.dumps(json_safe(reports), ensure_ascii=False)}\n"
                    'Return JSON only: {"final":"answer"}'
                ),
            }
        ],
    )
    if "final" not in decision:
        raise ValueError("CMAS manager synthesis omitted final")
    return str(decision["final"])


async def run_profile(ctx: RunContext) -> str:
    if ctx.environment.declaration_only:
        from .declaration import SINGLE_TURN_PROFILES, run_declaration
        if ctx.profile not in SINGLE_TURN_PROFILES:
            raise ValueError(f"{ctx.profile} requires a multi-response agent protocol; BFCL single-turn is incompatible")
        if ctx.profile != "multi-persona":
            return await run_declaration(ctx)
    if ctx.profile == "actor-only":
        return await _json_tool_loop(ctx, "actor")
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
