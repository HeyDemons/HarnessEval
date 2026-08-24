"""Magentic-One protocol reproduction.

The orchestrator prompts and state transitions are adapted from Microsoft AutoGen at
bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0 (MIT License). Benchmark-native tools replace
the upstream browser/file/terminal implementations at the participant boundary.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .api import Completion
from .core import RunContext, json_safe, tool_result_content


# These prompts are the MagenticOneGroupChat prompts at the profile's pinned AutoGen
# revision (bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0). Keep them here rather than
# paraphrasing the state machine: the facts/task/progress ledgers are part of the method,
# not benchmark-specific prompt tuning.
ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT = """Below I will present you a request. Before we begin addressing the request, please answer the following pre-survey to the best of your ability. Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.

Here is the request:

{task}

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""


ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT = """Fantastic. To address this request we have assembled the following team:

{team}

Based on the team composition, and known and unknown facts, please devise a short bullet-point plan for addressing the original request. Remember, there is no requirement to involve all team members -- a team member's particular expertise may not be needed for this task."""


ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT = """
We are working to address the following user request:

{task}


To answer this request we have assembled the following team:

{team}


Here is an initial fact sheet to consider:

{facts}


Here is the plan to follow as best as possible:

{plan}
"""


ORCHESTRATOR_PROGRESS_LEDGER_PROMPT = """
Recall we are working on the following request:

{task}

And we have assembled the following team:

{team}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and / or getting the same responses as before? Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a handful of times.
    - Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)
    - Who should speak next? (select from: {names})
    - What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need)

Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

    {{
       "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }},
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_progress_being_made": {{
            "reason": string,
            "answer": boolean
        }},
        "next_speaker": {{
            "reason": string,
            "answer": string (select from: {names})
        }},
        "instruction_or_question": {{
            "reason": string,
            "answer": string
        }}
    }}
"""


ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT = """As a reminder, we are working to solve the following task:

{task}

It's clear we aren't making as much progress as we would like, but we may have learned something new. Please rewrite the following fact sheet, updating it to include anything new we have learned that may be helpful. Example edits can include (but are not limited to) adding new guesses, moving educated guesses to verified facts if appropriate, etc. Updates may be made to any section of the fact sheet, and more than one section of the fact sheet can be edited. This is an especially good time to update educated guesses, so please at least add or update one educated guess or hunch, and explain your reasoning.

Here is the old fact sheet:

{facts}
"""


ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT = """Please briefly explain what went wrong on this last run (the root cause of the failure), and then come up with a new plan that takes steps and/or includes hints to overcome prior challenges and especially avoids repeating the same mistakes. As before, the new plan should be concise, be expressed in bullet-point form, and consider the following team composition (do not involve any other outside people since we cannot contact anyone else):

{team}
"""


ORCHESTRATOR_FINAL_ANSWER_PROMPT = """
We are working on the following task:
{task}

We have completed the task.

The above messages contain the conversation that took place to complete the task.

Based on the information gathered, provide the final answer to the original request.
The answer should be phrased as if you were speaking to the user.
"""


CODER_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done."""


ORCHESTRATOR_NAME = "MagenticOneOrchestrator"

WORKER_ROLES = (
    (
        "FileSurfer",
        "An agent that can handle local files.",
        ("read_file", "list_files"),
    ),
    (
        "WebSurfer",
        (
            "A helpful assistant with access to a web browser. Ask them to perform web "
            "searches, open pages, and summarize or answer questions from web content."
        ),
        ("web_search",),
    ),
    (
        "Coder",
        (
            "A helpful and general-purpose AI assistant that has strong language skills, "
            "Python skills, and Linux command line skills."
        ),
        ("run_command",),
    ),
    (
        "Executor",
        (
            "A computer terminal and action executor that carries out permitted tool calls "
            "for the team."
        ),
        (),
    ),
)


PARTICIPANT_SYSTEM_MESSAGES = {
    "FileSurfer": (
        "You are a helpful AI Assistant. When given a request from the orchestrator, use "
        "available file functions to help with that request. Complete one participant turn "
        "and return the observed result to the team."
    ),
    "WebSurfer": (
        "You are a helpful AI Assistant with web retrieval tools. Respond to the most recent "
        "orchestrator request by selecting an appropriate available tool, or answer directly "
        "when no tool is needed. Complete one participant turn and return the result."
    ),
    "Coder": CODER_SYSTEM_MESSAGE,
    "Executor": (
        "You are the team's action executor. Carry out the most recent orchestrator "
        "instruction using the available native tools. If no tool is required, report the "
        "result concisely. Complete one participant turn and return control to the team."
    ),
}


def _magentic_team(names: set[str]) -> dict[str, str]:
    """Assemble only participants whose official capability exists in this environment."""

    return {
        role: description
        for role, description, required in WORKER_ROLES
        if not required or names.intersection(required)
    }


def _magentic_worker_tools(role: str, names: set[str]) -> list[str]:
    """Map dynamic benchmark tools onto the pinned participant capability boundaries."""

    if role == "WebSurfer":
        selected = names.intersection({"web_search"})
    elif role == "FileSurfer":
        selected = names.intersection({"read_file", "list_files"})
    elif role == "Coder":
        # The pinned MagenticOneCoderAgent writes code; the separately selected Executor
        # runs it. Giving Coder run_command collapses two official participants into one.
        selected = set()
    else:
        # Benchmark-native domain actions have no exact paper specialist. The catch-all
        # Executor owns them, while the orchestrator/participant turn boundary stays exact.
        selected = set(names)
    return sorted(selected)


def _team_description(workers: dict[str, str]) -> str:
    return "\n".join(f"{name}: {description}" for name, description in workers.items())


def _message(source: str, content: str) -> dict[str, str]:
    return {"source": source, "content": content}


def _thread_context(thread: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Apply AutoGen's source-to-model-role conversion for the orchestrator."""

    return [
        {
            "role": "assistant" if item["source"] == ORCHESTRATOR_NAME else "user",
            "name": item["source"],
            "content": item["content"],
        }
        for item in thread
    ]


def _participant_context(
    speaker: str,
    thread: list[dict[str, str]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PARTICIPANT_SYSTEM_MESSAGES[speaker]}
    ]
    messages.extend(
        {
            "role": "assistant" if item["source"] == speaker else "user",
            "name": item["source"],
            "content": item["content"],
        }
        for item in thread
    )
    return messages


def _native_tool_schemas(ctx: RunContext, names: list[str]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for name in names:
        prompt_schema = ctx.environment.tools[name].prompt_schema()
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": prompt_schema["name"],
                    "description": prompt_schema["description"],
                    "parameters": prompt_schema["parameters"],
                },
            }
        )
    return schemas


def _assistant_message(completion: Completion) -> dict[str, Any]:
    try:
        message = completion.raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {"role": "assistant", "content": completion.content, "tool_calls": []}
    return message if isinstance(message, dict) else {
        "role": "assistant",
        "content": completion.content,
        "tool_calls": [],
    }


async def _execute_native_call(
    ctx: RunContext,
    call: Any,
    allowed_names: set[str],
) -> str:
    if not isinstance(call, dict):
        return "Error: specialist emitted a non-object native tool call"
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments") or {}
    try:
        arguments = (
            json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        )
    except json.JSONDecodeError as exc:
        result = {
            "ok": False,
            "error": "invalid_tool_arguments_json",
            "detail": f"{type(exc).__name__}: {exc}",
        }
        await ctx.trace.emit(
            "magentic_tool_error",
            name=name,
            arguments=raw_arguments,
            result=result,
        )
        return f"{name}({raw_arguments})\n{tool_result_content(result)}"
    if not isinstance(arguments, dict):
        result = {
            "ok": False,
            "error": "invalid_tool_arguments",
            "detail": "arguments must decode to one JSON object",
        }
    elif name not in allowed_names:
        result = {
            "ok": False,
            "error": "specialist_tool_not_available",
            "available_tools": sorted(allowed_names),
        }
    else:
        result = await ctx.environment.call(name, arguments)
    if name not in allowed_names or not isinstance(arguments, dict):
        await ctx.trace.emit(
            "magentic_tool_error",
            name=name,
            arguments=json_safe(arguments),
            result=result,
        )
    return (
        f"{name}({json.dumps(json_safe(arguments), ensure_ascii=False, sort_keys=True)})\n"
        f"{tool_result_content(result)}"
    )


async def _participant_turn(
    ctx: RunContext,
    speaker: str,
    thread: list[dict[str, str]],
) -> str:
    allowed_names = _magentic_worker_tools(speaker, set(ctx.environment.names))
    tools = _native_tool_schemas(ctx, allowed_names)
    completion = await ctx.complete_native(
        speaker,
        _participant_context(speaker, thread),
        tools=tools or None,
        tool_choice="auto" if tools else None,
    )
    assistant = _assistant_message(completion)
    calls = assistant.get("tool_calls") or []
    if calls:
        call_names = [
            str((call.get("function") or {}).get("name") or "")
            for call in calls
            if isinstance(call, dict)
        ]
        if "send_message_to_user" in call_names and len(calls) != 1:
            # The native episode lifecycle cannot mix a visible user turn with environment
            # actions in one assistant message. AutoGen surfaces participant tool errors as
            # the participant response, so reject the batch before side effects and let the
            # next progress ledger issue a corrected instruction.
            response = (
                "Error: send_message_to_user must be the only tool call in a participant "
                f"response; received {call_names!r}"
            )
            await ctx.trace.emit(
                "magentic_tool_error",
                name="send_message_to_user",
                arguments=None,
                result={"ok": False, "error": "mixed_user_message_tool_batch"},
            )
            await ctx.trace.emit(
                "magentic_participant_response",
                speaker=speaker,
                kind="tool_summary",
                tool_names=call_names,
                content=response,
            )
            return response
        summaries = await asyncio.gather(
            *(
                _execute_native_call(ctx, call, set(allowed_names))
                for call in calls
            )
        )
        response = "\n\n".join(summaries)
        await ctx.trace.emit(
            "magentic_participant_response",
            speaker=speaker,
            kind="tool_summary",
            tool_names=call_names,
            content=response,
        )
        return response
    content = assistant.get("content")
    response = str(content if content is not None else completion.content or "")
    await ctx.trace.emit(
        "magentic_participant_response",
        speaker=speaker,
        kind="text",
        tool_names=[],
        content=response,
    )
    return response


def _ledger_error(ledger: Any, workers: dict[str, str]) -> str | None:
    if not isinstance(ledger, dict):
        return "progress ledger is not an object"
    if len(workers) == 1:
        only = next(iter(workers))
        # The pinned orchestrator applies this before structural validation, so a
        # single-participant team cannot fail merely because the model omitted or misspelled
        # next_speaker.
        ledger["next_speaker"] = {
            "reason": "The team consists of only one agent.",
            "answer": only,
        }
    required = (
        "is_request_satisfied",
        "is_progress_being_made",
        "is_in_loop",
        "instruction_or_question",
        "next_speaker",
    )
    for key in required:
        field = ledger.get(key)
        if not isinstance(field, dict) or "answer" not in field or "reason" not in field:
            return f"progress ledger field {key!r} omitted answer or reason"
    if (
        ledger["is_request_satisfied"]["answer"] is not True
        and ledger["next_speaker"]["answer"] not in workers
    ):
        return (
            f"invalid next speaker {ledger['next_speaker']['answer']!r}; "
            f"participants are {list(workers)!r}"
        )
    return None


async def _progress_ledger(
    ctx: RunContext,
    workers: dict[str, str],
    team: str,
    thread: list[dict[str, str]],
) -> dict[str, Any]:
    prompt = ORCHESTRATOR_PROGRESS_LEDGER_PROMPT.format(
        task=ctx.prompt,
        team=team,
        names=", ".join(workers),
    )
    messages = [*_thread_context(thread), {"role": "user", "content": prompt}]
    max_retries = int(ctx.policy.get("magentic_json_retries", 10))
    if max_retries < 1:
        raise ValueError("magentic_json_retries must be positive")
    last_error = "unknown progress ledger error"
    for attempt in range(1, max_retries + 1):
        raw = await ctx.complete("orchestrator_ledger", messages, json_mode=True)
        try:
            ledger = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            error = _ledger_error(ledger, workers)
            if error is None:
                await ctx.trace.emit(
                    "magentic_progress_ledger",
                    attempt=attempt,
                    ledger=ledger,
                )
                return ledger
            last_error = error
        await ctx.trace.emit(
            "magentic_ledger_retry",
            attempt=attempt,
            max_retries=max_retries,
            error=last_error,
        )
    raise ValueError(
        f"Failed to parse Magentic-One progress ledger after {max_retries} attempts: "
        f"{last_error}"
    )


async def _update_task_ledger(
    ctx: RunContext,
    team: str,
    thread: list[dict[str, str]],
    facts: str,
) -> tuple[str, str]:
    context = _thread_context(thread)
    context.append(
        {
            "role": "user",
            "content": ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT.format(
                task=ctx.prompt,
                facts=facts,
            ),
        }
    )
    updated_facts = await ctx.complete("orchestrator_facts_update", context)
    context.extend(
        [
            {"role": "assistant", "content": updated_facts},
            {
                "role": "user",
                "content": ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT.format(
                    team=team
                ),
            },
        ]
    )
    updated_plan = await ctx.complete("orchestrator_plan_update", context)
    return updated_facts, updated_plan


async def _final_answer(
    ctx: RunContext,
    thread: list[dict[str, str]],
    reason: str,
) -> str:
    await ctx.trace.emit("magentic_termination", reason=reason)
    return await ctx.complete(
        "orchestrator_final",
        [
            *_thread_context(thread),
            {
                "role": "user",
                "content": ORCHESTRATOR_FINAL_ANSWER_PROMPT.format(task=ctx.prompt),
            },
        ],
    )


async def run_magentic_one(ctx: RunContext) -> str:
    """Run the pinned Magentic-One ledger/participant state machine.

    The benchmark tools replace AutoGen's browser/file/terminal implementations, but the
    collaboration boundary is unchanged: one selected participant publishes one response,
    and the orchestrator updates its progress ledger before any participant can act again.
    """

    workers = _magentic_team(set(ctx.environment.names))
    team = _team_description(workers)
    planning_context: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT.format(task=ctx.prompt),
        }
    ]
    facts = await ctx.complete("orchestrator_facts", planning_context)
    planning_context.extend(
        [
            {"role": "assistant", "content": facts},
            {
                "role": "user",
                "content": ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT.format(team=team),
            },
        ]
    )
    plan = await ctx.complete("orchestrator_plan", planning_context)
    thread = [
        _message(
            ORCHESTRATOR_NAME,
            ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT.format(
                task=ctx.prompt,
                team=team,
                facts=facts,
                plan=plan,
            ),
        )
    ]
    stalls = 0
    max_rounds = int(ctx.policy.get("magentic_max_rounds", 20))
    max_stalls = int(ctx.policy.get("magentic_max_stalls", 3))
    if max_rounds < 1 or max_stalls < 1:
        raise ValueError("Magentic-One round and stall limits must be positive")

    for round_id in range(1, max_rounds + 1):
        ledger = await _progress_ledger(ctx, workers, team, thread)
        if ledger["is_request_satisfied"]["answer"] is True:
            return await _final_answer(
                ctx,
                thread,
                str(ledger["is_request_satisfied"]["reason"]),
            )

        previous_stalls = stalls
        if (
            ledger["is_progress_being_made"]["answer"] is not True
            or ledger["is_in_loop"]["answer"] is True
        ):
            stalls += 1
        else:
            stalls = max(0, stalls - 1)
        await ctx.trace.emit(
            "magentic_stall_state",
            round=round_id,
            previous=previous_stalls,
            current=stalls,
            in_loop=ledger["is_in_loop"]["answer"],
            progress=ledger["is_progress_being_made"]["answer"],
        )

        if stalls >= max_stalls:
            facts, plan = await _update_task_ledger(ctx, team, thread, facts)
            thread = [
                _message(
                    ORCHESTRATOR_NAME,
                    ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT.format(
                        task=ctx.prompt,
                        team=team,
                        facts=facts,
                        plan=plan,
                    ),
                )
            ]
            await ctx.trace.emit(
                "magentic_replan",
                round=round_id,
                stalls=stalls,
                facts=facts,
                plan=plan,
            )
            continue

        speaker = str(ledger["next_speaker"]["answer"])
        instruction = str(ledger["instruction_or_question"]["answer"])
        thread.append(_message(ORCHESTRATOR_NAME, instruction))
        await ctx.trace.emit(
            "magentic_dispatch",
            round=round_id,
            speaker=speaker,
            instruction=instruction,
        )
        response = await _participant_turn(ctx, speaker, thread)
        thread.append(_message(speaker, response))

    return await _final_answer(ctx, thread, "Max rounds reached.")
