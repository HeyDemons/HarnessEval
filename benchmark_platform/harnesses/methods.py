from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .core import RunContext, extract_json


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
                {"role": "user", "content": "Observation: " + json.dumps(result, ensure_ascii=False)},
            ]
        )
    raise RuntimeError("Agent-loop turn budget exhausted without a final answer")


def _parse_react(text: str) -> dict[str, Any]:
    final = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    action = re.search(r"Action\s*:\s*([\w.-]+)", text, flags=re.IGNORECASE)
    if final and (action is None or final.start() < action.start()):
        return {"final": final.group(1).strip()}
    if not action:
        raise ValueError("Response is neither a ReAct action nor a final answer")
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
        return {"tool": action.group(1), "arguments": arguments}
    raise ValueError("ReAct action names a tool but supplies no JSON action input")


async def run_react(ctx: RunContext) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Solve the task by interleaving Thought, Action, and Observation, as in ReAct.\n"
                f"Available tools: {ctx.environment.schema}\n"
                "For a tool turn emit Thought, Action, and JSON Action Input. When complete emit Thought and Final Answer. "
                "Never invent an observation."
            ),
        },
        {"role": "user", "content": ctx.prompt},
    ]
    for _ in range(ctx.max_turns):
        raw = await ctx.complete("react", messages)
        try:
            action = _parse_react(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Protocol error: {exc}. Emit one complete ReAct step."},
                ]
            )
            continue
        if "final" in action:
            return str(action["final"])
        result = await ctx.environment.call(action["tool"], action["arguments"])
        messages.extend(
            [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Observation: " + json.dumps(result, ensure_ascii=False)},
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
                    f"Task: {ctx.prompt}\nReports: {json.dumps(reports, ensure_ascii=False)}\n"
                    'Return JSON only: {"final":"answer"}'
                ),
            }
        ],
    )
    if "final" not in decision:
        raise ValueError("CMAS manager synthesis omitted final")
    return str(decision["final"])


async def run_profile(ctx: RunContext) -> str:
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
        run_rewoo,
        run_sa,
    )

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
