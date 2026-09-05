"""Native-tool ReAct adapter: reason, act, observe, and explicitly finish.

Uses the same structured-message boundary as Inspect's ReAct agent. One
environment action per turn preserves the serial ReAct interaction pattern.
No intermediate scorer or gold-conditioned retry is used.
"""
from __future__ import annotations

import json

from .core import RunContext, tool_result_content


async def run_react_native(ctx: RunContext) -> str:
    finish = "react_finish"
    if finish in ctx.environment.names:
        raise ValueError("Benchmark tool collides with ReAct's react_finish control action")
    tools = [{"type": "function", "function": tool.prompt_schema()} for tool in ctx.environment.tools.values()]
    tools.append({"type": "function", "function": {"name": finish,
        "description": "Submit the final answer when the task has been completed.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"], "additionalProperties": False}}})
    messages = [{"role": "system", "content": (
        "Solve the task by reasoning, taking an action, and observing its actual result. "
        "Use exactly one native function call per turn. Never invent a tool observation. "
        "When the task is complete, call react_finish with your final answer. "
        "Do not output simulated Action/Observation transcripts; use the native tools.")},
        {"role": "user", "content": ctx.prompt}]
    await ctx.trace.emit("react_protocol", protocol="native", implementation="native-serial-tools-v1")
    for _ in range(ctx.max_turns):
        completion = await ctx.complete_native("react", messages, tools=tools, tool_choice="auto")
        choices = completion.raw.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise ValueError("ReAct provider did not return a native assistant message")
        message = {**choices[0]["message"], "role": "assistant"}
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": "Choose one native tool, or submit your answer with react_finish."})
            continue
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            if len(calls) != 1:
                result = {"ok": False, "error": "ReAct requires exactly one action per turn; no calls were executed"}
            else:
                try:
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    if name == finish:
                        if not isinstance(arguments.get("answer"), str):
                            raise ValueError("react_finish requires a string answer")
                        await ctx.trace.emit("react_finished", response_id=ctx.last_actor_response_id)
                        return arguments["answer"]
                    result = await ctx.environment.call(name, arguments)
                except (ValueError, json.JSONDecodeError) as error:
                    result = {"ok": False, "error": "invalid_arguments", "detail": str(error)}
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": tool_result_content(result)})
    raise RuntimeError("ReAct turn budget exhausted without a final answer")
