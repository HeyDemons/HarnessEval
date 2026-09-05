"""One native assistant response for single-turn function declaration tasks.

Multi-response algorithms are incompatible with this protocol; substituting a
single model call for their complete algorithm would mislabel the baseline.
"""
from __future__ import annotations

import json

from .core import RunContext

SINGLE_TURN_PROFILES = frozenset({"actor-only", "react", "sa", "multi-persona"})


async def run_declaration(ctx: RunContext) -> str:
    instructions = (
        "Select the function calls needed to satisfy the user's request using the supplied native tool schemas. "
        "Declare all needed calls in this one response. The functions will not execute and no tool observations "
        "will follow. If none of the functions is appropriate, do not call a function."
    )
    if ctx.profile == "react":
        instructions = "Reason about the request before selecting the appropriate actions. " + instructions
    # SA cannot pre-execute declaration-only functions; its Actor has the same
    # one-response interface as actor-only. No speculative provider call is made.
    schemas = [{"type": "function", "function": tool.prompt_schema()} for tool in ctx.environment.tools.values()]
    completion = await ctx.complete_native(
        f"{ctx.profile}_declaration", [{"role": "system", "content": instructions},
                                     {"role": "user", "content": ctx.prompt}],
        tools=schemas or None, tool_choice="auto" if schemas else None,
    )
    choices = completion.raw.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
        raise ValueError("Declaration response must contain exactly one native assistant message")
    message = choices[0]["message"]
    batch = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name, arguments = function.get("name"), function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("Declaration calls require a function name and JSON object arguments")
        batch.append((name, arguments, call.get("id")))
    # Parse the entire batch before committing any calls, so malformed trailing
    # arguments cannot turn a partially accepted batch into a successful result.
    for name, arguments, call_id in batch:
        await ctx.environment.call(name, arguments)
        await ctx.trace.emit("declaration_native_call", tool_call_id=call_id, name=name,
                             assistant_response_id=ctx.last_actor_response_id)
    await ctx.trace.emit("declaration_response_complete", response_id=ctx.last_actor_response_id,
                         call_count=len(batch), environment_calls=0,
                         implementation="native-single-response-v1")
    return completion.content
