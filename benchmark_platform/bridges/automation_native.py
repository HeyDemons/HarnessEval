"""AutomationBench's plain native-tool Actor, sharing the normal trace/budget boundary."""
from __future__ import annotations

import copy
import json

from benchmark_platform.harnesses.core import RunContext


async def run_native_actor(ctx: RunContext) -> str:
    # Only official tool definition fields go to the model; parallel/read_only
    # are controller metadata, not function schema extensions.
    tools = [{"type": "function", "function": {
        key: value for key, value in tool.prompt_schema().items()
        if key in {"name", "description", "parameters"}
    }} for tool in ctx.environment.tools.values()]
    messages = copy.deepcopy(ctx.task_messages)
    await ctx.trace.emit("automationbench_actor_protocol", protocol="native",
                         implementation="official-api-tools-v1")
    for turn in range(ctx.max_turns):
        finalizing = ctx.should_finalize(turn)
        if finalizing:
            messages.append({"role": "user", "content": (
                "The action budget is exhausted. Give your final answer based on the existing observations. "
                "No further tool calls are permitted.")})
            await ctx.trace.emit("budget_finalization", scope="actor", model_requests=ctx.model_budget.used)
        completion = await ctx.complete_native("actor", messages,
            tools=None if finalizing else tools, tool_choice="none" if finalizing else "auto")
        choices = completion.raw.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
            raise ValueError("AutomationBench Actor requires one native assistant message")
        message = {**copy.deepcopy(choices[0]["message"]), "role": "assistant"}
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return completion.content
        if finalizing:
            raise RuntimeError("Agent-loop turn budget exhausted: final response requested another tool")
        ids = [call.get("id") for call in calls]
        if any(not isinstance(call_id, str) or not call_id for call_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Native tool calls require distinct nonempty IDs")
        # StatefulToolEnv executes the entire assistant batch in order, including
        # multiple api_fetch writes; it does not reject or truncate a multi-call response.
        for call in calls:
            function = call.get("function") or {}
            try:
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                # Same optional-value sentinel as official update_tool_args.
                arguments = {key: value for key, value in arguments.items()
                             if not (isinstance(value, dict) and not value)}
                result = await ctx.environment.call(str(function.get("name", "")), arguments)
            except ValueError as error:
                result = {"ok": False, "error": "invalid_arguments", "detail": str(error)}
            # The official API returns strings. Keep its observation content verbatim;
            # the controller's ok/result envelope is retained in the trace only.
            content = result.get("result") if result.get("ok") else result
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        await ctx.trace.emit("automationbench_native_batch", response_id=ctx.last_actor_response_id,
                             tool_call_ids=ids, call_count=len(calls))
    raise RuntimeError("Agent-loop turn budget exhausted without a final answer")
