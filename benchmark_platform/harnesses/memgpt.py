from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from .core import RunContext


_NATIVE_USER_TOOL = "send_message_to_user"


_HEARTBEAT_PARAMETER = {
    "type": "boolean",
    "description": "Request an immediate heartbeat after function execution to chain another function.",
}


def _parameters(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_MEMORY_FUNCTIONS: dict[str, dict[str, Any]] = {
    "core_memory_append": {
        "description": "Append content to the persona or human section of core memory.",
        "parameters": _parameters(
            {"name": {"type": "string"}, "content": {"type": "string"}},
            ["name", "content"],
        ),
    },
    "core_memory_replace": {
        "description": "Replace an exact string in the persona or human section of core memory.",
        "parameters": _parameters(
            {
                "name": {"type": "string"},
                "old_content": {"type": "string"},
                "new_content": {"type": "string"},
            },
            ["name", "old_content", "new_content"],
        ),
    },
    "conversation_search": {
        "description": "Search recall memory using case-insensitive string matching.",
        "parameters": _parameters(
            {"query": {"type": "string"}, "page": {"type": "integer"}},
            ["query", "page"],
        ),
    },
    "conversation_search_date": {
        "description": "Search recall memory over an ISO date range.",
        "parameters": _parameters(
            {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "page": {"type": "integer"},
            },
            ["start_date", "end_date", "page"],
        ),
    },
    "archival_memory_insert": {
        "description": "Add a durable entry to archival memory.",
        "parameters": _parameters({"content": {"type": "string"}}, ["content"]),
    },
    "archival_memory_search": {
        "description": "Search archival memory using case-insensitive string matching.",
        "parameters": _parameters(
            {"query": {"type": "string"}, "page": {"type": "integer"}},
            ["query", "page"],
        ),
    },
    "pause_heartbeats": {
        "description": "Pause timed heartbeats. Immediate requested heartbeats are unaffected.",
        "parameters": _parameters({"minutes": {"type": "integer"}}, ["minutes"]),
    },
    "send_message": {
        "description": "Send a message to the human user.",
        "parameters": _parameters({"message": {"type": "string"}}, ["message"]),
    },
}

_NON_CHAINABLE_FUNCTIONS = {"pause_heartbeats", "send_message"}


def _with_required_heartbeat(function_schema: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(function_schema)
    parameters = schema.setdefault("parameters", {"type": "object", "properties": {}})
    properties = parameters.setdefault("properties", {})
    properties["request_heartbeat"] = copy.deepcopy(_HEARTBEAT_PARAMETER)
    required = list(parameters.get("required") or [])
    if "request_heartbeat" not in required:
        required.append("request_heartbeat")
    parameters["required"] = required
    return schema


def _memory_function_schemas() -> dict[str, dict[str, Any]]:
    return {
        name: copy.deepcopy(schema) if name in _NON_CHAINABLE_FUNCTIONS else _with_required_heartbeat(schema)
        for name, schema in _MEMORY_FUNCTIONS.items()
    }


def _benchmark_function_schemas(ctx: RunContext) -> list[dict[str, Any]]:
    # Benchmark tools play the same chainable role as MemGPT's message_chatgpt function.
    # A native episode exposes its user channel as a bridge tool. MemGPT already owns the
    # equivalent published function (`send_message`), so advertising both would let the
    # model bypass MemGPT's function history and memory lifecycle.
    declaration_only = bool(ctx.policy.get("declaration_only_tools"))
    return [
        copy.deepcopy(tool.prompt_schema())
        if declaration_only
        else _with_required_heartbeat(tool.prompt_schema())
        for tool in ctx.environment.tools.values()
        if tool.name != _NATIVE_USER_TOOL
    ]


def _event(role: str, content: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
    }


def _page(items: list[Any], page: int, count: int = 5) -> dict[str, Any]:
    if page < 0:
        raise ValueError("memory search page must be non-negative")
    start = page * count
    return {"matches": items[start : start + count], "total": len(items), "page": page}


def _system_message(
    ctx: RunContext,
    core: dict[str, str],
    recall: list[dict[str, Any]],
    archival: list[dict[str, Any]],
) -> str:
    benchmark_functions = _benchmark_function_schemas(ctx)
    if ctx.policy.get("declaration_only_tools"):
        benchmark_contract = (
            "Benchmark functions are declaration-only answer calls: use each published argument schema exactly "
            "and do not add request_heartbeat. A successful declaration-only call schedules an immediate "
            "heartbeat automatically."
        )
    else:
        benchmark_contract = "Every benchmark function requires boolean request_heartbeat inside arguments."
    return (
        "You are MemGPT, an LLM operating system with core, recall, and archival memory. Internal monologue stays "
        "inside the `thought` field; communicate with the user only through send_message. Execute exactly one "
        "function per turn. Function failure or request_heartbeat=true schedules an immediate heartbeat.\n"
        f"Core memory: {json.dumps(core, ensure_ascii=False)}\n"
        f"Recall memory contains {len(recall)} events. Archival memory contains {len(archival)} entries.\n"
        f"Memory functions: {json.dumps(_memory_function_schemas(), ensure_ascii=False)}\n"
        f"Benchmark functions: {json.dumps(benchmark_functions, ensure_ascii=False)}\n"
        'Return JSON only: {"thought":"internal monologue","function":"name","arguments":'
        '{"request_heartbeat":true}}. Every memory function except send_message and pause_heartbeats requires '
        f"boolean request_heartbeat inside arguments. {benchmark_contract}"
    )


async def run_memgpt(ctx: RunContext) -> str:
    """MemGPT's function executor, virtual memory tiers, and heartbeat queue."""
    collisions = sorted(set(_MEMORY_FUNCTIONS) & set(ctx.environment.names))
    if collisions:
        raise ValueError(f"MemGPT memory functions collide with benchmark tools: {collisions}")

    core = {
        "persona": "I am a persistent tool-using assistant that preserves important state through memory functions.",
        "human": "The human expects the benchmark task to be completed accurately.",
    }
    recall: list[dict[str, Any]] = [_event("user", ctx.prompt)]
    archival: list[dict[str, Any]] = []
    active: list[dict[str, str]] = [{"role": "user", "content": ctx.prompt}]
    memory_limit = int(ctx.policy.get("memgpt_core_memory_chars", 2000))
    warning_tokens = int(ctx.policy.get("memgpt_memory_warning_tokens", 7000))
    if min(memory_limit, warning_tokens) < 1:
        raise ValueError("MemGPT memory policy values must be positive")
    warned = False

    async def summarize_active_memory() -> None:
        nonlocal active, warned
        if len(active) <= 2:
            raise RuntimeError("MemGPT active context overflowed before any history could be summarized")
        summary = await ctx.complete(
            "memgpt_summarizer",
            [
                {
                    "role": "user",
                    "content": (
                        "Summarize the prior conversation events, preserving decisions, tool observations, and "
                        "unfinished work. Return only the summary.\n"
                        f"Events: {json.dumps(active[:-2], ensure_ascii=False)}"
                    ),
                }
            ],
        )
        active = [
            {"role": "user", "content": "Summary of prior events: " + summary},
            *active[-2:],
        ]
        warned = False
        await ctx.trace.emit("memgpt_active_memory_summarized", hidden_events=len(recall) - len(active))

    async def execute(function: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if function in ctx.environment.names:
            return await ctx.environment.call(function, arguments)
        if function == "core_memory_append":
            name = str(arguments.get("name", ""))
            if name not in core:
                raise ValueError("core_memory_append name must be persona or human")
            content = str(arguments.get("content", ""))
            proposed = core[name] + "\n" + content
            if len(proposed) > memory_limit:
                raise ValueError("core memory append exceeds the configured core-memory limit")
            core[name] = proposed
            return {"ok": True, "message": "core memory appended"}
        if function == "core_memory_replace":
            name = str(arguments.get("name", ""))
            if name not in core:
                raise ValueError("core_memory_replace name must be persona or human")
            old = str(arguments.get("old_content", ""))
            if old not in core[name]:
                raise ValueError("core memory replacement content was not found")
            proposed = core[name].replace(old, str(arguments.get("new_content", "")))
            if len(proposed) > memory_limit:
                raise ValueError("core memory replacement exceeds the configured core-memory limit")
            core[name] = proposed
            return {"ok": True, "message": "core memory replaced"}
        if function == "conversation_search":
            query = str(arguments.get("query", "")).casefold()
            matches = [item for item in recall if query in json.dumps(item["content"], ensure_ascii=False).casefold()]
            return {"ok": True, "result": _page(matches, int(arguments.get("page", 0)))}
        if function == "conversation_search_date":
            start = datetime.fromisoformat(str(arguments["start_date"])).date()
            end = datetime.fromisoformat(str(arguments["end_date"])).date()
            matches = [
                item
                for item in recall
                if start <= datetime.fromisoformat(item["timestamp"]).date() <= end
            ]
            return {"ok": True, "result": _page(matches, int(arguments.get("page", 0)))}
        if function == "archival_memory_insert":
            archival.append(_event("memory", str(arguments.get("content", ""))))
            return {"ok": True, "message": "archival memory inserted"}
        if function == "archival_memory_search":
            query = str(arguments.get("query", "")).casefold()
            matches = [item for item in archival if query in str(item["content"]).casefold()]
            return {"ok": True, "result": _page(matches, int(arguments.get("page", 0)))}
        if function == "pause_heartbeats":
            minutes = int(arguments.get("minutes", 0))
            if not 0 <= minutes <= 360:
                raise ValueError("pause_heartbeats minutes must be between 0 and 360")
            return {"ok": True, "message": f"timed heartbeats paused for {minutes} minutes"}
        raise ValueError(f"Unknown MemGPT function: {function}")

    for turn in range(ctx.max_turns):
        messages = [
            {"role": "system", "content": _system_message(ctx, core, recall, archival)},
            *active,
        ]
        prompt_tokens_before = ctx.prompt_tokens
        try:
            action = await ctx.complete_json("memgpt_processor", messages)
        except RuntimeError as exc:
            message = str(exc).casefold()
            if "context" not in message or ("maximum" not in message and "length" not in message):
                raise
            await summarize_active_memory()
            messages = [
                {"role": "system", "content": _system_message(ctx, core, recall, archival)},
                *active,
            ]
            action = await ctx.complete_json("memgpt_processor", messages)
        prompt_tokens = ctx.prompt_tokens - prompt_tokens_before
        thought = action.get("thought")
        function = action.get("function")
        arguments = action.get("arguments")
        if not isinstance(thought, str) or not isinstance(function, str) or not isinstance(arguments, dict):
            raise ValueError("MemGPT processor response omitted thought, function, or object arguments")
        active.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        recall.append(_event("assistant", action))

        if function == "send_message":
            if "message" not in arguments:
                raise ValueError("send_message omitted message")
            message = str(arguments["message"])
            if _NATIVE_USER_TOOL not in ctx.environment.names:
                return message

            # The original MemGPT CLI keeps one Agent object alive: send_message returns
            # control to the outer user loop, and the next user event is passed back into
            # that same Agent.step call history. Native conversational benchmarks have the
            # same boundary through send_message_to_user. Await it here instead of returning
            # from run_memgpt, otherwise every user turn reconstructs core, recall, and
            # archival memory from scratch.
            delivery = await ctx.environment.call(_NATIVE_USER_TOOL, {"content": message})
            if not delivery.get("ok"):
                packaged = {"function": function, "result": delivery}
                active.append({"role": "user", "content": json.dumps(packaged, ensure_ascii=False)})
                recall.append(_event("function", packaged))
                active.append({"role": "user", "content": '{"type":"heartbeat","reason":"Function call failed"}'})
                await ctx.trace.emit(
                    "memgpt_function",
                    turn=turn + 1,
                    function=function,
                    arguments={"message": message},
                    result=delivery,
                    request_heartbeat=None,
                )
                continue
            payload = delivery.get("result")
            user_message = payload.get("user_message") if isinstance(payload, dict) else None
            if not isinstance(user_message, str):
                raise RuntimeError("MemGPT native send_message returned no user message")
            packaged = {"function": function, "result": {"ok": True, "message": "message delivered"}}
            active.append({"role": "user", "content": json.dumps(packaged, ensure_ascii=False)})
            recall.append(_event("function", packaged))
            active.append({"role": "user", "content": user_message})
            recall.append(_event("user", user_message))
            await ctx.trace.emit(
                "memgpt_user_message",
                turn=turn + 1,
                message=user_message,
            )
            continue

        function_arguments = dict(arguments)
        request_heartbeat = function_arguments.pop("request_heartbeat", None)
        if not (isinstance(request_heartbeat, bool) or request_heartbeat is None):
            # The original executor warns and treats an invalid heartbeat value as absent;
            # schema validation should normally prevent this path.
            await ctx.trace.emit(
                "memgpt_invalid_heartbeat",
                turn=turn + 1,
                function=function,
                value=request_heartbeat,
            )
            request_heartbeat = None
        try:
            result = await execute(function, function_arguments)
            # The benchmark transport packages raised exceptions and validation
            # failures as ok=false. Upstream forces an error-handling heartbeat
            # for failed function execution, regardless of request_heartbeat.
            function_failed = isinstance(result, dict) and result.get("ok") is False
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            function_failed = True
        packaged = {"function": function, "result": result}
        active.append({"role": "user", "content": json.dumps(packaged, ensure_ascii=False)})
        recall.append(_event("function", packaged))
        await ctx.trace.emit(
            "memgpt_function",
            turn=turn + 1,
            function=function,
            arguments=function_arguments,
            result=result,
            request_heartbeat=request_heartbeat,
        )

        if prompt_tokens > warning_tokens and not warned:
            warning = (
                "Warning: the conversation history will soon reach its maximum length and be summarized. Save "
                "important information to core or archival memory before it leaves active context."
            )
            active.append({"role": "user", "content": warning})
            recall.append(_event("system", warning))
            warned = True
        declaration_only = bool(
            isinstance(result, dict)
            and result.get("ok") is True
            and isinstance(result.get("result"), dict)
            and result["result"].get("declaration_only") is True
        )
        if function_failed:
            active.append({"role": "user", "content": '{"type":"heartbeat","reason":"Function call failed"}'})
        elif declaration_only:
            active.append(
                {
                    "role": "user",
                    "content": '{"type":"heartbeat","reason":"Declaration-only tool call recorded"}',
                }
            )
        elif request_heartbeat:
            active.append({"role": "user", "content": '{"type":"heartbeat","reason":"AI requested"}'})
        else:
            # In the upstream event loop, omitting request_heartbeat after a successful
            # function call yields control to the caller; it is not an agent crash.  A
            # one-shot benchmark has no further external event to deliver, so finish this
            # harness invocation with no user-facing answer and let the benchmark's native
            # scorer/verifier judge the state the function calls produced.
            await ctx.trace.emit(
                "memgpt_yield",
                turn=turn + 1,
                function=function,
                reason="function completed without heartbeat request",
            )
            return ""

    raise RuntimeError("MemGPT turn budget exhausted without send_message")
