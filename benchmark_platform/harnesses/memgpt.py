from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .core import RunContext


_MEMORY_FUNCTIONS = {
    "core_memory_append": {
        "description": "Append content to the persona or human section of core memory.",
        "parameters": {"name": "persona or human", "content": "string"},
    },
    "core_memory_replace": {
        "description": "Replace an exact string in the persona or human section of core memory.",
        "parameters": {"name": "persona or human", "old_content": "string", "new_content": "string"},
    },
    "conversation_search": {
        "description": "Search recall memory using case-insensitive string matching.",
        "parameters": {"query": "string", "page": "integer"},
    },
    "conversation_search_date": {
        "description": "Search recall memory over an ISO date range.",
        "parameters": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "page": "integer"},
    },
    "archival_memory_insert": {
        "description": "Add a durable entry to archival memory.",
        "parameters": {"content": "string"},
    },
    "archival_memory_search": {
        "description": "Search archival memory using case-insensitive string matching.",
        "parameters": {"query": "string", "page": "integer"},
    },
    "pause_heartbeats": {
        "description": "Pause timed heartbeats. Immediate requested heartbeats are unaffected.",
        "parameters": {"minutes": "integer"},
    },
    "send_message": {
        "description": "Send the final response to the user.",
        "parameters": {"message": "string"},
    },
}


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
    benchmark_functions = [tool.prompt_schema() for tool in ctx.environment.tools.values()]
    return (
        "You are MemGPT, an LLM operating system with core, recall, and archival memory. Internal monologue stays "
        "inside the `thought` field; communicate with the user only through send_message. Execute exactly one "
        "function per turn. Function failure or request_heartbeat=true schedules an immediate heartbeat.\n"
        f"Core memory: {json.dumps(core, ensure_ascii=False)}\n"
        f"Recall memory contains {len(recall)} events. Archival memory contains {len(archival)} entries.\n"
        f"Memory functions: {json.dumps(_MEMORY_FUNCTIONS, ensure_ascii=False)}\n"
        f"Benchmark functions: {json.dumps(benchmark_functions, ensure_ascii=False)}\n"
        'Return JSON only: {"thought":"internal monologue","function":"name","arguments":'
        '{"request_heartbeat":true}}. Every function except send_message requires boolean request_heartbeat '
        "inside arguments."
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
            return str(arguments["message"])

        function_arguments = dict(arguments)
        request_heartbeat = function_arguments.pop("request_heartbeat", None)
        if not isinstance(request_heartbeat, bool):
            raise ValueError("MemGPT non-terminal functions require boolean request_heartbeat")
        try:
            result = await execute(function, function_arguments)
            function_failed = False
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
        if function_failed:
            active.append({"role": "user", "content": '{"type":"heartbeat","reason":"Function call failed"}'})
        elif request_heartbeat:
            active.append({"role": "user", "content": '{"type":"heartbeat","reason":"AI requested"}'})
        else:
            raise RuntimeError("MemGPT yielded without send_message or requesting a heartbeat")

    raise RuntimeError("MemGPT turn budget exhausted without send_message")
