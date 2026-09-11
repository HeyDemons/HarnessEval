"""BFCL's one-response native declaration boundary.

The benchmark functions are never executed.  A method's existing output-producing
node emits native calls, this module records that response, and the bridge publishes
the same batch without a finalizer or a proposal union.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .api import Completion
from .core import RunContext

PUBLISHER_PROTOCOL = "bfcl-native-declaration-boundary-v2"
NATIVE_SINGLE_RESPONSE_PROTOCOL = "bfcl-native-single-response-v1"
MULTI_MODEL_PROTOCOL = "multi-model-declaration-aggregation-v1"
METHOD_FINAL_PROTOCOL = "bfcl-method-final-native-v1"
TEXT_ONLY_PROTOCOL = "bfcl-text-only-empty-v1"


@dataclass(frozen=True)
class DeclarationOutput:
    response_id: int
    source_response_ids: tuple[int, ...]
    calls: tuple[tuple[str, dict[str, Any], str | None], ...]
    content: str
    protocol: str


def declaration_messages(
    ctx: RunContext,
    *,
    method_instruction: str | None = None,
    internal_context: str | None = None,
) -> list[dict[str, Any]]:
    """Preserve BFCL task roles and add only a method-owned final-node context."""

    task = copy.deepcopy(ctx.task_messages) or [
        {"role": "user", "content": ctx.prompt}
    ]
    leading: list[dict[str, Any]] = []
    while task and task[0].get("role") in {"system", "developer"}:
        leading.append(task.pop(0))
    messages = [*leading]
    if method_instruction:
        messages.append({"role": "system", "content": method_instruction})
    messages.extend(task)
    if internal_context:
        messages.append({"role": "user", "content": internal_context})
    return messages


def _assistant_message(completion: Completion) -> dict[str, Any]:
    choices = completion.raw.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("BFCL native response omitted its assistant choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("BFCL native response omitted its assistant message")
    return message


def parse_native_declarations(
    completion: Completion,
) -> list[tuple[str, dict[str, Any], str | None]]:
    """Read one assistant response's complete native call batch, preserving order."""

    batch: list[tuple[str, dict[str, Any], str | None]] = []
    for index, call in enumerate(_assistant_message(completion).get("tool_calls") or []):
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError(f"BFCL native tool call {index} is malformed")
        function = call["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"BFCL native tool call {index} has invalid JSON arguments"
                ) from exc
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError(
                f"BFCL native tool call {index} requires a name and object arguments"
            )
        call_id = call.get("id")
        batch.append((name, arguments, str(call_id) if call_id is not None else None))
    return batch


async def complete_native_declaration(
    ctx: RunContext,
    *,
    role: str,
    messages: list[dict[str, Any]],
    protocol: str,
) -> str:
    """Use the method's own final node to produce the scored native response."""

    output = await native_declaration_candidate(
        ctx,
        role=role,
        messages=messages,
        protocol=protocol,
    )
    await stage_declaration_output(ctx, output)
    return output.content


async def native_declaration_candidate(
    ctx: RunContext,
    *,
    role: str,
    messages: list[dict[str, Any]],
    protocol: str,
) -> DeclarationOutput:
    """Generate one native batch candidate without making it externally visible."""

    tools = [
        {"type": "function", "function": tool.native_schema()}
        for tool in ctx.environment.tools.values()
    ]
    completion = await ctx.complete_native(
        role,
        messages,
        tools=tools or None,
        tool_choice="auto" if tools else None,
    )
    response_id = ctx.last_actor_response_id
    if response_id is None:
        raise ValueError("BFCL method produced no authoritative Actor response")
    output = DeclarationOutput(
        response_id=response_id,
        source_response_ids=(response_id,),
        calls=tuple(parse_native_declarations(completion)),
        content=completion.content,
        protocol=protocol,
    )
    await ctx.trace.emit(
        "declaration_candidate_response",
        response_id=response_id,
        call_count=len(output.calls),
        protocol=protocol,
    )
    return output


async def stage_declaration_output(ctx: RunContext, output: DeclarationOutput) -> None:
    """Select an existing method response as the sole outward BFCL response."""

    if ctx.declaration_output is not None:
        raise ValueError("BFCL method produced more than one final declaration response")
    ctx.declaration_output = {
        "response_id": output.response_id,
        "source_response_ids": list(output.source_response_ids),
        "calls": list(output.calls),
        "content": output.content,
        "protocol": output.protocol,
    }
    await ctx.trace.emit(
        "method_declaration_response",
        response_id=output.response_id,
        source_response_ids=list(output.source_response_ids),
        call_count=len(output.calls),
        protocol=output.protocol,
    )


async def publish_method_declaration(
    ctx: RunContext,
    *,
    method_output: str,
    proposal_calls: list[dict[str, Any]],
    tool_capable: bool,
) -> str:
    """Publish a recorded final response without another model call or call union."""

    if not isinstance(method_output, str):
        raise ValueError("BFCL method must return final text")
    if tool_capable:
        if ctx.declaration_output is None:
            raise ValueError("BFCL tool-capable method omitted its native final response")
        response_id = int(ctx.declaration_output["response_id"])
        batch = list(ctx.declaration_output["calls"])
        source_response_ids = list(ctx.declaration_output["source_response_ids"])
        protocol = str(ctx.declaration_output["protocol"])
    else:
        response_id = ctx.last_actor_response_id
        batch = []
        source_response_ids = [response_id] if response_id is not None else []
        protocol = TEXT_ONLY_PROTOCOL
    if response_id is None:
        raise ValueError("BFCL method produced no authoritative Actor response")
    await ctx.environment.publish_declaration_batch(response_id, batch)
    await ctx.trace.emit(
        "declaration_response_complete",
        response_id=response_id,
        call_count=len(batch),
        environment_calls=0,
        proposal_count=len(proposal_calls),
        implementation=PUBLISHER_PROTOCOL,
        output_protocol=protocol,
        source_response_ids=source_response_ids,
        publisher_model_calls=0,
    )
    return method_output
