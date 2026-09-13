"""BFCL's one-nominated-response declaration boundary.

Official BFCL scores exactly one assistant response.  ``inference_single_turn_FC``
compiles every function schema, makes a single ``_query_FC`` call, and
``_parse_query_response_FC`` collects every ``function_call`` that one response carried.
There is no rule upstream for merging turns or for choosing among candidates because a
bare model never produces more than one candidate.

A multi-node method produces many, and the rule has to be supplied here.  This is the
only one that keeps the graded object identical across topologies: the method nominates
exactly one of its own Actor responses, and that response's batch is published verbatim
-- no union across turns, no deduplication, no runtime picking a response on the
method's behalf.  An empty batch is a valid nomination and is how a method answers that
no supplied function fits.

A ``DeclarationOutput`` can only be built from a single completion, so "one response"
is a property of the type rather than a check that a later caller can route around.

Benchmark functions are never executed.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .api import Completion
from .core import RunContext

PUBLISHER_PROTOCOL = "bfcl-one-nominated-response-v3"


@dataclass(frozen=True)
class DeclarationOutput:
    """One Actor response and the complete native batch it carried."""

    response_id: int
    calls: tuple[tuple[str, dict[str, Any], str | None], ...]
    content: str


def recorded_calls(ctx: RunContext) -> list[dict[str, Any]]:
    """Every call the method has proposed so far, deduplicated, in first-seen order.

    Without this the nominating node has to re-derive the batch from its workers' prose,
    which is how cmas, llmcompiler and dmas dropped calls their own agents had already
    produced.  Peer agents routinely propose the identical call, so first-seen order
    keeps the list readable without hiding a genuinely repeated call in one response --
    those are made here, not replayed from this list.
    """

    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for record in ctx.environment.proposal_calls:
        name, arguments = record.get("name"), record.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"name": name, "arguments": arguments})
    return unique


def declaration_messages(
    ctx: RunContext,
    *,
    method_instruction: str | None = None,
    internal_context: str | None = None,
) -> list[dict[str, Any]]:
    """Preserve BFCL's own task roles and add only the method's final-node context."""

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
    recorded = recorded_calls(ctx)
    if recorded:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Calls your method has already recorded, in order. None of them "
                    "executed and none returned an observation. Adopt the ones the "
                    "answer needs verbatim and drop the rest -- none of them is in your "
                    "batch until you make it in this response.\n"
                    + json.dumps(recorded, ensure_ascii=False)
                ),
            }
        )
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
    """Read one assistant response's complete native call batch, preserving order.

    Official BFCL collects every ``function_call`` in the response and never drops a
    repeat, so neither does this.
    """

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


async def native_declaration_candidate(
    ctx: RunContext,
    *,
    role: str,
    messages: list[dict[str, Any]],
) -> DeclarationOutput:
    """Generate one candidate batch without nominating it.

    Methods that choose among their agents' answers (DyLAN's network, DMAS's router)
    build candidates with this and nominate the one their own selection returns.
    """

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
        calls=tuple(parse_native_declarations(completion)),
        content=completion.content,
    )
    await ctx.trace.emit(
        "declaration_candidate_response",
        response_id=response_id,
        call_count=len(output.calls),
    )
    return output


async def stage_declaration_output(
    ctx: RunContext,
    output: DeclarationOutput,
    *,
    spanned_responses: int = 1,
) -> None:
    """Nominate one existing Actor response as the method's sole BFCL answer.

    ``spanned_responses`` is how many of the method's own responses carried calls. Above
    one, the nominated response is not the method's whole trajectory and the score has to
    be read as such, so it is recorded rather than quietly normalised to 1.
    """

    if ctx.declaration_output is not None:
        raise ValueError("BFCL method nominated more than one declaration response")
    ctx.declaration_output = {
        "response_id": output.response_id,
        "calls": list(output.calls),
        "content": output.content,
        "spanned_responses": spanned_responses,
    }
    await ctx.trace.emit(
        "method_declaration_response",
        response_id=output.response_id,
        call_count=len(output.calls),
        spanned_responses=spanned_responses,
    )


async def complete_native_declaration(
    ctx: RunContext,
    *,
    role: str,
    messages: list[dict[str, Any]],
) -> str:
    """Produce the nominated response from the method's own final node."""

    output = await native_declaration_candidate(ctx, role=role, messages=messages)
    await stage_declaration_output(ctx, output)
    return output.content


def _recorded_by_response(ctx: RunContext) -> dict[int, list[tuple[str, dict[str, Any], None]]]:
    """The method's recorded calls, grouped by the Actor response that made them."""

    grouped: dict[int, list[tuple[str, dict[str, Any], None]]] = {}
    for record in ctx.environment.proposal_calls:
        response_id = record.get("assistant_response_id")
        name, arguments = record.get("name"), record.get("arguments")
        if not isinstance(response_id, int) or not isinstance(name, str):
            continue
        if not isinstance(arguments, dict):
            continue
        grouped.setdefault(response_id, []).append((name, dict(arguments), None))
    return grouped


async def stage_recorded_declaration(ctx: RunContext, *, content: str) -> str:
    """Nominate the response a loop-shaped method last acted in.

    A loop has no synthesis node to restate its answer in, and giving it one would add a
    node the algorithm does not have.  It does not need one: every method whose turn may
    carry a batch already puts its whole answer in a single response, so the nomination
    is a check here rather than something this function manufactures.

    MemGPT is the exception, and deliberately so.  One function per step *is* MemGPT, so
    its answer can span turns and only the last one is nominated.  Under BFCL's
    one-response boundary that is a real limit of the method, not of this adapter, so the
    span is reported and the score is left to stand.
    """

    grouped = _recorded_by_response(ctx)
    response_id = max(grouped) if grouped else ctx.last_actor_response_id
    if response_id is None:
        raise ValueError("BFCL method produced no authoritative Actor response")
    calls = tuple(grouped.get(response_id) or ())
    await ctx.trace.emit(
        "declaration_recorded_span",
        responses=len(grouped),
        nominated=response_id,
        nominated_calls=len(calls),
        recorded_calls=sum(len(batch) for batch in grouped.values()),
    )
    await stage_declaration_output(
        ctx,
        DeclarationOutput(response_id=response_id, calls=calls, content=content),
        spanned_responses=len(grouped),
    )
    return content


async def publish_method_declaration(
    ctx: RunContext,
    *,
    method_output: str,
    proposal_calls: list[dict[str, Any]],
    tool_capable: bool,
) -> str:
    """Publish the nominated response's batch without another model call."""

    if not isinstance(method_output, str):
        raise ValueError("BFCL method must return final text")
    if tool_capable:
        if ctx.declaration_output is None:
            raise ValueError("BFCL tool-capable method nominated no declaration response")
        response_id = int(ctx.declaration_output["response_id"])
        batch = list(ctx.declaration_output["calls"])
    else:
        response_id = ctx.last_actor_response_id
        batch = []
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
        source_response_ids=[response_id],
        publisher_llm_calls=0,
    )
    return method_output
