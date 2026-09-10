"""Publish a method's own final BFCL declaration list without another LLM call."""
from __future__ import annotations

import json
import re
from typing import Any

from .core import RunContext

PUBLISHER_PROTOCOL = "bfcl-method-final-declarations-v1"
DECLARATIONS_OPEN = "<BFCL_DECLARATIONS>"
DECLARATIONS_CLOSE = "</BFCL_DECLARATIONS>"


def method_declaration_instruction(tool_schema: str) -> str:
    """Tell a tool-capable method how its own final response becomes BFCL output."""

    return f"""This is a BFCL single-turn function-selection task. The benchmark functions never execute.
Calls made while your method is reasoning are internal proposals only; their acknowledgements contain no environment observation or proof that an action succeeded. Use them to reason about the complete answer, not as executed results.

When your method would normally return its final answer, include exactly one block in that same final answer text:
{DECLARATIONS_OPEN}[{{"name":"function_name","arguments":{{}}}}]{DECLARATIONS_CLOSE}
The JSON value must be a list containing every required function invocation. Use [] when no function is appropriate. Parallel requests require multiple list items, including repeated calls to the same function when requested. Preserve exact names, arguments, values and multiplicity. Do not emit this block on intermediate planning, critique, routing or proposal turns.

Available BFCL function schemas:
{tool_schema}"""


def parse_method_declarations(method_output: str) -> list[tuple[str, dict[str, Any], None]]:
    """Parse the one declaration block emitted by the method's final response.

    Ordinary text with no block is a valid no-call response, matching BFCL's
    relevance/irrelevance boundary. A malformed or ambiguous block is an
    algorithm output error; no partial batch is published.
    """

    matches = re.findall(
        re.escape(DECLARATIONS_OPEN) + r"(.*?)" + re.escape(DECLARATIONS_CLOSE),
        method_output,
        flags=re.DOTALL,
    )
    if not matches:
        return []
    if len(matches) != 1:
        raise ValueError("BFCL method output must contain at most one declaration block")
    try:
        value = json.loads(matches[0].strip())
    except json.JSONDecodeError as exc:
        raise ValueError("BFCL declaration block must contain valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("BFCL declaration block must contain a JSON list")
    batch: list[tuple[str, dict[str, Any], None]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"name", "arguments"}:
            raise ValueError(
                f"BFCL declaration item {index} must contain exactly name and arguments"
            )
        name, arguments = item["name"], item["arguments"]
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError(
                f"BFCL declaration item {index} requires a nonempty name and object arguments"
            )
        batch.append((name, arguments, None))
    return batch


async def publish_method_declaration(
    ctx: RunContext,
    *,
    method_output: str,
    proposal_calls: list[dict[str, Any]],
    tool_capable: bool,
) -> str:
    """Turn the method's existing final response into the sole outward batch.

    This function is deterministic: it performs no provider request and adds no
    model cost. Text-only methods publish an empty call batch without receiving
    the benchmark schemas.
    """

    if not isinstance(method_output, str):
        raise ValueError("BFCL method must return final text")
    batch = parse_method_declarations(method_output) if tool_capable else []
    response_id = ctx.last_actor_response_id
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
        prompt_adapter=("method-final-contract-v1" if tool_capable else "text-only-none"),
        publisher_model_calls=0,
    )
    return method_output
