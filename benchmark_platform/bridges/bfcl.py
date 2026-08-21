from __future__ import annotations

import copy
import json
from typing import Any, Mapping


# Pinned BFCL revision 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 uses
# bfcl_eval.constants.type_mappings.GORILLA_TO_OPENAPI before sending native tools.
# Keep this bridge-side copy explicit: the benchmark package lives in the runtime image,
# while case preparation and baseline execution also run from the HarnessEval source tree.
GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

OPENAPI_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)


def _normalize_property_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    """Apply BFCL's official source-type conversion recursively.

    BFCL deliberately treats an unknown source type as a string. In particular, its
    official mapping defines ``any`` as ``string`` rather than as an unconstrained JSON
    Schema. Matching that behavior matters both to provider validation and to AST scoring.
    """

    schema = copy.deepcopy(dict(value))
    declared = schema.get("type")
    if declared is None:
        schema["type"] = "string"
    elif not isinstance(declared, str):
        raise ValueError(f"BFCL parameter type must be a string, got {declared!r}")
    else:
        schema["type"] = GORILLA_TO_OPENAPI.get(declared, "string")
        if declared == "float":
            schema["format"] = "float"
            suffix = "This is a float type value."
            description = str(schema.get("description") or "").strip()
            if suffix not in description:
                schema["description"] = f"{description} {suffix}".strip()

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        schema["properties"] = {
            str(name): _normalize_property_schema(child)
            for name, child in properties.items()
            if isinstance(child, Mapping)
        }
    items = schema.get("items")
    if isinstance(items, Mapping):
        schema["items"] = _normalize_property_schema(items)
    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        schema["additionalProperties"] = _normalize_property_schema(additional)
    return schema


def normalize_bfcl_parameters(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compile one BFCL function's parameters as the official OpenAI handler does."""

    parameters = copy.deepcopy(dict(value or {}))
    parameters["type"] = "object"
    properties = parameters.get("properties")
    parameters["properties"] = (
        {
            str(name): _normalize_property_schema(child)
            for name, child in properties.items()
            if isinstance(child, Mapping)
        }
        if isinstance(properties, Mapping)
        else {}
    )
    return parameters


def noncanonical_schema_types(value: Any) -> set[str]:
    """Return provider-invalid JSON Schema type names from a compiled schema."""

    invalid: set[str] = set()
    if isinstance(value, list):
        for item in value:
            invalid.update(noncanonical_schema_types(item))
    elif isinstance(value, dict):
        declared = value.get("type")
        if isinstance(declared, str) and declared not in OPENAPI_TYPES:
            invalid.add(declared)
        elif isinstance(declared, list):
            invalid.update(
                item for item in declared if isinstance(item, str) and item not in OPENAPI_TYPES
            )
        for item in value.values():
            invalid.update(noncanonical_schema_types(item))
    return invalid


def bfcl_single_turn_messages(question: Any) -> list[dict[str, Any]]:
    """Validate and preserve the official ``question[0]`` chat-message batch."""

    if not isinstance(question, list) or len(question) != 1 or not isinstance(question[0], list):
        raise ValueError("BFCL single-turn case requires question to contain exactly one message batch")
    messages = copy.deepcopy(question[0])
    if not messages:
        raise ValueError("BFCL single-turn case has no messages")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("BFCL message must be an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"BFCL message has unsupported role: {message.get('role')!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError("BFCL message content must be a string")
    return messages


def prepared_bfcl_messages(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read new prepared cases while remaining compatible with existing artifacts."""

    messages = value.get("messages")
    if isinstance(messages, list) and messages and all(isinstance(item, dict) for item in messages):
        return copy.deepcopy(messages)

    prompt = str(value.get("prompt") or "")
    try:
        decoded = json.loads(prompt)
    except json.JSONDecodeError:
        return [{"role": "user", "content": prompt}]
    try:
        return bfcl_single_turn_messages(decoded)
    except ValueError:
        return [{"role": "user", "content": prompt}]


def render_bfcl_prompt(messages: list[dict[str, Any]]) -> str:
    """Render official messages for generic agent profiles without JSON-stringifying them."""

    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0]["content"])
    labels = {"system": "System instruction", "user": "User request", "assistant": "Assistant context"}
    return "\n\n".join(
        f"{labels.get(str(message.get('role')), str(message.get('role')).title())}:\n{message['content']}"
        for message in messages
    )


def declaration_only_result(function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Acknowledge a BFCL answer call without pretending the declared function ran."""

    return {
        "recorded_function_call": function_name,
        "arguments": arguments,
        "declaration_only": True,
        "execution": "not_run",
        "terminate": True,
        "instruction": (
            "BFCL records this assistant response's function-call batch as the answer, "
            "does not execute the functions, and does not permit a later assistant turn."
        ),
    }
