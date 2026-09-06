"""Role-specific reply contracts, separate from benchmark tool-argument schemas."""
from __future__ import annotations

import math
from typing import Any

VERSION = "role-schema-v1"


def object_schema(properties: dict, required: list[str] | None = None, *, extra: bool = True) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": extra}


def action_schema(names: list[str], *, finalizing: bool = False) -> dict:
    optional = {"thought": {"type": "string"}, "reasoning": {"type": "string"}}
    final = object_schema({"final": {"type": "string"}, **optional}, ["final"], extra=False)
    if finalizing or not names:
        return final
    tool = object_schema({"tool": {"type": "string", "enum": names},
                          "arguments": {"type": "object"}, **optional}, ["tool", "arguments"], extra=False)
    return {"type": "object", "anyOf": [tool, final]}


def instruction_list_schema(key: str, *, nonempty: bool) -> dict:
    text = {"type": "string", "minLength": 1}
    item = {"anyOf": [text, {
        "type": "object", "properties": {name: text for name in ("instruction", "step", "task")},
        "anyOf": [{"required": [name]} for name in ("instruction", "step", "task")],
    }]}
    return object_schema({key: {"type": "array", "items": item, "minItems": 1 if nonempty else 0}})


def validate_reply(value: Any, schema: dict, path: str = "reply") -> None:
    """Validate the subset used by our reply contracts, without tightening tool schemas."""
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                validate_reply(value, branch, path)
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"{path} does not match any permitted reply shape")
    kind = schema.get("type")
    types = kind if isinstance(kind, list) else [kind] if kind else []
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str), "boolean": lambda x: isinstance(x, bool),
              "null": lambda x: x is None,
              "integer": lambda x: type(x) is int,
              "number": lambda x: type(x) in (int, float) and math.isfinite(x)}
    if types and not any(checks[t](value) for t in types):
        raise ValueError(f"{path} must have type {' or '.join(types)}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path} has unexpected fields: {sorted(set(value) - set(properties))}")
        for key, child in value.items():
            if key in properties:
                validate_reply(child, properties[key], f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path} must contain at least {schema['minItems']} item(s)")
        for index, child in enumerate(value):
            if "items" in schema:
                validate_reply(child, schema["items"], f"{path}[{index}]")
    if isinstance(value, str) and len(value.strip()) < schema.get("minLength", 0):
        raise ValueError(f"{path} must contain nonempty text")
