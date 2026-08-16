from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from .base import BridgeCase, native_spec, read_case, workspace_tools


def _tool_name(original: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", original).strip("_").lower() or "tool"
    return f"{stem[:48]}_{hashlib.sha256(original.encode('utf-8')).hexdigest()[:8]}"


def load_workspace(benchmark: str, case_id: str, root: Path) -> BridgeCase:
    value = read_case(root)
    writable = benchmark == "gdpval"
    specs, handlers = workspace_tools(
        root / "workspace",
        writable=writable,
        include_web=benchmark in {"gaia", "gdpval"},
    )
    return BridgeCase(benchmark, case_id, value["prompt"], specs, handlers, {key: item for key, item in value.items() if key != "prompt"})


def _parameter_schema(tool: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    type_map = {"STRING": "string", "NUMBER": "number", "INTEGER": "integer", "BOOLEAN": "boolean", "ARRAY": "array", "OBJECT": "object"}
    for field, is_required in (("required parameters", True), ("required_parameters", True), ("optional parameters", False), ("optional_parameters", False)):
        for parameter in tool.get(field, []) or []:
            name = str(parameter.get("name", ""))
            if not name:
                continue
            properties[name] = {"type": type_map.get(str(parameter.get("type", "STRING")).upper(), "string")}
            if parameter.get("description"):
                properties[name]["description"] = str(parameter["description"])
            if is_required:
                required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def load_trajectory(case_id: str, root: Path) -> BridgeCase:
    value = read_case(root)
    specs = []
    handlers = {}
    for original in value.get("tools", []):
        original_name = str(original.get("tool name", ""))
        name = _tool_name(original_name)
        specs.append(native_spec(name, f"{original_name}: {original.get('tool description', '')}", _parameter_schema(original), parallel=True, read_only=True))

        async def invoke(arguments: dict[str, Any], *, tool: dict[str, Any] = original) -> Any:
            service_url = os.environ.get("API_URL", "")
            key = os.environ.get("TOOLBENCH_KEY", "")
            if not service_url or not key:
                raise RuntimeError("TRAJECT tool execution requires API_URL and TOOLBENCH_KEY")
            payload = {
                "category": tool.get("domain name", ""),
                "tool_name": tool.get("parent tool name", tool.get("tool name", "")),
                "api_name": tool.get("API name", tool.get("tool name", "")),
                "tool_input": arguments,
                "toolbench_key": key,
            }
            request = urllib.request.Request(service_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "toolbench_key": key}, method="POST")
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))

        handlers[name] = invoke
    return BridgeCase("trajectory-bench", case_id, value["prompt"], specs, handlers, {"source_tools": len(specs)})


def load_bfcl(case_id: str, root: Path) -> BridgeCase:
    value = read_case(root)
    specs = []
    handlers = {}
    for item in value.get("functions", []):
        function = item.get("function", item)
        name = str(function["name"])
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        if parameters.get("type") == "dict":
            parameters = {**parameters, "type": "object"}
        specs.append(native_spec(name, str(function.get("description", "")), parameters, parallel=True, read_only=True))

        async def record(arguments: dict[str, Any], *, function_name: str = name) -> Any:
            return {"recorded_function_call": function_name, "arguments": arguments}

        handlers[name] = record
    return BridgeCase("bfcl", case_id, value["prompt"], specs, handlers, {"source": value.get("source")})


def load_case(benchmark: str, case_id: str, root: Path) -> BridgeCase:
    if benchmark in {"gaia", "gdpval"}:
        return load_workspace(benchmark, case_id, root)
    if benchmark == "trajectory-bench":
        return load_trajectory(case_id, root)
    if benchmark == "bfcl":
        return load_bfcl(case_id, root)
    raise ValueError(f"No single-turn bridge for benchmark: {benchmark}")
