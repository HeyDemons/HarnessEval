from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from .api import Completion, CompletionClient


JSON_SCHEMA_TYPE_ALIASES = {
    "bool": "boolean",
    "dict": "object",
    "double": "number",
    "float": "number",
    "int": "integer",
    "list": "array",
    "str": "string",
    "tuple": "array",
}


def normalize_json_schema(value: Any) -> Any:
    """Translate common source-schema aliases without changing constraints or fields."""
    if isinstance(value, list):
        return [normalize_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: normalize_json_schema(item) for key, item in value.items()}
    declared = normalized.get("type")
    if isinstance(declared, str):
        normalized["type"] = JSON_SCHEMA_TYPE_ALIASES.get(declared.lower(), declared)
    elif isinstance(declared, list):
        normalized["type"] = [
            JSON_SCHEMA_TYPE_ALIASES.get(item.lower(), item) if isinstance(item, str) else item
            for item in declared
        ]
    return normalized


def extract_json(text: str, expected_type: type | tuple[type, ...] | None = None) -> Any:
    """Decode one complete JSON value without slicing through quoted content."""
    decoder = json.JSONDecoder()
    stripped = text.strip()
    fenced = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    ]
    for source in [*fenced, stripped]:
        starts: list[int] = []
        in_string = False
        escaped = False
        for index, character in enumerate(source):
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "[{":
                starts.append(index)
        for start in starts:
            try:
                value, end = decoder.raw_decode(source[start:])
            except json.JSONDecodeError:
                continue
            remainder = source[start + end :].strip()
            if remainder:
                # A model may emit a whitespace-separated sequence of complete values,
                # narrating a whole trajectory in one turn. Accept the sequence only if
                # it tiles the remainder exactly, and take the FIRST value: the later
                # ones were produced without ever observing a tool result, so honouring
                # them would let a fabricated {"final": ...} end the episode with zero
                # tool calls. Anything else is still a protocol violation.
                if not _tiles_json_sequence(decoder, remainder):
                    continue
            if expected_type is not None and not isinstance(value, expected_type):
                continue
            return value
    raise ValueError("Response did not contain one complete JSON value")


def _tiles_json_sequence(decoder: json.JSONDecoder, text: str) -> bool:
    """True when text is exactly a whitespace-separated run of complete JSON values."""
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        try:
            _, end = decoder.raw_decode(text[cursor:])
        except json.JSONDecodeError:
            return False
        cursor += end
    return True


class JsonlTrace:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def emit(self, event: str, **fields: Any) -> None:
        line = json.dumps({"ts": time.time(), "event": event, **fields}, ensure_ascii=False, sort_keys=True)
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    command: tuple[str, ...]
    cwd: str | None = None
    parallel: bool = False
    read_only: bool = False
    pass_env: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ToolSpec":
        command = value.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise ValueError(f"Tool {value.get('name', '<unknown>')} requires a non-empty argv command")
        pass_env = value.get("pass_env", [])
        if not isinstance(pass_env, list) or not all(isinstance(item, str) for item in pass_env):
            raise ValueError("tool pass_env must be a list of environment variable names")
        return cls(
            name=str(value["name"]),
            description=str(value.get("description", "")),
            parameters=value.get("parameters") or {"type": "object"},
            command=tuple(command),
            cwd=str(value["cwd"]) if value.get("cwd") else None,
            parallel=bool(value.get("parallel", False)),
            read_only=bool(value.get("read_only", False)),
            pass_env=tuple(pass_env),
        )

    def prompt_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": normalize_json_schema(dict(self.parameters)),
            "parallel": self.parallel,
            "read_only": self.read_only,
        }


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


def validate_arguments(schema: Mapping[str, Any], value: Any, path: str = "arguments") -> list[str]:
    """Validate the JSON-Schema subset used by benchmark tool definitions."""
    errors: list[str] = []
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected] if expected else []
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    known_types = [item for item in expected_types if item in type_checks]
    if known_types and not any(type_checks[item](value) for item in known_types):
        errors.append(f"{path} must have type {' or '.join(known_types)}")
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']!r}")
    if isinstance(value, dict):
        required = schema.get("required") or []
        for name in required:
            if name not in value:
                errors.append(f"{path}.{name} is required")
        properties = schema.get("properties") or {}
        for name, item in value.items():
            child = properties.get(name)
            if isinstance(child, Mapping):
                errors.extend(validate_arguments(child, item, f"{path}.{name}"))
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            errors.extend(f"{path}.{name} is not allowed" for name in unknown)
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            errors.extend(validate_arguments(schema["items"], item, f"{path}[{index}]"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
    return errors


class ToolEnvironment:
    def __init__(
        self,
        tools: list[ToolSpec],
        trace: JsonlTrace,
        handlers: Mapping[str, ToolHandler] | None = None,
    ):
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate tool names: {names}")
        self.tools = {tool.name: tool for tool in tools}
        self.handlers = dict(handlers or {})
        unknown_handlers = sorted(set(self.handlers) - set(self.tools))
        if unknown_handlers:
            raise ValueError(f"Handlers reference unknown tools: {unknown_handlers}")
        self.trace = trace
        self.calls: list[dict[str, Any]] = []
        self._state_condition = asyncio.Condition()
        self._active_shared = 0
        self._exclusive_active = False
        self._waiting_exclusive = 0
        self._state_version = 0

    @property
    def names(self) -> list[str]:
        return list(self.tools)

    @property
    def schema(self) -> str:
        return json.dumps(
            [tool.prompt_schema() for tool in self.tools.values()],
            ensure_ascii=False,
            sort_keys=True,
        )

    @property
    def state_version(self) -> int:
        """Monotonic environment version advanced by every invoked mutating tool."""
        return self._state_version

    async def _enter_shared(self) -> None:
        async with self._state_condition:
            await self._state_condition.wait_for(
                lambda: not self._exclusive_active and self._waiting_exclusive == 0
            )
            self._active_shared += 1

    async def _leave_shared(self) -> None:
        async with self._state_condition:
            self._active_shared -= 1
            self._state_condition.notify_all()

    async def _enter_exclusive(self) -> None:
        async with self._state_condition:
            self._waiting_exclusive += 1
            try:
                await self._state_condition.wait_for(
                    lambda: not self._exclusive_active and self._active_shared == 0
                )
                self._exclusive_active = True
            finally:
                self._waiting_exclusive -= 1
                self._state_condition.notify_all()

    async def _leave_exclusive(self, *, mutated: bool) -> None:
        async with self._state_condition:
            if mutated:
                self._state_version += 1
            self._exclusive_active = False
            self._state_condition.notify_all()

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        await self.trace.emit("tool_request", name=name, arguments=arguments)
        state_before = self._state_version
        if tool is None:
            result = {"ok": False, "error": "unknown_tool", "available_tools": self.names}
        elif errors := validate_arguments(normalize_json_schema(tool.parameters), arguments):
            result = {"ok": False, "error": "invalid_arguments", "details": errors}
        elif tool.parallel and tool.read_only:
            await self._enter_shared()
            try:
                state_before = self._state_version
                result = await self._invoke(tool, arguments)
            finally:
                await self._leave_shared()
        else:
            await self._enter_exclusive()
            try:
                state_before = self._state_version
                result = await self._invoke(tool, arguments)
            finally:
                # A failed mutating tool may have applied a partial side effect. Treat
                # every invocation as a state boundary; invalid arguments never enter.
                await self._leave_exclusive(mutated=not tool.read_only)
        record = {
            "name": name,
            "arguments": arguments,
            "result": result,
            "state_version_before": state_before,
            "state_version_after": self._state_version,
        }
        self.calls.append(record)
        await self.trace.emit("tool_result", **record)
        return result

    async def _invoke(self, tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        if handler := self.handlers.get(tool.name):
            try:
                value = await handler(arguments)
            except Exception as exc:
                return {"ok": False, "error": "tool_handler_failed", "detail": f"{type(exc).__name__}: {exc}"}
            return value if isinstance(value, dict) and "ok" in value else {"ok": True, "result": value}
        allowed = {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", *tool.pass_env}
        environment = {name: value for name, value in os.environ.items() if name in allowed}
        process = await asyncio.create_subprocess_exec(
            *tool.command,
            cwd=tool.cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(json.dumps(arguments, ensure_ascii=False).encode("utf-8"))
        output = stdout.decode("utf-8", errors="replace")
        error_output = stderr.decode("utf-8", errors="replace")
        if process.returncode != 0:
            return {
                "ok": False,
                "error": "tool_process_failed",
                "returncode": process.returncode,
                "stdout": output,
                "stderr": error_output,
            }
        try:
            value = extract_json(output)
        except ValueError as exc:
            return {
                "ok": False,
                "error": "tool_result_not_json",
                "detail": str(exc),
                "stdout": output,
                "stderr": error_output,
            }
        result = {"ok": True, "result": value}
        if error_output:
            result["stderr"] = error_output
        return result


class RunContext:
    def __init__(
        self,
        profile: str,
        prompt: str,
        client: CompletionClient,
        environment: ToolEnvironment,
        trace: JsonlTrace,
        policy: dict[str, Any],
    ):
        self.profile = profile
        self.prompt = prompt
        self.client = client
        self.environment = environment
        self.trace = trace
        self.policy = policy
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def max_turns(self) -> int:
        value = int(self.policy.get("max_turns", 8))
        if value < 1:
            raise ValueError("policy.max_turns must be positive")
        return value

    @property
    def max_parallel(self) -> int | None:
        value = self.policy.get("max_parallel")
        if value is None:
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError("policy.max_parallel must be positive when set")
        return parsed

    async def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        await self.trace.emit(
            "llm_request",
            role=role,
            messages=messages,
            json_mode=json_mode,
            temperature=temperature,
        )
        completion: Completion = await self.client.complete(
            messages,
            json_mode=json_mode,
            temperature=temperature,
        )
        self.llm_calls += 1
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        await self.trace.emit(
            "llm_response",
            role=role,
            content=completion.content,
            raw=completion.raw,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            transport_retries=completion.transport_retries,
        )
        return completion.content

    async def complete_json(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        conversation = list(messages)
        protocol_repairs = int(self.policy.get("protocol_repairs", 1))
        for attempt in range(protocol_repairs + 1):
            raw = await self.complete(role, conversation, json_mode=True, temperature=temperature)
            try:
                value = extract_json(raw, expected_type=dict)
                return value
            except ValueError:
                if attempt >= protocol_repairs:
                    raise
                conversation.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": "Return one complete JSON object matching the requested schema. Preserve every field and argument.",
                        },
                    ]
                )
        raise AssertionError("unreachable")
