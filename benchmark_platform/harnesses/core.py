from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .api import Completion, OpenAICompatibleClient


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
            if source[start + end :].strip():
                continue
            if expected_type is not None and not isinstance(value, expected_type):
                continue
            return value
    raise ValueError("Response did not contain one complete JSON value")


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
            "parameters": dict(self.parameters),
            "parallel": self.parallel,
            "read_only": self.read_only,
        }


class ToolEnvironment:
    def __init__(self, tools: list[ToolSpec], trace: JsonlTrace):
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate tool names: {names}")
        self.tools = {tool.name: tool for tool in tools}
        self.trace = trace
        self.calls: list[dict[str, Any]] = []
        self._state_lock = asyncio.Lock()

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

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        await self.trace.emit("tool_request", name=name, arguments=arguments)
        if tool is None:
            result = {"ok": False, "error": "unknown_tool", "available_tools": self.names}
        elif tool.parallel:
            result = await self._invoke(tool, arguments)
        else:
            async with self._state_lock:
                result = await self._invoke(tool, arguments)
        record = {"name": name, "arguments": arguments, "result": result}
        self.calls.append(record)
        await self.trace.emit("tool_result", **record)
        return result

    async def _invoke(self, tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
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
        client: OpenAICompatibleClient,
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

    async def complete_json(self, role: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        conversation = list(messages)
        protocol_repairs = int(self.policy.get("protocol_repairs", 1))
        for attempt in range(protocol_repairs + 1):
            raw = await self.complete(role, conversation, json_mode=True)
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
