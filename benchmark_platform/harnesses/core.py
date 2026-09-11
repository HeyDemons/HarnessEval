from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from .api import Completion, CompletionClient
from .content import ToolImage, json_safe, tool_result_content
from ..measurement import TURN_DEFINITION, TURN_SCOPE, TOKEN_DEFINITION, METRICS_VERSION, add_tokens, normalize_usage, zero_tokens


# The relay sometimes leaks the model's internal tool-call channel into plain text: a
# ``to=<tool>`` header and/or a burst of rarely-trained glitch tokens (Cyrillic, Georgian,
# Thai, Hangul, CJK spam) appears immediately before the JSON payload the caller is about
# to parse. Measured 2026-09-07 over 1811 traces: 31 responses, all in roles that describe
# a tool call as text (rewoo_planner 20, magentic-one 4, plan-execute 2, dmas/dylan/cmas 1
# each). Native-tool roles -- actor-only and react -- were never affected, because their
# calls have a real channel to go to. Only the leaked prefix is removed; ``raw`` keeps the
# original, and every removal is traced so the rate stays countable.
_GLITCH = (
    "\u0400-\u052f\u0530-\u058f\u0590-\u05ff\u0600-\u06ff\u0900-\u097f"
    "\u0e00-\u0e7f\u10a0-\u10ff\u3010\u3011\u4e00-\u9fff\uac00-\ud7af"
)
# Junk never contains the sentence punctuation that real prose uses before a brace, so a
# legitimate "returns the following JSON:\n{...}" cannot match.
_CHANNEL_LEAK = re.compile(
    rf"(?:\bto=[A-Za-z_][\w.]*)?[^\n:.\"'{{}}\[\]]*[{_GLITCH}][^\n:.\"'{{}}\[\]]*"
    rf"(?:json)?\s*(?=[{{\[])"
)


def strip_channel_leak(text: str) -> tuple[str, list[str]]:
    """Drop leaked tool-channel prefixes, returning the text and what was removed."""
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        removed.append(match.group(0))
        return ""

    return _CHANNEL_LEAK.sub(replace, text), removed


_ASSISTANT_RESPONSE_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "harnesseval_assistant_response_id",
    default=None,
)
_ACTOR_FINAL_SLOT = contextvars.ContextVar("harnesseval_actor_final_slot", default=None)


class DeclarationOnlyComplete(RuntimeError):
    """The benchmark already received its one committed assistant call batch."""


# Every tool subprocess is handed this allowlist and nothing else, so no API credential can
# reach a command the model wrote. The proxy names are on it because the container's only route
# out is the proxy the host injects (http_proxy=http://10.0.2.2:7890 on the 4090 box): a curl
# through it answers 200 while a direct connection is unreachable. Dropping those names is what
# made GAIA look like it had no network at all -- web_search runs in the harness process and
# keeps its own environment, so it worked, while every run_command fetch the model tried died
# with "Network is unreachable": requests, wget, git clone, pip install, 52 calls across one
# sweep. Cases that need a page, a dataset file or a package are unsolvable without them.
SUBPROCESS_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
)


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
    fence_matches = list(
        re.finditer(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    )
    fenced = [match.group(1).strip() for match in fence_matches]
    # Preserve generation order.  A fenced block may occur after an earlier action in the
    # raw response, so prioritising fences lets a later ``{"final": ...}`` override the
    # tool call the model generated first.  Scan the complete response before falling back
    # to individual fences (the fallback only matters when unmatched prose quotes hide the
    # fence contents from the lightweight string scanner below).
    for source in [stripped, *fenced]:
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
        decoded_through = 0
        for start in starts:
            # Once a complete outer value has decoded, braces inside that value are not
            # independent candidates.  This matters when the outer value has the wrong
            # expected type: without the boundary a nested object could be accepted as if
            # it were the model's top-level response.
            if start < decoded_through:
                continue
            try:
                value, end = decoder.raw_decode(source[start:])
            except json.JSONDecodeError:
                continue
            decoded_through = max(decoded_through, start + end)
            if expected_type is not None and not isinstance(value, expected_type):
                continue
            # The first complete value is the only action generated before an environment
            # observation.  Ignore every later value or prose fragment in this turn: models
            # sometimes narrate an entire hypothetical trajectory as
            # ``tool + <think> + tool + final``. Executing or accepting anything after the
            # first value would fabricate observations and was responsible for zero-tool
            # false completions across several multi-agent profiles.
            return value
    raise ValueError("Response did not contain one complete JSON value")


def _extract_json_object_with_root_key(text: str, root_key: str) -> dict[str, Any]:
    """Find the first complete JSON object whose root contains ``root_key``.

    This is deliberately separate from ``extract_json``. Agent action loops must keep
    honouring the first action in a generated sequence, but LLMCompiler's upstream parser
    scans the complete planner response for legal plan actions. Its JSON adaptation therefore
    needs to skip incidental objects such as ``{"query": ...}`` and select the plan object.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()
    fenced = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    ]
    for source in [*fenced, stripped]:
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
                continue
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(source[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and root_key in value:
                return value
    raise ValueError(
        f"Response did not contain a JSON object with root field {root_key!r}"
    )


def _extract_single_json_object(
    text: str, root_key: str | None = None
) -> dict[str, Any]:
    """Decode exactly one object, optionally requiring a root field.

    Planner calls differ from agent action loops: a planner response is one complete
    program, not a speculative sequence whose first action can be executed safely.  If a
    provider returns several concatenated planner drafts, choosing either endpoint silently
    changes the plan.  Rejecting the sequence lets ``complete_json`` use its existing
    protocol-repair turn and obtain one authoritative plan instead.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()
    fenced = [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE
        )
    ]
    sources = fenced or [stripped]
    matches: list[dict[str, Any]] = []
    for source in sources:
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
            elif character == "{":
                starts.append(index)
        decoded_through = 0
        for start in starts:
            if start < decoded_through:
                continue
            try:
                value, end = decoder.raw_decode(source[start:])
            except json.JSONDecodeError:
                continue
            decoded_through = max(decoded_through, start + end)
            if not isinstance(value, dict):
                continue
            if root_key is not None and root_key not in value:
                continue
            matches.append(value)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Response contained multiple complete JSON objects")
    if root_key is not None:
        raise ValueError(
            f"Response did not contain exactly one JSON object with root field {root_key!r}"
        )
    raise ValueError("Response did not contain exactly one complete JSON object")


class JsonlTrace:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def emit(self, event: str, **fields: Any) -> None:
        line = json.dumps(
            json_safe({"ts": time.time(), "event": event, **fields}),
            ensure_ascii=False,
            sort_keys=True,
        )
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

    def native_schema(self) -> dict[str, Any]:
        """Provider function fields; execution metadata remains controller-owned."""
        return {key: value for key, value in self.prompt_schema().items()
                if key in {"name", "description", "parameters"}}


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
        *,
        declaration_only: bool = False,
        proposal_only: bool = False,
        expose_execution_metadata: bool = True,
        validate_schema: bool = True,
        isolated_calls_supported: bool = True,
        isolated_handlers: Mapping[str, ToolHandler] | None = None,
        commit_isolated_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ):
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate tool names: {names}")
        self.tools = {tool.name: tool for tool in tools}
        self.handlers = dict(handlers or {})
        self.isolated_handlers = dict(isolated_handlers or {})
        unknown_handlers = sorted(set(self.handlers) - set(self.tools))
        if unknown_handlers:
            raise ValueError(f"Handlers reference unknown tools: {unknown_handlers}")
        unknown_isolated = sorted(set(self.isolated_handlers) - set(self.tools))
        if unknown_isolated:
            raise ValueError(f"Isolated handlers reference unknown tools: {unknown_isolated}")
        self.trace = trace
        self.calls: list[dict[str, Any]] = []
        self.declaration_only = declaration_only
        self.proposal_only = proposal_only
        # Scheduling/mutability flags belong to the controller.  A benchmark such as
        # BFCL defines only native function name/description/parameters for the model;
        # exposing our local ``parallel``/``read_only`` annotations changes that prompt.
        self.expose_execution_metadata = expose_execution_metadata
        self.validate_schema = validate_schema
        # Read-only business data does not imply transcript/budget isolation.
        self.isolated_calls_supported = isolated_calls_supported
        self.commit_isolated_handler = commit_isolated_handler
        self._declaration_committed = False
        self._committed_response_id: int | None = None
        self._state_condition = asyncio.Condition()
        self._active_shared = 0
        self._exclusive_active = False
        self._waiting_exclusive = 0
        self._state_version = 0
        self._images: list[ToolImage] = []

    @property
    def names(self) -> list[str]:
        return list(self.tools)

    @property
    def schema(self) -> str:
        return json.dumps(
            [
                tool.prompt_schema() if self.expose_execution_metadata else tool.native_schema()
                for tool in self.tools.values()
            ],
            ensure_ascii=False,
            sort_keys=True,
        )

    @property
    def state_version(self) -> int:
        """Monotonic environment version advanced by every invoked mutating tool."""
        return self._state_version

    @property
    def declaration_committed(self) -> bool:
        return self._declaration_committed

    @property
    def declaration_response_id(self) -> int | None:
        return self._committed_response_id

    def commit_declaration_response(self, response_id: int) -> None:
        """Freeze the first assistant response, including an empty call batch."""
        if self.declaration_only:
            if self._declaration_committed and response_id != self._committed_response_id:
                raise DeclarationOnlyComplete("The first declaration response is already committed")
            self._declaration_committed = True
            self._committed_response_id = response_id

    def begin_declaration_publication(self) -> None:
        """End proposal collection and make the next Actor response the BFCL answer."""

        if self._declaration_committed:
            raise DeclarationOnlyComplete("The declaration response is already committed")
        self.proposal_only = False
        self.declaration_only = True

    async def publish_declaration_batch(
        self,
        response_id: int,
        batch: list[tuple[str, dict[str, Any], Any]],
    ) -> None:
        """Commit a parsed method-final batch without making another model call."""

        self.begin_declaration_publication()
        self.commit_declaration_response(response_id)
        token = _ASSISTANT_RESPONSE_ID.set(response_id)
        try:
            for name, arguments, tool_call_id in batch:
                await self.call(name, arguments)
                await self.trace.emit(
                    "declaration_native_call",
                    tool_call_id=tool_call_id,
                    name=name,
                    assistant_response_id=response_id,
                )
        finally:
            _ASSISTANT_RESPONSE_ID.reset(token)

    @property
    def proposal_calls(self) -> list[dict[str, Any]]:
        return [record for record in self.calls if record.get("proposal_only") is True]

    @property
    def committed_calls(self) -> list[dict[str, Any]]:
        if not self.declaration_only or not self._declaration_committed:
            return []
        return [
            {"name": str(record["name"]), "arguments": dict(record.get("arguments") or {})}
            for record in self.calls
            if (record.get("assistant_response_id") == self._committed_response_id
                and record.get("proposal_only") is not True)
        ]

    def _accept_declaration_call(self, response_id: int | None) -> bool:
        if not self.declaration_only:
            return True
        if not self._declaration_committed:
            self._declaration_committed = True
            self._committed_response_id = response_id
            return True
        return response_id == self._committed_response_id

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
        final_slot = _ACTOR_FINAL_SLOT.get()
        if (final_slot is not None and final_slot[0].environment is self and final_slot[1]
                and not self.declaration_only and name != "send_message_to_user"):
            from ..budgets import ModelBudgetExceeded
            await self.trace.emit("budget_action_rejected", name=name, scope="final_model_response")
            raise ModelBudgetExceeded("Final model response cannot commit a new environment action")
        response_id = _ASSISTANT_RESPONSE_ID.get()
        if not self._accept_declaration_call(response_id):
            result = {
                "ok": False,
                "error": "declaration_batch_already_committed",
                "committed_response_id": self._committed_response_id,
            }
            await self.trace.emit(
                "declaration_call_ignored",
                name=name,
                arguments=arguments,
                assistant_response_id=response_id,
                committed_response_id=self._committed_response_id,
            )
            return result
        tool = self.tools.get(name)
        proposal = self.proposal_only and not self.declaration_only
        event_prefix = "tool_proposal" if proposal else "tool"
        await self.trace.emit(
            f"{event_prefix}_request",
            name=name,
            arguments=arguments,
            assistant_response_id=response_id,
        )
        state_before = self._state_version
        if tool is None:
            result = {"ok": False, "error": "unknown_tool", "available_tools": self.names}
        elif self.validate_schema and (errors := validate_arguments(normalize_json_schema(tool.parameters), arguments)):
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
                await self._leave_exclusive(
                    mutated=not tool.read_only and not self.declaration_only and not self.proposal_only
                )
        record = {
            "name": name,
            "arguments": arguments,
            "result": result,
            "state_version_before": state_before,
            "state_version_after": self._state_version,
            "assistant_response_id": response_id,
            "proposal_only": proposal,
        }
        self._remember_images(result)
        self.calls.append(record)
        await self.trace.emit(f"{event_prefix}_result", **record)
        return result

    async def call_isolated(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        event_prefix: str = "lats",
        assistant_response_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one read-only search-branch call without publishing it as an answer.

        Tree-search profiles have to observe candidate actions while exploring, but only
        the calls on the selected trajectory are part of the profile's prediction.  In
        particular, BFCL scores the standard ``tool_request`` events themselves.  Keep
        exploratory calls in an explicitly LATS-namespaced trace and return their complete
        records so the winning path can be committed later without executing the tools a
        second time.
        """

        if not self.isolated_calls_supported:
            raise RuntimeError("This environment has no isolated tool execution/commit channel")
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Cannot isolate unknown tool {name!r}")
        if not tool.read_only:
            raise ValueError(f"Cannot isolate mutating tool {name!r}")
        token = _ASSISTANT_RESPONSE_ID.set(assistant_response_id) if assistant_response_id is not None else None
        try:
            await self.trace.emit(f"{event_prefix}_tool_request", name=name, arguments=arguments,
                                  assistant_response_id=_ASSISTANT_RESPONSE_ID.get())
            record = await self._execute_isolated_read_only_call(tool, arguments)
        finally:
            if token is not None:
                _ASSISTANT_RESPONSE_ID.reset(token)
        await self.trace.emit(f"{event_prefix}_tool_result", **record)
        return record["result"], record

    async def commit_isolated_calls(
        self,
        records: list[dict[str, Any]],
        *,
        assistant_response_id: int | None = None,
    ) -> None:
        """Publish selected reads without merging their model-response provenance.

        An explicit Actor id is used by SA to adopt one speculative read. It may
        never relabel a different response as a BFCL declaration.
        """
        if records and not self.isolated_calls_supported:
            raise RuntimeError("This environment has no isolated tool execution/commit channel")
        if self.declaration_only and records:
            source_ids = {record.get("assistant_response_id") for record in records}
            if len(source_ids) != 1 or None in source_ids:
                raise ValueError("Cannot merge different assistant responses into one declaration batch")
            source_id = next(iter(source_ids))
            if assistant_response_id is not None and assistant_response_id != source_id:
                raise ValueError("Cannot relabel a declaration response")
        for record in records:
            name = str(record["name"])
            arguments = record["arguments"]
            tool = self.tools.get(name)
            if tool is None or not tool.read_only:
                raise ValueError(f"Cannot commit non-isolated tool record {name!r}")
            response_id = record.get("assistant_response_id")
            if assistant_response_id is not None:
                record = {**record, "source_assistant_response_id": response_id,
                          "assistant_response_id": assistant_response_id}
                response_id = assistant_response_id
            if self.commit_isolated_handler is not None:
                # Native conversation adapters use this hook to publish the Actor's
                # selected call and adopt the already computed shadow result inside the
                # benchmark-owned transcript. It is deliberately absent from ordinary
                # workspace/task environments, whose local commit is already authoritative.
                authoritative = await self.commit_isolated_handler(record)
                record = {**record, "result": authoritative}
            if not self._accept_declaration_call(response_id):
                await self.trace.emit(
                    "declaration_call_ignored",
                    name=name,
                    arguments=arguments,
                    assistant_response_id=response_id,
                    committed_response_id=self._committed_response_id,
                )
                continue
            proposal = self.proposal_only and not self.declaration_only
            record = {**record, "proposal_only": proposal}
            event_prefix = "tool_proposal" if proposal else "tool"
            await self.trace.emit(
                f"{event_prefix}_request",
                name=name,
                arguments=arguments,
                assistant_response_id=response_id,
            )
            self._remember_images(record["result"])
            self.calls.append(record)
            await self.trace.emit(f"{event_prefix}_result", **record)

    async def _execute_isolated_read_only_call(
        self, tool: ToolSpec, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        state_before = self._state_version
        if self.validate_schema and (errors := validate_arguments(normalize_json_schema(tool.parameters), arguments)):
            result = {"ok": False, "error": "invalid_arguments", "details": errors}
        elif tool.parallel and tool.read_only:
            await self._enter_shared()
            try:
                state_before = self._state_version
                result = await self._invoke_isolated(tool, arguments)
            finally:
                await self._leave_shared()
        else:
            await self._enter_exclusive()
            try:
                state_before = self._state_version
                result = await self._invoke_isolated(tool, arguments)
            finally:
                # A failed mutating tool may have applied a partial side effect. Treat
                # every invocation as a state boundary; invalid arguments never enter.
                await self._leave_exclusive(mutated=not tool.read_only)
        record = {
            "name": tool.name,
            "arguments": arguments,
            "result": result,
            "state_version_before": state_before,
            "state_version_after": self._state_version,
            "assistant_response_id": _ASSISTANT_RESPONSE_ID.get(),
            "proposal_only": self.proposal_only and not self.declaration_only,
        }
        return record

    async def _invoke_isolated(self, tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        if handler := self.isolated_handlers.get(tool.name):
            try:
                value = await handler(arguments)
            except Exception as exc:
                return {"ok": False, "error": "isolated_tool_handler_failed",
                        "detail": f"{type(exc).__name__}: {exc}"}
            return value if isinstance(value, dict) and "ok" in value else {"ok": True, "result": value}
        return await self._invoke(tool, arguments)

    def _remember_images(self, value: Any) -> None:
        if isinstance(value, ToolImage):
            if all(existing is not value for existing in self._images):
                self._images.append(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                self._remember_images(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._remember_images(item)

    def with_images(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach tool-produced images to the latest user message for every later call."""
        if not self._images:
            return messages
        rendered = [dict(message) for message in messages]
        existing = {
            id(part.get("image"))
            for message in rendered
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
            if isinstance(part, dict) and isinstance(part.get("image"), ToolImage)
        }
        additions = [image for image in self._images if id(image) not in existing]
        if not additions:
            return rendered
        target = next(
            (index for index in range(len(rendered) - 1, -1, -1) if rendered[index].get("role") == "user"),
            None,
        )
        if target is None:
            rendered.append({"role": "user", "content": []})
            target = len(rendered) - 1
        content = rendered[target].get("content", "")
        parts = list(content) if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        parts.extend({"type": "image", "image": image} for image in additions)
        rendered[target]["content"] = parts
        return rendered

    async def _invoke(self, tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.declaration_only:
            from ..bridges.bfcl import declaration_only_result
            return {"ok": True, "result": declaration_only_result(tool.name, arguments)}
        if self.proposal_only:
            from ..bridges.bfcl import proposal_only_result
            return {"ok": True, "result": proposal_only_result(tool.name, arguments)}
        if handler := self.handlers.get(tool.name):
            try:
                value = await handler(arguments)
            except Exception as exc:
                return {"ok": False, "error": "tool_handler_failed", "detail": f"{type(exc).__name__}: {exc}"}
            return value if isinstance(value, dict) and "ok" in value else {"ok": True, "result": value}
        allowed = {*SUBPROCESS_ENV, *tool.pass_env}
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
        *,
        speculator_client: CompletionClient | None = None,
        task_messages: list[dict[str, Any]] | None = None,
    ):
        self.profile = profile
        self.prompt = prompt
        self.client = client
        self.environment = environment
        self.trace = trace
        self.policy = policy
        self.speculator_client = speculator_client
        # Benchmark-owned public instructions retain their role in every planner,
        # worker and Actor request. Other bridges keep their existing empty default.
        self.task_messages = copy.deepcopy(task_messages or [])
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.actor_llm_calls = 0
        self.actor_prompt_tokens = 0
        self.actor_completion_tokens = 0
        self.speculator_llm_calls = 0
        self.speculator_prompt_tokens = 0
        self.speculator_completion_tokens = 0
        self.last_actor_response_id: int | None = None
        self.last_actor_response_tokens = 0
        # Set exactly once by the method's own BFCL output-producing response.  The
        # deterministic bridge publisher consumes this record and never calls a model.
        self.declaration_output: dict[str, Any] | None = None
        from ..budgets import ModelResponseBudget
        self.model_budget = ModelResponseBudget(policy.get("model_response_limit"))
        self.channel_requests = {"actor": 0, "speculator": 0}
        self.channel_tokens = {"actor": zero_tokens(), "speculator": zero_tokens()}
        self.channel_usage_missing = {"actor": 0, "speculator": 0}
        # Inspect-style reporting counts completed generations in this Actor
        # scope, including planners/workers. Benchmark budgets remain separate.

    @property
    def max_turns(self) -> int:
        from ..budgets import DEFAULT_LOOP_SAFETY_LIMIT, positive_int
        return positive_int(self.policy.get("max_turns", DEFAULT_LOOP_SAFETY_LIMIT), "policy.max_turns")

    def should_finalize(self, loop_index: int) -> bool:
        """Reserve the last permitted generation for an answer, not another action."""
        return self.policy.get("finalize_on_loop_limit", False) is True and (
            loop_index >= self.max_turns - 1 or self.model_budget.final_response
        )

    @property
    def last_response_used_final_slot(self) -> bool:
        value = _ACTOR_FINAL_SLOT.get()
        return value is not None and value[0] is self and value[1]

    async def _reserve_model_response(self) -> None:
        from ..budgets import ModelBudgetExceeded
        try:
            self.model_budget.reserve()
        except ModelBudgetExceeded:
            await self.trace.emit("budget_exhausted", scope="model_responses",
                                  limit=self.model_budget.limit, used=self.model_budget.used)
            raise

    def with_task_instructions(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        instructions = [message for message in self.task_messages
                        if message.get("role") in {"system", "developer"} and message not in messages]
        return [*copy.deepcopy(instructions), *messages]

    @property
    def max_parallel(self) -> int | None:
        value = self.policy.get("max_parallel")
        if value is None:
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError("policy.max_parallel must be positive when set")
        return parsed

    async def evaluate_terminal(self, answer: str | None) -> float | None:
        """The environment's own reward for a terminal answer, or None when it has none.

        LATS' published loop reads an environment reward at a terminal state. No bridge here
        exposes one to the agent: BFCL's answer key stays on the host and is graded only
        after the arm exits, so a caller must fall back to its own value model.
        """

        return None

    async def complete(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        response_schema: dict | None = None,
    ) -> str:
        return await self._complete_with(
            self.client,
            "actor",
            role,
            messages,
            json_mode=json_mode,
            temperature=temperature,
            response_schema=response_schema,
        )

    async def complete_speculator(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        response_schema: dict | None = None,
    ) -> str:
        if self.speculator_client is None:
            raise RuntimeError(
                "Speculative Actions requires an independent Speculator client"
            )
        return await self._complete_with(
            self.speculator_client,
            "speculator",
            role,
            messages,
            json_mode=json_mode,
            temperature=temperature,
            response_schema=response_schema,
        )

    async def _complete_with(
        self,
        client: CompletionClient,
        channel: str,
        role: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool,
        temperature: float | None,
        response_schema: dict | None = None,
    ) -> str:
        if response_schema is not None:
            json_mode = True
        if self.environment.declaration_only and self.environment.declaration_committed:
            raise DeclarationOnlyComplete(
                "Declaration-only benchmark already received its committed call batch"
            )
        final_slot = (self.policy.get("finalize_on_loop_limit", False) is True and self.model_budget.final_response)
        if channel == "actor":
            await self._reserve_model_response()
        messages = self.environment.with_images(self.with_task_instructions(messages))
        await self.trace.emit(
            "llm_request",
            role=role,
            channel=channel,
            messages=messages,
            json_mode=json_mode,
            temperature=temperature,
            response_schema=response_schema,
        )
        self.channel_requests[channel] += 1
        constrained = getattr(client, "complete_constrained", None)
        if response_schema is not None and callable(constrained):
            completion = await constrained(messages, response_schema=response_schema, temperature=temperature)
        else:
            completion = await client.complete(messages, json_mode=json_mode, temperature=temperature)
        self.llm_calls += 1
        self._record_usage(channel, completion)
        if channel == "speculator":
            self.speculator_llm_calls += 1
            self.speculator_prompt_tokens += completion.prompt_tokens
            self.speculator_completion_tokens += completion.completion_tokens
        else:
            self.actor_llm_calls += 1
            self.actor_prompt_tokens += completion.prompt_tokens
            self.actor_completion_tokens += completion.completion_tokens
            self.last_actor_response_tokens = completion.prompt_tokens + completion.completion_tokens
        response_id = self.llm_calls
        if channel == "actor":
            self.last_actor_response_id = response_id
            _ACTOR_FINAL_SLOT.set((self, final_slot))
        _ASSISTANT_RESPONSE_ID.set(response_id)
        if channel == "actor":
            self.environment.commit_declaration_response(response_id)
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        content, leaked = strip_channel_leak(completion.content)
        if leaked:
            await self.trace.emit("channel_leak_stripped", response_id=response_id, role=role,
                                  channel=channel, removed=leaked)
        await self.trace.emit(
            "llm_response",
            response_id=response_id,
            role=role,
            channel=channel,
            content=content,
            raw=completion.raw,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            transport_retries=completion.transport_retries,
        )
        return content

    async def complete_native(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
    ) -> Completion:
        """Make one native assistant turn while preserving the common accounting boundary.

        Paper profiles normally use ``complete`` because their published protocols render
        actions as text. Magentic-One's specialists are different: the pinned AutoGen
        implementation gives participant agents native function tools, executes the calls
        from that one response, then returns a tool summary to the orchestrator. Calling the
        client directly would omit the request/response trace, token totals, response id and
        BFCL declaration boundary that every other profile records, so native turns enter
        through this sibling of ``complete``.
        """

        if self.environment.declaration_only and self.environment.declaration_committed:
            raise DeclarationOnlyComplete(
                "Declaration-only benchmark already received its committed call batch"
            )
        final_slot = self.policy.get("finalize_on_loop_limit", False) is True and self.model_budget.final_response
        await self._reserve_model_response()
        messages = self.environment.with_images(self.with_task_instructions(messages))
        await self.trace.emit(
            "llm_request",
            role=role,
            channel="actor",
            messages=messages,
            json_mode=False,
            temperature=temperature,
            tools=tools or [],
            tool_choice=tool_choice,
        )
        self.channel_requests["actor"] += 1
        completion: Completion = await self.client.complete_native(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        self.llm_calls += 1
        self._record_usage("actor", completion)
        self.actor_llm_calls += 1
        response_id = self.llm_calls
        self.last_actor_response_id = response_id
        _ACTOR_FINAL_SLOT.set((self, final_slot))
        _ASSISTANT_RESPONSE_ID.set(response_id)
        self.environment.commit_declaration_response(response_id)
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        self.actor_prompt_tokens += completion.prompt_tokens
        self.actor_completion_tokens += completion.completion_tokens
        self.last_actor_response_tokens = completion.prompt_tokens + completion.completion_tokens
        await self.trace.emit(
            "llm_response",
            response_id=response_id,
            role=role,
            channel="actor",
            content=completion.content,
            raw=completion.raw,
            elapsed_seconds=completion.elapsed_seconds,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            transport_retries=completion.transport_retries,
        )
        return completion

    @property
    def agent_turns(self) -> int:
        return self.actor_llm_calls

    def _record_usage(self, channel: str, completion: Completion) -> None:
        tokens, known = normalize_usage(completion.raw)
        add_tokens(self.channel_tokens[channel], tokens)
        self.channel_usage_missing[channel] += int(not known) + completion.transport_retries

    def usage_metrics(self) -> dict[str, Any]:
        coverage = {}
        for channel, responses in (("actor", self.actor_llm_calls), ("speculator", self.speculator_llm_calls)):
            missing = self.channel_usage_missing[channel] + max(0, self.channel_requests[channel] - responses)
            coverage[channel] = {"usage_missing_requests": missing, "usage_complete": missing == 0}
        return {
            "metrics_version": METRICS_VERSION,
            "agent_turns_definition": TURN_DEFINITION,
            "agent_turns_scope": TURN_SCOPE,
            "token_definition": TOKEN_DEFINITION,
            "actor_tokens": dict(self.channel_tokens["actor"]),
            "speculator_tokens": dict(self.channel_tokens["speculator"]),
            "usage_coverage": coverage,
            "model_requests": self.model_budget.used,
            "agent_turns": self.agent_turns,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "actor_llm_calls": self.actor_llm_calls,
            "actor_prompt_tokens": self.actor_prompt_tokens,
            "actor_completion_tokens": self.actor_completion_tokens,
            "speculator_llm_calls": self.speculator_llm_calls,
            "speculator_prompt_tokens": self.speculator_prompt_tokens,
            "speculator_completion_tokens": self.speculator_completion_tokens,
        }

    async def complete_json(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        required_root_key: str | None = None,
        strict_single_object: bool = False,
        response_schema: dict | None = None,
        validator: Callable[[dict], None] | None = None,
        final_response_schema: dict | None = None,
    ) -> dict[str, Any]:
        conversation = list(messages)
        from .reply_contracts import object_schema, validate_reply
        if response_schema is None and required_root_key is not None:
            response_schema = object_schema({required_root_key: {}})
        protocol_repairs = int(self.policy.get("protocol_repairs", 1))
        for attempt in range(protocol_repairs + 1):
            attempt_schema = (final_response_schema if final_response_schema is not None
                              and self.policy.get("finalize_on_loop_limit", False) is True
                              and self.model_budget.final_response else response_schema)
            raw = await self.complete(
                role, conversation, json_mode=True, temperature=temperature, response_schema=attempt_schema
            )
            try:
                if strict_single_object:
                    value = _extract_single_json_object(raw, required_root_key)
                elif required_root_key is not None:
                    value = _extract_json_object_with_root_key(raw, required_root_key)
                else:
                    value = extract_json(raw, expected_type=dict)
                if attempt_schema is not None:
                    validate_reply(value, attempt_schema)
                if validator is not None:
                    validator(value)
                return value
            except ValueError as error:
                if attempt >= protocol_repairs:
                    raise
                await self.trace.emit("json_reply_repair", role=role, attempt=attempt + 1,
                                      error=str(error), response_schema=attempt_schema)
                requirement = ("\nRequired reply schema: " + json.dumps(attempt_schema, ensure_ascii=False)
                               if attempt_schema is not None else "")
                conversation.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                f"Protocol error: {error}. Return one complete JSON object matching this "
                                f"caller's requested format. Preserve all required fields and arguments.{requirement}"
                            ),
                        },
                    ]
                )
        raise AssertionError("unreachable")
