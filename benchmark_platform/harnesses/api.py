from __future__ import annotations

import asyncio
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .content import ToolImage


RETRYABLE_HTTP_STATUS = {408, 409, 424, 425, 429}


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout_seconds: float = 180.0
    transport_retries: int = 3
    max_output_tokens: int | None = None
    api_type: str = "openai-completions"
    api_auth: str = "x-api-key"
    reasoning_effort: str | None = None
    # Off for a directly-constructed config (what the tests use); from_env turns it on,
    # and from_env is the only path a real run takes. See _reassemble_stream for why.
    stream: bool = False
    # urllib's default "Python-urllib/3.x" is on a Cloudflare bot-rule blocklist at one of
    # the relays in use, which answers every request with 403 error code 1010 -- that alone
    # killed 478 of 728 arms in an earlier sweep while the product harness, whose SDK sends
    # its own UA, sailed through. Only the string is checked; the TLS fingerprint is not.
    user_agent: str = "HarnessEval/0.1"

    @classmethod
    def from_env(cls) -> "ApiConfig":
        base_url = os.getenv("HARNESS_API_BASE", "").strip()
        api_key = os.getenv("HARNESS_API_KEY", "").strip()
        model = os.getenv("HARNESS_MODEL", "").strip()
        if not base_url or not api_key or not model:
            raise RuntimeError("Set HARNESS_API_BASE, HARNESS_API_KEY, and HARNESS_MODEL")
        raw_max = os.getenv("HARNESS_MAX_OUTPUT_TOKENS", "").strip()
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("HARNESS_TEMPERATURE", "0")),
            timeout_seconds=float(os.getenv("HARNESS_API_TIMEOUT_S", "180")),
            transport_retries=max(0, int(os.getenv("HARNESS_API_RETRIES", "3"))),
            max_output_tokens=int(raw_max) if raw_max else None,
            api_type=os.getenv("HARNESS_API_TYPE", "openai-completions").strip() or "openai-completions",
            api_auth=os.getenv("HARNESS_API_AUTH", "x-api-key").strip().lower() or "x-api-key",
            reasoning_effort=os.getenv("HARNESS_REASONING_EFFORT", "").strip() or None,
            stream=os.getenv("HARNESS_API_STREAM", "1").strip() not in {"0", "false", "no"},
            user_agent=(
                os.getenv("HARNESS_API_USER_AGENT", "").strip()
                or os.getenv("HARNESS_USER_AGENT", "").strip()
                or "HarnessEval/0.1"
            ),
        )

    @classmethod
    def from_sa_env(cls, actor: "ApiConfig") -> "ApiConfig":
        """Build the independent Speculative Actions client configuration.

        The paper's latency tradeoff requires a separately selected, faster Speculator.
        Endpoint and transport settings may inherit from the Actor, but the model selection
        must be explicit so an SA run can never silently collapse to two calls to the same
        model.
        """

        model = os.getenv("HARNESS_SA_MODEL", "").strip()
        if not model:
            raise RuntimeError(
                "Set HARNESS_SA_MODEL when running the sa profile; "
                "the published method requires an explicitly selected Speculator model"
            )
        raw_max = os.getenv("HARNESS_SA_MAX_OUTPUT_TOKENS", "").strip()
        raw_stream = os.getenv("HARNESS_SA_API_STREAM", "").strip().lower()
        return cls(
            base_url=(
                os.getenv("HARNESS_SA_API_BASE", "").strip().rstrip("/")
                or actor.base_url
            ),
            api_key=os.getenv("HARNESS_SA_API_KEY", "").strip() or actor.api_key,
            model=model,
            temperature=float(
                os.getenv("HARNESS_SA_TEMPERATURE", "").strip()
                or actor.temperature
            ),
            timeout_seconds=float(
                os.getenv("HARNESS_SA_API_TIMEOUT_S", "").strip()
                or actor.timeout_seconds
            ),
            transport_retries=max(
                0,
                int(
                    os.getenv("HARNESS_SA_API_RETRIES", "").strip()
                    or actor.transport_retries
                ),
            ),
            max_output_tokens=int(raw_max) if raw_max else actor.max_output_tokens,
            api_type=(
                os.getenv("HARNESS_SA_API_TYPE", "").strip()
                or actor.api_type
            ),
            api_auth=(
                os.getenv("HARNESS_SA_API_AUTH", "").strip().lower()
                or actor.api_auth
            ),
            reasoning_effort=(
                os.getenv("HARNESS_SA_REASONING_EFFORT", "").strip()
                or actor.reasoning_effort
            ),
            stream=(
                actor.stream
                if not raw_stream
                else raw_stream not in {"0", "false", "no"}
            ),
            user_agent=(
                os.getenv("HARNESS_SA_API_USER_AGENT", "").strip()
                or actor.user_agent
            ),
        )


class StreamInterrupted(Exception):
    """A stream that ended without a finish_reason, or carried an error frame instead.

    Both look like success at the HTTP layer -- status 200, a well-formed but short body --
    so without this the caller silently records an empty completion as the model's answer.
    It joins the transport-retry budget because that is exactly the transient it is.
    """


class ProviderError(RuntimeError):
    """A retryable provider-side failure that did not produce a model outcome."""

    def __init__(self, message: str, *, kind: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate internal image tool results to Chat Completions content parts."""
    rendered: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rendered.append(message)
            continue
        parts: list[dict[str, Any]] = []
        changed = False
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "image"
                and isinstance(part.get("image"), ToolImage)
            ):
                image = part["image"]
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image.data_uri, "detail": image.detail},
                    }
                )
                changed = True
            else:
                parts.append(part)
        rendered.append({**message, "content": parts} if changed else message)
    return rendered


def _reassemble_stream(response: Any) -> dict[str, Any]:
    """Fold an SSE chat-completions stream back into the non-streaming response shape.

    Streaming is not a feature here, it is the only way to place a long call. The relay sits
    behind Cloudflare, which closes a request that has produced no bytes for ~100s with a 524;
    at reasoning_effort=high that killed 3/3 non-streamed probes at 125s while the identical
    streamed calls returned at 156s and 196s. The product harness streams, so a non-streaming
    control also fails asymmetrically -- only the baseline arm loses its hard cases.

    Callers read completion.raw["choices"][0]["message"], tool_calls included (tau_episode.py),
    so the reassembled dict has to be that exact shape and not a stream-flavoured cousin.
    """
    content: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    envelope: dict[str, Any] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise StreamInterrupted(json.dumps(chunk["error"], ensure_ascii=False)[:400])
        for key in ("id", "model", "created", "system_fingerprint"):
            if chunk.get(key) is not None:
                envelope[key] = chunk[key]
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    int(call.get("index") or 0),
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                # Name arrives whole in the opening frame and empty thereafter, but a provider
                # is allowed to split it, so append rather than assign.
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
    if finish_reason is None:
        raise StreamInterrupted("stream ended without a finish_reason")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        message["content"] = message["content"] or None
    return {
        **envelope,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    transport_retries: int
    raw: dict[str, Any]


class CompletionClient(Protocol):
    config: ApiConfig

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> Completion: ...

    async def complete_native(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion: ...


class OpenAICompatibleClient:
    def __init__(self, config: ApiConfig):
        self.config = config

    @property
    def endpoint(self) -> str:
        if self.config.base_url.endswith("/chat/completions"):
            return self.config.base_url
        return f"{self.config.base_url}/chat/completions"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> Completion:
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
        )

    def complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Completion:
        """Blocking entry point for a caller that already has a thread of its own.

        A benchmark-owned simulator drives the agent from its own worker thread and its hook
        is an ordinary synchronous function. Reaching the async API from there costs an
        asyncio.run() per call -- a fresh event loop, and a fresh default thread pool inside
        it -- to end up running this very method, which blocks anyway. Nothing is gained and
        the loop churn is exactly the shape that leaves asyncio primitives created on one
        loop being awaited on another.
        """
        return self._complete_sync(
            messages,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
            tools=tools,
            tool_choice=tool_choice,
        )

    async def complete_native(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion:
        """Preserve native chat/tool messages for benchmark-owned simulators."""
        return await asyncio.to_thread(
            self.complete_sync,
            messages,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            json_mode=False,
            tools=tools,
            tool_choice=tool_choice,
        )

    def _complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        seed: int | None,
        reasoning_effort: str | None,
        json_mode: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": _chat_messages(messages),
            "stream": self.config.stream,
        }
        if self.config.stream:
            payload["stream_options"] = {"include_usage": True}
        # A reasoning model rejects or ignores temperature; the product harness omits it for
        # the same reason, so a matched control must omit it here too. That applies to the
        # default, not to a profile that names a temperature itself: dylan passes 1.0 and lats
        # passes lats_temperature because sampling diversity IS their published mechanism --
        # DyLAN's agents have nothing to debate if they all answer identically, LATS has
        # nothing to branch on -- and folding them into the same else-branch silently disabled
        # the method under HARNESS_REASONING_EFFORT. The two parameters are not exclusive at
        # the API: gpt-5.6-terra accepted reasoning_effort=high with temperature=1.0 and five
        # such calls answered Octopus/Octopus/Otter/Elephant/Octopus, so the diversity is real.
        effective_reasoning_effort = reasoning_effort or self.config.reasoning_effort
        if effective_reasoning_effort:
            payload["reasoning_effort"] = effective_reasoning_effort
        if temperature is not None:
            payload["temperature"] = temperature
        elif not effective_reasoning_effort:
            payload["temperature"] = self.config.temperature
        if seed is not None:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if self.config.max_output_tokens is not None:
            payload["max_tokens"] = self.config.max_output_tokens
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        retries = 0
        while True:
            request = urllib.request.Request(
                self.endpoint,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": self.config.user_agent,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw = (
                        _reassemble_stream(response)
                        if self.config.stream
                        else json.loads(response.read().decode("utf-8"))
                    )
                break
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read()
                finally:
                    exc.close()
                if exc.code in {401, 403}:
                    raise ProviderError(
                        f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}",
                        kind="http",
                        status_code=exc.code,
                    ) from exc
                if exc.code not in RETRYABLE_HTTP_STATUS and exc.code < 500:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
                if retries >= self.config.transport_retries:
                    raise ProviderError(
                        f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}",
                        kind="http",
                        status_code=exc.code,
                    ) from exc
            # http.client.RemoteDisconnected subclasses ConnectionResetError and
            # BadStatusLine, neither of which is a urllib.error.URLError, so it escaped
            # this handler and killed the episode outright. Large tool-schema payloads
            # (VitaBench cross-domain sends 66 schemas) make such transient disconnects
            # routine, which is precisely what the retry budget exists for.
            except (
                TimeoutError,
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                StreamInterrupted,
                json.JSONDecodeError,
            ) as exc:
                if retries >= self.config.transport_retries:
                    raise ProviderError(
                        f"API transport failed after {retries} retries: {type(exc).__name__}: {exc}",
                        kind="transport",
                    ) from exc
            retries += 1
            time.sleep(2 ** (retries - 1))

        message = raw["choices"][0]["message"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        usage = raw.get("usage") or {}
        return Completion(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            elapsed_seconds=time.perf_counter() - started,
            transport_retries=retries,
            raw=raw,
        )


class AnthropicMessagesClient:
    """Anthropic Messages transport with an OpenAI-shaped response adapter.

    Benchmark packages already normalize their native message objects to OpenAI
    chat dictionaries. Keeping that boundary stable avoids changing any benchmark
    prompt, simulator, tool schema, or evaluator logic when the provider transport
    is Anthropic Messages.
    """

    def __init__(self, config: ApiConfig):
        if config.max_output_tokens is None:
            raise RuntimeError(
                "HARNESS_MAX_OUTPUT_TOKENS is required for anthropic-messages because "
                "the protocol requires an explicit max_tokens value"
            )
        self.config = config

    def _auth_headers(self) -> dict[str, str]:
        if self.config.api_auth == "x-api-key":
            return {"x-api-key": self.config.api_key}
        if self.config.api_auth == "bearer":
            return {"Authorization": f"Bearer {self.config.api_key}"}
        raise RuntimeError(
            "Unsupported HARNESS_API_AUTH for anthropic-messages: "
            f"{self.config.api_auth}; expected x-api-key or bearer"
        )

    @property
    def endpoint(self) -> str:
        if self.config.base_url.endswith("/messages"):
            return self.config.base_url
        if self.config.base_url.endswith("/v1"):
            return f"{self.config.base_url}/messages"
        return f"{self.config.base_url}/v1/messages"

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        # Anthropic Messages has no response_format equivalent. Harness prompts
        # already carry their JSON contract and complete_json validates/repairs
        # the returned text, so json_mode is intentionally a transport no-op.
        return await self.complete_native(messages, temperature=temperature)

    async def complete_native(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
    ) -> Completion:
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

    @staticmethod
    def _text_content(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    blocks.append({"type": "text", "text": str(item.get("text") or "")})
                elif isinstance(item, str):
                    blocks.append({"type": "text", "text": item})
            return blocks
        return [{"type": "text", "text": str(content or "")}]

    @classmethod
    def _convert_messages(
        cls, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "")
            if role == "system":
                content = message.get("content")
                if isinstance(content, list):
                    system_parts.extend(
                        str(item.get("text") or "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                else:
                    system_parts.append(str(content or ""))
                continue
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id") or message.get("id") or ""),
                    "content": str(message.get("content") or ""),
                }
                if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
                continue
            if role not in {"user", "assistant"}:
                continue
            blocks = cls._text_content(message.get("content"))
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise TypeError("Tool-call arguments must decode to an object")
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "input": arguments,
                        }
                    )
            converted.append({"role": role, "content": blocks})
        return ("\n\n".join(system_parts) if system_parts else None), converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted = []
        for tool in tools:
            function = tool.get("function") if tool.get("type") == "function" else tool
            if not isinstance(function, dict):
                raise TypeError("Tool schema must be an object")
            converted.append(
                {
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return converted

    @staticmethod
    def _convert_tool_choice(tool_choice: str | None) -> dict[str, Any] | None:
        if not tool_choice or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice in {"required", "any"}:
            return {"type": "any"}
        if tool_choice == "none":
            return None
        return {"type": "tool", "name": tool_choice}

    def _complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        temperature: float | None,
    ) -> Completion:
        system, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": converted_messages,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if system:
            payload["system"] = system
        if converted_tools:
            payload["tools"] = converted_tools
            choice = self._convert_tool_choice(tool_choice)
            if choice is not None:
                payload["tool_choice"] = choice
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        retries = 0
        while True:
            request = urllib.request.Request(
                self.endpoint,
                data=encoded,
                headers={
                    **self._auth_headers(),
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                    "User-Agent": self.config.user_agent,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = response.read()
                raw = json.loads(body.decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read()
                finally:
                    exc.close()
                if exc.code in {401, 403}:
                    raise ProviderError(
                        f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}",
                        kind="http",
                        status_code=exc.code,
                    ) from exc
                if exc.code not in RETRYABLE_HTTP_STATUS and exc.code < 500:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
                if retries >= self.config.transport_retries:
                    raise ProviderError(
                        f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}",
                        kind="http",
                        status_code=exc.code,
                    ) from exc
            except (
                TimeoutError,
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                json.JSONDecodeError,
            ) as exc:
                if retries >= self.config.transport_retries:
                    raise ProviderError(
                        f"API transport failed after {retries} retries: {type(exc).__name__}: {exc}",
                        kind="transport",
                    ) from exc
            retries += 1
            time.sleep(2 ** (retries - 1))

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in raw.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    }
                )
        content = "\n".join(text_parts)
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        stop_reason = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }.get(raw.get("stop_reason"), raw.get("stop_reason"))
        usage = raw.get("usage") or {}
        openai_raw = {
            **raw,
            "choices": [{"message": message, "finish_reason": stop_reason}],
            "usage": {
                "prompt_tokens": int(usage.get("input_tokens") or 0),
                "completion_tokens": int(usage.get("output_tokens") or 0),
                "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
            },
        }
        return Completion(
            content=content,
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            elapsed_seconds=time.perf_counter() - started,
            transport_retries=retries,
            raw=openai_raw,
        )


def _completion_client(config: ApiConfig, *, variable: str) -> CompletionClient:
    if config.api_type == "openai-completions":
        return OpenAICompatibleClient(config)
    if config.api_type == "anthropic-messages":
        return AnthropicMessagesClient(config)
    raise RuntimeError(f"Unsupported {variable}: {config.api_type}")


def completion_client_from_env() -> CompletionClient:
    return _completion_client(ApiConfig.from_env(), variable="HARNESS_API_TYPE")


def sa_speculator_client_from_env(actor: CompletionClient) -> CompletionClient:
    return _completion_client(
        ApiConfig.from_sa_env(actor.config),
        variable="HARNESS_SA_API_TYPE",
    )
