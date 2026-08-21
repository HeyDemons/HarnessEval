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
            user_agent=os.getenv("HARNESS_API_USER_AGENT", "HarnessEval/0.1").strip() or "HarnessEval/0.1",
        )


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
        json_mode: bool = False,
    ) -> Completion: ...

    async def complete_native(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
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
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            temperature=temperature,
            json_mode=json_mode,
        )

    async def complete_native(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
    ) -> Completion:
        """Preserve native chat/tool messages for benchmark-owned simulators."""
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            temperature=temperature,
            json_mode=False,
            tools=tools,
            tool_choice=tool_choice,
        )

    def _complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        json_mode: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": False,
        }
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
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                body = exc.read()
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
                if retries >= self.config.transport_retries:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
            # http.client.RemoteDisconnected subclasses ConnectionResetError and
            # BadStatusLine, neither of which is a urllib.error.URLError, so it escaped
            # this handler and killed the episode outright. Large tool-schema payloads
            # (VitaBench cross-domain sends 66 schemas) make such transient disconnects
            # routine, which is precisely what the retry budget exists for.
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError) as exc:
                if retries >= self.config.transport_retries:
                    raise RuntimeError(
                        f"API transport failed after {retries} retries: {type(exc).__name__}: {exc}"
                    ) from exc
            retries += 1
            time.sleep(2 ** (retries - 1))

        raw = json.loads(body.decode("utf-8"))
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
                break
            except urllib.error.HTTPError as exc:
                body = exc.read()
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
                if retries >= self.config.transport_retries:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError) as exc:
                if retries >= self.config.transport_retries:
                    raise RuntimeError(
                        f"API transport failed after {retries} retries: {type(exc).__name__}: {exc}"
                    ) from exc
            retries += 1
            time.sleep(2 ** (retries - 1))

        raw = json.loads(body.decode("utf-8"))
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


def completion_client_from_env() -> CompletionClient:
    config = ApiConfig.from_env()
    if config.api_type == "openai-completions":
        return OpenAICompatibleClient(config)
    if config.api_type == "anthropic-messages":
        return AnthropicMessagesClient(config)
    raise RuntimeError(f"Unsupported HARNESS_API_TYPE: {config.api_type}")
