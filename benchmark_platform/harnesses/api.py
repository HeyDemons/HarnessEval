from __future__ import annotations

import asyncio
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout_seconds: float = 180.0
    transport_retries: int = 3
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    # Off for a directly-constructed config (what the tests use); from_env turns it on,
    # and from_env is the only path a real run takes. See _reassemble_stream for why.
    stream: bool = False
    # urllib's default "Python-urllib/3.x" is on a Cloudflare bot-rule blocklist at one of
    # the relays in use, which answers every request with 403 error code 1010 -- that alone
    # killed 478 of 728 arms in an earlier sweep while the product harness, whose SDK sends
    # its own UA, sailed through. Only the string is checked; the TLS fingerprint is not.
    user_agent: str = "HarnessEval/1.0"

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
            reasoning_effort=os.getenv("HARNESS_REASONING_EFFORT", "").strip() or None,
            stream=os.getenv("HARNESS_API_STREAM", "1").strip() not in {"0", "false", "no"},
            user_agent=os.getenv("HARNESS_USER_AGENT", "").strip() or "HarnessEval/1.0",
        )


class StreamInterrupted(Exception):
    """A stream that ended without a finish_reason, or carried an error frame instead.

    Both look like success at the HTTP layer -- status 200, a well-formed but short body --
    so without this the caller silently records an empty completion as the model's answer.
    It joins the transport-retry budget because that is exactly the transient it is.
    """


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

    def complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
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
    ) -> Completion:
        """Preserve native chat/tool messages for benchmark-owned simulators."""
        return await asyncio.to_thread(
            self.complete_sync,
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
            "stream": self.config.stream,
        }
        if self.config.stream:
            payload["stream_options"] = {"include_usage": True}
        # A reasoning model rejects or ignores temperature; the product harness omits it for
        # the same reason, so a matched control must omit it here too.
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        else:
            payload["temperature"] = self.config.temperature if temperature is None else temperature
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
            except (
                TimeoutError,
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                StreamInterrupted,
            ) as exc:
                if retries >= self.config.transport_retries:
                    raise RuntimeError(
                        f"API transport failed after {retries} retries: {type(exc).__name__}: {exc}"
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
