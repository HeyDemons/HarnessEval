"""Responses transport behind the existing, method-neutral Completion contract.

No hosted tools or stored conversations: benchmark tools still execute only in
the existing bridge. Text action parsing stays with each harness method.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import http.client
import json
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import warnings

from .api import (
    ApiConfig, Completion, OpenAICompatibleClient, ProviderError,
    RETRYABLE_HTTP_STATUS, StreamInterrupted, _chat_messages,
)


def _content(content: Any, *, assistant: bool = False) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Responses message content must be text or content parts")
    parts = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "output_text" if assistant else "input_text", "text": part["text"]})
        elif kind == "image_url" and not assistant:
            image = part["image_url"]
            parts.append({"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")})
        elif kind in {"input_text", "output_text", "input_image"}:
            parts.append(copy.deepcopy(part))
        else:
            raise ValueError(f"Unsupported Responses content part: {kind}")
    return parts


def _plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = _content(content)
    if any(p.get("type") not in {"input_text", "output_text"} for p in parts):
        raise ValueError("System instructions must contain only text")
    return "\n".join(p["text"] for p in parts)


def _signature(message: dict[str, Any]) -> dict[str, Any]:
    result = {"role": message.get("role"), "content": message.get("content") or None}
    if message.get("tool_call_id"):
        result["tool_call_id"] = message["tool_call_id"]
    if message.get("tool_calls"):
        calls = copy.deepcopy(message["tool_calls"])
        for call in calls:
            arguments = call.get("function", {}).get("arguments")
            if isinstance(arguments, str):
                try:
                    call["function"]["arguments"] = json.loads(arguments)
                except ValueError:
                    pass
        result["tool_calls"] = calls
    return result


def _replay_key(prefix: list[dict[str, Any]], message: dict[str, Any]) -> str:
    encoded = json.dumps([*map(_signature, prefix), _signature(message)], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _stream_response(response: Any) -> dict[str, Any]:
    """Use the terminal envelope once; delta/done events are not concatenated."""
    data: list[str] = []
    terminal = None

    def event() -> None:
        nonlocal terminal
        payload = "\n".join(data)
        data.clear()
        if not payload or payload == "[DONE]":
            return
        value = json.loads(payload)
        kind = value.get("type")
        if kind in {"error", "response.failed"}:
            _check_context_limit(json.dumps(value))
            raise StreamInterrupted("Responses stream reported failure")
        if kind in {"response.completed", "response.incomplete"}:
            terminal = value.get("response")
            if not isinstance(terminal, dict):
                raise StreamInterrupted("Responses terminal event omitted response")

    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            event()
            if terminal is not None:
                # The terminal envelope is authoritative. Do not wait for a
                # compatible server to close an otherwise idle HTTP stream.
                return terminal
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        event()
    if terminal is None:
        raise StreamInterrupted("Responses stream ended without a terminal envelope")
    return terminal


def _check_context_limit(text: str) -> None:
    # Preserve the existing distinction between a budget failure and lost API
    # measurement, even when a relay wraps the former in an HTTP 500 response.
    lower = text.lower()
    if "context_length_exceeded" in lower or "maximum context length" in lower:
        raise ValueError("Responses API context_length_exceeded")


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    status = raw.get("status")
    reason = (raw.get("incomplete_details") or {}).get("reason")
    if status != "completed" and not (status == "incomplete" and reason in {"max_output_tokens", "content_filter"}):
        raise StreamInterrupted(f"Responses did not complete: {status}")
    if not isinstance(raw.get("output"), list):
        raise StreamInterrupted("Responses envelope omitted output Items")
    texts, calls, output_types = [], [], []
    for item in raw["output"]:
        kind = item.get("type")
        output_types.append(kind)
        if kind == "message":
            # Join parts within an item, then separate messages in generation order.
            texts.append("".join(
                part.get("text", "") if part.get("type") == "output_text" else part.get("refusal", "")
                for part in item.get("content", []) if part.get("type") in {"output_text", "refusal"}
            ))
        elif kind == "function_call":
            if not item.get("call_id") or not item.get("name"):
                raise StreamInterrupted("Responses function call omitted call_id or name")
            calls.append({"id": item["call_id"], "type": "function", "function": {
                "name": item["name"], "arguments": item.get("arguments", "")}})
        elif kind != "reasoning":
            raise StreamInterrupted(f"Unexpected Responses output Item: {kind}")
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if calls:
        message["tool_calls"] = calls
    usage = raw.get("usage") or {}
    finish = "length" if reason == "max_output_tokens" else "content_filter" if reason == "content_filter" else "tool_calls" if calls else "stop"
    return {
        "id": raw.get("id"), "model": raw.get("model"), "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
            "prompt_tokens_details": usage.get("input_tokens_details") or {},
            "completion_tokens_details": usage.get("output_tokens_details") or {},
        },
        "responses_status": status, "responses_output_types": output_types,
        "responses_usage": usage, "api_type": "openai-responses",
    }


class OpenAIResponsesClient(OpenAICompatibleClient):
    """Use explicit instructions and stateless native-tool context replay."""

    def __init__(self, config: ApiConfig):
        super().__init__(config)
        self._replay: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._replay_lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[:-len("/chat/completions")]
        return base if base.endswith("/responses") else base + "/responses"

    def _request(self, messages, *, temperature, seed, json_mode, tools, tool_choice):
        if seed is not None:
            warnings.warn(
                "Responses does not support provider seed; the requested value is recorded but not sent. "
                "Native episode seeds are unchanged.", RuntimeWarning, stacklevel=2,
            )
        visible = _chat_messages(messages)
        instructions, items = [], []
        for index, message in enumerate(visible):
            role = message.get("role")
            if role == "system":
                instructions.append(_plain_text(message.get("content")))
                continue
            if role == "assistant" and not json_mode:
                key = _replay_key(visible[:index], message)
                with self._replay_lock:
                    cached = copy.deepcopy(self._replay.get(key))
                if cached is not None:
                    items.extend(cached)
                    continue
            if role == "tool":
                if not message.get("tool_call_id"):
                    raise ValueError("Tool result requires tool_call_id")
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                              "output": _content(message.get("content"))})
            elif role in {"user", "assistant", "developer"}:
                content = _content(message.get("content"), assistant=role == "assistant")
                if content or not message.get("tool_calls"):
                    items.append({"role": role, "content": content})
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    if not call.get("id"):
                        raise ValueError("Assistant tool call requires id")
                    arguments = function.get("arguments", "")
                    items.append({"type": "function_call", "call_id": call["id"], "name": function["name"],
                                  "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)})
            else:
                raise ValueError(f"Unsupported Responses message role: {role}")
        system_instructions = "\n\n".join(instructions)
        body: dict[str, Any] = {
            "model": self.config.model, "input": items,
            # The relay trims whitespace before choosing its default instructions.
            # A minimal non-whitespace sentinel avoids an injected agent prompt.
            "instructions": system_instructions if system_instructions.strip() else ".",
            "store": False, "stream": self.config.stream,
        }
        if self.config.reasoning_effort:
            body["reasoning"] = {"effort": self.config.reasoning_effort}
            body["include"] = ["reasoning.encrypted_content"]
        if temperature is not None:
            body["temperature"] = temperature
        elif not self.config.reasoning_effort:
            body["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            body["max_output_tokens"] = self.config.max_output_tokens
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}
            # Some compatible endpoints inspect input only, not instructions.
            if "json" not in json.dumps(items, ensure_ascii=False).lower():
                items.append({"role": "user", "content": "Return JSON."})
        if tools:
            converted = []
            for tool in tools:
                if tool.get("type") != "function":
                    raise ValueError("HarnessEval Responses supports benchmark function tools only")
                function = tool["function"]
                converted.append({"type": "function", **copy.deepcopy(function), "strict": function.get("strict", False)})
            body["tools"] = converted
            choice = tool_choice or "auto"
            body["tool_choice"] = ({"type": "function", "name": choice["function"]["name"]}
                                   if isinstance(choice, dict) else choice)
        return body, visible

    def _complete_sync(self, messages, *, temperature, seed, json_mode, tools=None, tool_choice=None):
        body, visible = self._request(messages, temperature=temperature, seed=seed,
                                      json_mode=json_mode, tools=tools, tool_choice=tool_choice)
        encoded = json.dumps(body, ensure_ascii=False).encode()
        started, retries = time.perf_counter(), 0
        while True:
            request = urllib.request.Request(self.endpoint, data=encoded, headers={
                "Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            }, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    original = _stream_response(response) if self.config.stream else json.loads(response.read().decode())
                raw = _normalize(original)
                break
            except urllib.error.HTTPError as exc:
                status = exc.code
                # Extract only known budget markers; never echo arbitrary bodies.
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                finally:
                    exc.close()
                _check_context_limit(error_body)
                if (status not in RETRYABLE_HTTP_STATUS and status < 500) or retries >= self.config.transport_retries:
                    raise ProviderError(f"Responses API HTTP {status}", kind="http", status_code=status) from exc
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    StreamInterrupted, json.JSONDecodeError, UnicodeDecodeError) as exc:
                if retries >= self.config.transport_retries:
                    raise ProviderError(f"Responses API transport failed after {retries} retries: {type(exc).__name__}",
                                        kind="transport") from exc
            retries += 1
            time.sleep(2 ** (retries - 1))
        message = raw["choices"][0]["message"]
        raw["responses_request"] = {"provider_seed_requested": seed, "provider_seed_applied": False,
                                    "store": False, "explicit_instructions": True}
        if not json_mode and original.get("status") == "completed":
            # Match the entire visible prefix as well as the assistant response:
            # concurrent branches/hidden-user calls cannot borrow one another's state.
            key = _replay_key(visible, message)
            replay = copy.deepcopy(original["output"])
            # Only encrypted reasoning is useful for replay; never persist plaintext CoT.
            replay = [x for x in replay if x.get("type") != "reasoning" or x.get("encrypted_content")]
            with self._replay_lock:
                self._replay[key] = replay
                self._replay.move_to_end(key)
                while len(self._replay) > 512:
                    self._replay.popitem(last=False)
        usage = raw["usage"]
        return Completion(content=message["content"], prompt_tokens=usage["prompt_tokens"],
                          completion_tokens=usage["completion_tokens"], elapsed_seconds=time.perf_counter() - started,
                          transport_retries=retries, raw=raw)
