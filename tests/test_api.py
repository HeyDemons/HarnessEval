from __future__ import annotations

import asyncio
import http.client
import json
import unittest
from unittest.mock import patch

from benchmark_platform.harnesses.api import (
    AnthropicMessagesClient,
    ApiConfig,
    OpenAICompatibleClient,
    completion_client_from_env,
)


class _Response:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


class NativeTransportTests(unittest.TestCase):
    def test_native_messages_and_tool_schema_are_not_rewritten_or_sliced(self) -> None:
        large_argument = "x" * 200_000
        messages = [
            {"role": "system", "content": "policy"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"query": large_argument}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "complete result"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        observed = {}

        def fake_urlopen(request, timeout):
            observed["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        client = OpenAICompatibleClient(
            ApiConfig("https://example.invalid/v1", "secret", "model", transport_retries=0)
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            completion = asyncio.run(
                client.complete_native(messages, tools=tools, tool_choice="required")
            )

        self.assertEqual(completion.content, "done")
        self.assertEqual(observed["payload"]["messages"], messages)
        self.assertEqual(observed["payload"]["tools"], tools)
        self.assertEqual(observed["payload"]["tool_choice"], "required")
        self.assertEqual(
            json.loads(observed["payload"]["messages"][1]["tool_calls"][0]["function"]["arguments"])["query"],
            large_argument,
        )

    def test_remote_disconnect_uses_transport_retry_budget(self) -> None:
        client = OpenAICompatibleClient(
            ApiConfig("https://example.invalid/v1", "secret", "model", transport_retries=1)
        )
        response = _Response(
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[http.client.RemoteDisconnected("closed"), response],
            ) as urlopen,
            patch("benchmark_platform.harnesses.api.time.sleep") as sleep,
        ):
            completion = asyncio.run(client.complete_native([{"role": "user", "content": "test"}]))

        self.assertEqual(completion.content, "done")
        self.assertEqual(completion.transport_retries, 1)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_anthropic_messages_preserves_tools_and_normalizes_response(self) -> None:
        large_argument = "x" * 200_000
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "lookup"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"query": large_argument}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "complete result"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        observed = {}

        def fake_urlopen(request, timeout):
            observed["url"] = request.full_url
            observed["payload"] = json.loads(request.data.decode("utf-8"))
            observed["authorization"] = request.get_header("Authorization")
            observed["user_agent"] = request.get_header("User-agent")
            return _Response(
                {
                    "id": "msg-1",
                    "model": "claude-sonnet-5",
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "call-2",
                            "name": "lookup",
                            "input": {"query": "next"},
                        },
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 7},
                }
            )

        client = AnthropicMessagesClient(
            ApiConfig(
                "https://example.invalid",
                "secret",
                "claude-sonnet-5",
                transport_retries=0,
                max_output_tokens=16_384,
                api_type="anthropic-messages",
                api_auth="bearer",
            )
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            completion = asyncio.run(
                client.complete_native(messages, tools=tools, tool_choice="required")
            )

        self.assertEqual(observed["url"], "https://example.invalid/v1/messages")
        self.assertEqual(observed["authorization"], "Bearer secret")
        self.assertEqual(observed["user_agent"], "HarnessEval/0.1")
        self.assertEqual(observed["payload"]["system"], "policy")
        self.assertEqual(observed["payload"]["tool_choice"], {"type": "any"})
        self.assertEqual(observed["payload"]["tools"][0]["input_schema"], tools[0]["function"]["parameters"])
        self.assertEqual(
            observed["payload"]["messages"][1]["content"][1]["input"]["query"],
            large_argument,
        )
        self.assertEqual(
            observed["payload"]["messages"][2]["content"][0],
            {"type": "tool_result", "tool_use_id": "call-1", "content": "complete result"},
        )
        self.assertEqual(completion.content, "checking")
        self.assertEqual(completion.prompt_tokens, 5)
        self.assertEqual(completion.completion_tokens, 7)
        self.assertEqual(
            json.loads(completion.raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]),
            {"query": "next"},
        )

    def test_anthropic_json_mode_uses_prompt_level_json_contract(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {
                    "id": "msg-json",
                    "model": "claude-sonnet-5",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": '{"assignments": ["lookup alpha"]}'}],
                    "usage": {"input_tokens": 4, "output_tokens": 6},
                }
            )

        client = AnthropicMessagesClient(
            ApiConfig(
                "https://example.invalid",
                "secret",
                "claude-sonnet-5",
                transport_retries=0,
                max_output_tokens=16_384,
                api_type="anthropic-messages",
                api_auth="bearer",
            )
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            completion = asyncio.run(
                client.complete(
                    [{"role": "user", "content": "Return JSON matching the supplied schema."}],
                    json_mode=True,
                )
            )

        self.assertEqual(completion.content, '{"assignments": ["lookup alpha"]}')
        self.assertNotIn("response_format", observed["payload"])

    def test_client_factory_selects_anthropic_transport(self) -> None:
        values = {
            "HARNESS_API_BASE": "https://example.invalid",
            "HARNESS_API_KEY": "secret",
            "HARNESS_MODEL": "claude-sonnet-5",
            "HARNESS_API_TYPE": "anthropic-messages",
            "HARNESS_MAX_OUTPUT_TOKENS": "16384",
        }
        with patch.dict("os.environ", values, clear=True):
            client = completion_client_from_env()
        self.assertIsInstance(client, AnthropicMessagesClient)


if __name__ == "__main__":
    unittest.main()
