from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from benchmark_platform.harnesses.api import ApiConfig, OpenAICompatibleClient


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


if __name__ == "__main__":
    unittest.main()
