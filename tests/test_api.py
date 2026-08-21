from __future__ import annotations

import asyncio
import http.client
import io
import json
import re
import unittest
import urllib.error
from unittest.mock import patch

from benchmark_platform.harnesses.api import (
    AnthropicMessagesClient,
    ApiConfig,
    OpenAICompatibleClient,
    ProviderError,
    completion_client_from_env,
)
from benchmark_platform.harnesses.content import ToolImage, tool_result_content


class _Response:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


class _RawResponse(_Response):
    def __init__(self, body: bytes):
        self.body = body


class NativeTransportTests(unittest.TestCase):
    def test_tool_image_is_sent_as_multimodal_content_not_base64_text(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        result = {
            "ok": True,
            "result": {
                "path": "figure.jpg",
                "image": ToolImage("image/jpeg", b"\xff\xd8\xff"),
            },
        }
        messages = [{"role": "user", "content": tool_result_content(result)}]
        self.assertNotIn("/9j/", json.dumps(messages, ensure_ascii=False))

        client = OpenAICompatibleClient(
            ApiConfig("https://example.invalid/v1", "secret", "model", transport_retries=0)
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            asyncio.run(client.complete(messages))

        content = observed["payload"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn('"bytes": 3', content[0]["text"])
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,/9j/")
        self.assertEqual(content[1]["image_url"]["detail"], "auto")

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

    def test_empty_json_response_uses_transport_retry_budget(self) -> None:
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
            patch("urllib.request.urlopen", side_effect=[_RawResponse(b""), response]) as urlopen,
            patch("benchmark_platform.harnesses.api.time.sleep") as sleep,
        ):
            completion = asyncio.run(
                client.complete_native([{"role": "user", "content": "test"}])
            )

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


class ReasoningEffortTests(unittest.TestCase):
    """A matched control must send the same reasoning knob the product harness sends."""

    def _payload(self, config: ApiConfig, temperature: float | None = None) -> dict:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })

        with patch("urllib.request.urlopen", fake_urlopen):
            asyncio.run(
                OpenAICompatibleClient(config).complete(
                    [{"role": "user", "content": "hi"}], temperature=temperature
                )
            )
        return captured

    def _config(self, **overrides) -> ApiConfig:
        return ApiConfig(base_url="https://provider.example/v1", api_key="k", model="m", **overrides)

    def test_reasoning_effort_replaces_temperature(self) -> None:
        payload = self._payload(self._config(reasoning_effort="high"))
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", payload)

    def test_temperature_is_sent_when_no_reasoning_effort_is_configured(self) -> None:
        payload = self._payload(self._config(temperature=0.0))
        self.assertEqual(payload["temperature"], 0.0)
        self.assertNotIn("reasoning_effort", payload)

    def test_a_profile_that_names_a_temperature_keeps_it_under_a_reasoning_effort(self) -> None:
        """dylan asks for 1.0 and lats for lats_temperature because sampling diversity is the
        method, not a preference: DyLAN's agents have nothing to debate if they all answer the
        same. Folding those into the effort branch disabled the method silently, and the API
        takes both parameters together."""
        payload = self._payload(self._config(reasoning_effort="high"), temperature=1.0)
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["temperature"], 1.0)

    def test_a_profile_that_names_no_temperature_still_sends_none_under_an_effort(self) -> None:
        """The actor-only control stays matched to the product harness, which sends none."""
        payload = self._payload(self._config(reasoning_effort="high", temperature=0.7))
        self.assertNotIn("temperature", payload)



class _StreamResponse:
    """Yields SSE lines the way http.client.HTTPResponse does: bytes, one per line."""

    def __init__(self, *frames: str):
        self.lines = [f"data: {frame}\n".encode("utf-8") for frame in frames]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self.lines)


def _chunk(**choice) -> str:
    return json.dumps({"id": "resp_1", "model": "m", "choices": [{"index": 0, **choice}]})


class StreamingTests(unittest.TestCase):
    """Frames below are copied from a live relay response, not invented."""

    def _client(self, **overrides) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            ApiConfig("https://example.invalid/v1", "secret", "model", stream=True, **overrides)
        )

    def test_text_deltas_and_trailing_usage_frame_rebuild_a_completion(self) -> None:
        response = _StreamResponse(
            _chunk(delta={"role": "assistant"}, finish_reason=None),
            _chunk(delta={"content": "4"}, finish_reason=None),
            _chunk(delta={"content": "048"}, finish_reason=None),
            _chunk(delta={"content": ""}, finish_reason="stop"),
            json.dumps({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 9}}),
            "[DONE]",
        )
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured.update(json.loads(request.data.decode("utf-8")))
            return response

        with patch("urllib.request.urlopen", fake_urlopen):
            completion = asyncio.run(self._client().complete([{"role": "user", "content": "hi"}]))

        self.assertTrue(captured["stream"])
        self.assertEqual(captured["stream_options"], {"include_usage": True})
        self.assertEqual(completion.content, "4048")
        self.assertEqual((completion.prompt_tokens, completion.completion_tokens), (7, 9))
        self.assertEqual(completion.raw["choices"][0]["finish_reason"], "stop")

    def test_tool_call_arguments_are_concatenated_across_frames(self) -> None:
        # The relay sends the name whole in the opening frame and "" in every frame after.
        response = _StreamResponse(
            _chunk(delta={"role": "assistant"}, finish_reason=None),
            _chunk(
                delta={"tool_calls": [{"index": 0, "id": "call_x", "type": "function",
                                       "function": {"name": "run_command", "arguments": ""}}]},
                finish_reason=None,
            ),
            _chunk(delta={"tool_calls": [{"index": 0, "function": {"name": "", "arguments": '{"argv":["ls'}}]},
                   finish_reason=None),
            _chunk(delta={"tool_calls": [{"index": 0, "function": {"name": "", "arguments": '"]}'}}]},
                   finish_reason=None),
            _chunk(delta={"content": ""}, finish_reason="tool_calls"),
            "[DONE]",
        )
        with patch("urllib.request.urlopen", lambda request, timeout=None: response):
            completion = asyncio.run(
                self._client().complete_native([{"role": "user", "content": "list /etc"}])
            )

        # tau_episode.py reads exactly this path, so assert on it rather than on a helper.
        call = completion.raw["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["id"], "call_x")
        self.assertEqual(call["function"]["name"], "run_command")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"argv": ["ls"]})
        self.assertIsNone(completion.raw["choices"][0]["message"]["content"])

    def test_truncated_stream_is_retried_rather_than_returned_as_an_empty_answer(self) -> None:
        truncated = _StreamResponse(_chunk(delta={"content": "partial"}, finish_reason=None))
        complete = _StreamResponse(
            _chunk(delta={"content": "whole"}, finish_reason="stop"),
            json.dumps({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
            "[DONE]",
        )
        with (
            patch("urllib.request.urlopen", side_effect=[truncated, complete]) as urlopen,
            patch("benchmark_platform.harnesses.api.time.sleep"),
        ):
            completion = asyncio.run(
                self._client(transport_retries=1).complete([{"role": "user", "content": "hi"}])
            )

        self.assertEqual(completion.content, "whole")
        self.assertEqual(completion.transport_retries, 1)
        self.assertEqual(urlopen.call_count, 2)

    def test_error_frame_on_a_200_response_is_not_mistaken_for_an_answer(self) -> None:
        error = _StreamResponse(json.dumps({"error": {"message": "upstream unavailable"}}))
        with (
            patch("urllib.request.urlopen", lambda request, timeout=None: error),
            patch("benchmark_platform.harnesses.api.time.sleep"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                asyncio.run(
                    self._client(transport_retries=0).complete([{"role": "user", "content": "hi"}])
                )
        self.assertIn("upstream unavailable", str(caught.exception))


class UserAgentTests(unittest.TestCase):
    def test_exhausted_retryable_http_error_is_structured_provider_error(self) -> None:
        error = urllib.error.HTTPError(
            "https://e.invalid/v1/chat/completions",
            503,
            "unavailable",
            {},
            io.BytesIO(b'{"error":"no capacity"}'),
        )
        client = OpenAICompatibleClient(
            ApiConfig("https://e.invalid/v1", "k", "m", transport_retries=0)
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
        self.assertEqual(caught.exception.kind, "http")
        self.assertEqual(caught.exception.status_code, 503)

    def test_nonretryable_client_error_remains_deterministic_runtime_error(self) -> None:
        error = urllib.error.HTTPError(
            "https://e.invalid/v1/chat/completions",
            400,
            "bad request",
            {},
            io.BytesIO(b'{"error":"invalid schema"}'),
        )
        client = OpenAICompatibleClient(
            ApiConfig("https://e.invalid/v1", "k", "m", transport_retries=0)
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
        self.assertNotIsInstance(caught.exception, ProviderError)

    def test_auth_or_relay_forbidden_error_is_structured_provider_error(self) -> None:
        error = urllib.error.HTTPError(
            "https://e.invalid/v1/chat/completions",
            403,
            "forbidden",
            {},
            io.BytesIO(b'{"error":"relay bot rule"}'),
        )
        client = OpenAICompatibleClient(
            ApiConfig("https://e.invalid/v1", "k", "m", transport_retries=0)
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
        self.assertEqual(caught.exception.status_code, 403)

    def test_a_non_default_user_agent_is_sent(self) -> None:
        """urllib's default UA is 403'd by a relay's bot rule; the header must be explicit."""
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured.update(request.headers)
            return _Response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

        with patch("urllib.request.urlopen", fake_urlopen):
            asyncio.run(
                OpenAICompatibleClient(ApiConfig("https://e.invalid/v1", "k", "m"))
                .complete([{"role": "user", "content": "hi"}])
            )
        # urllib title-cases header names it stores on the Request.
        self.assertEqual(captured.get("User-agent"), "HarnessEval/0.1")


class PassEnvAllowlistTests(unittest.TestCase):
    def test_every_variable_the_client_reads_is_passable_into_a_container(self) -> None:
        """The allowlist and ApiConfig.from_env drifted apart once and killed a whole sweep."""
        import inspect

        from benchmark_platform.engine import HARNESS_ENV
        from benchmark_platform.harnesses import api

        read = set(re.findall(r'os\.getenv\(\s*"(HARNESS_[A-Z_]+)"', inspect.getsource(api)))
        self.assertTrue(read, "expected ApiConfig.from_env to read HARNESS_* variables")
        self.assertEqual(read - HARNESS_ENV, set(), "client reads a variable no container may receive")


class SyncEntryPointTests(unittest.TestCase):
    def test_native_simulator_hook_needs_no_event_loop(self) -> None:
        """tau2/vitabench call their generation hook from an ordinary worker thread.

        Reaching the client through asyncio.run() there built a fresh event loop, and a
        fresh default thread pool inside it, once per conversation turn -- for a call that
        blocks regardless. One observed tau2 arm wedged with 64 threads parked on futexes
        and not one open socket. The hook must reach the client without a loop at all.
        """
        response = _Response({"choices": [{"message": {"content": "ok"}}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        client = OpenAICompatibleClient(ApiConfig("https://e.invalid/v1", "k", "m"))
        with patch("urllib.request.urlopen", lambda request, timeout=None: response):
            completion = client.complete_sync([{"role": "user", "content": "hi"}])
        self.assertEqual(completion.content, "ok")
        with self.assertRaises(RuntimeError):
            asyncio.get_running_loop()  # nothing above left a loop behind

    def test_the_native_episode_hooks_do_not_open_an_event_loop_per_turn(self) -> None:
        import inspect

        from benchmark_platform.bridges import tau_episode, vita_episode

        for module in (tau_episode, vita_episode):
            self.assertNotIn("asyncio.run(", inspect.getsource(module), module.__name__)
