from __future__ import annotations

import asyncio
import copy
import http.client
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from benchmark_platform.harnesses.api import ApiConfig, ProviderError, completion_client_from_env, sa_speculator_client_from_env
from benchmark_platform.harnesses.content import ToolImage, tool_result_content
from benchmark_platform.harnesses.core import extract_json
from benchmark_platform.harnesses.responses_api import OpenAIResponsesClient, _stream_response


def envelope(text="done", *, output=None, status="completed", **extra):
    return {"id": "resp_test", "model": "model", "status": status,
            "output": output if output is not None else [{"type": "message", "role": "assistant", "id": "msg_1",
                "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 100, "output_tokens": 12, "input_tokens_details": {"cached_tokens": 30},
                      "output_tokens_details": {"reasoning_tokens": 4}}, **extra}


class Response:
    def __init__(self, value):
        self.value = json.dumps(value).encode() if isinstance(value, dict) else value

    def read(self):
        return self.value

    def __iter__(self):
        return iter(self.value.splitlines(keepends=True))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def stream(*events):
    return Response("".join("data: " + json.dumps(e) + "\n\n" for e in events).encode())


def client(**kwargs):
    return OpenAIResponsesClient(ApiConfig("https://example.invalid/v1", "private-test-key", "model",
        api_type="openai-responses", transport_retries=kwargs.pop("transport_retries", 0), **kwargs))


class ResponsesTests(unittest.TestCase):
    def test_factory_and_sa_inheritance(self):
        env = {"HARNESS_API_BASE": "https://example.invalid/v1", "HARNESS_API_KEY": "test",
               "HARNESS_MODEL": "model", "HARNESS_API_TYPE": "openai-responses", "HARNESS_SA_MODEL": "fast"}
        with patch.dict("os.environ", env, clear=True):
            actor = completion_client_from_env()
            self.assertIsInstance(actor, OpenAIResponsesClient)
            sa = sa_speculator_client_from_env(actor)
            self.assertIsInstance(sa, OpenAIResponsesClient)
            self.assertEqual(sa.config.model, "fast")

    def test_endpoint_variants(self):
        for base in ["https://x/v1", "https://x/v1/", "https://x/v1/chat/completions", "https://x/v1/responses"]:
            self.assertEqual(OpenAIResponsesClient(ApiConfig(base, "test", "m")).endpoint, "https://x/v1/responses")

    def test_explicit_instructions_json_guard_and_usage(self):
        c = client(reasoning_effort="high", max_output_tokens=123)
        messages = [{"role": "system", "content": "Return an action as JSON."}, {"role": "user", "content": "task"}]
        before = copy.deepcopy(messages)
        with patch("urllib.request.urlopen", return_value=Response(envelope('{"final":"done"}'))) as urlopen:
            result = asyncio.run(c.complete(messages, json_mode=True))
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["instructions"], messages[0]["content"])
        self.assertEqual(body["input"], [messages[1], {"role": "user", "content": "Return JSON."}])
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertEqual(body["text"], {"format": {"type": "json_object"}})
        self.assertFalse(body["store"])
        self.assertNotIn("temperature", body)
        self.assertNotIn("previous_response_id", body)
        self.assertEqual(body["max_output_tokens"], 123)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.raw["usage"]["prompt_tokens_details"]["cached_tokens"], 30)
        self.assertEqual(result.raw["usage"]["completion_tokens_details"]["reasoning_tokens"], 4)
        self.assertEqual(messages, before)

    def test_empty_instructions_are_explicit_and_developer_role_preserved(self):
        with patch("urllib.request.urlopen", return_value=Response(envelope())) as urlopen:
            client().complete_sync([{"role": "developer", "content": "policy"}, {"role": "user", "content": "hello"}])
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["instructions"], " ")
        self.assertEqual(body["input"][0]["role"], "developer")

    def test_existing_json_in_input_needs_no_reminder(self):
        with patch("urllib.request.urlopen", return_value=Response(envelope())) as urlopen:
            client().complete_sync([{"role": "user", "content": "return JSON"}], json_mode=True)
        self.assertEqual(len(json.loads(urlopen.call_args.args[0].data)["input"]), 1)

    def test_native_tool_roundtrip_replays_reasoning_and_normalized_arguments(self):
        c = client(reasoning_effort="high")
        history = [{"role": "system", "content": "policy"}, {"role": "user", "content": "lookup"}]
        output = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []},
                  {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'}]
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        with patch("urllib.request.urlopen", return_value=Response(envelope(output=output))) as request:
            first = asyncio.run(c.complete_native(history, tools=[tool], tool_choice={"type": "function", "function": {"name": "lookup"}}))
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "lookup"})
        self.assertEqual(payload["tools"][0]["strict"], False)
        self.assertNotIn("opaque", json.dumps(first.raw))
        message = copy.deepcopy(first.raw["choices"][0]["message"])
        message["content"] = None
        message["tool_calls"][0]["function"]["arguments"] = '{"q": "x"}'
        history.extend([message, {"role": "tool", "tool_call_id": "call_1", "content": "found"}])
        with patch("urllib.request.urlopen", return_value=Response(envelope())) as request:
            c.complete_sync(history)
        items = json.loads(request.call_args.args[0].data)["input"]
        self.assertEqual(items[1:3], output)
        self.assertEqual(items[-1], {"type": "function_call_output", "call_id": "call_1", "output": "found"})

    def test_native_replay_is_prefix_scoped(self):
        c = client(reasoning_effort="high")
        with patch("urllib.request.urlopen", return_value=Response(envelope(output=[
            {"type": "reasoning", "id": "rs", "encrypted_content": "private", "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "same"}]}]))):
            c.complete_sync([{"role": "user", "content": "branch A"}])
        with patch("urllib.request.urlopen", return_value=Response(envelope())) as request:
            c.complete_sync([{"role": "user", "content": "branch B"}, {"role": "assistant", "content": "same"}])
        self.assertNotIn("private", request.call_args.args[0].data.decode())

    def test_text_json_loop_never_replays_hypothetical_later_actions(self):
        c = client(reasoning_effort="high")
        output = [envelope('{"tool":"lookup","arguments":{}}')["output"][0],
                  envelope('{"final":"invented"}')["output"][0]]
        with patch("urllib.request.urlopen", return_value=Response(envelope(output=output))):
            result = c.complete_sync([{"role": "user", "content": "JSON task"}], json_mode=True)
        self.assertEqual(extract_json(result.content, dict)["tool"], "lookup")
        with patch("urllib.request.urlopen", return_value=Response(envelope())) as request:
            c.complete_sync([{"role": "user", "content": "JSON task"},
                             {"role": "assistant", "content": '{"tool":"lookup","arguments":{}}'},
                             {"role": "user", "content": "real observation"}], json_mode=True)
        self.assertNotIn("invented", request.call_args.args[0].data.decode())

    def test_delta_done_completed_are_not_duplicated(self):
        answer = '{"final":"ok"}'
        response = stream({"type": "response.output_text.delta", "delta": answer},
                          {"type": "response.output_text.done", "text": answer},
                          {"type": "response.completed", "response": envelope(answer)})
        with patch("urllib.request.urlopen", return_value=response):
            result = client(stream=True).complete_sync([{"role": "user", "content": "JSON"}], json_mode=True)
        self.assertEqual(result.content, answer)

    def test_multiline_sse_and_eof_flush(self):
        text = json.dumps({"type": "response.completed", "response": envelope()}, indent=2)
        response = Response("\n".join("data: " + l for l in text.splitlines()).encode())
        self.assertEqual(_stream_response(response)["status"], "completed")

    def test_stream_failure_retry_and_missing_terminal(self):
        for failure in [stream({"type": "response.failed"}), stream({"type": "error"}),
                        stream({"type": "response.output_text.delta", "delta": "partial"})]:
            with self.subTest(failure=failure), patch("urllib.request.urlopen", side_effect=[failure,
                    stream({"type": "response.completed", "response": envelope()})]) as request, patch(
                    "benchmark_platform.harnesses.responses_api.time.sleep"):
                result = client(stream=True, transport_retries=1).complete_sync([{"role": "user", "content": "hi"}])
                self.assertEqual(result.transport_retries, 1)
                self.assertEqual(request.call_count, 2)

    def test_http_error_classification_and_no_key_echo(self):
        for status, attempts in [(400, 1), (401, 1), (429, 2), (500, 2)]:
            with self.subTest(status=status):
                def failure(*args, **kwargs):
                    raise urllib.error.HTTPError("https://x", status, "error", {}, io.BytesIO(b"private-test-key"))
                with patch("urllib.request.urlopen", side_effect=failure) as request, patch(
                        "benchmark_platform.harnesses.responses_api.time.sleep"), self.assertRaises(ProviderError) as caught:
                    client(transport_retries=1).complete_sync([{"role": "user", "content": "hi"}])
                self.assertEqual(request.call_count, attempts)
                self.assertEqual(caught.exception.status_code, status)
                self.assertNotIn("private-test-key", str(caught.exception))

    def test_remote_disconnect_retry(self):
        with patch("urllib.request.urlopen", side_effect=[http.client.RemoteDisconnected("closed"), Response(envelope())]), patch(
                "benchmark_platform.harnesses.responses_api.time.sleep"):
            self.assertEqual(client(transport_retries=1).complete_sync([{"role": "user", "content": "hi"}]).transport_retries, 1)

    def test_context_overflow_is_not_retried_as_provider_infra(self):
        error = urllib.error.HTTPError("https://x", 500, "error", {}, io.BytesIO(b'{"code":"context_length_exceeded"}'))
        with patch("urllib.request.urlopen", side_effect=error) as request, self.assertRaisesRegex(ValueError, "context_length_exceeded"):
            client(transport_retries=3).complete_sync([{"role": "user", "content": "large"}])
        self.assertEqual(request.call_count, 1)

    def test_budget_incomplete_is_not_retried_as_infra(self):
        raw = envelope('partial', status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        with patch("urllib.request.urlopen", return_value=stream({"type": "response.incomplete", "response": raw})) as request:
            result = client(stream=True, transport_retries=3).complete_sync([{"role": "user", "content": "hi"}])
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result.raw["choices"][0]["finish_reason"], "length")

    def test_malformed_native_call_is_infra(self):
        raw = envelope(output=[{"type": "function_call", "name": "lookup", "arguments": "{}"}])
        with patch("urllib.request.urlopen", return_value=Response(raw)), self.assertRaises(ProviderError):
            client().complete_sync([{"role": "user", "content": "hi"}])

    def test_native_tool_images_are_typed(self):
        image = ToolImage("image/png", b"image")
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": tool_result_content({"image": image})}]
        with patch("urllib.request.urlopen", return_value=Response(envelope())) as request:
            client().complete_sync(messages)
        parts = json.loads(request.call_args.args[0].data)["input"][0]["output"]
        self.assertEqual(parts[0]["type"], "input_text")
        self.assertEqual(parts[1], {"type": "input_image", "image_url": image.data_uri, "detail": "auto"})
        self.assertNotIn(image.data_uri, json.dumps(messages))

    def test_temperature_preserved_and_seed_not_silently_discarded(self):
        with patch("urllib.request.urlopen", return_value=Response(envelope())) as request:
            client(reasoning_effort="high").complete_sync([{"role": "user", "content": "hi"}], temperature=0)
        self.assertEqual(json.loads(request.call_args.args[0].data)["temperature"], 0)
        with patch("urllib.request.urlopen") as request, self.assertRaisesRegex(ValueError, "provider seed"):
            client().complete_sync([{"role": "user", "content": "hi"}], seed=42)
        request.assert_not_called()

    def test_invalid_roles_and_hosted_tools_rejected(self):
        for messages, tools in [([{"role": "unexpected", "content": "hi"}], None),
                                ([{"role": "user", "content": "hi"}], [{"type": "web_search"}])]:
            with patch("urllib.request.urlopen") as request, self.assertRaises(ValueError):
                client().complete_sync(messages, tools=tools)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
