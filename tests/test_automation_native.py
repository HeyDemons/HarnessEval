import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_platform.bridges.automation_native import run_native_actor
from benchmark_platform.harnesses.api import Completion, ApiConfig
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec, JsonlTrace
from benchmark_platform.harnesses.responses_api import OpenAIResponsesClient


def call(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete_native(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), kwargs))
        message = next(self.replies)
        return Completion(message.get("content", ""), 1, 1, 0, 0, {"choices": [{"message": message}]})

    async def complete(self, messages, **kwargs):
        return await self.complete_native(messages, **kwargs)


class NativeAutomationTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_keeps_order_ids_raw_observations_and_one_generation(self):
        calls = [call("api_fetch", {"number": 1, "optional": {}}, "a"), call("api_fetch", {"number": 2}, "b")]
        client = Client([{"tool_calls": calls}, {"content": "done"}])
        seen = []
        async def handler(args):
            seen.append(args)
            return '{"observed":' + str(len(seen)) + '}'
        messages = [{"role": "system", "content": "OFFICIAL_POLICY"}, {"role": "user", "content": "TASK"}]
        with tempfile.TemporaryDirectory() as tmp:
            trace = JsonlTrace(Path(tmp) / "trace.jsonl")
            env = ToolEnvironment([ToolSpec("api_fetch", "API", {"type": "object"}, ())], trace, {"api_fetch": handler})
            ctx = RunContext("actor-only", "TASK", client, env, trace, {}, task_messages=messages)
            self.assertEqual(await run_native_actor(ctx), "done")
            self.assertEqual(seen, [{"number": 1}, {"number": 2}])
            self.assertEqual(ctx.agent_turns, 2)
            self.assertEqual(len(env.calls), 2)
            self.assertEqual([c["assistant_response_id"] for c in env.calls], [1, 1])
            followup = client.requests[1][0]
            self.assertEqual(followup[:2], messages)
            self.assertEqual(followup[-2:], [
                {"role": "tool", "tool_call_id": "a", "content": '{"observed":1}'},
                {"role": "tool", "tool_call_id": "b", "content": '{"observed":2}'}])
            self.assertEqual(set(client.requests[0][1]["tools"][0]["function"]), {"name", "description", "parameters"})
            transport = OpenAIResponsesClient(ApiConfig("http://unused.invalid/v1", "test", "model", api_type="openai-responses"))
            with patch.dict("os.environ", {"HARNESS_RESPONSES_JSON_TRANSPORT": "function"}):
                body, _ = transport._request(followup, json_mode=False, temperature=None, seed=None,
                                             tools=client.requests[1][1]["tools"], tool_choice="auto")
            self.assertEqual(body["instructions"], "OFFICIAL_POLICY")
            self.assertEqual([t["name"] for t in body["tools"]], ["api_fetch"])
            self.assertNotIn("submit_benchmark_json", json.dumps(body))
            self.assertEqual([m["call_id"] for m in body["input"] if m.get("type") == "function_call_output"], ["a", "b"])

    async def test_parse_and_execution_errors_return_observations_and_continue_batch(self):
        client = Client([{"tool_calls": [call("work", "not JSON", "bad"), call("work", {}, "fails")]}, {"content": "done"}])
        async def handler(args):
            raise RuntimeError("simulated failure")
        with tempfile.TemporaryDirectory() as tmp:
            trace = JsonlTrace(Path(tmp) / "trace.jsonl")
            env = ToolEnvironment([ToolSpec("work", "work", {"type": "object"}, ())], trace, {"work": handler})
            ctx = RunContext("actor-only", "TASK", client, env, trace, {}, task_messages=[{"role": "user", "content": "TASK"}])
            self.assertEqual(await run_native_actor(ctx), "done")
            self.assertEqual(ctx.agent_turns, 2)
            observations = [m for m in client.requests[1][0] if m["role"] == "tool"]
            self.assertEqual([m["tool_call_id"] for m in observations], ["bad", "fails"])
            self.assertIn("invalid_arguments", observations[0]["content"])
            self.assertIn("simulated failure", observations[1]["content"])

    async def test_last_budget_response_cannot_act(self):
        client = Client([{"tool_calls": [call("work", {}, "bad")]}])
        with tempfile.TemporaryDirectory() as tmp:
            trace = JsonlTrace(Path(tmp) / "trace.jsonl")
            env = ToolEnvironment([ToolSpec("work", "work", {"type": "object"}, ())], trace)
            ctx = RunContext("actor-only", "TASK", client, env, trace,
                             {"model_response_limit": 1, "finalize_on_loop_limit": True})
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                await run_native_actor(ctx)
            self.assertEqual(env.calls, [])
            self.assertEqual(ctx.agent_turns, 1)
            self.assertEqual(client.requests[0][1]["tool_choice"], "none")

    async def test_public_instructions_reach_workers_and_speculator_without_mutation(self):
        client = Client([{"content": "ok"}] * 3)
        task_messages = [{"role": "system", "content": "DOMAIN_POLICY"}, {"role": "user", "content": "TASK"}]
        planner = [{"role": "system", "content": "PLAN_PROTOCOL"}, {"role": "user", "content": "TASK"}]
        before = copy.deepcopy(planner)
        with tempfile.TemporaryDirectory() as tmp:
            trace = JsonlTrace(Path(tmp) / "trace.jsonl")
            ctx = RunContext("plan-execute", "TASK", client, ToolEnvironment([], trace), trace, {},
                             task_messages=task_messages, speculator_client=client)
            await ctx.complete("planner", planner)
            await ctx.complete_native("worker", planner)
            await ctx.complete_speculator("spec", planner)
            self.assertEqual(ctx.agent_turns, 2)
            self.assertEqual(ctx.speculator_llm_calls, 1)
            self.assertEqual(planner, before)
            for messages, _ in client.requests:
                self.assertEqual(messages[0], task_messages[0])
                self.assertEqual(messages[1:], planner)


if __name__ == "__main__":
    unittest.main()
