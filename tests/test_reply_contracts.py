import asyncio
import json
import unittest
from unittest.mock import patch

from benchmark_platform.budgets import ModelBudgetExceeded
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.dmas import _agents_from_policy, _route_after_split
from benchmark_platform.harnesses.reply_contracts import object_schema, validate_reply
from test_dylan_fidelity import Trace
from test_harnesses import ScriptedClient
from test_responses_api import client as response_client, envelope, Response


def context(profile, replies, policy=None, handler=None):
    trace = Trace()
    async def write(args):
        return {"done": True}
    env = ToolEnvironment([ToolSpec("write", "write", {"type": "object"}, ())], trace, {"write": handler or write})
    return RunContext(profile, "PUBLIC_TASK", ScriptedClient(replies), env, trace, policy or {}), trace


class ReplyContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_planners_repair_wrong_root_and_wrong_field_type(self):
        for method, key in (("plan-execute", "steps"), ("cmas", "assignments")):
            for bad in ('{"status":"in_progress"}', json.dumps({key: "not an array"})):
                with self.subTest(method=method, bad=bad):
                    plan = {key: [{"instruction": "Answer"}]} if method == "plan-execute" else {key: []}
                    ctx, trace = context(method, [bad, json.dumps(plan), '{"final":"done"}'])
                    self.assertEqual(await run_profile(ctx), "done")
                    self.assertEqual(ctx.agent_turns, 3)
                    self.assertEqual(ctx.environment.calls, [])
                    repair = next(e for e in trace.events if e["event"] == "json_reply_repair")
                    self.assertIn(key, json.dumps(repair["response_schema"]))

    async def test_dmas_status_is_a_valid_role_specific_reply_after_repair(self):
        ctx, trace = context("dmas", ["not JSON", '{"status":"completed"}'])
        agents = _agents_from_policy({})
        reply = await _route_after_split(ctx, agents[0], agents, remaining="public", progress=[])
        self.assertEqual(reply["status"], "completed")
        feedback = ctx.client.messages[-1][-1]["content"]
        self.assertNotIn("status, progress or summary object is not", feedback)
        self.assertIn('"status"', feedback)
        self.assertEqual(ctx.agent_turns, 2)

    async def test_invalid_action_objects_never_reach_environment(self):
        bads = ['{"status":"in_progress","arguments":{}}', '{"final":null}',
                '{"final":"fake","tool":"write","arguments":{}}',
                '{"tool":"missing","arguments":{}}']
        for method in ("actor-only", "sa"):
            for bad in bads:
                with self.subTest(method=method, bad=bad):
                    ctx, _ = context(method, [bad, '{"final":"done"}'])
                    self.assertEqual(await run_profile(ctx), "done")
                    self.assertEqual(ctx.environment.calls, [])
                    self.assertEqual(ctx.agent_turns, 2)

    async def test_memgpt_json_repair_cannot_write_with_last_response(self):
        ctx, _ = context("memgpt", ["not JSON", json.dumps({"thought": "write", "function": "write",
            "arguments": {"request_heartbeat": False}})],
            {"model_response_limit": 2, "finalize_on_loop_limit": True})
        with self.assertRaises((ValueError, ModelBudgetExceeded)):
            await run_profile(ctx)
        self.assertEqual(ctx.environment.calls, [])
        self.assertEqual(ctx.agent_turns, 2)

    async def test_explicitly_disabled_finalization_keeps_its_existing_policy(self):
        ctx, _ = context("memgpt", ["not JSON", json.dumps({"thought": "write", "function": "write",
            "arguments": {"request_heartbeat": False}})],
            {"model_response_limit": 2, "finalize_on_loop_limit": False})
        self.assertEqual(await run_profile(ctx), "")
        self.assertEqual(len(ctx.environment.calls), 1)
        self.assertEqual(ctx.agent_turns, 2)

    async def test_final_slot_is_task_local_and_does_not_block_an_earlier_worker(self):
        ctx, _ = context("cmas", ["first", "last"], {"model_response_limit": 2, "finalize_on_loop_limit": True})
        earlier_done = asyncio.Event()
        last_done = asyncio.Event()
        async def earlier():
            await ctx.complete("earlier", [{"role": "user", "content": "first"}])
            earlier_done.set()
            await last_done.wait()
            await ctx.environment.call("write", {"worker": "earlier"})
        async def last():
            await earlier_done.wait()
            await ctx.complete("last", [{"role": "user", "content": "last"}])
            with self.assertRaises(ModelBudgetExceeded):
                await ctx.environment.call("write", {"worker": "last"})
            last_done.set()
        await asyncio.gather(earlier(), last())
        self.assertEqual([r["arguments"] for r in ctx.environment.calls], [{"worker": "earlier"}])

    async def test_provider_failure_during_repair_is_not_hidden_as_protocol_error(self):
        from benchmark_platform.harnesses.api import ProviderError
        ctx, _ = context("plan-execute", [])
        count = 0
        async def complete(messages, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise ProviderError("transport unavailable", kind="transport")
            from benchmark_platform.harnesses.api import Completion
            return Completion('{"status":"in_progress"}', 1, 1, 0, 0, {})
        ctx.client.complete = complete
        with self.assertRaises(ProviderError):
            await run_profile(ctx)
        self.assertEqual(ctx.agent_turns, 1)
        self.assertEqual(ctx.environment.calls, [])


class ResponsesContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_calls_carry_their_own_function_reply_schema(self):
        provider = response_client()
        bodies = []
        def respond(request, **kwargs):
            body = json.loads(request.data)
            bodies.append(body)
            schema = body["tools"][0]["parameters"]["properties"]["response"]
            key = schema["required"][0]
            return Response(envelope(output=[{"type": "function_call", "call_id": key,
                "name": "submit_benchmark_json", "arguments": json.dumps({"response": {key: "ok"}})}]))
        with patch.dict("os.environ", {"HARNESS_RESPONSES_JSON_TRANSPORT": "function"}), \
                patch("urllib.request.urlopen", side_effect=respond):
            replies = await asyncio.gather(*(provider.complete_constrained(
                [{"role": "user", "content": "Return JSON"}], response_schema=object_schema({key: {"type": "string"}}))
                for key in ("steps", "status")))
        self.assertEqual([json.loads(r.content) for r in replies], [{"steps": "ok"}, {"status": "ok"}])
        self.assertEqual({b["tools"][0]["parameters"]["properties"]["response"]["required"][0] for b in bodies}, {"steps", "status"})
        self.assertTrue(all(b["tool_choice"]["name"] == "submit_benchmark_json" for b in bodies))

    def test_text_transport_uses_named_json_schema_and_no_hosted_tools(self):
        schema = object_schema({"status": {"type": "string"}})
        with patch.dict("os.environ", {"HARNESS_RESPONSES_JSON_TRANSPORT": "text"}):
            body, _ = response_client()._request([{"role": "user", "content": "Return JSON"}],
                json_mode=True, tools=None, tool_choice=None, temperature=None, seed=None, response_schema=schema)
        self.assertEqual(body["text"]["format"]["schema"]["properties"]["response"], schema)
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(body["tools"], [])

    async def test_text_schema_envelope_is_unwrapped_without_losing_usage(self):
        with patch.dict("os.environ", {"HARNESS_RESPONSES_JSON_TRANSPORT": "text"}), \
                patch("urllib.request.urlopen", return_value=Response(envelope(text='{"response":{"status":"completed"}}'))):
            reply = await response_client().complete_constrained([{"role": "user", "content": "Return JSON"}],
                response_schema=object_schema({"status": {"type": "string"}}))
        self.assertEqual(json.loads(reply.content), {"status": "completed"})
        self.assertEqual(reply.prompt_tokens, 100)
