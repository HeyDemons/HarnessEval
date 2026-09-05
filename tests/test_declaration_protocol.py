import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_platform.bridges import runner
from benchmark_platform.harnesses.api import Completion, ProviderError
from benchmark_platform.harnesses.core import DeclarationOnlyComplete, RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import PROFILES
from benchmark_platform.harnesses.declaration import SINGLE_TURN_PROFILES
from test_bridges import make_case


class Trace:
    def __init__(self):
        self.events = []

    async def emit(self, event, **data):
        self.events.append({"event": event, **data})


class Client:
    def __init__(self, message):
        self.message = message
        self.requests = []

    async def complete_native(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        if isinstance(self.message, Exception):
            raise self.message
        return Completion(self.message.get("content") or "", 10, 5, 0, 0,
                          {"choices": [{"message": self.message}]})

    async def complete(self, messages, **kwargs):
        return await self.complete_native(messages, **kwargs)


def message(calls):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": str(i), "type": "function", "function": {"name": "lookup_item", "arguments": json.dumps({"id": value})}}
        for i, value in enumerate(calls)]}


def context(profile, response):
    trace = Trace()
    async def forbidden(args):
        raise AssertionError("declaration executed a handler")
    tool = ToolSpec("lookup_item", "lookup", {"type": "object"}, (), read_only=False)
    return RunContext(profile, "lookup two ids", Client(response),
                      ToolEnvironment([tool], trace, {"lookup_item": forbidden}, declaration_only=True), trace, {})


class DeclarationTests(unittest.IsolatedAsyncioTestCase):
    async def test_bridge_scores_only_one_native_response_and_no_synthetic_followup(self):
        for profile in ("actor-only", "react", "sa"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / "input"; source.mkdir()
                job = root / "job"; job.mkdir(); make_case(source, "bfcl")
                client = Client(message(["a", "b"]))
                with patch.object(runner, "completion_client_from_env", return_value=client):
                    result = await runner.execute("bfcl", profile, "case", source, job, {})
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["tool_calls"], 2)
                self.assertEqual(result["agent_turns"], 1)
                self.assertEqual(len(client.requests), 1)
                events = [json.loads(x) for x in (job / "harness_trace.jsonl").read_text().splitlines()]
                ids = [e["assistant_response_id"] for e in events if e["event"] == "tool_result"]
                self.assertEqual(ids, [1, 1])

    async def test_native_batch_all_supported_tool_profiles_without_execution(self):
        for profile in ("actor-only", "react", "sa"):
            ctx = context(profile, message(["a", "b"]))
            await run_profile(ctx)
            self.assertEqual(ctx.llm_calls, 1)
            self.assertEqual(ctx.speculator_llm_calls, 0)
            self.assertEqual(ctx.environment.state_version, 0)
            self.assertEqual([c["arguments"]["id"] for c in ctx.environment.committed_calls], ["a", "b"])
            self.assertTrue(all(c["result"]["result"]["execution"] == "not_run" for c in ctx.environment.calls))
            with self.assertRaises(DeclarationOnlyComplete):
                await ctx.complete("extra", [])
            self.assertEqual(len(ctx.client.requests), 1)

    async def test_empty_first_response_is_final_and_does_not_search_for_later_calls(self):
        ctx = context("actor-only", {"role": "assistant", "content": "No relevant function"})
        self.assertEqual(await run_profile(ctx), "No relevant function")
        self.assertEqual(ctx.environment.committed_calls, [])
        with self.assertRaises(DeclarationOnlyComplete):
            await ctx.complete_native("second", [])
        self.assertEqual(ctx.llm_calls, 1)

    async def test_multi_response_methods_rejected_before_provider(self):
        for profile in PROFILES:
            if profile.id in SINGLE_TURN_PROFILES:
                continue
            ctx = context(profile.id, message(["a"]))
            with self.assertRaisesRegex(ValueError, "multi-response"):
                await run_profile(ctx)
            self.assertFalse(ctx.client.requests)

    async def test_cross_response_records_are_rejected_without_partial_commit(self):
        ctx = context("actor-only", message([]))
        records = [{"name": "lookup_item", "arguments": {"id": str(i)},
                    "assistant_response_id": i, "result": {"ok": True}} for i in (1, 3)]
        with self.assertRaisesRegex(ValueError, "different assistant responses"):
            await ctx.environment.commit_isolated_calls(records)
        self.assertEqual(ctx.environment.calls, [])
        with self.assertRaisesRegex(ValueError, "relabel"):
            await ctx.environment.commit_isolated_calls(records[:1], assistant_response_id=7)

    async def test_malformed_trailing_call_does_not_commit_partial_batch(self):
        response = message(["a", "b"])
        response["tool_calls"][1]["function"]["arguments"] = "not-json"
        ctx = context("actor-only", response)
        with self.assertRaises(ValueError):
            await run_profile(ctx)
        self.assertEqual(ctx.environment.calls, [])

    async def test_bridge_preserves_unexpected_failure_instead_of_masking_it(self):
        async def broken(ctx):
            await ctx.complete("first", [])
            await ctx.environment.call("lookup_item", {"id": "a"})
            raise RuntimeError("unexpected after commit")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "input"; source.mkdir()
            job = root / "job"; job.mkdir(); make_case(source, "bfcl")
            with patch.object(runner, "completion_client_from_env", return_value=Client(message([]))), \
                 patch.object(runner, "run_profile", new=broken):
                result = await runner.execute("bfcl", "actor-only", "case", source, job, {})
        self.assertEqual(result["status"], "failed")
        self.assertIn("unexpected after commit", result["error"])
        self.assertEqual(len(result["committed_calls"]), 1)
