import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_platform.bridges import runner
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import (
    DeclarationOnlyComplete,
    RunContext,
    ToolEnvironment,
    ToolSpec,
)
from benchmark_platform.harnesses.declaration import (
    DECLARATIONS_CLOSE,
    DECLARATIONS_OPEN,
    PUBLISHER_PROTOCOL,
    parse_method_declarations,
    publish_method_declaration,
)
from test_bridges import make_case


class Trace:
    def __init__(self):
        self.events = []

    async def emit(self, event, **data):
        self.events.append({"event": event, **data})


class Client:
    def __init__(self, message):
        self.messages = iter(message if isinstance(message, list) else [message])
        self.requests = []

    async def complete_native(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        message = next(self.messages)
        if isinstance(message, Exception):
            raise message
        if not isinstance(message, dict):
            message = {"role": "assistant", "content": str(message)}
        return Completion(
            message.get("content") or "",
            10,
            5,
            0,
            0,
            {"choices": [{"message": message}]},
        )

    async def complete(self, messages, **kwargs):
        return await self.complete_native(messages, **kwargs)


def declaration_text(*ids: str) -> str:
    calls = [
        {"name": "lookup_item", "arguments": {"id": value}}
        for value in ids
    ]
    return DECLARATIONS_OPEN + json.dumps(calls) + DECLARATIONS_CLOSE


def actor_final(*ids: str) -> str:
    return json.dumps({"final": declaration_text(*ids)})


def context(profile, responses):
    trace = Trace()

    async def forbidden(_args):
        raise AssertionError("declaration executed a handler")

    tool = ToolSpec(
        "lookup_item",
        "lookup",
        {"type": "object"},
        (),
        parallel=True,
        read_only=True,
    )
    return RunContext(
        profile,
        "lookup two ids",
        Client(responses),
        ToolEnvironment(
            [tool], trace, {"lookup_item": forbidden}, proposal_only=True
        ),
        trace,
        {},
    )


class DeclarationTests(unittest.IsolatedAsyncioTestCase):
    async def test_method_final_response_publishes_parallel_batch_without_extra_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            job = root / "job"
            job.mkdir()
            make_case(source, "bfcl")
            client = Client(actor_final("a", "b"))
            with patch.object(runner, "completion_client_from_env", return_value=client):
                result = await runner.execute("bfcl", "actor-only", "case", source, job, {})

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["agent_turns"], 1)
        self.assertEqual(result["internal_llm_calls"], 1)
        self.assertEqual(result["publisher_llm_calls"], 0)
        self.assertEqual(result["declaration_protocol"], PUBLISHER_PROTOCOL)
        self.assertEqual(len(client.requests), 1)
        self.assertIn("BFCL_DECLARATIONS", str(client.requests[0][0]))
        self.assertEqual(
            [(call["name"], call["arguments"]) for call in result["committed_calls"]],
            [
                ("lookup_item", {"id": "a"}),
                ("lookup_item", {"id": "b"}),
            ],
        )

    async def test_deterministic_publication_never_executes_handlers_or_adds_cost(self):
        ctx = context("react", declaration_text("a", "b"))
        output = await ctx.complete("method-final", [{"role": "user", "content": "finish"}])
        before = ctx.llm_calls
        await publish_method_declaration(
            ctx,
            method_output=output,
            proposal_calls=[],
            tool_capable=True,
        )
        self.assertEqual(ctx.llm_calls, before)
        self.assertEqual(ctx.environment.state_version, 0)
        self.assertEqual(
            [call["arguments"]["id"] for call in ctx.environment.committed_calls],
            ["a", "b"],
        )
        self.assertTrue(
            all(call["result"]["result"]["execution"] == "not_run" for call in ctx.environment.calls)
        )

    async def test_plain_final_text_publishes_empty_batch_and_freezes_boundary(self):
        ctx = context("react", "No relevant function")
        output = await ctx.complete("method-final", [{"role": "user", "content": "finish"}])
        self.assertEqual(
            await publish_method_declaration(
                ctx,
                method_output=output,
                proposal_calls=[],
                tool_capable=True,
            ),
            "No relevant function",
        )
        self.assertEqual(ctx.environment.committed_calls, [])
        with self.assertRaises(DeclarationOnlyComplete):
            await ctx.complete_native("second", [])
        self.assertEqual(ctx.llm_calls, 1)

    async def test_text_only_method_never_publishes_calls(self):
        ctx = context("multi-persona", declaration_text("a"))
        output = await ctx.complete("method-final", [{"role": "user", "content": "finish"}])
        await publish_method_declaration(
            ctx,
            method_output=output,
            proposal_calls=[],
            tool_capable=False,
        )
        self.assertEqual(ctx.environment.committed_calls, [])

    async def test_sa_prelaunches_and_adopts_only_the_actor_selected_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            job = root / "job"
            job.mkdir()
            make_case(source, "bfcl")
            actor = Client([
                '{"tool":"lookup_item","arguments":{"id":"a"}}',
                actor_final("a", "b"),
            ])
            speculator = Client([
                '{"actions":[{"tool":"lookup_item","arguments":{"id":"a"}}]}',
                '{"actions":[]}',
            ])
            with patch.object(
                runner, "completion_client_from_env", return_value=actor
            ), patch.object(
                runner, "sa_speculator_client_from_env", return_value=speculator
            ):
                result = await runner.execute("bfcl", "sa", "case", source, job, {})

            events = [
                json.loads(line)
                for line in (job / "harness_trace.jsonl").read_text().splitlines()
            ]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["speculator_llm_calls"], 2)
        self.assertEqual(len(result["proposal_calls"]), 1)
        self.assertEqual(result["proposal_calls"][0]["arguments"], {"id": "a"})
        self.assertEqual(len(result["committed_calls"]), 2)
        self.assertTrue(any(row["event"] == "sa_cache_hit" for row in events))
        self.assertEqual(result["publisher_llm_calls"], 0)

    def test_parser_preserves_duplicates_and_rejects_partial_or_ambiguous_batches(self):
        parsed = parse_method_declarations(declaration_text("a", "a"))
        self.assertEqual([item[1]["id"] for item in parsed], ["a", "a"])
        malformed = DECLARATIONS_OPEN + '[{"name":"lookup_item"}]' + DECLARATIONS_CLOSE
        with self.assertRaises(ValueError):
            parse_method_declarations(malformed)
        with self.assertRaises(ValueError):
            parse_method_declarations(declaration_text("a") + declaration_text("b"))

    async def test_malformed_batch_commits_nothing(self):
        malformed = DECLARATIONS_OPEN + '[{"name":"lookup_item"}]' + DECLARATIONS_CLOSE
        ctx = context("actor-only", malformed)
        output = await ctx.complete("method-final", [{"role": "user", "content": "finish"}])
        with self.assertRaises(ValueError):
            await publish_method_declaration(
                ctx,
                method_output=output,
                proposal_calls=[],
                tool_capable=True,
            )
        self.assertEqual(ctx.environment.calls, [])

    async def test_bridge_preserves_internal_failure_without_publication(self):
        async def broken(ctx):
            await ctx.complete("first", [])
            await ctx.environment.call("lookup_item", {"id": "a"})
            raise RuntimeError("unexpected before final output")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            job = root / "job"
            job.mkdir()
            make_case(source, "bfcl")
            with patch.object(
                runner, "completion_client_from_env", return_value=Client("internal")
            ), patch.object(runner, "run_profile", new=broken):
                result = await runner.execute(
                    "bfcl", "plan-execute", "case", source, job, {}
                )
        self.assertEqual(result["status"], "failed")
        self.assertIn("unexpected before final output", result["error"])
        self.assertEqual(result["committed_calls"], [])
        self.assertEqual(len(result["proposal_calls"]), 1)
