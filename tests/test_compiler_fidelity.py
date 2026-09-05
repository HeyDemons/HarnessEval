"""Regression cases for the pinned LLMCompiler TaskFetchingUnit contract."""
import asyncio
import json
import unittest

from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.paper_methods import _resolve_reference
from benchmark_platform.harnesses.compiler_references import resolve_arguments


class Trace:
    def __init__(self):
        self.events = []

    async def emit(self, event, **data):
        self.events.append({"event": event, **data})


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def complete(self, messages, **kwargs):
        value = next(self.responses)
        text = value if isinstance(value, str) else json.dumps(value)
        return Completion(text, 1, 1, 0, 0, {})


def make_context(tasks, handlers):
    trace = Trace()
    specs = [ToolSpec(name, name, {"type": "object"}, (), parallel=True, read_only=True) for name in handlers]
    env = ToolEnvironment(specs, trace, handlers)
    return RunContext("llmcompiler", "synthetic task", Client([
        {"tasks": tasks}, {"action": "finish", "answer": "done"},
    ]), env, trace, {})


class CompilerFidelityTests(unittest.IsolatedAsyncioTestCase):
    def test_upstream_preserves_literal_suffixes_and_string_types(self):
        results = {"1": "report", "10": {"ok": True}}
        self.assertEqual(resolve_arguments("read $1.txt ${1}.csv", [1], results),
                         "read report.txt report.csv")
        self.assertEqual(resolve_arguments("${10}", [10], results), "{'ok': True}")
        self.assertEqual(resolve_arguments(["$10", "$1", "$PATH", "${HOME}"], [1, 10], results),
                         ["{'ok': True}", "report", "$PATH", "${HOME}"])

    def test_only_declared_dependencies_are_substituted(self):
        self.assertEqual(resolve_arguments({"q": "$2 ${2} $1"}, [1], {"1": "yes", "2": "no"}),
                         {"q": "$2 ${2} yes"})
        self.assertEqual(resolve_arguments("$1", [1], {"1": None}), "$1")

    async def test_default_runner_uses_pinned_text_substitution(self):
        received = []
        async def lookup(args):
            return "report"
        async def consume(args):
            received.append(args)
            return "done"
        ctx = make_context([
            {"id": "1", "tool": "lookup", "arguments": {}, "dependencies": []},
            {"id": "2", "tool": "consume", "arguments": {"q": "$1.txt", "other": "$9"}, "dependencies": ["1"]},
        ], {"lookup": lookup, "consume": consume})
        await run_profile(ctx)
        self.assertEqual(received, [{"q": "{'ok': True, 'result': 'report'}.txt", "other": "$9"}])
        config = next(e for e in ctx.trace.events if e["event"] == "llmcompiler_config")
        self.assertEqual(config["reference_mode"], "upstream")

    def test_interpolates_braced_nested_and_embedded_references_without_shell_variables(self):
        values = {"1": {"result": {"city": "Paris", "n": 42}}, "10": {"result": ["France"]}}
        result = _resolve_reference({
            "query": "population of $1.result.city, ${10}.result[0]",
            "number": "$1.result.n",
            "whole": "${1}",
            "command": ["bash", "-c", "echo $PATH ${PATH} $HOME; echo ${1}.result.n"],
            "env": "$PATH",
        }, values, interpolate_strings=True)
        self.assertEqual(result["query"], "population of Paris, France")
        self.assertEqual(result["number"], 42)
        self.assertEqual(result["whole"], values["1"])
        self.assertEqual(result["command"][-1], "echo $PATH ${PATH} $HOME; echo 42")
        self.assertEqual(result["env"], "$PATH")

    async def test_successor_starts_before_unrelated_task_finishes(self):
        release = asyncio.Event()
        successor = asyncio.Event()
        received = []

        async def fast(args):
            return {"city": "Paris"}

        async def slow(args):
            await release.wait()
            return "slow"

        async def dependent(args):
            received.append(args)
            successor.set()
            return "done"

        ctx = make_context([
            {"id": "1", "tool": "fast", "arguments": {}, "dependencies": []},
            {"id": "2", "tool": "slow", "arguments": {}, "dependencies": []},
            {"id": "3", "tool": "dependent", "arguments": {"q": "population of $1.result.city"}, "dependencies": ["1"]},
        ], {"fast": fast, "slow": slow, "dependent": dependent})
        ctx.policy["llmcompiler_reference_mode"] = "legacy-json-fields"
        task = asyncio.create_task(run_profile(ctx))
        try:
            await asyncio.wait_for(successor.wait(), 1)
            self.assertFalse(task.done())
            self.assertEqual(received, [{"q": "population of Paris"}])
        finally:
            release.set()
            await task

    async def test_cancellation_awaits_running_tool_cleanup(self):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def block(args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        ctx = make_context([{"id": "1", "tool": "block", "arguments": {}}], {"block": block})
        task = asyncio.create_task(run_profile(ctx))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cleaned.is_set())

    async def test_cycle_is_reported_without_hanging(self):
        ctx = make_context([
            {"id": "1", "tool": "none", "dependencies": ["2"]},
            {"id": "2", "tool": "none", "dependencies": ["1"]},
        ], {})
        self.assertEqual(await asyncio.wait_for(run_profile(ctx), 1), "done")
        event = next(x for x in ctx.trace.events if x["event"] == "llmcompiler_dag_complete")
        self.assertEqual(event["results"]["_scheduler"]["error"], "dependency_deadlock")
