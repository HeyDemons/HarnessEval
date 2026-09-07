"""Exercise SA adoption through production bridge entrypoints and handlers.

Provider replies and the Docker process boundary are scripted; GAIA file reads
are real. These are deterministic integration tests, not scored experiments.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_platform.budgets import baseline_limits
from benchmark_platform.bridges import runner, terminal_episode, automation_episode
from benchmark_platform.harnesses.core import ToolSpec
from tests.test_bridges import make_case
from tests.test_episode import ScriptedClient
from tests.test_automation_episode import FakeEpisode


class BridgeSpeculationTests(unittest.IsolatedAsyncioTestCase):
    async def test_hit_and_miss_publish_only_the_actor_read(self):
        for benchmark in ("gaia", "terminal-bench-2", "automationbench"):
            for hit in (True, False):
                with self.subTest(benchmark=benchmark, hit=hit), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    job = root / "job"
                    job.mkdir()
                    tool = "api_search" if benchmark == "automationbench" else "read_file"
                    key = "query" if benchmark == "automationbench" else "path"
                    actor_args = {key: "evidence.txt"}
                    predicted_args = actor_args if hit else {key: "wrong.txt"}
                    actor = ScriptedClient([json.dumps({"tool": tool, "arguments": actor_args}),
                                            '{"final":"done"}'])
                    speculator = ScriptedClient([json.dumps({"actions": [
                        {"tool": tool, "arguments": predicted_args}]})])
                    # Last slot finalization is production behavior; a small loop
                    # boundary makes it testable without hundreds of fake replies.
                    policy = {**baseline_limits(benchmark), "max_turns": 2}
                    executions = []
                    if benchmark == "automationbench":
                        episode = FakeEpisode([])
                        episode.tools = [ToolSpec(tool, "Search public API documentation",
                            {"type": "object", "properties": {key: {"type": "string"}}, "required": [key]},
                            (), read_only=True, parallel=True)]
                        async def search(args):
                            executions.append(args)
                            return {"document": args[key]}
                        episode.handlers = lambda: {tool: search}
                        with patch.object(automation_episode, "sa_speculator_client_from_env",
                                          side_effect=AssertionError("unexpected env lookup")):
                            result = await automation_episode.run_episode("sa", "sales:1", policy, job,
                                episode=episode, client=actor, speculator_client=speculator)
                        self.assertEqual(executions, [predicted_args] if hit else [predicted_args, actor_args])
                    elif benchmark == "gaia":
                        source = root / "source"
                        source.mkdir()
                        make_case(source, benchmark)
                        (source / "workspace/wrong.txt").write_text("wrong evidence")
                        load_case = runner.load_case
                        def tracked_case(*args):
                            bridge = load_case(*args)
                            read = bridge.handlers[tool]
                            async def tracked_read(arguments):
                                executions.append(arguments)
                                return await read(arguments)
                            bridge.handlers = {**bridge.handlers, tool: tracked_read}
                            return bridge
                        with patch.object(runner, "completion_client_from_env", return_value=actor), \
                             patch.object(runner, "sa_speculator_client_from_env", return_value=speculator), \
                             patch.object(runner, "load_case", side_effect=tracked_case):
                            result = await runner.execute(benchmark, "sa", "case", source, job, policy)
                        self.assertEqual(executions, [predicted_args] if hit else [predicted_args, actor_args])
                    else:
                        async def docker_read(command, **kwargs):
                            executions.append(command)
                            return {"ok": True, "result": {"stdout": "complete evidence"}}
                        with patch.object(terminal_episode, "completion_client_from_env", return_value=actor), \
                             patch.object(terminal_episode, "sa_speculator_client_from_env", return_value=speculator), \
                             patch.object(terminal_episode, "_async_completed", side_effect=docker_read):
                            result = await terminal_episode.execute(profile_id="sa", prompt="Read evidence.txt",
                                policy=policy, container="scripted-container", trace_path=job / "harness_trace.jsonl")
                        self.assertEqual(len(executions), 1 if hit else 2)
                    self.assertEqual(result["status"], "completed", result.get("error"))
                    self.assertEqual(result["tool_calls"], 1)
                    self.assertEqual(result["actor_llm_calls"], 2)
                    self.assertEqual(result["speculator_llm_calls"], 1)
                    events = [json.loads(line) for line in (job / "harness_trace.jsonl").read_text().splitlines()]
                    published = [e for e in events if e["event"] == "tool_result"]
                    predicted = [e for e in events if e["event"] == "sa_speculative_tool_result"]
                    self.assertEqual(len(published), 1)
                    self.assertEqual(published[0]["arguments"], actor_args)
                    self.assertTrue(published[0]["result"]["ok"])
                    self.assertEqual(len(predicted), 1)
                    self.assertEqual(predicted[0]["arguments"], predicted_args)
                    self.assertEqual(sum(e["event"] == "sa_cache_hit" for e in events), int(hit))
                    actor_id = next(e["response_id"] for e in events
                                    if e["event"] == "llm_response" and e["role"] == "sa_actor")
                    self.assertEqual(published[0]["assistant_response_id"], actor_id)
                    self.assertTrue(any(e["event"] == "budget_finalization" for e in events))
                    if not hit:
                        self.assertNotIn("wrong.txt", json.dumps(actor.requests))

    async def test_gaia_final_slot_rejects_new_tool_without_executing_it(self):
        for method in ("actor-only", "sa"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, job = root / "source", root / "job"
                source.mkdir()
                job.mkdir()
                make_case(source, "gaia")
                actor = ScriptedClient(['{"tool":"read_file","arguments":{"path":"evidence.txt"}}'])
                speculator = ScriptedClient([])
                with patch.object(runner, "completion_client_from_env", return_value=actor), \
                     patch.object(runner, "sa_speculator_client_from_env", return_value=speculator):
                    result = await runner.execute("gaia", method, "case", source, job,
                        {**baseline_limits("gaia"), "max_turns": 1})
                self.assertEqual(result["status"], "failed")
                self.assertIn("budget exhausted", result["error"])
                self.assertEqual(result["tool_calls"], 0)
                self.assertEqual(result["actor_llm_calls"], 1)
                self.assertEqual(result["speculator_llm_calls"], 0)
