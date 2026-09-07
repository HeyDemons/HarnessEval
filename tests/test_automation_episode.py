import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_platform.bridges.automation_episode import AutomationEpisode, api_specs, public_messages, public_prompt, run_episode
from benchmark_platform.harnesses.api import Completion, ProviderError
from benchmark_platform.harnesses.core import ToolSpec, ToolEnvironment, JsonlTrace
from benchmark_platform.catalog import Catalog
from benchmark_platform.compatibility import compatibility_rows
from benchmark_platform.harnesses.profiles import PROFILES
from tests.test_episode import ScriptedClient, PROFILE_RESPONSES


def native_call(name, arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class FakeEpisode:
    def __init__(self, order):
        self.prompt = "Complete the public task"
        self.messages = [{"role": "system", "content": "PUBLIC_SYSTEM_POLICY"},
                         {"role": "user", "content": self.prompt}]
        self.metadata = {"task_contract_sha256": "test-contract"}
        self.tools = [ToolSpec("work", "work", {"type": "object"}, ())]
        self.order = order
        self.info = {"assertions": ["PRIVATE_ASSERTION"]}

    def handlers(self):
        async def work(args):
            self.order.append("tool")
            return {"done": True}
        return {"work": work}

    def finalize(self):
        self.order.append("scorer")
        return {"partial_credit": 0.5, "task_completed_correctly": 0.0, "assertion_results": ["PRIVATE_ASSERTION"]}


class Client:
    def __init__(self, order, fail=False):
        self.order, self.fail, self.calls = order, fail, 0

    async def complete(self, messages, **kwargs):
        assert "PRIVATE_ASSERTION" not in json.dumps(messages)
        assert messages[0] == {"role": "system", "content": "PUBLIC_SYSTEM_POLICY"}
        assert all("PUBLIC_SYSTEM_POLICY" not in m["content"] for m in messages if m["role"] == "user")
        if self.fail:
            self.order.append("provider_failed")
            raise ProviderError("test provider failure", kind="transport")
        self.calls += 1
        self.order.append("agent_tool" if self.calls == 1 else "agent_final")
        text = '{"tool":"work","arguments":{}}' if self.calls == 1 else '{"final":"done"}'
        return Completion(text, 1, 1, 0, 0, {})

    async def complete_native(self, messages, **kwargs):
        assert "PRIVATE_ASSERTION" not in json.dumps(messages)
        assert messages[0] == {"role": "system", "content": "PUBLIC_SYSTEM_POLICY"}
        assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["work"]
        if self.fail:
            self.order.append("provider_failed")
            raise ProviderError("test provider failure", kind="transport")
        self.calls += 1
        self.order.append("agent_tool" if self.calls == 1 else "agent_final")
        message = {"role": "assistant", "content": "done"}
        if self.calls == 1:
            message.update(content="", tool_calls=[{"id": "call_1", "type": "function", "function": {
                "name": "work", "arguments": "{}"}}])
        else:
            assert messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"done": true}'}
        return Completion(message["content"], 1, 1, 0, 0, {"choices": [{"message": message}]})


class AutomationEpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runnable_matrix_executes_tools_before_native_scoring(self):
        root = Path(__file__).resolve().parents[1]
        catalog = Catalog(root / "catalog/benchmarks.json", root, root.parent)
        rows = compatibility_rows(PROFILES, [catalog.get("automationbench")])
        action = '{"tool":"work","arguments":{}}'
        responses = {
            "actor-only": [native_call("work", {}), "done"],
            "react": [native_call("work", {}), native_call("react_finish", {"answer": "done"})],
            "plan-execute": [PROFILE_RESPONSES["plan-execute"][0], action, '{"final":"done"}'],
            "cmas": [PROFILE_RESPONSES["cmas"][0], action, *PROFILE_RESPONSES["cmas"][1:]],
            "dmas": [*PROFILE_RESPONSES["dmas"][:3], action, PROFILE_RESPONSES["dmas"][3]],
            "memgpt": ['{"thought":"work","function":"work","arguments":{"request_heartbeat":true}}',
                       *PROFILE_RESPONSES["memgpt"]],
            "aflow-tools": [action, '{"final":"done"}'],
            "dylan": [action] * 4 + ['{"final":"done"}'] * 4,
            "llmcompiler": ['{"tasks":[{"id":1,"tool":"work","arguments":{},"dependencies":[]}]}',
                            '{"action":"finish","answer":"done"}'],
            "rewoo": ['Plan: work\n#E1 = work[{}]', "done"],
            "sa": [action, '{"final":"done"}'],
            "aflow": ["done"],
            "multi-persona": ["Final answer: done"],
            "magentic-one": [
                "facts", "plan", PROFILE_RESPONSES["magentic-one"][2].replace("FileSurfer", "WebSurfer"),
                native_call("work", {}), PROFILE_RESPONSES["magentic-one"][4], "done",
            ],
        }
        self.assertEqual({r["baseline"] for r in rows if r["runnable"]}, set(responses))
        for method, replies in responses.items():
            with self.subTest(profile=method), tempfile.TemporaryDirectory() as tmp:
                order = []
                actor = ScriptedClient(replies)
                policy = {"react_protocol": "native"}
                if method in {"aflow", "aflow-tools"}:
                    from benchmark_platform.harnesses.aflow import make_artifact as make_qa_artifact
                    from benchmark_platform.harnesses.aflow_tools import make_artifact as make_tool_artifact
                    # Initial graph is a protocol fixture, not an optimized evaluation artifact.
                    policy.update(aflow_artifact=(make_tool_artifact() if method == "aflow-tools"
                                                  else make_qa_artifact()),
                                  aflow_allow_initialization=True)
                with patch("benchmark_platform.bridges.automation_episode.sa_speculator_client_from_env",
                           side_effect=AssertionError("injected clients must not read provider configuration")):
                    result = await run_episode(method, "sales:1", policy, Path(tmp),
                        episode=FakeEpisode(order), client=actor, speculator_client=ScriptedClient([]))
                self.assertEqual(result["status"], "completed", result.get("error"))
                expected_tools = 0 if method in {"aflow", "multi-persona"} else 1
                self.assertEqual(order, ["tool"] * expected_tools + ["scorer"])
                self.assertEqual(result["tool_calls"], expected_tools)
                self.assertEqual(result["native_score"], 0)
                self.assertEqual(result["native_partial_credit"], 0.5)
                self.assertEqual(result["policy"]["model_response_limit"], 50)
                transcript = json.dumps(actor.requests)
                self.assertNotIn("PRIVATE_ASSERTION", transcript)
                self.assertIn("PUBLIC_SYSTEM_POLICY", transcript)

    async def test_native_strict_score_is_used_after_agent_and_partial_is_separate(self):
        order = []
        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode("actor-only", "sales:1", {}, Path(tmp), episode=FakeEpisode(order), client=Client(order))
            self.assertEqual(order, ["agent_tool", "tool", "agent_final", "scorer"])
            self.assertEqual(result["native_score"], 0)
            self.assertEqual(result["native_partial_credit"], 0.5)
            self.assertEqual(result["llm_calls"], 2)
            self.assertNotIn("PRIVATE_ASSERTION", (Path(tmp) / "bridge_manifest.json").read_text())
            self.assertTrue((Path(tmp) / "official_score.json").exists())

    async def test_provider_failure_cannot_acquire_score_from_native_scorer(self):
        order = []
        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode("actor-only", "sales:1", {}, Path(tmp), episode=FakeEpisode(order), client=Client(order, True))
            self.assertEqual(order, ["provider_failed", "scorer"])
            self.assertEqual(result["failure_kind"], "provider_error")
            self.assertIsNone(result["native_score"])

    async def test_dataset_setup_consumes_arm_budget_and_scoring_still_persists(self):
        order = []
        def initialize(case_id):
            time.sleep(0.03)
            return FakeEpisode(order)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmark_platform.bridges.automation_episode.AutomationEpisode", side_effect=initialize
        ):
            result = await run_episode("actor-only", "sales:1",
                {"automationbench_arm_timeout_s": 0.01, "automationbench_finalize_grace_s": 0},
                Path(tmp), client=Client(order))
            self.assertEqual(order, ["scorer"])
            self.assertEqual(result["failure_kind"], "agent_timeout")
            self.assertTrue((Path(tmp) / "official_score.json").exists())
            self.assertEqual(result["llm_calls"], 0)

    def test_prompt_does_not_render_private_state_or_assertions(self):
        row = {"prompt": [{"role": "user", "content": "public task"}], "info": {"assertions": "PRIVATE_ASSERTION", "initial_state": "PRIVATE_WORLD"}}
        self.assertEqual(public_prompt(row), "[USER]\npublic task")

    def test_system_role_is_separate_from_text_algorithm_task(self):
        row = {"prompt": [{"role": "system", "content": "PUBLIC_SYSTEM_POLICY", "private": "NO"},
                          {"role": "user", "content": "public task"}], "info": {"assertions": "PRIVATE"}}
        self.assertEqual(public_prompt(row), "[USER]\npublic task")
        self.assertEqual(public_messages(row), [{"role": "system", "content": "PUBLIC_SYSTEM_POLICY"},
                                                 {"role": "user", "content": "public task"}])

    async def test_json_control_also_keeps_task_system_instructions(self):
        order = []
        with tempfile.TemporaryDirectory() as tmp:
            result = await run_episode("actor-only", "sales:1", {"automationbench_actor_protocol": "json"},
                                       Path(tmp), episode=FakeEpisode(order), client=Client(order))
            self.assertEqual(order, ["agent_tool", "tool", "agent_final", "scorer"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["policy"]["automationbench_prompt_protocol"], "native-task-roles-v1")

    def test_stateful_api_fetch_is_not_safe_for_speculation(self):
        with patch("benchmark_platform.bridges.automation_episode.official_tool_definition",
                   return_value=SimpleNamespace(description="official description", parameters={"type": "object"})):
            specs = {tool.name: tool for tool in api_specs({name: lambda: None for name in ["api_search", "api_fetch", "base64_encode"]})}
        self.assertFalse(specs["api_fetch"].read_only)
        self.assertFalse(specs["api_fetch"].parallel)
        self.assertTrue(specs["api_search"].read_only)
        self.assertTrue(specs["base64_encode"].read_only)

    async def test_official_optional_defaults_survive_strict_schema_and_empty_object(self):
        episode = object.__new__(AutomationEpisode)
        observed = []
        def search(query, top_k=5):
            observed.append((query, top_k))
            return "official result"
        episode.functions = {"api_search": search}
        spec = ToolSpec("api_search", "search", {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query", "top_k"]}, ())
        with tempfile.TemporaryDirectory() as tmp:
            trace = JsonlTrace(Path(tmp) / "trace.jsonl")
            env = ToolEnvironment([spec], trace, episode.handlers(), validate_schema=False)
            for arguments in ({"query": "x"}, {"query": "x", "top_k": {}}, {"query": "x", "top_k": 0}):
                self.assertTrue((await env.call("api_search", arguments))["ok"])
            self.assertEqual(observed, [("x", 5), ("x", 5), ("x", 0)])
            rejected = await env.call("api_search", {"query": "x", "world": {"injected": True}})
            self.assertFalse(rejected["ok"])
            self.assertIn("controller-owned", rejected["detail"])
            regular = ToolEnvironment([spec], trace, episode.handlers())
            self.assertEqual((await regular.call("api_search", {"query": "x"}))["error"], "invalid_arguments")
