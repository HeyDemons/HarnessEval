import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_platform.bridges.automation_episode import api_specs, public_messages, public_prompt, run_episode
from benchmark_platform.harnesses.api import Completion, ProviderError
from benchmark_platform.harnesses.core import ToolSpec


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
