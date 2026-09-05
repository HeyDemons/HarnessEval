import json
import tempfile
import unittest
from pathlib import Path

from benchmark_platform.bridges.automation_episode import api_specs, public_prompt, run_episode
from benchmark_platform.harnesses.api import Completion, ProviderError
from benchmark_platform.harnesses.core import ToolSpec


class FakeEpisode:
    def __init__(self, order):
        self.prompt = "Complete the public task"
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
        if self.fail:
            self.order.append("provider_failed")
            raise ProviderError("test provider failure", kind="transport")
        self.calls += 1
        self.order.append("agent_tool" if self.calls == 1 else "agent_final")
        text = '{"tool":"work","arguments":{}}' if self.calls == 1 else '{"final":"done"}'
        return Completion(text, 1, 1, 0, 0, {})


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

    def test_prompt_does_not_render_private_state_or_assertions(self):
        row = {"prompt": [{"role": "user", "content": "public task"}], "info": {"assertions": "PRIVATE_ASSERTION", "initial_state": "PRIVATE_WORLD"}}
        self.assertEqual(public_prompt(row), "[USER]\npublic task")

    def test_stateful_api_fetch_is_not_safe_for_speculation(self):
        specs = {tool.name: tool for tool in api_specs({name: lambda: None for name in ["api_search", "api_fetch", "base64_encode"]})}
        self.assertFalse(specs["api_fetch"].read_only)
        self.assertFalse(specs["api_fetch"].parallel)
        self.assertTrue(specs["api_search"].read_only)
        self.assertTrue(specs["base64_encode"].read_only)
