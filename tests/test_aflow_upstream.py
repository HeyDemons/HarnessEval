"""Behavior of the frozen workflow executor restored from HarnessEval upstream."""
import json
import unittest

from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile


class Trace:
    async def emit(self, event, **data):
        pass


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    async def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return Completion(next(self.replies), 1, 1, 0, 0, {})


class AFlowUpstreamTests(unittest.IsolatedAsyncioTestCase):
    def context(self, workflow, replies):
        async def lookup(args):
            return {"value": args["key"]}

        trace = Trace()
        env = ToolEnvironment([ToolSpec("lookup", "lookup", {"type": "object"}, (), read_only=True)], trace, {"lookup": lookup})
        policy = {} if workflow is None else {"aflow_workflow": workflow}
        return RunContext("aflow", "retrieve a value", Client(replies), env, trace, policy)

    async def test_custom_answer_generate_and_ensemble(self):
        ctx = self.context(["Custom", "AnswerGenerate", "ScEnsemble"], [
            '{"tool":"lookup","arguments":{"key":"alpha"}}', '{"final":"candidate alpha"}',
            '{"tool":"lookup","arguments":{"key":"beta"}}', '{"final":"candidate beta"}',
            "selected beta",
        ])
        self.assertEqual(await run_profile(ctx), "selected beta")
        self.assertEqual([r["arguments"]["key"] for r in ctx.environment.calls], ["alpha", "beta"])
        ensemble_prompt = json.dumps(ctx.client.messages[-1])
        self.assertIn("candidate alpha", ensemble_prompt)
        self.assertIn("candidate beta", ensemble_prompt)
        self.assertEqual(ctx.llm_calls, 5)

    async def test_returns_last_candidate_without_ensemble(self):
        ctx = self.context(["Custom", "AnswerGenerate"], ['{"final":"alpha"}', '{"final":"beta"}'])
        self.assertEqual(await run_profile(ctx), "beta")

    async def test_missing_empty_or_unsupported_workflow_is_rejected(self):
        for workflow in (None, [], "Custom", ["ScEnsemble"], ["Unknown"]):
            with self.subTest(workflow=workflow):
                ctx = self.context(workflow, [])
                with self.assertRaises(ValueError):
                    await run_profile(ctx)
                self.assertEqual(ctx.llm_calls, 0)
