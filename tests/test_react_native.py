import json
import unittest

from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from test_harnesses import ScriptedClient, native_tool_call
from test_declaration_protocol import Trace


class NativeReactTests(unittest.IsolatedAsyncioTestCase):
    def context(self, replies):
        trace = Trace()
        async def read(args):
            return {"value": "observed"}
        return RunContext("react", "Read then answer", ScriptedClient(replies),
            ToolEnvironment([ToolSpec("read", "read", {"type": "object"}, ())], trace, {"read":read}),
            trace, {"react_protocol":"native"})

    async def test_native_action_observation_finish(self):
        ctx = self.context([native_tool_call("read", {}), native_tool_call("react_finish", {"answer":"observed"})])
        self.assertEqual(await run_profile(ctx), "observed")
        self.assertEqual(len(ctx.environment.calls), 1)
        self.assertEqual(ctx.llm_calls, 2)
        observation = next(m for m in ctx.client.messages[1] if m["role"] == "tool")
        self.assertIn("observed", observation["content"])

    async def test_multiple_actions_are_rejected_before_any_side_effect(self):
        batch = native_tool_call("read", {})
        batch["tool_calls"].append({**batch["tool_calls"][0], "id":"other"})
        ctx = self.context([batch, native_tool_call("react_finish", {"answer":"done"})])
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual(ctx.environment.calls, [])
        self.assertEqual(len([m for m in ctx.client.messages[1] if m["role"] == "tool"]), 2)

    async def test_bad_arguments_get_an_error_observation(self):
        call = native_tool_call("read", {})
        call["tool_calls"][0]["function"]["arguments"] = "{bad"
        ctx = self.context([call, native_tool_call("react_finish", {"answer":"done"})])
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual(ctx.environment.calls, [])
        self.assertIn("invalid_arguments", json.dumps(ctx.client.messages[1]))
