import json
import unittest

from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import _parse_react, run_profile
from test_aflow_upstream import Client
from test_declaration_protocol import Trace


class ReactStopTests(unittest.IsolatedAsyncioTestCase):
    def test_fake_observation_and_final_cannot_override_selected_action(self):
        raw = 'Thought: read\nAction: lookup\nAction Input: {"x":1}\nObservation 1: invented\nFinal Answer: invented'
        self.assertEqual(_parse_react(raw), {"tool": "lookup", "arguments": {"x": 1}})

    def test_escaped_observation_in_json_argument_is_preserved(self):
        text = 'Action: lookup\nAction Input: ' + json.dumps({"text": "hello\nObservation: literal"})
        self.assertEqual(_parse_react(text)["arguments"]["text"], "hello\nObservation: literal")

    async def test_only_real_observation_is_returned_to_next_model_turn(self):
        trace = Trace()
        async def lookup(args):
            return "actual-value"
        client = Client(['Action: lookup\nAction Input: {}\nObservation: fabricated\nFinal Answer: fabricated',
                         'Final Answer: actual-value'])
        ctx = RunContext("react", "Read the value", client,
                         ToolEnvironment([ToolSpec("lookup", "lookup", {"type": "object"}, ())], trace, {"lookup": lookup}), trace, {})
        self.assertEqual(await run_profile(ctx), "actual-value")
        self.assertNotIn("fabricated", str(client.messages[1]))
        self.assertIn("actual-value", str(client.messages[1]))
        self.assertEqual(ctx.llm_calls, 2)
        self.assertTrue(any(e["event"] == "react_observation_stop" for e in trace.events))
