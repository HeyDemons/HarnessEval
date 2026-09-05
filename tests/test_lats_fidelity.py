import unittest

from benchmark_platform.harnesses.lats import _Node, _SearchMemory, _value
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment
from test_aflow_upstream import Trace, Client


class LatsCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_replay_preserves_terminal_success_and_vote_result(self):
        trace = Trace()
        client = Client(['{"score":1,"success":true,"feedback":"valid"}',
                         '{"score":0.8,"success":true,"feedback":"supported"}',
                         '{"score":0.2,"success":false,"feedback":"dissent"}'])
        ctx = RunContext("lats", "synthetic", client, ToolEnvironment([], trace), trace, {})
        memory = _SearchMemory()
        node = _Node(None, action={"final": "42"}, terminal=True, answer="42")
        first = await _value(ctx, node, 3, memory, 0)
        cached = await _value(ctx, node, 3, memory, 0)
        self.assertTrue(first[1])
        self.assertEqual(cached, first)
        self.assertEqual(ctx.llm_calls, 3)
