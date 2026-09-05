import unittest

from benchmark_platform.harnesses.lats import _Node, _SearchMemory, _value, _expand, _candidate_key
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from test_aflow_upstream import Trace, Client


class LatsCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_proposals_keep_their_own_response_ids(self):
        trace = Trace()
        client = Client(['{"thought":"first","tool":"read","arguments":{"id":1}}',
                         '{"thought":"second","tool":"read","arguments":{"id":2}}'])
        async def read(args):
            return args
        env = ToolEnvironment([ToolSpec("read", "read", {"type":"object"}, (), read_only=True)], trace, {"read": read})
        ctx = RunContext("lats", "synthetic", client, env, trace, {})
        nodes = await _expand(ctx, _Node(None), 2, 1, _SearchMemory(), 5, 3, 7, 1)
        records = [n.call_records[0] for n in nodes]
        self.assertEqual([r["assistant_response_id"] for r in records], [1, 2])
        await env.commit_isolated_calls(records)
        self.assertEqual([r["assistant_response_id"] for r in env.calls], [1, 2])
        self.assertEqual(_candidate_key({"tool":"read", "assistant_response_id":1}),
                         _candidate_key({"tool":"read", "assistant_response_id":2}))

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
