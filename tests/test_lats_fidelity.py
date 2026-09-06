import unittest
from unittest.mock import AsyncMock

from benchmark_platform.harnesses.lats import _Node, _SearchMemory, _value, _expand, _candidate_key
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from test_aflow_upstream import Trace, Client


class LatsCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_value_cache_is_not_a_cached_assistant_generation(self):
        trace = Trace()
        trace.emit = AsyncMock()
        ctx = RunContext('lats', 'synthetic', Client(['{"score":0.5,"success":false,"feedback":"partial"}']),
                         ToolEnvironment([], trace), trace, {})
        node, memory = _Node(None), _SearchMemory()
        first = await _value(ctx, node, 1, memory, 0.0)
        before = ctx.usage_metrics()
        self.assertEqual(first, await _value(ctx, node, 1, memory, 0.0))
        self.assertEqual(ctx.usage_metrics(), before)
        self.assertEqual(ctx.agent_turns, 1)
        event = next(call for call in trace.emit.call_args_list if call.args[0] == 'lats_value_cache_hit')
        self.assertFalse(event.kwargs['model_generation'])
        self.assertEqual(event.kwargs['agent_turn_increment'], 0)

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
