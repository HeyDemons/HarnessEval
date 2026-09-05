import random
import unittest

from benchmark_platform.harnesses.dylan import Node, backward, edge_weights, parse_ranks, run_dylan
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment
from test_dylan_fidelity import Trace, Client


class RecordingClient(Client):
    def __init__(self, replies):
        super().__init__(replies)
        self.messages = []

    async def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return await super().complete(messages, **kwargs)


def context(replies, **policy):
    trace = Trace()
    return RunContext("dylan-query-local", "Task", RecordingClient(replies), ToolEnvironment([], trace), trace, policy)


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    def test_backward_uses_reply_as_bleu_hypothesis(self):
        answer = " ".join(f"word{i}" for i in range(10))
        longer = answer + " extra"
        # BLEU(answer, longer)=90.48 but BLEU(longer, answer)=89.32.
        # Only the exact-answer node supports this terminal result upstream.
        layers = [[Node(0, answer), Node(1, longer)]]
        self.assertEqual(backward(layers, answer, 2), [1.0, 0.0])

    def test_backward_propagates_edge_weights_and_sums_by_agent(self):
        layers = [[Node(0, "x"), Node(1, "y")],
                  [Node(0, "oak", {0: .75, 1: .25}), Node(1, "pine", {0: .1, 1: .9})]]
        self.assertEqual(backward(layers, "oak", 2), [1.75, .25])
        self.assertEqual(sum(n.importance for n in layers[0]), 1)
        self.assertEqual(sum(n.importance for n in layers[1]), 1)

    def test_weights_and_bad_ranking(self):
        self.assertEqual(edge_weights("[[5, 1]]", 2), [5 / 6, 1 / 6])
        self.assertEqual(edge_weights("[[0, -2]]", 2), [.5, .5])
        self.assertEqual(edge_weights("[[5]]", 2), [.5, .5])
        for value in ('{"top":null}', '[1,1]', '[0,9]', 'invalid'):
            ranks, fallback = parse_ranks(value, 4, random.Random(0))
            self.assertTrue(fallback)
            self.assertEqual(len(set(ranks)), 2)

    async def test_explicit_query_local_variant_solves_without_trial_messages(self):
        ctx = context(["trial-answer"] * 3 + ["solve-answer"] * 2)
        state = random.getstate()
        self.assertEqual(await run_dylan(ctx), "solve-answer")
        self.assertEqual(random.getstate(), state)
        selected = next(e for e in ctx.trace.events if e["event"] == "dylan_team_selected")
        self.assertEqual(len(selected["active_agents"]), 2)
        self.assertEqual(ctx.llm_calls, 5)
        self.assertNotIn("trial-answer", str(ctx.client.messages[3:]))

    async def test_four_rounds_keep_selected_agent_ids_and_all_predecessors(self):
        ctx = context(["red", "blue", "green", "yellow", "cat", "dog", "bird", "fish",
                       "[1,1]", "oak", "pine", "sun", "moon"],
                      dylan_rounds=4, dylan_team_optimization=False)
        await run_dylan(ctx)
        event = next(e for e in ctx.trace.events if e["event"] == "dylan_activation")
        selected = set(event["active_agents"])
        self.assertEqual(len(selected), 2)
        self.assertTrue(event["fallback"])
        nodes = [e for e in ctx.trace.events if e["event"] == "dylan_node"]
        for round_id in (3, 4):
            self.assertEqual({n["agent"] for n in nodes if n["round"] == round_id}, selected)
        for node in nodes:
            if node["round"] == 3:
                self.assertEqual(len(node["predecessor_weights"]), 4)
            if node["round"] == 4:
                self.assertEqual(len(node["predecessor_weights"]), 2)

    async def test_invalid_config_fails_before_model_calls(self):
        for policy in ({"dylan_agents": 0}, {"dylan_rounds": 0}, {"dylan_team_size": 5},
                       {"dylan_roles": ["missing"] * 4}, {"dylan_team_optimization": "false"}):
            ctx = context([], **policy)
            with self.assertRaises(ValueError):
                await run_dylan(ctx)
            self.assertEqual(ctx.llm_calls, 0)
