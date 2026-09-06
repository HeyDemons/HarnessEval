import json
import random
import unittest

from benchmark_platform.budgets import ModelBudgetExceeded
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.dylan_policy import (
    PAPER_TEAMS, PolicyNode, deliberate, importance, parse_action, visible_state_kind,
)
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import PROFILES, get_profile
from test_dylan_network import RecordingClient
from test_dylan_fidelity import Trace


def action(value):
    return json.dumps({"action": {"tool": "lookup", "arguments": {"value": value}}, "ratings": []})


def context(replies, policy=None):
    trace = Trace()
    async def lookup(args):
        return {"value": "actual observation " + args["value"]}
    env = ToolEnvironment([ToolSpec("lookup", "lookup", {"type": "object"}, ())], trace, {"lookup": lookup})
    ctx = RunContext("dylan", "PUBLIC_TASK", RecordingClient(replies), env, trace, policy or {},
                     task_messages=[{"role": "system", "content": "PUBLIC_POLICY"}])
    return ctx, trace


class DyLANPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_only_one_public_profile_and_no_artifact_gate(self):
        self.assertEqual([p.id for p in PROFILES if p.id.startswith("dylan")], ["dylan"])
        self.assertEqual(get_profile("dylan").tool_contract, "dynamic")
        for name in ("dylan-dm", "dylan-inference", "dylan-query-local"):
            with self.assertRaises((KeyError, ValueError)):
                get_profile(name)

    async def test_four_proposals_commit_one_action_then_observe_and_finish(self):
        final = '{"action":{"final":"done"},"ratings":[]}'
        ctx, trace = context([action("alpha")] * 4 + [final] * 4)
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual(ctx.llm_calls, 8)
        self.assertEqual(len(ctx.environment.calls), 1)
        self.assertTrue(all(messages[0]["content"] == "PUBLIC_POLICY" for messages in ctx.client.messages))
        self.assertNotIn("actual observation alpha", json.dumps(ctx.client.messages[0]))
        self.assertIn("actual observation alpha", json.dumps(ctx.client.messages[-1]))
        teams = [e for e in trace.events if e["event"] == "dylan_team_selected"]
        self.assertEqual([e["state_kind"] for e in teams], ["searching", "item"])
        self.assertEqual(tuple(teams[0]["roles"]), PAPER_TEAMS["searching"])

    async def test_rank_layer_copies_messages_instead_of_generating_again(self):
        replies = [action(v) for v in "abcdwxyz"] + ["[1,2]", action("winner"), action("winner")]
        ctx, trace = context(replies)
        chosen = await deliberate(ctx, "PUBLIC_TASK", PAPER_TEAMS["searching"], random.Random(0), 1, finalizing=False)
        self.assertEqual(json.loads(chosen)["arguments"], {"value": "winner"})
        self.assertEqual(ctx.llm_calls, 11)  # 4 + 4 + ranker + 2
        self.assertEqual(ctx.environment.calls, [])
        reformation = next(e for e in trace.events if e["event"] == "dylan_reformation")
        self.assertTrue(reformation["message_copy"])
        self.assertEqual(len(set(reformation["active_agents"])), 2)
        scores = next(e["scores"] for e in trace.events if e["event"] == "dylan_importance")
        self.assertAlmostEqual(sum(scores.values()), 4.0)

    def test_action_identity_does_not_merge_different_arguments(self):
        one, _ = parse_action(action("alpha"), ["lookup"])
        other, _ = parse_action(action("beta"), ["lookup"])
        self.assertNotEqual(one, other)
        for raw in ('{"status":"in_progress"}', '{"tool":"missing","arguments":{}}',
                    '{"final":"done","tool":"lookup","arguments":{}}'):
            self.assertIsNone(parse_action(raw, ["lookup"])[0])
        self.assertIsNone(parse_action(action("alpha"), ["lookup"], finalizing=True)[0])

    def test_router_uses_only_visible_committed_observations(self):
        self.assertEqual(visible_state_kind([]), "searching")
        self.assertEqual(visible_state_kind([{"name": "api_search", "result": {"ok": True, "result": "docs"}}]), "exploring")
        self.assertEqual(visible_state_kind([{"name": "api_fetch", "result": {"ok": True, "result": '{"files":[]}'}}]), "exploring")
        self.assertEqual(visible_state_kind([{"name": "api_fetch", "result": {"ok": True, "result": '{"value":2}'}}]), "item")
        self.assertEqual(visible_state_kind([{"name": "send_message_to_user", "result": {"ok": True}}]), "searching")

    def test_importance_propagates_exact_action_supporters(self):
        layers = [[PolicyNode("a", "a", "A"), PolicyNode("b", "b", "B")],
                  [PolicyNode("a", "next", "A", {"a": .75, "b": .25}),
                   PolicyNode("b", "next", "B", {"a": .1, "b": .9})]]
        self.assertEqual(importance(layers, "A"), {"a": 1.75, "b": .25})

    async def test_invalid_layer_is_not_a_consensus(self):
        ctx, trace = context(['{"status":"in_progress"}'] * 4 + ['{"final":"done"}'] * 4)
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual(ctx.llm_calls, 8)
        self.assertEqual(ctx.environment.calls, [])
        self.assertEqual([e["layer"] for e in trace.events if e["event"] == "dylan_early_stop"], [2])

    async def test_budget_never_allows_last_response_to_commit_a_tool(self):
        ctx, _ = context([action("alpha")] * 4, {"model_response_limit": 4, "finalize_on_loop_limit": True})
        with self.assertRaises(ModelBudgetExceeded):
            await run_profile(ctx)
        self.assertEqual(ctx.llm_calls, 4)
        self.assertEqual(ctx.environment.calls, [])

    async def test_old_frozen_configuration_is_rejected_before_model_calls(self):
        ctx, _ = context([], {"dylan_team_artifact": {"old": "artifact"}})
        with self.assertRaisesRegex(ValueError, "legacy overrides"):
            await run_profile(ctx)
        self.assertEqual(ctx.llm_calls, 0)

    async def test_penultimate_response_can_still_commit_a_decision(self):
        ctx, _ = context([action("alpha")] * 4 + ['{"final":"done"}'],
                         {"model_response_limit": 5, "finalize_on_loop_limit": True})
        with self.assertRaises(ModelBudgetExceeded):
            await run_profile(ctx)
        self.assertEqual(ctx.llm_calls, 5)
        self.assertEqual(len(ctx.environment.calls), 1)
