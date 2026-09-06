"""The decision-making network is the policy, not the actor (2310.02170v2, B.3).

Agents deliberate over the task plus the concatenated observations of previous
actions, one consensus action is executed outside the network, and consistency
is exact identity rather than the BLEU threshold used for open-ended answers.
"""
import json
import unittest

from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.dylan import DM_DEFAULT_ROLES, DM_ROLE_PROMPTS, canonical_action
from benchmark_platform.harnesses.dylan import run_dylan_dm as run_profile
from test_dylan_fidelity import Client, Trace


SCHEMA = {"type": "object", "properties": {"key": {"type": "string"}}}


class RecordingClient(Client):
    def __init__(self, replies):
        super().__init__(replies)
        self.messages = []

    async def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return await super().complete(messages, **kwargs)


async def _lookup(arguments):
    return {"value": f"seen {arguments.get('key')}"}


def context(replies, **policy):
    trace = Trace()
    environment = ToolEnvironment(
        [ToolSpec("lookup", "lookup a key", SCHEMA, ("true",), parallel=True, read_only=True)],
        trace,
        {"lookup": _lookup},
    )
    client = RecordingClient(replies)
    return RunContext("dylan-dm", "Task", client, environment, trace, policy), trace, environment, client


def act(**arguments):
    return json.dumps({"tool": "lookup", "arguments": arguments})


def _bleu(candidate: str, other: str) -> float:
    from sacrebleu import sentence_bleu
    pair = [canonical_action(reply, ["lookup"]) for reply in (candidate, other)]
    return sentence_bleu(pair[0], [pair[1]], lowercase=True).score


class DecisionMakingTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_action_per_deliberation_and_the_observation_feeds_the_next(self):
        ctx, trace, environment, client = context(
            [act(key="alpha")] * 3 + ['{"final":"alpha is seen alpha"}'] * 3
        )
        self.assertEqual(await run_profile(ctx), "alpha is seen alpha")
        # One environment call per consensus action: no node holds a tool, so
        # four parallel agents cannot interleave writes on a stateful world.
        self.assertEqual(len(environment.calls), 1)
        self.assertEqual(ctx.llm_calls, 6)
        self.assertEqual([x["action"] for x in trace.events if x["event"] == "dylan_dm_action"],
                         [{"tool": "lookup", "arguments": {"key": "alpha"}},
                          {"final": "alpha is seen alpha"}])

    async def test_the_next_decision_sees_the_previous_action_and_observation(self):
        ctx, _, _, client = context([act(key="alpha")] * 3 + ['{"final":"done"}'] * 3)
        await run_profile(ctx)
        first, last = client.messages[0][-1]["content"], client.messages[-1][-1]["content"]
        self.assertNotIn("Observation:", first)
        self.assertIn('Action: {"arguments":{"key":"alpha"},"tool":"lookup"}', last)
        self.assertIn("seen alpha", last)

    async def test_consistency_is_identity_not_bleu(self):
        # sacrebleu scores these canonical actions at 92.79, above the 0.9
        # open-ended threshold, so BLEU consensus would merge them and stop at
        # the third agent. As decisions they are not consistent, so no 2/3
        # majority forms until the fourth agent repeats the first action.
        near = "the quick brown fox jumps over the lazy dog near the old stone bridge"
        first, other = act(key=f"{near} alpha"), act(key=f"{near} beta")
        self.assertGreaterEqual(_bleu(first, other), 90)
        ctx, _, environment, _ = context([first, other, first, first] + ['{"final":"ok"}'] * 3)
        self.assertEqual(await run_profile(ctx), "ok")
        self.assertEqual(ctx.llm_calls, 7)  # no early stop at the third agent
        self.assertEqual(environment.calls[0]["arguments"], {"key": f"{near} alpha"})

    async def test_invalid_outputs_are_skipped_rather_than_voting_for_each_other(self):
        ctx, trace, environment, _ = context(
            ["not json", "not json either", act(key="a"), act(key="a")] + ['{"final":"ok"}'] * 3
        )
        self.assertEqual(await run_profile(ctx), "ok")
        # Two unparseable replies are identical strings but are not a majority.
        self.assertEqual(environment.calls, [])
        self.assertFalse([x for x in trace.events if x["event"] == "dylan_dm_invalid_layer"])

    async def test_a_wholly_invalid_layer_is_recorded_and_the_query_changes(self):
        ctx, trace, _, client = context(["nope", '{"final":"ok"}'], dylan_agents=1, dylan_rounds=1)
        self.assertEqual(await run_profile(ctx), "ok")
        self.assertEqual([x["step"] for x in trace.events if x["event"] == "dylan_dm_invalid_layer"], [1])
        self.assertIn("Protocol error", client.messages[-1][-1]["content"])

    async def test_an_unsupported_role_fails_before_any_model_call(self):
        ctx, _, _, _ = context([], dylan_roles=["Assistant"] * 4)
        with self.assertRaises(ValueError):
            await run_profile(ctx)
        self.assertEqual(ctx.llm_calls, 0)

    def test_the_eight_paper_candidates_are_registered(self):
        self.assertEqual(len(DM_ROLE_PROMPTS), 8)
        self.assertTrue(set(DM_DEFAULT_ROLES) <= set(DM_ROLE_PROMPTS))
        self.assertEqual(len(DM_DEFAULT_ROLES), 4)  # paper sets N=4 after team optimization


if __name__ == "__main__":
    unittest.main()
