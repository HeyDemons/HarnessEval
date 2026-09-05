"""DyLAN's open-ended consensus uses sacrebleu 2.3.1, not exact strings."""
import unittest

from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.paper_methods import _dylan_most_frequent


class Trace:
    def __init__(self):
        self.events = []

    async def emit(self, event, **data):
        self.events.append({"event": event, **data})


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)

    async def complete(self, messages, **kwargs):
        return Completion(next(self.replies), 1, 1, 0, 0, {})


class DyLANFidelityTests(unittest.IsolatedAsyncioTestCase):
    async def test_case_variants_reach_consensus_after_three_calls(self):
        answer = "The answer is Paris because it is the capital city of France."
        trace = Trace()
        ctx = RunContext("dylan", "synthetic task", Client([answer, answer.lower(), answer]),
                         ToolEnvironment([], trace), trace, {"dylan_team_optimization": False})
        self.assertEqual(await run_profile(ctx), answer)
        self.assertEqual(ctx.llm_calls, 3)
        self.assertTrue(any(x["event"] == "dylan_early_stop" for x in trace.events))

    def test_high_bleu_without_case_equivalence_forms_majority(self):
        a = ("The answer is Paris because it is the capital city of France and the location asked for in this question. "
             "The supplied information identifies the country and asks specifically for the name of its capital city.")
        b = a.replace("this question", "that question")
        self.assertEqual(_dylan_most_frequent(["Berlin", a, b, a]), (a, 3))

    def test_tie_keeps_first_candidate_and_compound_answers_stay_complete(self):
        self.assertEqual(_dylan_most_frequent(["Berlin", "Paris"]), ("Berlin", 1))
        self.assertEqual(_dylan_most_frequent(["7, 9", "7, 9", "1, 3"]), ("7, 9", 2))

    async def test_final_vote_also_uses_bleu(self):
        a = "The answer is Paris because it is the capital city of France."
        trace = Trace()
        ctx = RunContext("dylan", "synthetic task", Client(["Berlin", a, a.lower(), "London"]),
                         ToolEnvironment([], trace), trace, {"dylan_rounds": 1, "dylan_team_optimization": False})
        self.assertEqual(await run_profile(ctx), a)
        self.assertEqual(ctx.llm_calls, 4)
        self.assertFalse(any(x["event"] == "dylan_early_stop" for x in trace.events))
