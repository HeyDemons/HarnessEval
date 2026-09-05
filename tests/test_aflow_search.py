import json
from pathlib import Path
import tempfile
import unittest

from benchmark_platform.harnesses.aflow import INITIAL_GRAPH, validate_artifact
from benchmark_platform.harnesses.aflow_search import optimize, selection_probabilities
from test_aflow_upstream import Client


SPLIT = {"benchmark": "synthetic", "optimization_case_ids": ["train"], "evaluation_case_ids": ["heldout"]}


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def test_official_score_mixture(self):
        self.assertEqual(selection_probabilities([.5, .5]), [.5, .5])
        high, low = selection_probabilities([1, 0])
        self.assertAlmostEqual(high, .85, places=7)
        self.assertAlmostEqual(low, .15, places=7)

    async def test_evaluates_repeatedly_records_parent_and_freezes_best_not_last(self):
        scores = iter([.2, .4, .9, .7, .1, .3])
        seen = []

        async def evaluate(artifact):
            seen.append(artifact)
            return {"score": next(scores), "feedback": "optimization-only diagnostic"}

        replies = [f"<graph>{INITIAL_GRAPH}\n# candidate-{n}</graph><prompt></prompt><modification>change-{n}</modification>" for n in (1, 2)]
        client = Client(replies)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            artifact = await optimize(client, evaluate, SPLIT, output, rounds=2, validation_rounds=2)
            validate_artifact(artifact, benchmark="synthetic", case_id="heldout")
            self.assertEqual(artifact["provenance"]["selected_round"], 2)
            self.assertEqual(artifact["provenance"]["validation_score"], .8)
            history = json.loads((output / "history.json").read_text())
            self.assertEqual(len(history), 3)
            self.assertIn(history[-1]["parent"], [1, 2])
        self.assertEqual(len(seen), 6)
        self.assertNotIn("heldout", str(client.messages))
        self.assertNotIn("validation_score", str(seen))

    async def test_split_overlap_fails_before_evaluation_or_model(self):
        async def evaluate(artifact):
            self.fail("overlapping split reached evaluator")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "overlap"):
                await optimize(Client([]), evaluate, {**SPLIT, "evaluation_case_ids": ["train"]}, Path(directory) / "search")

    async def test_provider_failure_is_not_scored_as_zero(self):
        async def evaluate(artifact):
            raise RuntimeError("provider unavailable")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "search"
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await optimize(Client([]), evaluate, SPLIT, target)
            self.assertFalse((target / "frozen.json").exists())
