import json
from pathlib import Path
import tempfile
import unittest

from benchmark_platform.harnesses.aflow import INITIAL_GRAPH, validate_artifact
from benchmark_platform.harnesses.aflow_search import convergence, optimize, selection_probabilities
from test_aflow_upstream import Client


SPLIT = {"benchmark": "synthetic", "optimization_case_ids": ["train"], "evaluation_case_ids": ["heldout"]}


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def test_convergence_counts_only_scored_rounds_and_resets_on_improvement(self):
        self.assertFalse(convergence([{"round": i, "score": .5} for i in range(1, 6)])["converged"])
        rows = [{"round": i, "score": .5} for i in range(1, 7)]
        self.assertEqual(convergence(rows), {"converged": True, "start_round": 2, "final_round": 6})
        rows[3]["score"] = None
        self.assertFalse(convergence(rows)["converged"])
        rows[3]["score"] = .7
        self.assertFalse(convergence(rows)["converged"])

    async def test_convergence_stops_search_and_can_be_disabled(self):
        async def evaluate(artifact):
            return {"score": .5}
        replies = [f"<graph>{INITIAL_GRAPH}\n# {n}</graph><prompt></prompt><modification>change-{n}</modification>"
                   for n in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            for enabled, expected in ((True, 5), (False, 8)):
                client = Client(replies)
                result = await optimize(client, evaluate, SPLIT, Path(directory) / str(enabled),
                                        rounds=8, validation_rounds=1, check_convergence=enabled)
                self.assertEqual(len(client.messages), expected)
                self.assertEqual(result["provenance"]["completed_rounds"], expected)
                self.assertEqual(result["provenance"]["stop_reason"], "converged" if enabled else "round_budget")

    async def test_repeated_modification_regenerates_without_consuming_round(self):
        scores = iter([.9, .1, .2])
        async def evaluate(artifact):
            return {"score": next(scores)}
        replies = [f"<graph>{INITIAL_GRAPH}\n# {n}</graph><prompt></prompt><modification>change-{n}</modification>"
                   for n in (1, 1, 2)]
        client = Client(replies)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            result = await optimize(client, evaluate, SPLIT, output, rounds=2, validation_rounds=1, sample=1)
            history = json.loads((output / "history.json").read_text())
            generations = json.loads((output / "generations.json").read_text())
            self.assertEqual(len(history), 3)
            self.assertEqual([row["round"] for row in generations], [2, 3, 3])
            self.assertEqual(generations[1]["rejection"], "repeated_modification")
            self.assertEqual(result["provenance"]["generation_calls"], 3)
            self.assertTrue((output / "expansion-3-attempt-1.txt").exists())
            self.assertTrue((output / "expansion-3-attempt-2.txt").exists())

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
