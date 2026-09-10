import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_platform.harnesses.aflow import validate_artifact, validate_runtime_artifact
from benchmark_platform.harnesses.aflow_official import INITIAL_GRAPH as OFFICIAL_INITIAL_GRAPH
from benchmark_platform.harnesses.aflow_tools import INITIAL_GRAPH, INITIAL_PROMPT
from benchmark_platform.harnesses.aflow_search import convergence, optimize, selection_probabilities
from test_aflow_upstream import Client


SPLIT = {"benchmark": "synthetic", "optimization_case_ids": ["train"], "evaluation_case_ids": ["heldout"]}


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_search_is_official_qa_not_the_benchmark_adapter(self):
        scores = iter([.2, .8])
        async def evaluate(artifact):
            validate_artifact(artifact, allow_initialization=True)
            return {"score": next(scores)}
        reply = (
            f"<graph>{OFFICIAL_INITIAL_GRAPH}\n# official candidate</graph>"
            "<prompt></prompt><modification>add official candidate marker</modification>"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await optimize(
                Client([reply]), evaluate, SPLIT, Path(directory) / "official",
                rounds=1, validation_rounds=1,
            )
        self.assertEqual(result["operator_profile"], "qa")
        self.assertEqual(result["provenance"]["operator_profile"], "qa")
        self.assertFalse(result["provenance"]["benchmark_adapter"])
        self.assertEqual(result["provenance"]["selected_round"], 2)

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
        replies = [f"<graph>{INITIAL_GRAPH}\n# {n}</graph><prompt>{INITIAL_PROMPT}</prompt><modification>change-{n}</modification>"
                   for n in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            for enabled, expected in ((True, 5), (False, 8)):
                client = Client(replies)
                result = await optimize(client, evaluate, SPLIT, Path(directory) / str(enabled),
                                        rounds=8, validation_rounds=1, check_convergence=enabled,
                                        operator_profile="benchmark-tools")
                self.assertEqual(len(client.messages), expected)
                self.assertEqual(result["provenance"]["completed_rounds"], expected)
                self.assertEqual(result["provenance"]["stop_reason"], "converged" if enabled else "round_budget")

    async def test_repeated_modification_regenerates_without_consuming_round(self):
        scores = iter([.9, .1, .2])
        async def evaluate(artifact):
            return {"score": next(scores)}
        replies = [f"<graph>{INITIAL_GRAPH}\n# {n}</graph><prompt>{INITIAL_PROMPT}</prompt><modification>change-{n}</modification>"
                   for n in (1, 1, 2)]
        client = Client(replies)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            result = await optimize(client, evaluate, SPLIT, output, rounds=2, validation_rounds=1,
                                    sample=1, operator_profile="benchmark-tools")
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

        replies = [f"<graph>{INITIAL_GRAPH}\n# candidate-{n}</graph><prompt>{INITIAL_PROMPT}</prompt><modification>change-{n}</modification>" for n in (1, 2)]
        client = Client(replies)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            artifact = await optimize(client, evaluate, SPLIT, output, rounds=2, validation_rounds=2,
                                      operator_profile="benchmark-tools")
            validate_runtime_artifact(artifact, benchmark="synthetic", case_id="heldout")
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
                await optimize(Client([]), evaluate, {**SPLIT, "evaluation_case_ids": ["train"]},
                               Path(directory) / "search", operator_profile="benchmark-tools")

    async def test_provider_failure_is_not_scored_as_zero(self):
        async def evaluate(artifact):
            raise RuntimeError("provider unavailable")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "search"
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await optimize(Client([]), evaluate, SPLIT, target, operator_profile="benchmark-tools")
            self.assertFalse((target / "frozen.json").exists())

    async def test_resume_retries_pending_evaluation_without_regenerating(self):
        calls = 0
        async def interrupted(artifact):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"score": .2}
            raise RuntimeError("provider unavailable")
        reply = (
            f"<graph>{INITIAL_GRAPH}\n# pending</graph>"
            f"<prompt>{INITIAL_PROMPT}</prompt><modification>pending-change</modification>"
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "search"
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await optimize(Client([reply]), interrupted, SPLIT, target,
                               rounds=1, validation_rounds=1, operator_profile="benchmark-tools")
            self.assertEqual(len(json.loads((target / "history.json").read_text())), 1)
            self.assertTrue((target / "expansion-2.txt").is_file())

            resumed_client = Client([])
            result = await optimize(
                resumed_client,
                lambda artifact: asyncio.sleep(0, result={"score": .9}),
                SPLIT,
                target,
                rounds=1,
                validation_rounds=1,
                resume=True,
                operator_profile="benchmark-tools",
            )
            history = json.loads((target / "history.json").read_text())
            self.assertEqual([row["round"] for row in history], [1, 2])
            self.assertTrue(history[1]["resumed_pending_evaluation"])
            self.assertEqual(result["provenance"]["selected_round"], 2)
            self.assertEqual(result["provenance"]["resume_count"], 1)
            self.assertEqual(resumed_client.messages, [])
