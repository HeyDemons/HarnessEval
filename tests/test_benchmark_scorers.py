from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drivers.gdpval_score import rubric_items
from drivers.portable_smoke import gdpval
from drivers.trajectory_score import grade, tool_name


class BenchmarkScorerTests(unittest.TestCase):
    def test_gdpval_snapshot_probe_checks_assets_and_frozen_sha(self) -> None:
        class Frame:
            columns = [
                "task_id",
                "prompt",
                "reference_files",
                "deliverable_files",
                "rubric_json",
            ]

            def __len__(self) -> int:
                return 1

            def to_dict(self, *, orient: str):
                if orient != "records":
                    raise AssertionError(f"unexpected orient: {orient}")
                return [row]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "data.parquet"
            parquet.write_bytes(b"frozen parquet")
            reference = root / "reference_files" / "hash" / "source.xlsx"
            deliverable = root / "deliverable_files" / "gold" / "answer.pdf"
            reference.parent.mkdir(parents=True)
            deliverable.parent.mkdir(parents=True)
            reference.write_bytes(b"reference")
            deliverable.write_bytes(b"gold")
            row = {
                "task_id": "case",
                "prompt": "create",
                "reference_files": [str(reference.relative_to(root))],
                "deliverable_files": [str(deliverable.relative_to(root))],
                "rubric_json": '[{"score":1,"criterion":"complete"}]',
            }
            suite = root / "suite.json"
            suite.write_text(
                json.dumps(
                    {
                        "source": {
                            "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest()
                        },
                        "cases": [{"id": "case", "reference_count": 1}],
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "drivers.portable_smoke.parquet_probe",
                return_value=(parquet, Frame()),
            ):
                self.assertEqual(gdpval(root, suite)["scores"]["dataset_integrity"], 1.0)
                reference.unlink()
                self.assertEqual(gdpval(root, suite)["scores"]["dataset_integrity"], 0.0)

    def test_trajectory_parallel_exact_ignores_order(self) -> None:
        case = {"tools": [{"tool name": "Weather.Now"}, {"tool name": "Maps.Route"}]}
        gold = {
            "tool_list": case["tools"],
            "final_answer": "done",
            "trajectory_type": "parallel",
        }
        calls = [{"name": tool_name("Weather.Now")}, {"name": tool_name("Maps.Route")}]
        verdict = grade(case, gold, calls, "done")
        self.assertEqual(verdict["score"], 1.0)
        self.assertTrue(verdict["answer_exact"])
        reversed_verdict = grade(case, gold, list(reversed(calls)), "done")
        self.assertEqual(reversed_verdict["score"], 1.0)
        self.assertTrue(reversed_verdict["tool_set_exact"])

    def test_trajectory_sequential_exact_preserves_order(self) -> None:
        case = {"tools": [{"tool name": "Weather.Now"}, {"tool name": "Maps.Route"}]}
        gold = {"tool_list": case["tools"], "trajectory_type": "sequential"}
        calls = [{"name": tool_name("Maps.Route")}, {"name": tool_name("Weather.Now")}]
        self.assertEqual(grade(case, gold, calls, "")["score"], 0.0)

    def test_trajectory_inclusion_uses_original_ground_truth_length(self) -> None:
        weather = {"tool name": "Weather.Now"}
        route = {"tool name": "Maps.Route"}
        case = {"tools": [weather, route]}
        gold = {
            "tool_list": [weather, weather, route],
            "trajectory_type": "parallel",
        }
        verdict = grade(case, gold, [{"name": tool_name("Weather.Now")}], "")
        self.assertAlmostEqual(verdict["tool_inclusion"], 1 / 3)

    def test_trajectory_usage_compares_normalized_arguments(self) -> None:
        definition = {
            "tool name": "Weather.Now",
            "required parameters": [{"name": "days", "value": "42"}],
            "optional parameters": [],
        }
        case = {"tools": [definition]}
        gold = {"tool_list": [definition], "trajectory_type": "parallel"}
        correct = grade(
            case,
            gold,
            [{"name": tool_name("Weather.Now"), "arguments": {"days": 42}}],
            "",
        )
        wrong = grade(
            case,
            gold,
            [{"name": tool_name("Weather.Now"), "arguments": {"days": 41}}],
            "",
        )
        self.assertEqual(correct["tool_traj_usage"], [True])
        self.assertEqual(wrong["tool_traj_usage"], [False])

    def test_gdpval_rubric_shapes_are_normalized(self) -> None:
        self.assertEqual(len(rubric_items('[{"criterion":"a"}]')), 1)
        self.assertEqual(len(rubric_items({"criteria": ["a", "b"]})), 2)
        with self.assertRaisesRegex(ValueError, "missing"):
            rubric_items(None)
        with self.assertRaisesRegex(ValueError, "empty"):
            rubric_items([])


if __name__ == "__main__":
    unittest.main()
