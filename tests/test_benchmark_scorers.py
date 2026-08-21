from __future__ import annotations

import unittest

from drivers.gdpval_score import rubric_items
from drivers.trajectory_score import grade, tool_name


class BenchmarkScorerTests(unittest.TestCase):
    def test_trajectory_primary_score_requires_ordered_tool_trajectory(self) -> None:
        case = {"tools": [{"tool name": "Weather.Now"}, {"tool name": "Maps.Route"}]}
        gold = {"tool_list": case["tools"], "final_answer": "done"}
        calls = [{"name": tool_name("Weather.Now")}, {"name": tool_name("Maps.Route")}]
        verdict = grade(case, gold, calls, "done")
        self.assertEqual(verdict["score"], 1.0)
        self.assertTrue(verdict["answer_exact"])
        reversed_verdict = grade(case, gold, list(reversed(calls)), "done")
        self.assertEqual(reversed_verdict["score"], 0.0)
        self.assertTrue(reversed_verdict["tool_set_exact"])

    def test_gdpval_rubric_shapes_are_normalized(self) -> None:
        self.assertEqual(len(rubric_items('[{"criterion":"a"}]')), 1)
        self.assertEqual(len(rubric_items({"criteria": ["a", "b"]})), 2)


if __name__ == "__main__":
    unittest.main()
