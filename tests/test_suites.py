from __future__ import annotations

import unittest
from collections import Counter
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from benchmark_platform.catalog import Catalog
from benchmark_platform.cli import build_parser
from benchmark_platform.cli import _main as cli_main
from benchmark_platform.suites import SUITE_MODES, SuiteCatalog


ROOT = Path(__file__).resolve().parents[1]


# The BFCL cohort measured before the 2026-09-18 widening to 156.  Committed here
# because the generator that produced it needs the unpinned .sources checkout, and
# the retention claim has to be checkable from the repo alone.
HISTORICAL_BFCL_65 = (
    'irrelevance_130',
    'irrelevance_221',
    'irrelevance_134',
    'irrelevance_39',
    'irrelevance_100',
    'live_irrelevance_166-21-2',
    'live_irrelevance_322-76-20',
    'live_irrelevance_14-2-2',
    'live_irrelevance_780-289-0',
    'live_irrelevance_793-302-0',
    'live_multiple_426-141-15',
    'live_multiple_1002-232-1',
    'live_multiple_290-129-5',
    'live_multiple_460-145-11',
    'live_multiple_837-178-12',
    'live_parallel_14-10-0',
    'live_parallel_13-9-0',
    'live_parallel_6-3-0',
    'live_parallel_2-0-2',
    'live_parallel_3-0-3',
    'live_parallel_multiple_23-20-0',
    'live_parallel_multiple_4-3-0',
    'live_parallel_multiple_1-1-0',
    'live_parallel_multiple_13-11-0',
    'live_parallel_multiple_0-0-0',
    'live_relevance_11-11-0',
    'live_relevance_6-6-0',
    'live_relevance_10-10-0',
    'live_relevance_7-7-0',
    'live_relevance_9-9-0',
    'live_simple_0-0-0',
    'live_simple_49-21-1',
    'live_simple_58-27-0',
    'live_simple_202-116-10',
    'live_simple_178-103-1',
    'multiple_28',
    'multiple_51',
    'multiple_174',
    'multiple_191',
    'multiple_120',
    'parallel_109',
    'parallel_56',
    'parallel_196',
    'parallel_103',
    'parallel_34',
    'parallel_multiple_2',
    'parallel_multiple_43',
    'parallel_multiple_5',
    'parallel_multiple_109',
    'parallel_multiple_108',
    'simple_java_53',
    'simple_java_6',
    'simple_java_9',
    'simple_java_91',
    'simple_java_22',
    'simple_javascript_48',
    'simple_javascript_21',
    'simple_javascript_40',
    'simple_javascript_32',
    'simple_javascript_33',
    'simple_python_228',
    'simple_python_181',
    'simple_python_65',
    'simple_python_179',
    'simple_python_138',
)


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = Catalog(ROOT / "catalog" / "benchmarks.json", ROOT, ROOT.parent)
        cls.suites = SuiteCatalog(ROOT / "catalog" / "suites.json", ROOT, cls.catalog.ids())

    def test_modes_cover_the_complete_benchmark_catalog(self) -> None:
        self.assertEqual(self.suites.modes(), SUITE_MODES)
        for mode in SUITE_MODES:
            self.assertEqual(set(self.suites.ids(mode)), set(self.catalog.ids()))

    def test_light_manifests_are_frozen_and_outcome_independent(self) -> None:
        for benchmark_id in ("gaia", "gdpval", "trajectory-bench", "vitabench", "tau2", "bfcl", "terminal-bench-2", "automationbench"):
            suite = self.suites.get(benchmark_id, "light")
            self.assertEqual(suite["status"], "ready")
            self.assertEqual(suite["declared_count"], len(suite["cases"]))
            self.assertEqual(len(suite["cases"]), len({case["id"] for case in suite["cases"]}))
            policy = suite["selection_policy"]
            self.assertTrue(policy["frozen"])
            self.assertFalse(policy["model_outcomes_used"])
            if benchmark_id != "bfcl":
                self.assertFalse(policy.get("historical_runs_used", policy.get("baseline_runs_used")))
            self.assertEqual(len(suite["manifest_sha256"]), 64)

    def test_bfcl_widening_kept_every_historical_case(self) -> None:
        """The 156-case cohort declares historical_runs_used; this is why that is allowed.

        The flag exists to catch a cohort steered by how runs turned out.  Here it records
        only that the 65 already-measured ids were retained wholesale when the per-category
        count went 5 -> 12, which is what lets their rows be reused.  No outcome entered the
        choice -- model_outcomes_used stays false -- so membership is still independent of
        which cases any baseline passed.
        """

        suite = self.suites.get("bfcl", "light")
        ids = {case["id"] for case in suite["cases"]}
        self.assertLessEqual(set(HISTORICAL_BFCL_65), ids)
        policy = suite["selection_policy"]
        self.assertTrue(policy["historical_runs_used"])
        self.assertFalse(policy["model_outcomes_used"])

    def test_gaia_requested_level_mix_and_scoreable_denominator(self) -> None:
        suite = self.suites.get("gaia", "light")
        self.assertEqual(suite["declared_count"], 56)
        self.assertEqual(Counter(case["level"] for case in suite["cases"]), {1: 10, 2: 20, 3: 26})
        self.assertEqual(suite["locally_scoreable_count"], 56)
        self.assertEqual(
            Counter(case["scoreability"] for case in suite["cases"]),
            {"local_official": 56},
        )

    def test_gdpval_is_three_cases_per_sector(self) -> None:
        suite = self.suites.get("gdpval", "light")
        self.assertEqual(suite["declared_count"], 27)
        sector_counts = Counter(case["sector"] for case in suite["cases"])
        self.assertEqual(len(sector_counts), 9)
        self.assertEqual(set(sector_counts.values()), {3})

    def test_vitabench_and_tau_domain_balance(self) -> None:
        vita = self.suites.get("vitabench", "light")
        self.assertEqual(vita["declared_count"], 60)
        self.assertEqual(set(Counter(case["domain"] for case in vita["cases"]).values()), {15})
        tau = self.suites.get("tau2", "light")
        self.assertEqual(tau["declared_count"], 60)
        self.assertEqual(Counter(case["domain"] for case in tau["cases"]), {"airline": 20, "retail": 20, "telecom": 20})

    def test_trajectory_inventory_is_executable_and_balanced(self) -> None:
        suite = self.suites.get("trajectory-bench", "light")
        self.assertEqual(suite["declared_count"], 100)
        self.assertEqual(set(Counter(case["domain"] for case in suite["cases"]).values()), {10})
        self.assertEqual(
            Counter(case["sample_stratum"] for case in suite["cases"]),
            {"parallel_hard": 25, "parallel_simple": 25, "sequential_hard": 25, "sequential_simple": 25},
        )
        self.assertTrue(all(":" in case["id"] and case["id"].split(":", 1)[0].endswith(".json") for case in suite["cases"]))
        self.assertTrue(all(case["live_status"] == "all_strict_success" for case in suite["cases"]))
        self.assertFalse(suite["selection_policy"]["model_outcomes_used"])
        self.assertTrue(suite["selection_policy"]["endpoint_probe_used"])

    def test_bfcl_uses_real_task_categories_not_auxiliary_index(self) -> None:
        suite = self.suites.get("bfcl", "light")
        categories = Counter(case["category"] for case in suite["cases"])
        self.assertEqual(suite["declared_count"], 156)
        self.assertEqual(len(categories), 13)
        self.assertEqual(set(categories.values()), {12})
        self.assertNotIn("format_sensitivity", categories)
        self.assertFalse(any(category.startswith("multi_turn") for category in categories))
        self.assertNotIn("memory", categories)
        self.assertNotIn("web_search", categories)
        self.assertTrue(all(isinstance(case["format_sensitive"], bool) for case in suite["cases"]))

    def test_terminal_selection_uses_public_short_task_metadata(self) -> None:
        suite = self.suites.get("terminal-bench-2", "light")
        self.assertEqual(suite["declared_count"], 31)
        self.assertEqual(len(suite["cases"]), suite["declared_count"])
        self.assertTrue(all(case["expert_time_estimate_min"] <= 30 for case in suite["cases"]))
        self.assertTrue(all(case["agent_timeout_sec"] > 0 and case["verifier_timeout_sec"] > 0 for case in suite["cases"]))
        # The four removals are host capability and model-download exclusions, so they must
        # stay declared rather than shrinking the suite silently.
        excluded = suite["selection_policy"]["excluded_after_selection"]
        self.assertEqual(excluded["previous_declared_count"], 34)
        self.assertEqual(len(excluded["reason_by_case"]), 3)
        # A case put back must say why, so a reinstatement cannot look like the
        # original selection and hide that an obstacle was measured, not assumed.
        self.assertIn("hf-model-inference", excluded["reinstated"])
        self.assertFalse({case["id"] for case in suite["cases"]} & set(excluded["reason_by_case"]))
        self.assertFalse(suite["selection_policy"]["model_outcomes_used"])

    def test_automationbench_has_six_formal_cases_per_domain(self) -> None:
        suite = self.suites.get("automationbench", "light")
        self.assertEqual(suite["declared_count"], 36)
        self.assertEqual(set(Counter(case["domain"] for case in suite["cases"]).values()), {6})
        self.assertNotIn("simple", {case["domain"] for case in suite["cases"]})

    def test_trajectory_full_and_swe_modes_are_explicitly_held(self) -> None:
        trajectory = self.suites.get("trajectory-bench", "full")
        self.assertEqual(trajectory["status"], "held")
        self.assertTrue(trajectory["reason"])
        for mode in SUITE_MODES:
            suite = self.suites.get("swe-bench-verified", mode)
            self.assertEqual(suite["status"], "held")
            self.assertEqual(suite["cases"], [])
            self.assertTrue(suite["reason"])

    def test_full_mode_delegates_enumeration_to_benchmark(self) -> None:
        for benchmark_id in ("gaia", "gdpval", "vitabench", "tau2", "bfcl", "terminal-bench-2"):
            suite = self.suites.get(benchmark_id, "full")
            self.assertEqual(suite["selection"], "official_full")
            self.assertIsNone(suite["declared_count"])
            self.assertEqual(suite["cases"], [])

    def test_suite_cli_parses_mode_and_ids_only(self) -> None:
        args = build_parser().parse_args(["suite", "gaia", "--mode", "light", "--ids-only"])
        self.assertEqual(args.action, "suite")
        self.assertEqual(args.benchmarks, ["gaia"])
        self.assertEqual(args.mode, "light")
        self.assertTrue(args.ids_only)

    def test_ids_only_rejects_unmaterialized_full_suite(self) -> None:
        with patch("sys.argv", ["harnesseval", "suite", "gaia", "--mode", "full", "--ids-only"]):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli_main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
