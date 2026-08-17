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
        for benchmark_id in ("gaia", "gdpval", "vitabench", "tau2", "bfcl", "terminal-bench-2"):
            suite = self.suites.get(benchmark_id, "light")
            self.assertEqual(suite["status"], "ready")
            self.assertEqual(suite["declared_count"], len(suite["cases"]))
            self.assertEqual(len(suite["cases"]), len({case["id"] for case in suite["cases"]}))
            policy = suite["selection_policy"]
            self.assertTrue(policy["frozen"])
            self.assertFalse(policy["model_outcomes_used"])
            self.assertFalse(policy["historical_runs_used"])
            self.assertEqual(len(suite["manifest_sha256"]), 64)

    def test_gaia_requested_level_mix_and_scoreable_denominator(self) -> None:
        suite = self.suites.get("gaia", "light")
        self.assertEqual(suite["declared_count"], 60)
        self.assertEqual(Counter(case["level"] for case in suite["cases"]), {1: 10, 2: 20, 3: 30})
        self.assertEqual(suite["locally_scoreable_count"], 56)
        self.assertEqual(
            Counter(case["scoreability"] for case in suite["cases"]),
            {"local_official": 56, "official_submission_only": 4},
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
        self.assertEqual(tau["declared_count"], 30)
        self.assertEqual(Counter(case["domain"] for case in tau["cases"]), {"airline": 10, "retail": 10, "telecom": 10})

    def test_bfcl_uses_real_task_categories_not_auxiliary_index(self) -> None:
        suite = self.suites.get("bfcl", "light")
        categories = Counter(case["category"] for case in suite["cases"])
        self.assertEqual(suite["declared_count"], 95)
        self.assertEqual(len(categories), 19)
        self.assertEqual(set(categories.values()), {5})
        self.assertNotIn("format_sensitivity", categories)
        self.assertTrue(all(isinstance(case["format_sensitive"], bool) for case in suite["cases"]))

    def test_terminal_selection_is_frozen_but_runner_limit_is_disclosed(self) -> None:
        suite = self.suites.get("terminal-bench-2", "light")
        self.assertEqual(suite["declared_count"], 20)
        self.assertIn("adapter", suite["runner_note"])
        self.assertEqual(self.suites.get("terminal-bench-2", "full")["runner_status"], "adapter_expansion_required")

    def test_trajectory_and_swe_are_explicitly_held(self) -> None:
        for benchmark_id in ("trajectory-bench", "swe-bench-verified"):
            for mode in SUITE_MODES:
                suite = self.suites.get(benchmark_id, mode)
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
