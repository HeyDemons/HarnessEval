from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from benchmark_platform.catalog import Catalog
from benchmark_platform.cli import build_parser
from benchmark_platform.engine import Platform, docker_socket_source, terminal_agent_command
from benchmark_platform.scorers.gaia import question_score
from benchmark_platform.store import CaseStore
from benchmark_platform.util import atomic_json, slug


ROOT = Path(__file__).resolve().parents[1]


class PlatformTests(unittest.TestCase):
    def test_catalog_is_unique_and_has_no_inspect_adapter(self) -> None:
        catalog = Catalog(ROOT / "catalog" / "benchmarks.json", ROOT, ROOT.parent)
        self.assertEqual(len(catalog.ids()), len(set(catalog.ids())))
        self.assertNotIn("inspect", {item.adapter["kind"] for item in catalog})

    def test_atomic_json_and_case_resume_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "value.json"
            atomic_json(path, {"text": "完整内容", "items": list(range(1000))})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["items"][-1], 999)
            store = CaseStore(Path(directory), "bench", "case/id")
            _, attempt = store.next_attempt()
            request = {"started_at": "now"}
            store.start(attempt, request)
            store.finish(attempt, {"status": "completed"})
            self.assertEqual(store.existing()["status"], "completed")

    def test_case_lock_rejects_duplicate_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = CaseStore(Path(directory), "bench", "same-case")
            second = CaseStore(Path(directory), "bench", "same-case")
            with first.lock():
                with self.assertRaises(RuntimeError):
                    with second.lock():
                        pass
            with second.lock():
                pass

    def test_slug_is_stable_and_collision_resistant(self) -> None:
        self.assertEqual(slug("abc-123"), "abc-123")
        self.assertNotEqual(slug("a/b"), slug("a b"))

    def test_run_options_parse_after_benchmark(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run", "gaia", "--case", "case-1", "--run-dir", "/tmp/run", "--no-build"]
        )
        self.assertEqual(args.benchmark, "gaia")
        self.assertEqual(args.case, "case-1")
        self.assertTrue(args.no_build)

    def test_gaia_public_scorer_semantics(self) -> None:
        self.assertTrue(question_score("$1,234", "1234"))
        self.assertTrue(question_score("Soups and Stews", "soups-and-stews"))
        self.assertTrue(question_score("7; 9", "7,9"))
        self.assertFalse(question_score("0.0429", "0.0424"))

    def test_environment_secrets_are_explicit_opt_in(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
        benchmark = platform.catalog.get("gaia")
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "not-serialized"
        try:
            self.assertEqual(platform._allowed_env(benchmark, []), [])
            self.assertEqual(platform._allowed_env(benchmark, ["OPENAI_API_KEY"]), ["OPENAI_API_KEY"])
            with self.assertRaises(ValueError):
                platform._allowed_env(benchmark, ["UNLISTED_SECRET"])
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous

    def test_root_execution_is_limited_to_docker_socket_controller(self) -> None:
        catalog = Catalog(ROOT / "catalog" / "benchmarks.json", ROOT, ROOT.parent)
        root_entries = [item for item in catalog if item.adapter.get("run_as_root")]
        self.assertEqual([item.id for item in root_entries], ["swe-bench-verified"])
        self.assertTrue(root_entries[0].adapter.get("docker_socket"))
        self.assertIn("docker:28-cli", root_entries[0].adapter.get("pre_pull", []))

    def test_colima_uses_daemon_side_socket_path(self) -> None:
        context = {
            "Name": "colima",
            "Endpoints": {"docker": {"Host": "unix:///Users/example/.colima/default/docker.sock"}},
        }
        self.assertEqual(docker_socket_source(context), "/var/run/docker.sock")

    def test_terminal_oracle_keeps_tests_out_of_agent_command(self) -> None:
        command = terminal_agent_command(None, smoke=True)
        self.assertEqual(command, "bash /solution/solve.sh")
        self.assertNotIn("/tests", command or "")
        self.assertEqual(
            terminal_agent_command(["python", "agent.py", "--mode", "test"], smoke=False),
            "python agent.py --mode test",
        )
        self.assertIsNone(terminal_agent_command(None, smoke=False))

    def test_harness_profiles_have_unique_ids(self) -> None:
        from benchmark_platform.harnesses import PROFILES

        ids = [profile.id for profile in PROFILES]
        self.assertEqual(len(ids), len(set(ids)))
        react = next(profile for profile in PROFILES if profile.id == "react")
        self.assertEqual(len(react.revision or ""), 40)
        for profile in PROFILES:
            if profile.revision is not None:
                self.assertEqual(len(profile.revision), 40, profile.id)

    def test_compatibility_matrix_is_complete_and_does_not_overclaim_scores(self) -> None:
        from benchmark_platform.compatibility import compatibility_rows
        from benchmark_platform.harnesses import PROFILES

        catalog = Catalog(ROOT / "catalog" / "benchmarks.json", ROOT, ROOT.parent)
        rows = compatibility_rows(PROFILES, catalog)
        self.assertEqual(len(rows), len(PROFILES) * len(catalog.ids()))
        self.assertFalse(any(row["publishable_score"] for row in rows))
        configured = [row for row in rows if row["benchmark"] == "swe-bench-verified"]
        self.assertTrue(configured)
        self.assertTrue(all(row["runnable"] for row in configured))

    def test_adapter_fingerprint_is_content_based(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
        benchmark = platform.catalog.get("swe-bench-verified")
        fingerprint = platform.adapter_fingerprint(benchmark.adapter)
        self.assertIsNotNone(fingerprint)
        self.assertEqual(len(fingerprint or ""), 64)

    def test_external_adapter_fingerprint_is_checkout_path_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first = parent / "first" / "adapter"
            second = parent / "renamed" / "adapter"
            first.mkdir(parents=True)
            (first / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
            (first / "source.txt").write_text("same content\n", encoding="utf-8")
            (first / "node_modules").mkdir()
            (first / "node_modules" / "local.js").write_text("ignored\n", encoding="utf-8")
            shutil.copytree(first, second)
            platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
            left = platform.adapter_fingerprint(
                {"dockerfile": str(first / "Dockerfile"), "fingerprint_paths": [str(first)]}
            )
            right = platform.adapter_fingerprint(
                {"dockerfile": str(second / "Dockerfile"), "fingerprint_paths": [str(second)]}
            )
            self.assertEqual(left, right)

    def test_catalog_dir_variable_is_relative_to_custom_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "nested" / "catalog.json"
            catalog_path.parent.mkdir()
            catalog_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmarks": [
                            {
                                "id": "portable",
                                "name": "Portable",
                                "source": {},
                                "adapter": {
                                    "kind": "docker-image",
                                    "dockerfile": "${CATALOG_DIR}/Dockerfile",
                                },
                                "scoring": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            catalog = Catalog(catalog_path, ROOT, ROOT.parent)
            self.assertEqual(
                Path(catalog.get("portable").adapter["dockerfile"]),
                catalog_path.parent.resolve() / "Dockerfile",
            )


if __name__ == "__main__":
    unittest.main()
