from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_platform.catalog import Catalog
from benchmark_platform.cli import build_parser
from benchmark_platform.engine import (
    Platform,
    docker_add_host_flags,
    docker_host_gateway_flags,
    docker_socket_source,
    terminal_agent_command,
)
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

    def test_implementation_identity_records_git_and_content_state(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
        identity = platform.implementation_identity()
        self.assertRegex(identity["harnesseval_worktree_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(identity["harnesseval_git_sha"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(identity["harnesseval_git_dirty"], bool)
        self.assertEqual(identity, platform.implementation_identity())

    def test_runtime_proxy_has_explicit_inherit_direct_and_url_modes(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("benchmark_platform.engine.local_proxy_url", return_value=None),
        ):
            automatic = platform._egress_env("bridge")
        automatic_env = {
            automatic[index + 1].split("=", 1)[0]: automatic[index + 1].split("=", 1)[1]
            for index in range(0, len(automatic), 2)
        }
        self.assertEqual(automatic_env["HTTPS_PROXY"], "")
        self.assertEqual(automatic_env["NO_PROXY"], "*")

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "benchmark_platform.engine.local_proxy_url",
                return_value="http://host.docker.internal:7890",
            ),
        ):
            automatic_proxy = platform._egress_env("bridge")
        self.assertIn("HTTP_PROXY=http://host.docker.internal:7890", automatic_proxy)

        with patch.dict(os.environ, {"BENCHMARK_RUN_PROXY": "inherit"}, clear=True):
            self.assertEqual(platform._egress_env("bridge"), [])
        with patch.dict(os.environ, {"BENCHMARK_RUN_PROXY": "direct"}, clear=True):
            direct = platform._egress_env("bridge")
        direct_env = {
            direct[index + 1].split("=", 1)[0]: direct[index + 1].split("=", 1)[1]
            for index in range(0, len(direct), 2)
        }
        self.assertEqual(direct_env["HTTPS_PROXY"], "")
        self.assertEqual(direct_env["NO_PROXY"], "*")

        with patch.dict(
            os.environ,
            {
                "BENCHMARK_RUN_PROXY": "http://127.0.0.1:7890",
                "BENCHMARK_RUN_NO_PROXY": "localhost,service.internal",
            },
            clear=True,
        ):
            proxied = platform._egress_env("bridge")
        proxied_env = {
            proxied[index + 1].split("=", 1)[0]: proxied[index + 1].split("=", 1)[1]
            for index in range(0, len(proxied), 2)
        }
        self.assertEqual(proxied_env["HTTP_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(proxied_env["NO_PROXY"], "localhost,service.internal")
        self.assertEqual(platform._egress_env("none"), [])

        with patch.dict(os.environ, {"BENCHMARK_RUN_PROXY": "not-a-url"}, clear=True):
            with self.assertRaises(ValueError):
                platform._egress_env("bridge")

    def test_build_proxy_defaults_to_runtime_auto_policy(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("benchmark_platform.engine.local_proxy_url", return_value=None),
        ):
            direct = platform._build_egress_args()
        direct_values = [direct[index + 1] for index in range(0, len(direct), 2)]
        self.assertIn("HTTP_PROXY=", direct_values)
        self.assertIn("ALL_PROXY=", direct_values)
        self.assertIn("NO_PROXY=*", direct_values)

        with patch.dict(
            os.environ,
            {"BENCHMARK_RUN_PROXY": "http://127.0.0.1:7890"},
            clear=True,
        ):
            mirrored = platform._build_egress_args()
        mirrored_values = [mirrored[index + 1] for index in range(0, len(mirrored), 2)]
        self.assertIn("HTTPS_PROXY=http://host.docker.internal:7890", mirrored_values)

        with patch.dict(os.environ, {"BENCHMARK_BUILD_PROXY": "inherit"}, clear=True):
            self.assertEqual(platform._build_egress_args(), [])
    def test_runtime_extra_hosts_are_explicit_validated_and_network_scoped(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BENCHMARK_DOCKER_ADD_HOSTS": (
                    "ai.centos.hk:202.160.129.37,api.example.test=192.0.2.8 "
                    "ai.centos.hk:202.160.129.37"
                )
            },
            clear=True,
        ):
            self.assertEqual(
                docker_add_host_flags("bridge"),
                [
                    "--add-host",
                    "ai.centos.hk:202.160.129.37",
                    "--add-host",
                    "api.example.test:192.0.2.8",
                ],
            )
            self.assertEqual(docker_add_host_flags("none"), [])

        for invalid in (
            "missing-address",
            "bad_host:192.0.2.1",
            "example.test:not-an-ip",
            "example.test:192.0.2.1:443",
        ):
            with patch.dict(
                os.environ,
                {"BENCHMARK_DOCKER_ADD_HOSTS": invalid},
                clear=True,
            ):
                with self.assertRaises(ValueError, msg=invalid):
                    docker_add_host_flags("bridge")

    def test_host_local_proxy_gets_linux_docker_gateway_mapping(self) -> None:
        self.assertEqual(
            docker_host_gateway_flags(
                "bridge",
                ["https://api.example.test/v1", "http://host.docker.internal:7890"],
            ),
            ["--add-host", "host.docker.internal:host-gateway"],
        )
        self.assertEqual(
            docker_host_gateway_flags("bridge", ["https://api.example.test/v1"]),
            [],
        )
        self.assertEqual(
            docker_host_gateway_flags("none", ["http://host.docker.internal:7890"]),
            [],
        )

    def test_pre_pull_uses_local_base_unless_refresh_is_explicit(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
        adapter = {"pre_pull": ["example/base:fixed"]}
        with (
            patch.object(platform, "image_exists", return_value=True),
            patch("benchmark_platform.engine.subprocess.run") as run,
        ):
            self.assertIsNone(platform._ensure_base_images(adapter, pull=False))
            run.assert_not_called()

        with (
            patch.object(platform, "image_exists", return_value=True),
            patch("benchmark_platform.engine.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            self.assertIsNone(platform._ensure_base_images(adapter, pull=True))
            run.assert_called_once_with(["docker", "pull", "example/base:fixed"], check=False)

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

    def test_generic_terminal_run_resolves_the_requested_task_case(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
        benchmark = SimpleNamespace(adapter={}, id="terminal-bench-2")
        metadata = {
            "environment": {},
            "verifier": {"environment_mode": "separate"},
        }
        store = SimpleNamespace(case_id="prove-plus-comm")
        with (
            patch.object(
                platform, "_terminal_metadata", return_value=(Path("/task"), metadata)
            ) as resolved,
            patch.object(
                platform, "_record_blocked", return_value={"status": "blocked"}
            ),
        ):
            platform._run_terminal_task(
                store, benchmark, smoke=False, command_override=["true"]
            )
        resolved.assert_called_once_with(benchmark, "prove-plus-comm")

        with (
            patch.object(
                platform, "_terminal_metadata", return_value=(Path("/task"), metadata)
            ) as resolved,
            patch.object(
                platform, "_record_blocked", return_value={"status": "blocked"}
            ),
        ):
            platform._run_terminal_task(
                store, benchmark, smoke=True, command_override=None
            )
        resolved.assert_called_once_with(benchmark, None)

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
        self.assertTrue(all(row["runnable"] for row in configured if row["baseline"] != "lats"))
        lats = next(row for row in configured if row["baseline"] == "lats")
        self.assertFalse(lats["runnable"])
        self.assertEqual(lats["baseline_requirement"], "branch_snapshot_or_all_tools_read_only")
        lats_trajectory = next(
            row
            for row in rows
            if row["baseline"] == "lats" and row["benchmark"] == "trajectory-bench"
        )
        self.assertTrue(lats_trajectory["runnable"])
        lats_gaia = next(
            row
            for row in rows
            if row["baseline"] == "lats" and row["benchmark"] == "gaia"
        )
        self.assertFalse(lats_gaia["runnable"])
        self.assertEqual(lats_gaia["baseline_requirement"], "branch_snapshot_or_all_tools_read_only")

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
