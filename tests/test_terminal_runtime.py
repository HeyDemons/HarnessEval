from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_platform.engine import Platform
from benchmark_platform.terminal_runtime import (
    docker_resource_flags,
    run_shared_verifier,
    terminal_agent_prompt,
    terminal_outer_timeout_sec,
    terminal_task_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class TerminalRuntimeTests(unittest.TestCase):
    def metadata(self, **environment):
        return {
            "agent": {"timeout_sec": 900},
            "verifier": {"timeout_sec": 1800, "env": {}},
            "environment": {
                "allow_internet": True,
                "cpus": 4,
                "memory_mb": 8192,
                "storage_mb": 10240,
                **environment,
            },
        }

    def test_agent_prompt_includes_official_root_without_sudo_guidance(self) -> None:
        prompt = terminal_agent_prompt("Complete the task.")
        self.assertIn("running as root inside a Docker container", prompt)
        self.assertIn("Do not use sudo", prompt)
        self.assertTrue(prompt.endswith("Complete the task."))

    def test_task_limits_and_outer_envelope_come_from_task_toml(self) -> None:
        settings = terminal_task_settings(self.metadata())
        self.assertEqual(settings.agent_timeout_sec, 900)
        self.assertEqual(settings.verifier_timeout_sec, 1800)
        self.assertEqual(
            docker_resource_flags(settings),
            ["--cpus", "4", "--memory", "8192m"],
        )
        self.assertEqual(terminal_outer_timeout_sec(self.metadata()), 2820)

    def test_create_preserves_oci_workdir_and_applies_cpu_memory(self) -> None:
        platform = Platform(ROOT, ROOT.parent, ROOT / "catalog" / "benchmarks.json")
        benchmark = SimpleNamespace(adapter={"platform": "linux/amd64"})
        with patch.object(platform, "_egress_env", return_value=[]):
            command = platform._terminal_create_command(
                benchmark=benchmark,
                metadata=self.metadata(),
                image="task:image",
                container="task-container",
                labels=["test=1"],
            )
        self.assertIn("--cpus", command)
        self.assertIn("--memory", command)
        self.assertNotIn("-w", command)
        self.assertEqual(
            command[-4:], ["task:image", "sh", "-lc", "while :; do sleep 3600; done"]
        )

        with patch.object(platform, "_egress_env", return_value=[]):
            overridden = platform._terminal_create_command(
                benchmark=benchmark,
                metadata=self.metadata(workdir="/workspace"),
                image="task:image",
                container="task-container",
                labels=[],
            )
        self.assertEqual(overridden[overridden.index("-w") + 1], "/workspace")

    def test_zero_reward_is_a_completed_shared_measurement(self) -> None:
        """Covers the nginx/R-package lifecycle and FEAL's writable /tests requirement.

        The only grading process is docker exec in the original task container; tests are
        copied after the agent instead of mounted read-only, so services and global installs
        survive and native extensions can be built beneath /tests.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "task"
            (task_dir / "tests").mkdir(parents=True)
            (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
            logs = root / "logs"
            commands = []

            def fake_run(command, *, timeout_sec=None):
                commands.append(command)
                if command[:3] == ["docker", "cp", "task-container:/logs/verifier/."]:
                    (logs / "reward.txt").write_text("0\n")
                return completed(command)

            with patch(
                "benchmark_platform.terminal_runtime.run_captured", side_effect=fake_run
            ):
                result = run_shared_verifier(
                    docker=lambda *args: ["docker", *args],
                    container="task-container",
                    task_dir=task_dir,
                    logs_dir=logs,
                    timeout_sec=30,
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scores"], {"reward": 0.0})
        upload = next(
            command
            for command in commands
            if command[1:2] == ["cp"] and "/tests/." in command[2]
        )
        self.assertEqual(upload[-1], "task-container:/tests")
        self.assertFalse(any(":ro" in item for command in commands for item in command))
        verifier = next(command for command in commands if "/tests/test.sh" in command)
        self.assertEqual(verifier[0:2], ["docker", "exec"])
        self.assertIn("task-container", verifier)
        self.assertFalse(any(command[1:2] == ["run"] for command in commands))
        self.assertFalse(
            any("-w" in command for command in commands if "/tests/test.sh" in command)
        )

    def test_transient_verifier_network_error_retries_three_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "task"
            (task_dir / "tests").mkdir(parents=True)
            (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
            logs = root / "logs"
            attempts = 0

            def fake_run(command, *, timeout_sec=None):
                nonlocal attempts
                if "/tests/test.sh" in command:
                    attempts += 1
                    if attempts < 3:
                        return completed(command, 1, stderr="apt: 502 Bad Gateway")
                if command[:3] == ["docker", "cp", "task-container:/logs/verifier/."]:
                    (logs / "reward.txt").write_text("1\n")
                return completed(command)

            with (
                patch(
                    "benchmark_platform.terminal_runtime.run_captured",
                    side_effect=fake_run,
                ),
                patch("benchmark_platform.terminal_runtime.time.sleep"),
            ):
                result = run_shared_verifier(
                    docker=lambda *args: ["docker", *args],
                    container="task-container",
                    task_dir=task_dir,
                    logs_dir=logs,
                    timeout_sec=30,
                )

        self.assertEqual(attempts, 3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scores"]["reward"], 1.0)

    def test_refused_local_task_service_is_a_zero_not_verifier_infrastructure(self) -> None:
        """A task that failed to start localhost is a model outcome, not lost grading."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "task"
            (task_dir / "tests").mkdir(parents=True)
            (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
            logs = root / "logs"
            attempts = 0

            def fake_run(command, *, timeout_sec=None):
                nonlocal attempts
                if "/tests/test.sh" in command:
                    attempts += 1
                    return completed(
                        command,
                        1,
                        stdout=(
                            "test_nginx_running FAILED\n"
                            "HTTPConnection(host='localhost', port=8080): "
                            "ConnectionRefusedError: [Errno 111] Connection refused\n"
                        ),
                    )
                if command[:3] == ["docker", "cp", "task-container:/logs/verifier/."]:
                    (logs / "reward.txt").write_text("0\n")
                return completed(command)

            with patch(
                "benchmark_platform.terminal_runtime.run_captured",
                side_effect=fake_run,
            ):
                result = run_shared_verifier(
                    docker=lambda *args: ["docker", *args],
                    container="task-container",
                    task_dir=task_dir,
                    logs_dir=logs,
                    timeout_sec=30,
                )

        self.assertEqual(attempts, 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scores"], {"reward": 0.0})

    def test_persistent_network_error_is_infrastructure_not_model_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "task"
            (task_dir / "tests").mkdir(parents=True)
            (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
            logs = root / "logs"

            def fake_run(command, *, timeout_sec=None):
                if "/tests/test.sh" in command:
                    return completed(
                        command, 1, stderr="Temporary failure in name resolution"
                    )
                if command[:3] == ["docker", "cp", "task-container:/logs/verifier/."]:
                    # Some test.sh files write zero after a failed install. It is not a valid
                    # answer verdict when the verifier output proves its setup lost network.
                    (logs / "reward.txt").write_text("0\n")
                return completed(command)

            with (
                patch(
                    "benchmark_platform.terminal_runtime.run_captured",
                    side_effect=fake_run,
                ),
                patch("benchmark_platform.terminal_runtime.time.sleep"),
            ):
                result = run_shared_verifier(
                    docker=lambda *args: ["docker", *args],
                    container="task-container",
                    task_dir=task_dir,
                    logs_dir=logs,
                    timeout_sec=30,
                )

        self.assertEqual(result["status"], "infra_failed")
        self.assertEqual(result["termination_reason"], "verifier_network_error")
        self.assertEqual(result["scores"], {})


if __name__ == "__main__":
    unittest.main()
