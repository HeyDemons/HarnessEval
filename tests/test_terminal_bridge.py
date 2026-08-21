from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmark_platform.bridges.terminal_episode import (
    _completed,
    _docker_exec,
    _tool_specs,
    _workspace_path,
)


class TerminalBridgeTests(unittest.TestCase):
    def test_workspace_path_rejects_escape(self) -> None:
        self.assertEqual(_workspace_path("regex.txt"), "/app/regex.txt")
        self.assertEqual(_workspace_path("/app/sub/file.txt"), "/app/sub/file.txt")
        with self.assertRaises(ValueError):
            _workspace_path("/tests/test.sh")
        with self.assertRaises(ValueError):
            _workspace_path("/app/../tests/test.sh")
        self.assertEqual(
            _workspace_path("src/file.py", "/testbed"), "/testbed/src/file.py"
        )
        with self.assertRaises(ValueError):
            _workspace_path("/app/file.py", "/testbed")

    def test_terminal_bench_root_access_keeps_native_relative_workdir(self) -> None:
        # nginx is configured outside /app, prove-plus-comm starts in /workspace, and
        # sanitize-git-repo starts in /app/dclm. Absolute machine paths must be reachable
        # without changing where relative paths resolve.
        self.assertEqual(
            _workspace_path("/etc/nginx/nginx.conf", "/", "/app"),
            "/etc/nginx/nginx.conf",
        )
        self.assertEqual(
            _workspace_path("proof.v", "/", "/workspace"), "/workspace/proof.v"
        )
        self.assertEqual(_workspace_path(".git", "/", "/app/dclm"), "/app/dclm/.git")

    def test_run_command_uses_native_workdir_and_a_cancellable_wrapper(self) -> None:
        command, token = _docker_exec(
            "task-container",
            ["bash", "-lc", "make"],
            timeout_sec=900,
            cwd="/workspace",
        )
        self.assertEqual(
            command[:5], ["docker", "exec", "-w", "/workspace", "task-container"]
        )
        self.assertIn("timeout --signal", command[7])
        self.assertIn(token, command)

    def test_run_command_has_no_model_selectable_timeout(self) -> None:
        run_command = next(tool for tool in _tool_specs() if tool.name == "run_command")
        self.assertNotIn("timeout_sec", run_command.parameters["properties"])

    def test_command_failure_is_not_reported_as_success(self) -> None:
        completed = type(
            "Completed", (), {"returncode": 7, "stdout": "full", "stderr": "error"}
        )()
        with patch("subprocess.run", return_value=completed):
            result = _completed(["docker", "exec", "container", "false"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["detail"]["returncode"], 7)
        self.assertEqual(result["detail"]["stdout"], "full")

    def test_oversized_task_output_is_middle_truncated(self) -> None:
        """One `cat` of a file the task ships reached 1,760,345 characters on gcode-to-text and
        6,900,728 on mcmc-sampling-stan; the next request came back context_length_exceeded and
        every arm that read the file lost the case. Follow inspect_ai: cap at its
        max_tool_output default, keep half the characters from each end the way truncate_str
        does, and say so with its START/END wrapper instead of a spliced-in marker."""
        stdout = "A" * 40_000 + "B" * 40_000
        completed = type(
            "Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""}
        )()
        with patch("subprocess.run", return_value=completed):
            result = _completed(["docker", "exec", "container", "cat", "/app/big"])
        self.assertTrue(result["ok"])
        clipped = result["result"]["stdout"]
        self.assertLess(len(clipped), len(stdout))
        self.assertIn("<START_TOOL_OUTPUT>", clipped)
        self.assertIn("<END_TOOL_OUTPUT>", clipped)
        # Middle truncation keeps both ends, so the error at the bottom of a build log survives.
        self.assertIn("A" * 100, clipped)
        self.assertIn("B" * 100, clipped)

    def test_output_within_the_limit_is_untouched(self) -> None:
        completed = type(
            "Completed", (), {"returncode": 0, "stdout": "small", "stderr": ""}
        )()
        with patch("subprocess.run", return_value=completed):
            result = _completed(["docker", "exec", "container", "cat", "/app/small"])
        self.assertEqual(result["result"]["stdout"], "small")


if __name__ == "__main__":
    unittest.main()
