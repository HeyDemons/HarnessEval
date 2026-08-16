from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmark_platform.bridges.terminal_episode import _completed, _workspace_path


class TerminalBridgeTests(unittest.TestCase):
    def test_workspace_path_rejects_escape(self) -> None:
        self.assertEqual(_workspace_path("regex.txt"), "/app/regex.txt")
        self.assertEqual(_workspace_path("/app/sub/file.txt"), "/app/sub/file.txt")
        with self.assertRaises(ValueError):
            _workspace_path("/tests/test.sh")
        with self.assertRaises(ValueError):
            _workspace_path("/app/../tests/test.sh")
        self.assertEqual(_workspace_path("src/file.py", "/testbed"), "/testbed/src/file.py")
        with self.assertRaises(ValueError):
            _workspace_path("/app/file.py", "/testbed")

    def test_command_failure_is_not_reported_as_success(self) -> None:
        completed = type("Completed", (), {"returncode": 7, "stdout": "full", "stderr": "error"})()
        with patch("subprocess.run", return_value=completed):
            result = _completed(["docker", "exec", "container", "false"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["detail"]["returncode"], 7)
        self.assertEqual(result["detail"]["stdout"], "full")


if __name__ == "__main__":
    unittest.main()
