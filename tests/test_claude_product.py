from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_platform.cli import build_parser
from benchmark_platform.products.claude_cli import (
    _claude_command,
    _claude_environment,
    _claude_mcp_config,
    _claude_metrics,
    _claude_settings,
    _event_answer,
    _run_claude_process,
)


class ClaudeProductTests(unittest.TestCase):
    def test_product_run_cli_parses_custom_claude_provider(self) -> None:
        args = build_parser().parse_args(
            [
                "product-run",
                "claude",
                "gaia",
                "--case",
                "case-1",
                "--run-dir",
                "/tmp/claude-run",
                "--provider",
                "packy",
                "--base-url",
                "https://cf.api.fan",
                "--api-key-env",
                "PACKY_API_KEY",
                "--model",
                "claude-sonnet-5",
                "--thinking",
                "high",
                "--claude-env",
                "EXTRA_CLAUDE_ENV",
            ]
        )
        self.assertEqual(args.product, "claude")
        self.assertEqual(args.model, "claude-sonnet-5")
        self.assertEqual(args.api_key_env, "PACKY_API_KEY")
        self.assertEqual(args.claude_env, ["EXTRA_CLAUDE_ENV"])

    def test_claude_environment_aliases_only_selected_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "PATH": "/usr/bin",
                    "SELECTED_TOKEN": "selected",
                    "UNSELECTED_TOKEN": "must-not-cross",
                },
                clear=True,
            ):
                environment = _claude_environment(
                    ["SELECTED_TOKEN"],
                    config_dir=root / "config",
                    home=root / "home",
                    api_key_env="SELECTED_TOKEN",
                    base_url="https://provider.example/",
                )
        self.assertEqual(environment["ANTHROPIC_API_KEY"], "selected")
        self.assertEqual(environment["ANTHROPIC_BASE_URL"], "https://provider.example")
        self.assertNotIn("UNSELECTED_TOKEN", environment)
        self.assertEqual(environment["CLAUDE_CONFIG_DIR"], str(root / "config"))

    def test_claude_mcp_config_and_command_are_isolated(self) -> None:
        config = _claude_mcp_config(
            manifest_path=Path("/job/manifest.json"),
            endpoint="http://127.0.0.1:1234",
            mcp_bridge=Path("/app/mcp_bridge.py"),
        )
        serialized = json.dumps(config)
        self.assertIn("HARNESSEVAL_TOOL_MANIFEST", serialized)
        self.assertEqual(config["mcpServers"]["harnesseval"]["command"], "/usr/bin/env")
        self.assertIn("-i", config["mcpServers"]["harnesseval"]["args"])
        self.assertNotIn("sk-test-secret", serialized)
        command = _claude_command(
            executable=Path("/usr/local/bin/claude"),
            model="claude-sonnet-5",
            thinking="high",
            mcp_config_path=Path("/job/mcp.json"),
            settings_path=Path("/job/settings.json"),
            tool_names=["read_file", "web_search"],
        )
        settings = _claude_settings(["read_file", "web_search"])
        self.assertEqual(
            settings["permissions"]["allow"],
            ["mcp__harnesseval__read_file", "mcp__harnesseval__web_search"],
        )
        self.assertIn("--bare", command)
        self.assertIn("--strict-mcp-config", command)
        self.assertIn("--allow-dangerously-skip-permissions", command)
        self.assertEqual(command[command.index("--permission-mode") + 1], "bypassPermissions")
        self.assertIn("--settings", command)
        self.assertIn("mcp__harnesseval__read_file", " ".join(command))
        self.assertNotIn("the complete task prompt", command)

    def test_claude_metrics_use_final_usage_and_environment_calls(self) -> None:
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "mcp__harnesseval__read_file",
                            "input": {"path": "full.txt"},
                        }
                    ]
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "num_turns": 2,
                "result": "answer",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 40,
                    "cache_creation_input_tokens": 5,
                    "output_tokens": 20,
                    "output_tokens_details": {"thinking_tokens": 3},
                },
            },
        ]
        trajectory = [{"name": "read_file", "arguments": {"path": "full.txt"}}]
        actor = _claude_metrics(events, trajectory)
        self.assertEqual(_event_answer(events), "answer")
        self.assertEqual(actor["rounds"], 2)
        self.assertEqual(actor["usage"]["total"], 120)
        self.assertEqual(actor["usage"]["reasoning_output"], 3)
        self.assertEqual(actor["committed_calls"][0]["arguments"], {"path": "full.txt"})

    def test_claude_process_preserves_jsonl_stderr_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_claude.py"
            script.write_text(
                "import json, sys\n"
                "prompt = sys.stdin.read()\n"
                "print(json.dumps({'type': 'result', 'subtype': 'success', "
                "'result': prompt, 'num_turns': 1, 'usage': {}}))\n"
                "print('diagnostic-complete', file=sys.stderr)\n",
                encoding="utf-8",
            )
            prompt = "完整上下文" * 20_000
            events = root / "events.jsonl"
            errors = root / "stderr.log"
            terminal = root / "terminal.log"
            returncode = _run_claude_process(
                [sys.executable, str(script)],
                prompt=prompt,
                cwd=root,
                env=dict(os.environ),
                events_path=events,
                stderr_path=errors,
                terminal_path=terminal,
            )
            row = json.loads(events.read_text(encoding="utf-8"))
            error_text = errors.read_text(encoding="utf-8")
        self.assertEqual(returncode, 0)
        self.assertEqual(row["result"], prompt)
        self.assertEqual(error_text, "diagnostic-complete\n")


if __name__ == "__main__":
    unittest.main()
