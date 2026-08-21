from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_platform.cli import build_parser
from benchmark_platform.products.codex_cli import (
    _codex_config,
    _codex_environment,
    _codex_metrics,
    _run_codex_process,
)
from benchmark_platform.products.codex_mcp_bridge import _result


ROOT = Path(__file__).resolve().parents[1]


class CodexProductTests(unittest.TestCase):
    def test_product_run_cli_parses_custom_codex_provider(self) -> None:
        args = build_parser().parse_args(
            [
                "product-run",
                "codex",
                "gaia",
                "--case",
                "case-1",
                "--run-dir",
                "/tmp/codex-run",
                "--provider",
                "packy",
                "--base-url",
                "https://cf.api.fan/v1",
                "--api-key-env",
                "PACKY_API_KEY",
                "--model",
                "gpt-5.6-terra",
                "--thinking",
                "high",
                "--codex-env",
                "EXTRA_CODEX_ENV",
            ]
        )
        self.assertEqual(args.product, "codex")
        self.assertEqual(args.model, "gpt-5.6-terra")
        self.assertEqual(args.api_key_env, "PACKY_API_KEY")
        self.assertEqual(args.codex_env, ["EXTRA_CODEX_ENV"])

    def test_codex_config_isolated_tools_and_never_contains_key_value(self) -> None:
        config = _codex_config(
            model="gpt-5.6-terra",
            provider="packy",
            base_url="https://cf.api.fan/v1",
            api_key_env="PACKY_API_KEY",
            thinking="high",
            manifest_path=Path("/job/manifest.json"),
            endpoint="http://127.0.0.1:1234",
            mcp_bridge=ROOT / "benchmark_platform" / "products" / "codex_mcp_bridge.py",
        )
        self.assertIn('env_key = "PACKY_API_KEY"', config)
        self.assertIn("shell_tool = false", config)
        self.assertIn("multi_agent = false", config)
        self.assertIn("[mcp_servers.harnesseval]", config)
        self.assertNotIn("sk-test-secret", config)

    def test_codex_environment_copies_only_explicit_secret_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "HOME": "/tmp/home",
                    "PATH": "/usr/bin",
                    "SELECTED_TOKEN": "selected",
                    "UNSELECTED_TOKEN": "must-not-cross",
                },
                clear=True,
            ):
                environment = _codex_environment(["SELECTED_TOKEN"], Path(directory))
        self.assertEqual(environment["SELECTED_TOKEN"], "selected")
        self.assertNotIn("UNSELECTED_TOKEN", environment)
        self.assertEqual(environment["CODEX_HOME"], directory)

    def test_mcp_bridge_lists_complete_schema_and_proxies_complete_arguments(self) -> None:
        manifest = {
            "tools": [
                {
                    "name": "web_search",
                    "description": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "read_only": True,
                }
            ]
        }
        listed = _result(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, manifest, "http://bridge"
        )
        self.assertEqual(listed["result"]["tools"][0]["inputSchema"], manifest["tools"][0]["parameters"])
        arguments = {"query": "完整参数" * 10_000}
        with patch(
            "benchmark_platform.products.codex_mcp_bridge._execute",
            return_value=({"ok": True, "result": "完整结果" * 10_000}, False),
        ) as execute:
            called = _result(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "web_search", "arguments": arguments},
                },
                manifest,
                "http://bridge",
            )
        execute.assert_called_once_with("http://bridge", "web_search", arguments)
        self.assertIn("完整结果" * 10_000, called["result"]["content"][0]["text"])

    def test_codex_metrics_preserve_usage_and_environment_calls(self) -> None:
        events = [
            {"type": "turn.started"},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 3,
                },
            },
        ]
        trajectory = [{"name": "read_file", "arguments": {"path": "full.txt"}}]
        actor = _codex_metrics(events, trajectory)
        self.assertEqual(actor["rounds"], 1)
        self.assertEqual(actor["usage"]["total"], 120)
        self.assertEqual(actor["committed_calls"][0]["arguments"], {"path": "full.txt"})

    def test_codex_process_preserves_jsonl_stderr_prompt_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_codex.py"
            script.write_text(
                "import json, sys\n"
                "prompt = sys.stdin.read()\n"
                "print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': prompt}}))\n"
                "print('diagnostic-complete', file=sys.stderr)\n",
                encoding="utf-8",
            )
            prompt = "完整上下文" * 20_000
            events = root / "events.jsonl"
            errors = root / "stderr.log"
            terminal = root / "terminal.log"
            returncode = _run_codex_process(
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
        self.assertEqual(row["item"]["text"], prompt)
        self.assertEqual(error_text, "diagnostic-complete\n")


if __name__ == "__main__":
    unittest.main()
