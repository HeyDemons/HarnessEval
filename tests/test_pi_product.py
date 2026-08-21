from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import patch

from benchmark_platform.cli import build_parser
from benchmark_platform.products.pi_cli import (
    _actor_metrics,
    _assistant_text,
    _jsonl,
    _pi_command,
    _pi_environment,
    _request_json,
    _run_pi_process,
    _tool_results,
)


ROOT = Path(__file__).resolve().parents[1]


class PiProductTests(unittest.TestCase):
    def test_bridge_startup_disconnect_is_retriable_transport_failure(self) -> None:
        with patch(
            "benchmark_platform.products.pi_cli.urlopen",
            side_effect=RemoteDisconnected("bridge is still starting"),
        ):
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                _request_json("http://127.0.0.1:1/manifest")

    def test_product_run_cli_parses_local_pi_configuration(self) -> None:
        args = build_parser().parse_args(
            [
                "product-run",
                "pi",
                "gaia",
                "--case",
                "case-1",
                "--run-dir",
                "/tmp/pi-run",
                "--provider",
                "deepseek",
                "--model",
                "deepseek-v4-flash",
                "--thinking",
                "high",
                "--pi-env",
                "CUSTOM_PROVIDER_KEY",
                "--no-build",
            ]
        )
        self.assertEqual(args.product, "pi")
        self.assertEqual(args.benchmark, "gaia")
        self.assertEqual(args.model, "deepseek-v4-flash")
        self.assertEqual(args.thinking, "high")
        self.assertEqual(args.pi_env, ["CUSTOM_PROVIDER_KEY"])
        self.assertTrue(args.no_build)

    def test_pi_command_disables_ambient_resources_and_keeps_complete_prompt(self) -> None:
        prompt = "instruction:" + "完整上下文" * 20_000
        command = _pi_command(
            executable=Path("/opt/pi"),
            extension=ROOT / "benchmark_platform" / "products" / "pi_tool_bridge.ts",
            tools=["web_search", "python"],
            prompt=prompt,
            provider="deepseek",
            model="deepseek-v4-flash",
            thinking="high",
        )
        for flag in (
            "--no-session",
            "--offline",
            "--no-context-files",
            "--no-skills",
            "--no-prompt-templates",
            "--no-extensions",
            "--no-builtin-tools",
        ):
            self.assertIn(flag, command)
        self.assertEqual(command[-2:], ["-p", prompt])
        self.assertEqual(command[command.index("--tools") + 1], "web_search,python")

    def test_pi_environment_is_explicit_and_does_not_copy_unselected_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            with patch.dict(
                os.environ,
                {
                    "HOME": "/tmp/pi-home",
                    "PATH": "/usr/bin",
                    "SELECTED_TOKEN": "selected",
                    "UNSELECTED_TOKEN": "must-not-cross",
                },
                clear=True,
            ):
                environment = _pi_environment(
                    ["SELECTED_TOKEN"], manifest, "http://127.0.0.1:1234"
                )
        self.assertEqual(environment["SELECTED_TOKEN"], "selected")
        self.assertNotIn("UNSELECTED_TOKEN", environment)
        self.assertEqual(environment["HARNESSEVAL_TOOL_MANIFEST"], str(manifest))
        self.assertEqual(environment["HARNESSEVAL_TOOL_ENDPOINT"], "http://127.0.0.1:1234")
        self.assertNotIn("PI_SYSTEM_DATE", environment)

    def test_event_parser_aggregates_all_rounds_calls_usage_and_final_text(self) -> None:
        events = [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "toolUse",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "web_search",
                            "arguments": {"query": "complete query"},
                        }
                    ],
                    "usage": {"input": 101, "output": 20, "cacheRead": 3, "totalTokens": 124},
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "content": [{"type": "text", "text": "Final answer: complete result"}],
                    "usage": {"input": 50, "output": 7, "cacheRead": 11, "totalTokens": 68},
                },
            },
        ]
        actor = _actor_metrics(events)
        self.assertEqual(actor["rounds"], 2)
        self.assertEqual(actor["committed_calls"][0]["arguments"]["query"], "complete query")
        self.assertEqual(actor["usage"], {"input": 151, "output": 27, "cache_read": 14, "cache_write": 0, "total": 192})
        self.assertEqual(_assistant_text(events), "Final answer: complete result")

    def test_failed_run_can_recover_complete_environment_trajectory_from_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "tool_trace.jsonl"
            result = {
                "event": "tool_result",
                "name": "web_search",
                "arguments": {"query": "unabridged query"},
                "result": {"ok": True, "result": {"body": "x" * 20_000}},
                "state_version_before": 0,
                "state_version_after": 0,
            }
            trace.write_text(json.dumps(result) + "\n", encoding="utf-8")
            calls, malformed = _tool_results(trace)
        self.assertEqual(malformed, 0)
        self.assertEqual(calls, [{key: result[key] for key in result if key != "event"}])

    def test_pi_process_preserves_jsonl_and_stderr_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = "result-" + "x" * 20_000
            script = root / "fake_pi.py"
            script.write_text(
                "import json, sys\n"
                f"print(json.dumps({{'type': 'message_end', 'message': {{'role': 'assistant', "
                f"'content': [{{'type': 'text', 'text': {payload!r}}}], 'stopReason': 'stop'}}}}))\n"
                "print('diagnostic-complete', file=sys.stderr)\n",
                encoding="utf-8",
            )
            events = root / "events.jsonl"
            errors = root / "stderr.log"
            terminal = root / "terminal.log"
            returncode = _run_pi_process(
                [sys.executable, str(script)],
                cwd=root,
                env=dict(os.environ),
                events_path=events,
                stderr_path=errors,
                terminal_path=terminal,
            )
            rows, malformed = _jsonl(events)
            self.assertEqual(returncode, 0)
            self.assertEqual(malformed, 0)
            self.assertEqual(_assistant_text(rows), payload)
            self.assertEqual(errors.read_text(encoding="utf-8"), "diagnostic-complete\n")
            self.assertIn(payload, terminal.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
