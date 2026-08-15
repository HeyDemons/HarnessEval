from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment, ToolSpec, extract_json
from benchmark_platform.harnesses.methods import run_profile


ROOT = Path(__file__).resolve().parents[1]


class ScriptedClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    async def complete(self, messages, *, temperature=None, json_mode=False):
        content = next(self.responses)
        return Completion(content, 1, 1, 0.0, 0, {"choices": [{"message": {"content": content}}]})


def tool_specs() -> list[ToolSpec]:
    script = str(ROOT / "examples" / "tools" / "arithmetic.py")
    schema = {"type": "object"}
    return [
        ToolSpec("lookup", "lookup", schema, (sys.executable, script, "lookup"), parallel=True, read_only=True),
        ToolSpec("multiply", "multiply", schema, (sys.executable, script, "multiply"), parallel=True, read_only=True),
    ]


class HarnessTests(unittest.TestCase):
    def run_profile(self, profile: str, responses: list[str]) -> tuple[str, ToolEnvironment]:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            context = RunContext(
                profile,
                "retrieve alpha and beta, multiply them",
                ScriptedClient(responses),
                environment,
                trace,
                {"max_turns": 8},
            )
            answer = asyncio.run(run_profile(context))
            self.assertTrue(trace.path.read_text(encoding="utf-8"))
            return answer, environment

    def test_actor_only_dynamic_tools(self) -> None:
        answer, environment = self.run_profile(
            "actor-only",
            [
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"tool":"lookup","arguments":{"key":"beta"}}',
                '{"tool":"multiply","arguments":{"a":6,"b":7}}',
                '{"final":"42"}',
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(len(environment.calls), 3)

    def test_react_protocol(self) -> None:
        answer, _ = self.run_profile(
            "react",
            [
                'Thought: get alpha\nAction: lookup\nAction Input: {"key":"alpha"}',
                'Thought: get beta\nAction: lookup\nAction Input: {"key":"beta"}',
                'Thought: multiply\nAction: multiply\nAction Input: {"a":6,"b":7}',
                "Thought: done\nFinal Answer: 42",
            ],
        )
        self.assertEqual(answer, "42")

    def test_plan_execute_nested_observation_reference(self) -> None:
        answer, environment = self.run_profile(
            "plan-execute",
            [
                json.dumps(
                    {
                        "steps": [
                            {"id": "s1", "tool": "lookup", "arguments": {"key": "alpha"}},
                            {"id": "s2", "tool": "lookup", "arguments": {"key": "beta"}},
                            {
                                "id": "s3",
                                "tool": "multiply",
                                "arguments": {"a": "$s1.result.value", "b": "$s2.result.value"},
                            },
                        ]
                    }
                ),
                "42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(environment.calls[-1]["result"]["result"]["product"], 42)

    def test_cmws_parallel_wave(self) -> None:
        answer, environment = self.run_profile(
            "cmws",
            [
                json.dumps(
                    {
                        "assignments": [
                            {"id": "w1", "instruction": "alpha", "tool": "lookup", "arguments": {"key": "alpha"}},
                            {"id": "w2", "instruction": "beta", "tool": "lookup", "arguments": {"key": "beta"}},
                        ]
                    }
                ),
                '{"tool":"multiply","arguments":{"a":6,"b":7}}',
                "42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(len(environment.calls), 3)

    def test_json_parser_preserves_large_complete_value(self) -> None:
        value = {"text": "x" * 200_000, "tail": [1, 2, 3]}
        self.assertEqual(extract_json(json.dumps(value)), value)

    def test_tool_does_not_inherit_api_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "env.py"
            script.write_text(
                "import json, os, sys\njson.load(sys.stdin)\nprint(json.dumps({'secret_visible': 'HARNESS_API_KEY' in os.environ}))\n",
                encoding="utf-8",
            )
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(
                [ToolSpec("env", "env", {"type": "object"}, (sys.executable, str(script)))],
                trace,
            )
            previous = os.environ.get("HARNESS_API_KEY")
            os.environ["HARNESS_API_KEY"] = "not-for-tools"
            try:
                result = asyncio.run(environment.call("env", {}))
            finally:
                if previous is None:
                    os.environ.pop("HARNESS_API_KEY", None)
                else:
                    os.environ["HARNESS_API_KEY"] = previous
            self.assertFalse(result["result"]["secret_visible"])


if __name__ == "__main__":
    unittest.main()
