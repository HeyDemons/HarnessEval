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
        self.messages = []

    async def complete(self, messages, *, temperature=None, json_mode=False):
        self.messages.append(messages)
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
    def run_profile(
        self,
        profile: str,
        responses: list[str],
        *,
        policy: dict | None = None,
    ) -> tuple[str, ToolEnvironment]:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            context = RunContext(
                profile,
                "retrieve alpha and beta, multiply them",
                ScriptedClient(responses),
                environment,
                trace,
                {"max_turns": 8, **(policy or {})},
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

    def test_aflow_frozen_custom_operator(self) -> None:
        answer, environment = self.run_profile(
            "aflow",
            [
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"final":"6"}',
            ],
            policy={"aflow_workflow": ["Custom"]},
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 1)

    def test_dylan_published_text_network_has_no_hidden_tool_loop(self) -> None:
        answer, environment = self.run_profile("dylan", ["42", "42", "42"])
        self.assertEqual(answer, "42")
        self.assertEqual(environment.calls, [])

    def test_multi_persona_published_single_model_protocol(self) -> None:
        answer, environment = self.run_profile("multi-persona", ["Final answer: 42"])
        self.assertEqual(answer, "Final answer: 42")
        self.assertEqual(environment.calls, [])

    def test_magentic_one_ledger_worker_and_delivery(self) -> None:
        answer, environment = self.run_profile(
            "magentic-one",
            [
                "Known facts",
                "Use the executor",
                '{"satisfied":false,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"look up alpha"}',
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"satisfied":true,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"deliver"}',
                "6",
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 1)

    def test_llmcompiler_dependency_ready_wave(self) -> None:
        answer, environment = self.run_profile(
            "llmcompiler",
            [
                json.dumps(
                    {
                        "tasks": [
                            {"id": "1", "tool": "lookup", "arguments": {"key": "alpha"}, "dependencies": []},
                            {"id": "2", "tool": "lookup", "arguments": {"key": "beta"}, "dependencies": []},
                            {
                                "id": "3",
                                "tool": "multiply",
                                "arguments": {"a": "$1.result.value", "b": "$2.result.value"},
                                "dependencies": ["1", "2"],
                            },
                        ]
                    }
                ),
                "42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(environment.calls[-1]["result"]["result"]["product"], 42)

    def test_rewoo_plan_evidence_solver(self) -> None:
        answer, environment = self.run_profile(
            "rewoo",
            [
                '{"steps":[{"id":"E1","tool":"lookup","arguments":{"key":"alpha"}}]}',
                "6",
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 1)

    def test_sa_exact_action_cache_hit(self) -> None:
        answer, environment = self.run_profile(
            "sa",
            [
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"actions":[{"tool":"lookup","arguments":{"key":"alpha"}}]}',
                '{"final":"6"}',
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 1)

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
