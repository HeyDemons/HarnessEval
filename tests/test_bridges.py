from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_platform.bridges import base, prepare as bridge_prepare, runner as bridge_runner
from benchmark_platform.bridges.adapters import load_case
from benchmark_platform.bridges.bfcl import (
    noncanonical_schema_types,
    normalize_bfcl_parameters,
)
from benchmark_platform.bridges.episode import NativeTool
from benchmark_platform.bridges.product_episode import ProductEpisodeBridge
from benchmark_platform.bridges.product_server import (
    PRODUCT_WORKSPACE_ROOT,
    ProductBridge,
    translate_product_workspace_arguments,
)
from benchmark_platform.bridges.prepare import _trajectory_source_tools, _trajectory_tool
from benchmark_platform.bridges.tau_episode import _visible_history as _tau_visible_history
from benchmark_platform.bridges.vita_episode import (
    _actor_language_directive,
    _message_text,
    _render_domain_policy,
    _visible_history,
)
from benchmark_platform.bridges.task_product_server import TaskProductBridge
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment, ToolImage
from benchmark_platform.harnesses.content import WIRE_IMAGE_MARKER, json_safe, wire_tool_result
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import PROFILES


class RecordingClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.requests: list[list[dict[str, str]]] = []
        self.native_tools: list[list[dict]] = []

    async def complete(self, messages, *, temperature=None, json_mode=False):
        self.requests.append(messages)
        return Completion(next(self.responses), 1, 1, 0.0, 0, {})

    async def complete_native(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        temperature=None,
    ):
        self.requests.append(messages)
        self.native_tools.append(tools or [])
        content = next(self.responses)
        return Completion(
            content,
            1,
            1,
            0.0,
            0,
            {"choices": [{"message": {"role": "assistant", "content": content}}]},
        )


RESPONSES = {
    "actor-only": ['{"final":"ok"}'],
    "react": ["Thought: complete\nFinal Answer: ok"],
    "plan-execute": [
        '{"steps":[{"id":"s1","instruction":"inspect available evidence"}]}',
        '{"final":"inspection complete"}',
    ],
    "cmas": [
        '{"assignments":[{"id":"w1","instruction":"inspect available evidence"}]}',
        '{"final":"inspection complete"}',
        '{"final":"ok"}',
    ],
    "dmas": [
        '{"requirements":{"reasoning":1.0}}',
        '{"decision":"execute","reason":"complete locally","next_agent_id":null,'
        '"executable":null,"remaining":null,"description":"inspect and answer"}',
        "Inspect the evidence and answer.",
        '{"final":"ok"}',
    ],
    "lats": [
        '{"thought":"complete","final":"ok"}',
        '{"score":1.0,"success":true,"feedback":"complete"}',
    ],
    "memgpt": ['{"thought":"complete","function":"send_message","arguments":{"message":"ok"}}'],
    "aflow-custom-init": ["ok"],
    "dylan": ["ok", "ok", "ok"],
    "magentic-one": [
        "facts",
        "plan",
        json.dumps(
            {
                "is_request_satisfied": {"reason": "test", "answer": False},
                "is_progress_being_made": {"reason": "test", "answer": True},
                "is_in_loop": {"reason": "test", "answer": False},
                "instruction_or_question": {"reason": "test", "answer": "inspect"},
                "next_speaker": {"reason": "test", "answer": "Executor"},
            }
        ),
        "inspection complete",
        json.dumps(
            {
                "is_request_satisfied": {"reason": "test", "answer": True},
                "is_progress_being_made": {"reason": "test", "answer": True},
                "is_in_loop": {"reason": "test", "answer": False},
                "instruction_or_question": {"reason": "test", "answer": "deliver"},
                "next_speaker": {"reason": "test", "answer": "Executor"},
            }
        ),
        "ok",
    ],
    "multi-persona": ["Final answer: ok"],
    "llmcompiler": ['{"tasks":[]}', '{"action":"finish","answer":"ok"}'],
    "rewoo": [
        "Plan: obtain direct evidence\n#E1 = LLM[Return ok]",
        "ok",
        "ok",
    ],
    "sa": ['{"final":"ok"}'],
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_case(root: Path, benchmark: str) -> None:
    if benchmark in {"gaia", "gdpval"}:
        write_json(root / "case.json", {"prompt": "Use the workspace", "case_id": "case"})
        (root / "workspace").mkdir()
        (root / "workspace" / "evidence.txt").write_text("complete evidence", encoding="utf-8")
        return
    if benchmark == "trajectory-bench":
        write_json(
            root / "case.json",
            {
                "prompt": "Call the declared API",
                "tools": [
                    {
                        "tool name": "lookup item",
                        "tool description": "look up an item",
                        "required_parameters": [{"name": "id", "type": "STRING"}],
                    }
                ],
            },
        )
        return
    if benchmark == "bfcl":
        write_json(
            root / "case.json",
            {
                "prompt": "Call the function",
                "functions": [
                    {
                        "name": "lookup_item",
                        "description": "look up an item",
                        "parameters": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                            "required": ["id"],
                        },
                    }
                ],
            },
        )
        return
    raise AssertionError(benchmark)


class BridgeMatrixTests(unittest.TestCase):
    def test_gdpval_prepare_requires_the_complete_v2_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gdpval"
            reference = root / "reference_files" / "hash" / "source.xlsx"
            deliverable = root / "deliverable_files" / "gold" / "answer.pdf"
            reference.parent.mkdir(parents=True)
            deliverable.parent.mkdir(parents=True)
            reference.write_bytes(b"reference")
            deliverable.write_bytes(b"gold")
            row = {
                "task_id": "case",
                "prompt": "Create the requested deliverable.",
                "reference_files": [str(reference.relative_to(root))],
                "deliverable_files": [str(deliverable.relative_to(root))],
                "rubric_json": '[{"score":2,"criterion":"complete"}]',
            }
            output = Path(directory) / "prepared"
            with (
                patch.object(bridge_prepare, "GDPVAL_DATA_ROOT", root),
                patch.object(bridge_prepare, "_parquet_row", return_value=row),
            ):
                bridge_prepare.prepare_gdpval("case", output)
                case = json.loads((output / "input" / "case.json").read_text())
                gold = json.loads((output / "authority" / "gold.json").read_text())
                self.assertEqual(case["reference_files"], ["source.xlsx"])
                self.assertEqual(gold["deliverable_files"], ["deliverable_files/gold/answer.pdf"])
                reference.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "source.xlsx"):
                    bridge_prepare.prepare_gdpval("case", Path(directory) / "missing")

    def test_bfcl_uses_official_type_mapping_and_declaration_only_results(self) -> None:
        parameters = normalize_bfcl_parameters(
            {
                "type": "dict",
                "properties": {
                    "anything": {"type": "any"},
                    "count": {"type": "long"},
                    "names": {"type": "Array", "items": {"type": "String"}},
                    "rows": {
                        "type": "ArrayList",
                        "items": {
                            "type": "dict",
                            "properties": {
                                "ratio": {"type": "float", "description": "A ratio."}
                            },
                        },
                    },
                    "title": {"type": "String"},
                },
            }
        )

        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["properties"]["anything"]["type"], "string")
        self.assertEqual(parameters["properties"]["count"]["type"], "integer")
        self.assertEqual(parameters["properties"]["names"]["type"], "array")
        self.assertEqual(parameters["properties"]["names"]["items"]["type"], "string")
        ratio = parameters["properties"]["rows"]["items"]["properties"]["ratio"]
        self.assertEqual(ratio["type"], "number")
        self.assertEqual(ratio["format"], "float")
        self.assertIn("This is a float type value.", ratio["description"])
        self.assertEqual(parameters["properties"]["title"]["type"], "string")
        self.assertEqual(noncanonical_schema_types(parameters), set())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            messages = [
                {"role": "system", "content": "Use only the declared functions."},
                {"role": "user", "content": "Look up item 7."},
            ]
            write_json(
                root / "case.json",
                {
                    "prompt": "legacy JSON prompt must not win",
                    "messages": messages,
                    "functions": [
                        {
                            "name": "lookup.item",
                            "description": "look up an item",
                            "parameters": {
                                "type": "dict",
                                "properties": {"id": {"type": "long"}},
                                "required": ["id"],
                            },
                        }
                    ],
                },
            )
            bridge = load_case("bfcl", "case", root)
            trace = JsonlTrace(root / "bfcl.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            result = asyncio.run(environment.call("lookup_item", {"id": 7}))

        self.assertNotIn('[[{"role"', bridge.prompt)
        self.assertIn("Use only the declared functions.", bridge.prompt)
        self.assertIn("Look up item 7.", bridge.prompt)
        self.assertEqual(bridge.metadata["messages"], messages)
        self.assertTrue(result["result"]["declaration_only"])
        self.assertEqual(result["result"]["execution"], "not_run")
        self.assertTrue(result["result"]["terminate"])

    def test_bfcl_product_does_not_prelaunch_declaration_only_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            job = root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            bridge = ProductBridge("bfcl", "case", source, job)
            try:
                manifest = bridge.manifest()
            finally:
                bridge.close()

        self.assertEqual(
            manifest["metadata"]["lifecycle"], "single_turn_declaration_only"
        )
        self.assertEqual(manifest["safe_tools"], [])

    def test_trajectory_product_does_not_prelaunch_unverified_remote_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            job = root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "trajectory-bench")
            bridge = ProductBridge("trajectory-bench", "case", source, job)
            try:
                manifest = bridge.manifest()
            finally:
                bridge.close()

        self.assertEqual(manifest["metadata"]["mutability_contract"], "unverified")
        self.assertEqual(manifest["safe_tools"], [])

    def test_bfcl_runner_keeps_committed_batch_when_profile_termination_fails(self) -> None:
        messages = [
            {"role": "system", "content": "Use the declarations exactly."},
            {"role": "user", "content": "Call lookup_item."},
        ]

        async def call_then_fail(context):
            self.assertTrue(context.policy["declaration_only_tools"])
            await context.environment.call("lookup_item", {"id": "7"})
            raise RuntimeError("profile terminal protocol was not satisfied")

        async def fail_without_call(context):
            self.assertTrue(context.policy["declaration_only_tools"])
            raise RuntimeError("provider failed before a prediction")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_root = root / "input"
            case_root.mkdir()
            make_case(case_root, "bfcl")
            case = json.loads((case_root / "case.json").read_text(encoding="utf-8"))
            case["messages"] = messages
            write_json(case_root / "case.json", case)
            environment = {
                "HARNESS_API_BASE": "http://example.invalid/v1",
                "HARNESS_API_KEY": "test-key",
                "HARNESS_MODEL": "test-model",
            }

            completed_job = root / "completed-job"
            completed_job.mkdir()
            with (
                patch.dict(os.environ, environment),
                patch.object(bridge_runner, "completion_client_from_env", return_value=object()),
                patch.object(bridge_runner, "run_profile", new=call_then_fail),
            ):
                completed = asyncio.run(
                    bridge_runner.execute(
                        "bfcl", "actor-only", "case", case_root, completed_job, {}
                    )
                )
            manifest = json.loads(
                (completed_job / "bridge_manifest.json").read_text(encoding="utf-8")
            )
            events = [
                json.loads(line)
                for line in (completed_job / "harness_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            failed_job = root / "failed-job"
            failed_job.mkdir()
            with (
                patch.dict(os.environ, environment),
                patch.object(bridge_runner, "completion_client_from_env", return_value=object()),
                patch.object(bridge_runner, "run_profile", new=fail_without_call),
            ):
                failed = asyncio.run(
                    bridge_runner.execute(
                        "bfcl", "actor-only", "case", case_root, failed_job, {}
                    )
                )

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["tool_calls"], 1)
        self.assertEqual(
            completed["committed_calls"],
            [{"name": "lookup_item", "arguments": {"id": "7"}}],
        )
        self.assertNotIn("error", completed)
        self.assertEqual(
            completed["termination"]["kind"],
            "profile_error_after_declaration_commit",
        )
        self.assertIn("terminal protocol", completed["termination"]["error"])
        self.assertEqual(manifest["metadata"]["messages"], messages)
        self.assertTrue(
            any(
                event["event"] == "bridge_warning"
                and event["kind"] == "profile_error_after_declaration_commit"
                for event in events
            )
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["tool_calls"], 0)
        self.assertIn("provider failed before a prediction", failed["error"])

    def test_bfcl_actor_stops_before_a_second_assistant_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            job = root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            client = RecordingClient(
                [
                    '{"tool":"lookup_item","arguments":{"id":"first"}}',
                    '{"tool":"lookup_item","arguments":{"id":"must-not-run"}}',
                ]
            )
            with patch.object(
                bridge_runner,
                "completion_client_from_env",
                return_value=client,
            ):
                result = asyncio.run(
                    bridge_runner.execute(
                        "bfcl",
                        "actor-only",
                        "case",
                        source,
                        job,
                        {"max_turns": 4},
                    )
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(
            result["committed_calls"],
            [{"name": "lookup_item", "arguments": {"id": "first"}}],
        )
        self.assertEqual(result["termination"]["kind"], "declaration_batch_committed")

    def test_bfcl_keeps_parallel_calls_from_one_assistant_response(self) -> None:
        async def one_response_batch(context):
            await context.complete("planner", [{"role": "user", "content": "plan"}])
            await asyncio.gather(
                context.environment.call("lookup_item", {"id": "a"}),
                context.environment.call("lookup_item", {"id": "b"}),
            )
            await context.complete("must_not_run", [{"role": "user", "content": "again"}])
            raise AssertionError("unreachable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            job = root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            client = RecordingClient(["planned", "must-not-run"])
            with (
                patch.object(bridge_runner, "completion_client_from_env", return_value=client),
                patch.object(bridge_runner, "run_profile", new=one_response_batch),
            ):
                result = asyncio.run(
                    bridge_runner.execute("bfcl", "llmcompiler", "case", source, job, {})
                )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(
            result["committed_calls"],
            [
                {"name": "lookup_item", "arguments": {"id": "a"}},
                {"name": "lookup_item", "arguments": {"id": "b"}},
            ],
        )

    def test_bfcl_representative_profiles_commit_without_a_followup_turn(self) -> None:
        scenarios = {
            "actor-only": (
                ['{"tool":"lookup_item","arguments":{"id":"a"}}'],
                [{"name": "lookup_item", "arguments": {"id": "a"}}],
            ),
            "react": (
                ['Thought: call it\nAction: lookup_item\nAction Input: {"id":"a"}'],
                [{"name": "lookup_item", "arguments": {"id": "a"}}],
            ),
            "plan-execute": (
                [
                    '{"steps":[{"id":"s1","instruction":"lookup a"}]}',
                    '{"tool":"lookup_item","arguments":{"id":"a"}}',
                ],
                [{"name": "lookup_item", "arguments": {"id": "a"}}],
            ),
            "cmas": (
                [
                    '{"assignments":[{"id":"w1","instruction":"lookup a"}]}',
                    '{"tool":"lookup_item","arguments":{"id":"a"}}',
                ],
                [{"name": "lookup_item", "arguments": {"id": "a"}}],
            ),
            "memgpt": (
                [
                    '{"thought":"lookup","function":"lookup_item",'
                    '"arguments":{"id":"a"}}'
                ],
                [{"name": "lookup_item", "arguments": {"id": "a"}}],
            ),
            "multi-persona": (
                ["Final answer: no function is relevant"],
                [],
            ),
            "llmcompiler": (
                [
                    '{"tasks":['
                    '{"id":"1","tool":"lookup_item","arguments":{"id":"a"},"dependencies":[]},'
                    '{"id":"2","tool":"lookup_item","arguments":{"id":"b"},"dependencies":[]}'
                    ']}'
                ],
                [
                    {"name": "lookup_item", "arguments": {"id": "a"}},
                    {"name": "lookup_item", "arguments": {"id": "b"}},
                ],
            ),
            "rewoo": (
                [
                    "Plan: first lookup\n#E1 = lookup_item[{\"id\":\"a\"}]\n"
                    "Plan: second lookup\n#E2 = lookup_item[{\"id\":\"b\"}]"
                ],
                [
                    {"name": "lookup_item", "arguments": {"id": "a"}},
                    {"name": "lookup_item", "arguments": {"id": "b"}},
                ],
            ),
        }
        for profile, (responses, expected) in scenarios.items():
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "input"
                job = root / "job"
                source.mkdir()
                job.mkdir()
                make_case(source, "bfcl")
                client = RecordingClient(responses)
                with patch.object(
                    bridge_runner,
                    "completion_client_from_env",
                    return_value=client,
                ):
                    result = asyncio.run(
                        bridge_runner.execute(
                            "bfcl",
                            profile,
                            "case",
                            source,
                            job,
                            {"max_turns": 4},
                        )
                    )
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["committed_calls"], expected)
                self.assertEqual(len(client.requests), len(responses))

    def test_bfcl_lats_commits_the_winning_trajectory_as_one_declaration_batch(self) -> None:
        # LATS explores with isolated calls, so the branches it rejects never reach BFCL,
        # and the calls it keeps come from separate proposal responses -- BFCL still scores
        # them as the one assistant batch the profile answered with.
        responses = [
            '{"thought":"declare a","tool":"lookup_item","arguments":{"id":"a"}}',
            '{"thought":"declare other","tool":"lookup_item","arguments":{"id":"other"}}',
            '{"score":0.9,"success":false,"feedback":"continue"}',
            '{"score":0.1,"success":false,"feedback":"wrong branch"}',
            '{"thought":"declare b","tool":"lookup_item","arguments":{"id":"b"}}',
            '{"score":0.9,"success":false,"feedback":"finish"}',
            '{"thought":"answer","final":"done"}',
            '{"score":1.0,"success":true,"feedback":"complete"}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            job = root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            client = RecordingClient(responses)
            with patch.object(
                bridge_runner,
                "completion_client_from_env",
                return_value=client,
            ):
                result = asyncio.run(
                    bridge_runner.execute(
                        "bfcl",
                        "lats",
                        "case",
                        source,
                        job,
                        {
                            "lats_iterations": 1,
                            "lats_generate_samples": 2,
                            "lats_value_samples": 1,
                            "lats_rollout_width": 1,
                            "lats_tree_depth": 3,
                            "lats_rollout_depth": 3,
                        },
                    )
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_answer"], "done")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(
            result["committed_calls"],
            [
                {"name": "lookup_item", "arguments": {"id": "a"}},
                {"name": "lookup_item", "arguments": {"id": "b"}},
            ],
        )
        self.assertEqual(len(client.requests), len(responses))

    def test_vita_visible_history_structures_tool_calls_without_runtime_timestamp(self) -> None:
        class ToolCall:
            id = "call-1"
            name = "lookup"
            arguments = {"key": "value"}
            requestor = "assistant"

        class Message:
            role = "assistant"
            content = None
            tool_calls = [ToolCall()]
            tool_messages = None

            def __str__(self) -> str:
                return "timestamp: 20260817_094500"

        rendered = _visible_history([Message()])
        self.assertIn('"name": "lookup"', rendered)
        self.assertNotIn("timestamp", rendered)
        self.assertNotIn("20260817", rendered)
        self.assertEqual(json.loads(_message_text(Message()))[0]["arguments"], {"key": "value"})

    def test_vita_actor_language_directive_is_english_only(self) -> None:
        class Environment:
            @staticmethod
            def get_policy() -> str:
                return "Official policy at {time}"

        with patch(
            "benchmark_platform.bridges.vita_episode._task_clock",
            return_value=("2026-08-23 00:00:00", "rendered clock"),
        ):
            english, _ = _render_domain_policy(Environment(), "english")
            chinese, _ = _render_domain_policy(Environment(), "chinese")

        directive = _actor_language_directive("english")
        self.assertIsNotNone(directive)
        self.assertIn("Official policy at rendered clock", english)
        self.assertIn("# Language", english)
        self.assertIn(str(directive), english)
        self.assertIn("Do not translate English entity names into Chinese", english)
        self.assertEqual(chinese, "Official policy at rendered clock")
        self.assertIsNone(_actor_language_directive("chinese"))

    def test_tau_visible_history_structures_tool_calls_without_runtime_timestamp(self) -> None:
        class ToolCall:
            id = "call-1"
            name = "lookup"
            arguments = {"key": "value"}
            requestor = "assistant"

        class Message:
            role = "assistant"
            content = None
            tool_calls = [ToolCall()]
            tool_messages = None

            def __str__(self) -> str:
                return "timestamp: 20260817_094500"

        rendered = _tau_visible_history([Message()])
        self.assertIn('"name": "lookup"', rendered)
        self.assertNotIn("timestamp", rendered)
        self.assertNotIn("20260817", rendered)

    def test_task_product_bridge_only_prelaunches_read_only_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = TaskProductBridge(
                benchmark="terminal-bench-2",
                case_id="case",
                prompt="Edit the task workspace",
                container="unused-until-a-tool-is-called",
                workspace_root="/app",
                job=Path(directory),
            )
            try:
                manifest = bridge.manifest()
                self.assertEqual(manifest["safe_tools"], ["read_file", "list_files"])
                self.assertEqual(
                    [tool["name"] for tool in manifest["tools"]],
                    ["read_file", "list_files", "write_file", "run_command"],
                )
            finally:
                bridge.close()

    def test_trajectory_tool_keeps_replay_data_separate_from_public_schema(self) -> None:
        visible = {
            "tool name": "Weather lookup",
            "tool description": "Get weather",
            "required parameters": [{"name": "city", "value": "Nanjing"}],
            "executed_output": "hidden answer",
        }
        catalog = [
            {
                "tool name": "Weather lookup",
                "domain name": "Weather",
                "parent tool name": "Weather API",
                "API name": "Current",
                "required_parameters": [
                    {"name": "city", "type": "STRING", "description": "City name"}
                ],
            }
        ]
        merged = _trajectory_tool(visible, catalog)
        self.assertEqual(merged["domain name"], "Weather")
        self.assertEqual(merged["required parameters"][0]["value"], "Nanjing")
        self.assertNotIn("executed_output", merged)
        self.assertEqual(merged["replay arguments"], {"city": "Nanjing"})
        self.assertEqual(merged["replay output"], "hidden answer")

    def test_trajectory_replay_requires_exact_recorded_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                root / "case.json",
                {
                    "prompt": "Call the declared API",
                    "tools": [
                        {
                            "tool name": "lookup item",
                            "tool description": "look up an item",
                            "required_parameters": [{"name": "id", "type": "STRING"}],
                            "replay arguments": {"id": "42"},
                            "replay output": "complete recorded result",
                        }
                    ],
                },
            )
            previous = os.environ.get("TRAJECT_TOOL_MODE")
            os.environ["TRAJECT_TOOL_MODE"] = "replay"
            try:
                bridge = load_case("trajectory-bench", "case", root)
                accepted = asyncio.run(bridge.handlers[bridge.tools[0].name]({"id": "42"}))
                rejected = asyncio.run(bridge.handlers[bridge.tools[0].name]({"id": "43"}))
            finally:
                if previous is None:
                    os.environ.pop("TRAJECT_TOOL_MODE", None)
                else:
                    os.environ["TRAJECT_TOOL_MODE"] = previous
            self.assertEqual(accepted["response"], "complete recorded result")
            self.assertEqual(accepted["transport"], "dataset_recorded_replay")
            self.assertEqual(rejected["error"], "trajectory_replay_arguments_mismatch")

    def test_trajectory_duplicate_endpoint_declares_one_tool_and_replays_each_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                root / "case.json",
                {
                    "prompt": "Call the same endpoint twice",
                    "tools": [
                        {
                            "tool name": "lookup item",
                            "required_parameters": [{"name": "id", "type": "STRING"}],
                            "replay arguments": {"id": "first"},
                            "replay output": "first result",
                        },
                        {
                            "tool name": "lookup item",
                            "required_parameters": [{"name": "id", "type": "STRING"}],
                            "replay arguments": {"id": "second"},
                            "replay output": "second result",
                        },
                    ],
                },
            )
            with patch.dict(os.environ, {"TRAJECT_TOOL_MODE": "replay"}, clear=True):
                bridge = load_case("trajectory-bench", "case", root)
                environment = ToolEnvironment(
                    bridge.tools,
                    JsonlTrace(root / "trace.jsonl"),
                    bridge.handlers,
                )
                async def replay_both():
                    first = await environment.call(bridge.tools[0].name, {"id": "first"})
                    second = await environment.call(bridge.tools[0].name, {"id": "second"})
                    return first, second

                first, second = asyncio.run(replay_both())

            self.assertEqual(len(bridge.tools), 1)
            self.assertEqual(bridge.metadata["source_tools"], 2)
            self.assertEqual(bridge.metadata["declared_tools"], 1)
            self.assertEqual(first["result"]["response"], "first result")
            self.assertEqual(second["result"]["response"], "second result")

    def test_trajectory_accepts_both_official_tool_list_keys(self) -> None:
        spaced = [{"tool name": "one"}]
        underscored = [{"tool name": "two"}]
        self.assertIs(_trajectory_source_tools({"tool list": spaced}, "case-a"), spaced)
        self.assertIs(_trajectory_source_tools({"tool_list": underscored}, "case-b"), underscored)
        with self.assertRaises(ValueError):
            _trajectory_source_tools({}, "case-c")

    def test_trajectory_allows_stabletoolbench_empty_key(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"ok": true, "result": "simulated"}'

        observed = {}

        def fake_urlopen(request, timeout=None):
            observed["url"] = request.full_url
            observed["payload"] = json.loads(request.data.decode("utf-8"))
            observed["headers"] = {name.lower(): value for name, value in request.header_items()}
            observed["timeout"] = timeout
            return Response()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "trajectory-bench")
            bridge = load_case("trajectory-bench", "case", root)
            with patch.dict(os.environ, {"API_URL": "http://host.docker.internal:8080/virtual"}, clear=True), \
                 patch("benchmark_platform.bridges.adapters.urllib.request.urlopen", fake_urlopen):
                result = asyncio.run(bridge.handlers[bridge.tools[0].name]({"id": "42"}))

        self.assertEqual(result, {"ok": True, "result": "simulated"})
        self.assertEqual(observed["url"], "http://host.docker.internal:8080/virtual")
        self.assertEqual(observed["payload"]["toolbench_key"], "")
        self.assertEqual(json.loads(observed["payload"]["tool_input"]), {"id": "42"})
        self.assertEqual(observed["timeout"], 60.0)
        self.assertNotIn("toolbench_key", observed["headers"])

    def test_trajectory_sends_nonempty_key_in_body_and_header(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"ok": true}'

        observed = {}

        def fake_urlopen(request, timeout=None):
            observed["payload"] = json.loads(request.data.decode("utf-8"))
            observed["headers"] = {name.lower(): value for name, value in request.header_items()}
            return Response()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "trajectory-bench")
            bridge = load_case("trajectory-bench", "case", root)
            with patch.dict(os.environ, {"API_URL": "https://tools.example/virtual", "TOOLBENCH_KEY": "secret"}, clear=True), \
                 patch("benchmark_platform.bridges.adapters.urllib.request.urlopen", fake_urlopen):
                asyncio.run(bridge.handlers[bridge.tools[0].name]({"id": "42"}))

        self.assertEqual(observed["payload"]["toolbench_key"], "secret")
        self.assertEqual(observed["headers"]["toolbench_key"], "secret")

    def test_product_episode_rendezvous_and_speculation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = ProductEpisodeBridge("native", "case", Path(directory))
            bridge.publish_manifest(
                prompt="visible prompt",
                tools=[
                    NativeTool(
                        "lookup",
                        "read state",
                        {"type": "object", "properties": {}},
                        read_only=True,
                        parallel=True,
                    )
                ],
                metadata={"native_lifecycle": True},
                allow_speculation=False,
            )
            self.assertEqual(bridge.manifest()["safe_tools"], [])
            self.assertFalse(bridge.execute("lookup", {}, speculative=True)["ok"])

            def native_side() -> None:
                action = bridge.next_action()
                bridge.resolve(action, {"ok": True, "result": "value"})
                final = bridge.next_action()
                bridge.complete({"status": "completed", "native_score": 1.0}, final)

            thread = threading.Thread(target=native_side)
            thread.start()
            self.assertEqual(bridge.execute("lookup", {}, speculative=False)["result"], "value")
            result = bridge.finalize({"profile": "actor-only", "answer": "done", "committed_calls": []})
            thread.join(timeout=1)
            self.assertEqual(result["native_score"], 1.0)
            self.assertFalse(thread.is_alive())

    def test_product_episode_completion_releases_inflight_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = ProductEpisodeBridge("native", "case", Path(directory))
            bridge.publish_manifest(
                prompt="visible prompt",
                tools=[],
                metadata={"native_lifecycle": True},
            )
            action_ready = threading.Event()

            def native_side() -> None:
                bridge.next_action()
                action_ready.set()
                bridge.complete({"status": "completed", "native_score": 1.0}, None)

            thread = threading.Thread(target=native_side)
            thread.start()
            result = bridge.execute("send_message_to_user", {"content": "done"}, speculative=False)
            action_ready.wait(timeout=1)
            thread.join(timeout=1)

            self.assertEqual(result, {"ok": False, "error": "native_episode_already_finished"})
            self.assertFalse(thread.is_alive())

    def test_product_episode_plain_message_continues_native_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = ProductEpisodeBridge("native", "case", Path(directory))
            bridge.publish_manifest(
                prompt="visible prompt",
                tools=[],
                metadata={"native_lifecycle": True},
            )

            def native_side() -> None:
                action = bridge.next_action()
                self.assertEqual(action.kind, "message")
                self.assertEqual(action.arguments, {"content": "Please confirm"})
                bridge.resolve(action, {"ok": True, "result": {"user_message": "Confirmed"}})

            thread = threading.Thread(target=native_side)
            thread.start()
            response = bridge.continue_conversation("Please confirm")
            thread.join(timeout=1)

            self.assertEqual(response["result"]["user_message"], "Confirmed")
            self.assertEqual(bridge.calls, [])
            self.assertFalse(thread.is_alive())

    def test_product_episode_native_user_stop_releases_plain_message_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = ProductEpisodeBridge("native", "case", Path(directory))
            bridge.publish_manifest(
                prompt="visible prompt",
                tools=[],
                metadata={"native_lifecycle": True},
            )

            def native_side() -> None:
                bridge.next_action()
                bridge.complete({"status": "completed", "native_score": 1.0}, None)

            thread = threading.Thread(target=native_side)
            thread.start()
            response = bridge.continue_conversation("Anything else?")
            thread.join(timeout=1)

            self.assertEqual(response, {"ok": True, "episode_complete": True})
            self.assertFalse(thread.is_alive())

    def test_product_episode_commits_and_replays_speculative_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = ProductEpisodeBridge("native", "case", Path(directory))
            bridge.publish_manifest(
                prompt="visible prompt",
                tools=[
                    NativeTool(
                        "lookup",
                        "read state",
                        {"type": "object", "properties": {}},
                        read_only=True,
                        parallel=True,
                    )
                ],
                metadata={"native_lifecycle": True},
                speculative_executor=lambda _name, _arguments: {"ok": True, "result": "cached"},
            )
            def native_side() -> None:
                action = bridge.next_action()
                self.assertEqual(bridge.replay_response(action.id), {"ok": True, "result": "cached"})
                bridge.resolve(action, {"ok": True, "result": "cached"})

            thread = threading.Thread(target=native_side)
            thread.start()
            speculative = bridge.execute("lookup", {"key": "a"}, speculative=True)
            speculation_id = speculative["_harnesseval_speculation_id"]
            committed = bridge.commit(speculation_id, "lookup", {"key": "a"})
            thread.join(timeout=1)

            self.assertEqual(committed, {"ok": True, "result": "cached"})
            self.assertTrue(bridge.calls[0]["replayed_speculation"])
            self.assertFalse(thread.is_alive())

    def test_all_profiles_load_every_single_turn_bridge(self) -> None:
        async def exercise(root: Path, benchmark: str, profile_id: str) -> tuple[str, RecordingClient, str]:
            bridge = load_case(benchmark, "case", root)
            trace = JsonlTrace(root / f"{profile_id}.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            client = RecordingClient(list(RESPONSES[profile_id]))
            policy = {"max_turns": 4}
            if profile_id == "aflow-custom-init":
                policy["aflow_workflow"] = ["Custom"]
            if profile_id == "lats":
                policy.update(
                    {
                        "lats_iterations": 1,
                        "lats_generate_samples": 1,
                        "lats_value_samples": 1,
                    }
                )
            speculator_client = (
                RecordingClient(['{"actions":[]}'])
                if profile_id == "sa"
                and any(tool.read_only and tool.parallel for tool in bridge.tools)
                else None
            )
            context = RunContext(
                profile_id,
                bridge.prompt,
                client,
                environment,
                trace,
                policy,
                speculator_client=speculator_client,
            )
            answer = await run_profile(context)
            return answer, client, environment.schema

        for benchmark in ("gaia", "gdpval", "trajectory-bench", "bfcl"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_case(root, benchmark)
                for profile in PROFILES:
                    with self.subTest(benchmark=benchmark, profile=profile.id):
                        if profile.id == "lats":
                            bridge = load_case(benchmark, "case", root)
                            if any(not tool.read_only for tool in bridge.tools):
                                trace = JsonlTrace(root / f"{profile.id}-unsupported.jsonl")
                                environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
                                context = RunContext(
                                    profile.id,
                                    bridge.prompt,
                                    RecordingClient(list(RESPONSES[profile.id])),
                                    environment,
                                    trace,
                                    {"lats_iterations": 1, "lats_generate_samples": 1},
                                )
                                with self.assertRaisesRegex(ValueError, "branch-isolated"):
                                    asyncio.run(run_profile(context))
                                continue
                        answer, client, schema = asyncio.run(exercise(root, benchmark, profile.id))
                        self.assertTrue(answer)
                        tool_name = json.loads(schema)[0]["name"]
                        transcript = json.dumps(
                            {"requests": client.requests, "native_tools": client.native_tools},
                            ensure_ascii=False,
                        )
                        if profile.tool_contract == "no-external-tools":
                            self.assertNotIn(tool_name, transcript)
                        else:
                            self.assertIn(tool_name, transcript)

    def test_workspace_case_tools_are_benchmark_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            gaia = load_case("gaia", "case", root)
            self.assertIn("web_search", [tool.name for tool in gaia.tools])
            self.assertIn("run_command", [tool.name for tool in gaia.tools])
            self.assertNotIn("write_file", [tool.name for tool in gaia.tools])
            gdp = load_case("gdpval", "case", root)
            self.assertIn("write_file", [tool.name for tool in gdp.tools])
            self.assertIn("run_command", [tool.name for tool in gdp.tools])
            self.assertIn("web_search", [tool.name for tool in gdp.tools])

    def test_workspace_list_files_matches_complete_relative_globs(self) -> None:
        async def listed(root: Path, arguments: dict) -> list[str]:
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "list-files.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            result = await environment.call("list_files", arguments)
            self.assertTrue(result["ok"])
            return result["result"]["files"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            workspace = root / "workspace"
            (workspace / "root.json").write_text("{}", encoding="utf-8")
            (workspace / "nested").mkdir()
            (workspace / "nested" / "child.json").write_text("{}", encoding="utf-8")

            recursive = asyncio.run(listed(root, {}))
            double_star = asyncio.run(listed(root, {"pattern": "**/*"}))
            nested_json = asyncio.run(listed(root, {"pattern": "nested/*.json"}))

        self.assertIn("root.json", recursive)
        self.assertIn("nested/child.json", recursive)
        self.assertIn("root.json", double_star)
        self.assertIn("nested/child.json", double_star)
        self.assertEqual(nested_json, ["nested/child.json"])

    def test_product_workspace_absolute_paths_are_translated_to_bridge_relative_paths(self) -> None:
        self.assertEqual(
            translate_product_workspace_arguments(
                "run_command", {"argv": ["pwd"], "cwd": PRODUCT_WORKSPACE_ROOT}
            ),
            {"argv": ["pwd"], "cwd": "."},
        )
        self.assertEqual(
            translate_product_workspace_arguments(
                "read_file", {"path": PRODUCT_WORKSPACE_ROOT + "/nested/evidence.txt"}
            ),
            {"path": "nested/evidence.txt"},
        )
        self.assertEqual(
            translate_product_workspace_arguments("read_file", {"path": "/tmp/evidence.txt"}),
            {"path": "/tmp/evidence.txt"},
        )

    def test_workspace_command_middle_truncates_long_output_without_losing_its_tail(self) -> None:
        async def exercise(root: Path):
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "command.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            return await environment.call(
                "run_command",
                {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import os; print('x' * 200000); print(os.getenv('HARNESS_API_KEY', 'absent'))",
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            previous = os.environ.get("HARNESS_API_KEY")
            os.environ["HARNESS_API_KEY"] = "not-for-command-tools"
            try:
                result = asyncio.run(exercise(root))
            finally:
                if previous is None:
                    os.environ.pop("HARNESS_API_KEY", None)
                else:
                    os.environ["HARNESS_API_KEY"] = previous
            self.assertTrue(result["ok"])
            stdout = result["result"]["stdout"]
            self.assertLess(len(stdout), 200000)
            self.assertIn("<START_TOOL_OUTPUT>", stdout)
            self.assertIn("<END_TOOL_OUTPUT>", stdout)
            self.assertTrue(stdout.startswith("The stdout of your command was too long"))
            self.assertIn("x" * 100, stdout)
            self.assertIn("absent", stdout)
            self.assertTrue(stdout.endswith("<END_TOOL_OUTPUT>"))

    def test_workspace_read_file_middle_truncates_large_utf8_content(self) -> None:
        async def exercise(root: Path):
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "large-file.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            return await environment.call("read_file", {"path": "large.txt"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            (root / "workspace" / "large.txt").write_text(
                "BEGIN" + "x" * 100_000 + "END", encoding="utf-8"
            )
            result = asyncio.run(exercise(root))

        text = result["result"]["text"]
        self.assertLess(len(text), 100_008)
        self.assertIn("<START_TOOL_OUTPUT>", text)
        self.assertIn("BEGIN", text)
        self.assertIn("END", text)

    def test_memgpt_next_request_does_not_embed_an_unbounded_command_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "memgpt-large-command.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            client = RecordingClient(
                [
                    json.dumps(
                        {
                            "thought": "inspect a large source",
                            "function": "run_command",
                            "arguments": {
                                "argv": [sys.executable, "-c", "print('x' * 100000)"],
                                "request_heartbeat": True,
                            },
                        }
                    ),
                    '{"thought":"deliver","function":"send_message","arguments":{"message":"done"}}',
                ]
            )
            context = RunContext(
                "memgpt", bridge.prompt, client, environment, trace, {"max_turns": 2}
            )
            answer = asyncio.run(run_profile(context))

        next_request = json.dumps(client.requests[1], ensure_ascii=False)
        self.assertEqual(answer, "done")
        self.assertLess(len(next_request), 100_000)
        self.assertIn("<START_TOOL_OUTPUT>", next_request)
        self.assertIn("<END_TOOL_OUTPUT>", next_request)

    def test_workspace_command_inherits_the_proxy_but_still_not_a_secret(self) -> None:
        """The container's only route out is the proxy the host injects, and stripping it made
        GAIA look like it had no network: web_search kept the harness process's own environment
        and worked, while every run_command fetch died with "Network is unreachable". The
        allowlist has to carry the proxy and still withhold credentials."""

        async def exercise(root: Path):
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "proxy.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            return await environment.call(
                "run_command",
                {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import os; print(os.getenv('https_proxy', 'absent')); "
                        "print(os.getenv('HARNESS_API_KEY', 'absent'))",
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            previous = {name: os.environ.get(name) for name in ("https_proxy", "HARNESS_API_KEY")}
            os.environ["https_proxy"] = "http://10.0.2.2:7890"
            os.environ["HARNESS_API_KEY"] = "not-for-command-tools"
            try:
                result = asyncio.run(exercise(root))
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            self.assertTrue(result["ok"])
            self.assertEqual(
                result["result"]["stdout"].split(), ["http://10.0.2.2:7890", "absent"]
            )

    def test_workspace_command_that_hangs_is_bounded_and_reaped(self) -> None:
        """An unbounded command can eat a whole episode: one GAIA case spent 1678s on curls
        that never resolved and was cancelled with no score. The timeout has to reach the
        grandchild too, since the model reaches the network through a shell."""

        async def exercise(root: Path):
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "timeout.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            marker = root / "workspace" / "grandchild-still-running"
            argv = [
                "bash",
                "-c",
                f"({sys.executable} -c \"import time; time.sleep(30); open({str(marker)!r},'w')\" &) ; sleep 30",
            ]
            return await environment.call("run_command", {"argv": argv}), marker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            previous = base.COMMAND_TIMEOUT_S
            base.COMMAND_TIMEOUT_S = 1.0
            try:
                started = time.monotonic()
                result, marker = asyncio.run(exercise(root))
                elapsed = time.monotonic() - started
            finally:
                base.COMMAND_TIMEOUT_S = previous
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "command_timeout")
            self.assertEqual(result["timeout_seconds"], 1.0)
            # Same keys as command_failed: a reference-resolving profile must not hit a
            # KeyError just because the command timed out instead of exiting nonzero.
            self.assertEqual(result["stdout"], "")
            self.assertIn("killed", result["stderr"])
            self.assertLess(elapsed, 20)
            time.sleep(1.0)
            self.assertFalse(marker.exists(), "the backgrounded grandchild outlived the kill")

    def test_workspace_image_is_structured_content_and_trace_stays_small(self) -> None:
        async def exercise(root: Path):
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "image.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            result = await environment.call("read_file", {"path": "evidence.jpg"})
            client = RecordingClient(["done"])
            context = RunContext("react", bridge.prompt, client, environment, trace, {})
            await context.complete("image_followup", [{"role": "user", "content": "Inspect it"}])
            return result, client.requests, trace.path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            payload = b"\xff\xd8\xff" + b"image-bytes" * 1000
            (root / "workspace" / "evidence.jpg").write_bytes(payload)
            result, requests, trace_path = asyncio.run(exercise(root))

            image = result["result"]["image"]
            self.assertIsInstance(image, ToolImage)
            self.assertEqual(image.data, payload)
            self.assertNotIn("content", result["result"])
            self.assertNotIn("base64", json.dumps(result, ensure_ascii=False))
            content = requests[0][-1]["content"]
            self.assertIsInstance(content, list)
            self.assertIs(content[-1]["image"], image)
            self.assertLess(trace_path.stat().st_size, 5000)
            self.assertNotIn("image-bytes", trace_path.read_text(encoding="utf-8"))

    def test_product_image_wire_encoding_is_confined_to_the_response_payload(self) -> None:
        image = ToolImage("image/png", b"\x89PNG\r\n")
        result = {"ok": True, "result": {"path": "figure.png", "image": image}}

        encoded = wire_tool_result(result)
        wire_image = encoded["result"]["image"]
        self.assertEqual(wire_image["type"], "image")
        self.assertEqual(wire_image["bytes"], len(image.data))
        self.assertEqual(wire_image[WIRE_IMAGE_MARKER]["mime_type"], "image/png")
        self.assertEqual(wire_image[WIRE_IMAGE_MARKER]["data"], "iVBORw0K")
        self.assertNotIn(WIRE_IMAGE_MARKER, json.dumps(json_safe(result)))
        self.assertNotIn("iVBORw0K", json.dumps(json_safe(result)))

    def test_workspace_non_image_binary_is_never_inlined(self) -> None:
        async def exercise(root: Path):
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "binary.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            return await environment.call("read_file", {"path": "archive.bin"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            (root / "workspace" / "archive.bin").write_bytes(b"\x00\xff" * 1000)
            result = asyncio.run(exercise(root))["result"]
            self.assertTrue(result["binary"])
            self.assertEqual(result["bytes"], 2000)
            self.assertNotIn("content", result)
            self.assertNotIn("base64", json.dumps(result, ensure_ascii=False))

    def test_workspace_command_exposes_nonzero_exit_as_tool_failure(self) -> None:
        async def exercise(root: Path):
            bridge = load_case("gaia", "case", root)
            trace = JsonlTrace(root / "failed-command.jsonl")
            environment = ToolEnvironment(bridge.tools, trace, bridge.handlers)
            return await environment.call(
                "run_command",
                {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import sys; print('complete failure output'); sys.exit(7)",
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_case(root, "gaia")
            result = asyncio.run(exercise(root))
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "command_failed")
            self.assertEqual(result["returncode"], 7)
            self.assertEqual(result["stdout"], "complete failure output\n")


if __name__ == "__main__":
    unittest.main()
