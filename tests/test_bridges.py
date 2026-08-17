from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from benchmark_platform.bridges.adapters import load_case
from benchmark_platform.bridges.episode import NativeTool
from benchmark_platform.bridges.product_episode import ProductEpisodeBridge
from benchmark_platform.bridges.prepare import _trajectory_tool
from benchmark_platform.bridges.tau_episode import _visible_history as _tau_visible_history
from benchmark_platform.bridges.vita_episode import _message_text, _visible_history
from benchmark_platform.bridges.task_product_server import TaskProductBridge
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import PROFILES


class RecordingClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.requests: list[list[dict[str, str]]] = []

    async def complete(self, messages, *, temperature=None, json_mode=False):
        self.requests.append(messages)
        return Completion(next(self.responses), 1, 1, 0.0, 0, {})


RESPONSES = {
    "actor-only": ['{"final":"ok"}'],
    "react": ["Thought: complete\nFinal Answer: ok"],
    "plan-execute": ['{"steps":[]}', "ok"],
    "cmws": ['{"assignments":[]}', '{"final":"ok"}'],
    "aflow": ['{"final":"ok"}'],
    "dylan": ["ok", "ok", "ok"],
    "magentic-one": [
        "facts",
        "plan",
        '{"satisfied":false,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"inspect"}',
        '{"report":"inspection complete"}',
        '{"satisfied":true,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"deliver"}',
        "ok",
    ],
    "multi-persona": ["Final answer: ok"],
    "llmcompiler": ['{"tasks":[]}', "ok"],
    "rewoo": ['{"steps":[]}', "ok"],
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

    def test_trajectory_tool_joins_public_execution_metadata_without_output(self) -> None:
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
            if profile_id == "aflow":
                policy["aflow_workflow"] = ["Custom"]
            context = RunContext(profile_id, bridge.prompt, client, environment, trace, policy)
            answer = await run_profile(context)
            return answer, client, environment.schema

        for benchmark in ("gaia", "gdpval", "trajectory-bench", "bfcl"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_case(root, benchmark)
                for profile in PROFILES:
                    with self.subTest(benchmark=benchmark, profile=profile.id):
                        answer, client, schema = asyncio.run(exercise(root, benchmark, profile.id))
                        self.assertTrue(answer)
                        tool_name = json.loads(schema)[0]["name"]
                        transcript = json.dumps(client.requests, ensure_ascii=False)
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

    def test_workspace_command_returns_complete_output_without_api_secret(self) -> None:
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
            self.assertTrue(stdout.startswith("x" * 200000))
            self.assertTrue(stdout.rstrip().endswith("absent"))

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
