from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from benchmark_platform.bridges.terminal_episode import _tool_specs
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import JsonlTrace, RunContext, ToolEnvironment
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import PROFILES
from tests.test_episode import PROFILE_RESPONSES


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


def handlers():
    async def complete(arguments):
        return {"ok": True, "result": {"arguments": arguments}}

    return {tool.name: complete for tool in _tool_specs()}


class TaskProfileMatrixTests(unittest.TestCase):
    def test_every_profile_loads_each_task_container_bridge(self) -> None:
        async def exercise(root: Path, benchmark: str, profile_id: str):
            trace = JsonlTrace(root / f"{benchmark}-{profile_id}.jsonl")
            environment = ToolEnvironment(_tool_specs(), trace, handlers())
            client = RecordingClient(list(PROFILE_RESPONSES[profile_id]))
            speculator_client = (
                RecordingClient(['{"actions":[]}'])
                if profile_id == "sa"
                and any(tool.read_only and tool.parallel for tool in _tool_specs())
                else None
            )
            policy = {"max_turns": 4}
            if profile_id == "aflow":
                from benchmark_platform.harnesses.aflow import make_artifact
                policy.update(aflow_artifact=make_artifact(), aflow_allow_initialization=True)
            context = RunContext(
                profile_id,
                f"complete the {benchmark} task container episode",
                client,
                environment,
                trace,
                policy,
                speculator_client=speculator_client,
            )
            answer = await run_profile(context)
            return answer, client, environment.schema

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for benchmark in ("terminal-bench-2", "swe-bench-verified"):
                for profile in PROFILES:
                    with self.subTest(benchmark=benchmark, profile=profile.id):
                        if profile.id == "lats":
                            with self.assertRaisesRegex(ValueError, "branch-isolated"):
                                asyncio.run(exercise(root, benchmark, profile.id))
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


if __name__ == "__main__":
    unittest.main()
