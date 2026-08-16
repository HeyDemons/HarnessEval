from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_platform.bridges.episode import (
    SEND_MESSAGE_TOOL,
    EpisodeBroker,
    FinalResponse,
    NativeTool,
)
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.profiles import PROFILES


PROFILE_RESPONSES = {
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


class ScriptedClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.requests: list[list[dict[str, str]]] = []

    async def complete(self, messages, *, temperature=None, json_mode=False):
        self.requests.append(messages)
        return Completion(next(self.responses), 1, 1, 0.0, 0, {})


class EpisodeBrokerTests(unittest.TestCase):
    def test_every_profile_runs_inside_native_episode_broker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for benchmark in ("vitabench", "tau2"):
                for profile in PROFILES:
                    with self.subTest(benchmark=benchmark, profile=profile.id):
                        client = ScriptedClient(list(PROFILE_RESPONSES[profile.id]))
                        policy = {"max_turns": 4}
                        if profile.id == "aflow":
                            policy["aflow_workflow"] = ["Custom"]
                        broker = EpisodeBroker(
                            profile=profile.id,
                            prompt=f"complete the {benchmark} native episode",
                            tools=[
                                NativeTool(
                                    "native_lookup",
                                    "look up native state",
                                    {"type": "object", "properties": {}},
                                )
                            ],
                            trace_path=root / f"{benchmark}-{profile.id}.jsonl",
                            policy=policy,
                            client=client,
                        )
                        broker.start()
                        result = broker.next_wave()
                        self.assertIsInstance(result, FinalResponse)
                        transcript = json.dumps(client.requests, ensure_ascii=False)
                        if profile.tool_contract == "no-external-tools":
                            self.assertNotIn("native_lookup", transcript)
                        else:
                            self.assertIn("native_lookup", transcript)

    def test_parallel_profile_wave_is_one_native_wave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broker = EpisodeBroker(
                profile="cmws",
                prompt="look up both values",
                tools=[
                    NativeTool("lookup", "lookup", {"type": "object"}),
                ],
                trace_path=Path(directory) / "trace.jsonl",
                policy={},
                client=ScriptedClient(
                    [
                        json.dumps(
                            {
                                "assignments": [
                                    {"id": "a", "tool": "lookup", "arguments": {"key": "a"}},
                                    {"id": "b", "tool": "lookup", "arguments": {"key": "b"}},
                                ]
                            }
                        ),
                        '{"final":"done"}',
                    ]
                ),
            )
            broker.start()
            wave = broker.next_wave()
            self.assertEqual(len(wave), 2)
            result = broker.next_wave(
                tool_results={item.id: ({"value": item.arguments["key"]}, False) for item in wave}
            )
            self.assertIsInstance(result, FinalResponse)
            self.assertEqual(result.answer, "done")

    def test_user_simulator_reply_returns_to_same_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broker = EpisodeBroker(
                profile="actor-only",
                prompt="ask the hidden user",
                tools=[],
                trace_path=Path(directory) / "trace.jsonl",
                policy={},
                client=ScriptedClient(
                    [
                        json.dumps({"tool": SEND_MESSAGE_TOOL, "arguments": {"content": "Which option?"}}),
                        '{"final":"Option B"}',
                    ]
                ),
            )
            broker.start()
            wave = broker.next_wave()
            self.assertEqual([item.name for item in wave], [SEND_MESSAGE_TOOL])
            result = broker.next_wave(user_message="B")
            self.assertIsInstance(result, FinalResponse)
            self.assertEqual(result.answer, "Option B")


if __name__ == "__main__":
    unittest.main()
