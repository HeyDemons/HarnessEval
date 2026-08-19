from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import subprocess
import sys
import tempfile
import textwrap
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]

from benchmark_platform.bridges.episode import (
    SEND_MESSAGE_TOOL,
    EpisodeBroker,
    FinalResponse,
    NativeTool,
)
from benchmark_platform.bridges.tau_episode import _native_tools as tau_native_tools
from benchmark_platform.bridges.vita_episode import _native_tools as vita_native_tools
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.profiles import PROFILES


PROFILE_RESPONSES = {
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
                        if profile.id == "lats":
                            with self.assertRaisesRegex(RuntimeError, "branch-isolated"):
                                broker.next_wave()
                            continue
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
                profile="cmas",
                prompt="look up both values",
                tools=[
                    NativeTool(
                        "lookup",
                        "lookup",
                        {"type": "object"},
                        read_only=True,
                        parallel=True,
                    ),
                ],
                trace_path=Path(directory) / "trace.jsonl",
                policy={},
                client=ScriptedClient(
                    [
                        json.dumps(
                            {
                                "assignments": [
                                    {"id": "a", "instruction": "look up a"},
                                    {"id": "b", "instruction": "look up b"},
                                ]
                            }
                        ),
                        '{"tool":"lookup","arguments":{"key":"a"}}',
                        '{"tool":"lookup","arguments":{"key":"b"}}',
                        '{"final":"a report"}',
                        '{"final":"b report"}',
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

    def test_abandoned_request_does_not_block_interpreter_exit(self) -> None:
        """A native adapter may stop calling next_wave with a request still pending:
        step budget exhausted, episode ended, adapter raised. The awaiting coroutine
        must not hold a concurrent.futures worker, because that executor is joined by
        an atexit hook before daemon threads are killed, hanging the process at exit
        after the episode has already finished."""
        script = textwrap.dedent(
            """
            import json, sys, tempfile
            from pathlib import Path
            sys.path.insert(0, %r)
            from tests.test_episode import ScriptedClient
            from benchmark_platform.bridges.episode import EpisodeBroker, NativeTool

            with tempfile.TemporaryDirectory() as directory:
                broker = EpisodeBroker(
                    profile="actor-only",
                    prompt="look up a value",
                    tools=[NativeTool("lookup", "lookup", {"type": "object"})],
                    trace_path=Path(directory) / "trace.jsonl",
                    policy={},
                    client=ScriptedClient(
                        [json.dumps({"tool": "lookup", "arguments": {"key": "a"}})]
                    ),
                )
                broker.start()
                wave = broker.next_wave()
                assert len(wave) == 1
                # Abandon it: never supply the result, exactly as the native loop does
                # when it stops early. The process must still exit.
                print("abandoned", flush=True)
            """
        ) % str(REPO_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertIn("abandoned", completed.stdout)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_native_tool_contracts_default_deny_unsafe_prelaunch(self) -> None:
        class ToolType(Enum):
            READ = "read"
            WRITE = "write"
            GENERIC = "generic"

        def declared(name: str, tool_type: ToolType | None, mutates: bool | None = None):
            def function():
                return None

            if tool_type is not None:
                setattr(function, "__tool_type__", tool_type)
            if mutates is not None:
                setattr(function, "__mutates_state__", mutates)
            return SimpleNamespace(
                _func=function,
                openai_schema={
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object"},
                    }
                },
            )

        vita = vita_native_tools(
            [
                declared("read", ToolType.READ),
                declared("write", ToolType.WRITE),
                declared("generic", ToolType.GENERIC),
                declared("unknown", None),
            ]
        )
        self.assertEqual(
            [(tool.name, tool.read_only, tool.parallel) for tool in vita],
            [
                ("read", True, True),
                ("write", False, False),
                ("generic", False, False),
                ("unknown", False, False),
            ],
        )

        tau = tau_native_tools(
            [
                declared("safe_read", ToolType.READ, False),
                declared("mutating_read", ToolType.READ, True),
                declared("nonmutating_write", ToolType.WRITE, False),
                declared("unknown", None),
            ]
        )
        self.assertEqual(
            [(tool.name, tool.read_only, tool.parallel) for tool in tau],
            [
                ("safe_read", True, True),
                ("mutating_read", False, False),
                ("nonmutating_write", False, False),
                ("unknown", False, False),
            ],
        )


if __name__ == "__main__":
    unittest.main()


class HandshakeTimeoutTests(unittest.TestCase):
    """Neither side of the native handshake may wait without a bound.

    The adapter is allowed to stop asking for the next wave with requests still pending, and
    the coroutine awaiting that reply then waits forever while the adapter waits for the
    message only that coroutine can produce. Neither side is on a socket, so nothing times
    out on its own: an observed arm sat for an hour with 64 threads in futex_wait_queue and
    the sweep sat behind it.
    """

    def _broker(self, tmp: Path):
        sys.path.insert(0, str(REPO_ROOT))
        from benchmark_platform.bridges import episode

        broker = episode.EpisodeBroker.__new__(episode.EpisodeBroker)
        broker._events = queue.Queue()
        broker._ready = threading.Event()
        broker._pending = {}
        broker._broken = None
        return episode, broker

    def test_an_unanswered_request_fails_instead_of_waiting_forever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode, broker = self._broker(Path(tmp))
            with patch.object(episode, "HANDSHAKE_TIMEOUT_S", 0.25):
                started = time.monotonic()
                with self.assertRaises(RuntimeError) as caught:
                    asyncio.run(broker._request("lookup", {}))
                waited = time.monotonic() - started
            self.assertIn("never answered", str(caught.exception))
            self.assertLess(waited, 5, "must give up on its own, not hang")
            # A later call must fail at once rather than wait the timeout all over again.
            with patch.object(episode, "HANDSHAKE_TIMEOUT_S", 30):
                started = time.monotonic()
                with self.assertRaises(RuntimeError):
                    asyncio.run(broker._request("lookup", {}))
                self.assertLess(time.monotonic() - started, 1)

    def test_a_silent_baseline_fails_the_wave_instead_of_waiting_forever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode, broker = self._broker(Path(tmp))
            with patch.object(episode, "HANDSHAKE_TIMEOUT_S", 0.25):
                started = time.monotonic()
                with self.assertRaises(RuntimeError) as caught:
                    broker.next_wave()
                waited = time.monotonic() - started
            self.assertIn("no action or answer", str(caught.exception))
            self.assertLess(waited, 5, "must give up on its own, not hang")
