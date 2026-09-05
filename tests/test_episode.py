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
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]

from benchmark_platform.bridges import tau_episode
from benchmark_platform.bridges.episode import (
    SEND_MESSAGE_TOOL,
    EpisodeBroker,
    FinalResponse,
    NativeTool,
)
from benchmark_platform.bridges.product_episode import ProductEpisodeBridge
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
    "aflow": ["ok"],
    "dylan": ["ok"] * 5,
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


class ScriptedClient:
    def __init__(self, responses: list[object]):
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
        response = next(self.responses)
        message = (
            response
            if isinstance(response, dict)
            else {"role": "assistant", "content": response}
        )
        content = message.get("content") or ""
        return Completion(
            content,
            1,
            1,
            0.0,
            0,
            {"choices": [{"message": message}]},
        )


class EpisodeBrokerTests(unittest.TestCase):
    def test_product_episode_status_exposes_pre_manifest_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = ProductEpisodeBridge("tau2", "case", Path(directory))
            self.assertEqual(bridge.status()["state"], "starting")
            bridge.fail(RuntimeError("provider unavailable"), None)
            status = bridge.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("provider unavailable", status["error"])

    def test_every_profile_runs_inside_native_episode_broker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for benchmark in ("vitabench", "tau2"):
                for profile in PROFILES:
                    with self.subTest(benchmark=benchmark, profile=profile.id):
                        client = ScriptedClient(list(PROFILE_RESPONSES[profile.id]))
                        policy = {"max_turns": 4}
                        if profile.id == "aflow":
                            from benchmark_platform.harnesses.aflow import make_artifact
                            policy.update(aflow_artifact=make_artifact(), aflow_allow_initialization=True)
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
                        if profile.id == "memgpt":
                            self.assertEqual([item.name for item in result], [SEND_MESSAGE_TOOL])
                            continue
                        self.assertIsInstance(result, FinalResponse)
                        transcript = json.dumps(
                            {"requests": client.requests, "native_tools": client.native_tools},
                            ensure_ascii=False,
                        )
                        if profile.tool_contract == "no-external-tools":
                            self.assertNotIn("native_lookup", transcript)
                        else:
                            self.assertIn("native_lookup", transcript)

    def test_magentic_user_reply_returns_to_progress_ledger(self) -> None:
        def ledger(satisfied: bool, instruction: str) -> str:
            return json.dumps(
                {
                    "is_request_satisfied": {"reason": "test", "answer": satisfied},
                    "is_progress_being_made": {"reason": "test", "answer": True},
                    "is_in_loop": {"reason": "test", "answer": False},
                    "instruction_or_question": {
                        "reason": "test",
                        "answer": instruction,
                    },
                    "next_speaker": {"reason": "test", "answer": "Executor"},
                }
            )
        message_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "message-call",
                    "type": "function",
                    "function": {
                        "name": SEND_MESSAGE_TOOL,
                        "arguments": '{"content":"Which option do you prefer?"}',
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "magentic-user.jsonl"
            client = ScriptedClient(
                [
                    "facts",
                    "plan",
                    ledger(False, "ask the user"),
                    message_call,
                    ledger(True, "deliver"),
                    "final answer",
                ]
            )
            broker = EpisodeBroker(
                profile="magentic-one",
                prompt="complete the native conversation",
                tools=[],
                trace_path=trace_path,
                policy={"magentic_max_rounds": 3},
                client=client,
            )
            broker.start()
            wave = broker.next_wave()
            self.assertEqual(len(wave), 1)
            self.assertEqual(wave[0].name, SEND_MESSAGE_TOOL)
            final = broker.next_wave(user_message="I prefer option B")
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertIsInstance(final, FinalResponse)
        self.assertEqual(final.answer, "final answer")
        roles = [event.get("role") for event in events if event["event"] == "llm_request"]
        self.assertEqual(
            roles,
            [
                "orchestrator_facts",
                "orchestrator_plan",
                "orchestrator_ledger",
                "Executor",
                "orchestrator_ledger",
                "orchestrator_final",
            ],
        )
        second_ledger = json.dumps(client.requests[4], ensure_ascii=False)
        self.assertIn("I prefer option B", second_ledger)

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
            self.assertEqual(broker.metrics()["tool_calls"], 0)
            self.assertEqual(broker.metrics()["user_messages"], 1)

    def test_memgpt_user_reply_preserves_internal_memory_in_same_broker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = ScriptedClient(
                [
                    '{"thought":"remember","function":"archival_memory_insert",'
                    '"arguments":{"content":"the pending option","request_heartbeat":true}}',
                    '{"thought":"ask","function":"send_message",'
                    '"arguments":{"message":"Which option?"}}',
                    '{"thought":"check memory","function":"archival_memory_search",'
                    '"arguments":{"query":"pending option","page":0,"request_heartbeat":true}}',
                    '{"thought":"reply","function":"send_message",'
                    '"arguments":{"message":"I kept the memory and received B."}}',
                ]
            )
            broker = EpisodeBroker(
                profile="memgpt",
                prompt="ask for an option and remember why",
                tools=[],
                trace_path=Path(directory) / "trace.jsonl",
                policy={"max_turns": 4},
                client=client,
            )
            broker.start()

            first = broker.next_wave()
            self.assertEqual([item.name for item in first], [SEND_MESSAGE_TOOL])
            self.assertEqual(first[0].arguments["content"], "Which option?")
            self.assertNotIn(SEND_MESSAGE_TOOL, client.requests[0][0]["content"])

            second = broker.next_wave(user_message="B")
            self.assertEqual([item.name for item in second], [SEND_MESSAGE_TOOL])
            self.assertEqual(second[0].arguments["content"], "I kept the memory and received B.")
            self.assertEqual(len(client.requests), 4)
            final_request = "\n".join(str(item.get("content", "")) for item in client.requests[-1])
            self.assertIn("the pending option", final_request)
            self.assertIn('"matches"', final_request)
            self.assertIn("B", final_request)

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


class TauTurnLifecycleTests(unittest.TestCase):
    def test_plain_profile_answers_are_consecutive_assistant_turns(self) -> None:
        """A paper profile returning text ends one assistant turn, not the tau2 episode."""
        from benchmark_platform.bridges import tau_episode

        class HalfDuplexAgent:
            @classmethod
            def __class_getitem__(cls, item):
                return cls

            def __init__(self, tools, domain_policy):
                self.tools = tools
                self.domain_policy = domain_policy

        class AssistantMessage:
            def __init__(self, *, role, content=None, tool_calls=None, cost=0.0):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
                self.tool_messages = None
                self.cost = cost

        class UserMessage:
            def __init__(self, content):
                self.role = "user"
                self.content = content
                self.tool_calls = None
                self.tool_messages = None

            def is_tool_call(self):
                return False

        class ToolMessage:
            pass

        class MultiToolMessage:
            pass

        class ToolCall:
            def __init__(self, *, id, name, arguments, requestor):
                self.id = id
                self.name = name
                self.arguments = arguments
                self.requestor = requestor

        class TextRunConfig:
            def __init__(self, **values):
                self.__dict__.update(values)

        class Registry:
            def __init__(self):
                self.factories = {}

            def register_agent_factory(self, factory, name):
                self.factories[name] = factory

        class Simulation:
            reward_info = None
            termination_reason = SimpleNamespace(value="user_stop")
            duration = 1.0

            def __init__(self, messages):
                self.messages = messages

            def get_messages(self):
                return self.messages

            def model_dump(self, *, mode):
                return {"messages": len(self.messages), "mode": mode}

        class Environment:
            @staticmethod
            def get_policy():
                return "policy"

        created: list[object] = []

        class FakeBroker:
            def __init__(self, *, prompt, **kwargs):
                self.prompt = prompt
                self.next_calls = 0
                created.append(self)

            def start(self):
                return None

            def next_wave(self, **kwargs):
                self.next_calls += 1
                if self.next_calls != 1:
                    raise RuntimeError("a completed per-turn broker was reused")
                return FinalResponse(f"assistant turn {len(created)}")

            @staticmethod
            def metrics():
                return {
                    "agent_turns": 1,
                    "llm_calls": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "tool_calls": 3,
                }

        registry = Registry()

        class Orchestrator:
            def __init__(self, agent):
                self.agent = agent
                self.environment = Environment()

            def run(self):
                greeting = AssistantMessage(role="assistant", content="Hello")
                state = self.agent.get_init_state([greeting])
                first_user = UserMessage("first identifier")
                first_answer, state = self.agent.generate_next_message(first_user, state)
                second_user = UserMessage("second detail")
                second_answer, state = self.agent.generate_next_message(second_user, state)
                return Simulation([greeting, first_user, first_answer, second_user, second_answer])

        configs: list[object] = []

        def build_text_orchestrator(config, task, *, seed):
            configs.append(config)
            agent = registry.factories[config.agent](tools=[], domain_policy="domain policy")
            return Orchestrator(agent)

        def module(name, **attributes):
            value = ModuleType(name)
            value.__dict__.update(attributes)
            if name.rpartition(".")[2] in {"tau2", "agent", "data_model", "evaluator", "runner"}:
                value.__path__ = []
            return value

        fake_modules = {
            "tau2": module("tau2"),
            "tau2.agent": module("tau2.agent"),
            "tau2.agent.base_agent": module("tau2.agent.base_agent", HalfDuplexAgent=HalfDuplexAgent),
            "tau2.data_model": module("tau2.data_model"),
            "tau2.data_model.message": module(
                "tau2.data_model.message",
                AssistantMessage=AssistantMessage,
                UserMessage=UserMessage,
                ToolMessage=ToolMessage,
                MultiToolMessage=MultiToolMessage,
                ToolCall=ToolCall,
            ),
            "tau2.data_model.simulation": module(
                "tau2.data_model.simulation", TextRunConfig=TextRunConfig
            ),
            "tau2.evaluator": module("tau2.evaluator"),
            "tau2.evaluator.evaluator": module(
                "tau2.evaluator.evaluator", EvaluationType=lambda value: value
            ),
            "tau2.registry": module("tau2.registry", registry=registry),
            "tau2.runner": module("tau2.runner"),
            "tau2.runner.build": module(
                "tau2.runner.build",
                _build_env_kwargs=lambda config, task: {},
                build_text_orchestrator=build_text_orchestrator,
            ),
            "tau2.runner.simulation": module(
                "tau2.runner.simulation", run_simulation=lambda *args, **kwargs: None
            ),
        }

        profiles = ("cmas", "memgpt", "dylan", "multi-persona", "llmcompiler", "rewoo")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, fake_modules), patch.object(
            tau_episode, "EpisodeBroker", FakeBroker
        ), patch.object(tau_episode, "_load_task", return_value=("retail", object())), patch.object(
            tau_episode, "_patch_tau_generation"
        ), patch.object(
            tau_episode, "completion_client_from_env", return_value=object()
        ):
            for profile in profiles:
                with self.subTest(profile=profile):
                    created.clear()
                    result = tau_episode.run_episode(
                        profile,
                        "retail:85",
                        {"native_evaluate": False},
                        Path(directory),
                    )
                    self.assertEqual(len(created), 2)
                    self.assertIn("user: first identifier", created[0].prompt)
                    self.assertNotIn("second detail", created[0].prompt)
                    self.assertIn("assistant: assistant turn 1", created[1].prompt)
                    self.assertIn("user: second detail", created[1].prompt)
                    self.assertEqual(result["llm_calls"], 2)
                    self.assertEqual(result["agent_turns"], 2)
                    self.assertEqual(result["tool_calls"], 6)
                    self.assertEqual(configs[-1].llm_args_user, {"temperature": 0.0})
                    self.assertEqual(configs[-1].seed, 300)
                    self.assertEqual(configs[-1].max_steps, 200)


class TauGenerationTests(unittest.TestCase):
    def test_tau_generation_hides_inline_reasoning_from_native_evaluator(self) -> None:
        observed: dict[str, object] = {}

        class AssistantMessage:
            def __init__(self, *, role, content=None, tool_calls=None, **kwargs):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls

        class ToolCall:
            def __init__(self, *, id, name, arguments, requestor):
                self.id = id
                self.name = name
                self.arguments = arguments
                self.requestor = requestor

        class Client:
            def complete_sync(self, *args, **kwargs):
                observed.update(kwargs)
                return Completion(
                    '<think>internal evaluator reasoning</think>{"results":[]}',
                    3,
                    2,
                    0.1,
                    0,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '<think>internal evaluator reasoning</think>{"results":[]}',
                                }
                            }
                        ]
                    },
                )

        def module(name: str, **attributes: object) -> ModuleType:
            value = ModuleType(name)
            value.__dict__.update(attributes)
            if name in {"tau2", "tau2.agent", "tau2.evaluator", "tau2.user", "tau2.utils"}:
                value.__path__ = []
            return value

        fake_modules = {
            "tau2": module("tau2"),
            "tau2.agent": module("tau2.agent"),
            "tau2.agent.llm_agent": module("tau2.agent.llm_agent", generate=None),
            "tau2.environment": module("tau2.environment"),
            "tau2.environment.utils": module("tau2.environment.utils"),
            "tau2.environment.utils.interface_agent": module(
                "tau2.environment.utils.interface_agent", generate=None
            ),
            "tau2.evaluator": module("tau2.evaluator"),
            "tau2.evaluator.auth_classifier": module(
                "tau2.evaluator.auth_classifier", generate=None
            ),
            "tau2.evaluator.evaluator_nl_assertions": module(
                "tau2.evaluator.evaluator_nl_assertions", generate=None
            ),
            "tau2.evaluator.hallucination_reviewer": module(
                "tau2.evaluator.hallucination_reviewer", generate=None
            ),
            "tau2.evaluator.review_llm_judge": module(
                "tau2.evaluator.review_llm_judge", generate=None
            ),
            "tau2.evaluator.review_llm_judge_user_only": module(
                "tau2.evaluator.review_llm_judge_user_only", generate=None
            ),
            "tau2.user": module("tau2.user"),
            "tau2.user.user_simulator": module("tau2.user.user_simulator", generate=None),
            "tau2.data_model": module("tau2.data_model"),
            "tau2.data_model.message": module(
                "tau2.data_model.message", AssistantMessage=AssistantMessage, ToolCall=ToolCall
            ),
            "tau2.utils": module("tau2.utils"),
            "tau2.utils.llm_utils": module(
                "tau2.utils.llm_utils", to_litellm_messages=lambda messages: messages
            ),
        }
        with patch.dict(sys.modules, fake_modules):
            tau_episode._patch_tau_generation(Client())
            result = fake_modules["tau2.evaluator.evaluator_nl_assertions"].generate(
                model="evaluator", messages=[], temperature=0.0, seed=300
            )

        self.assertEqual(result.content, '{"results":[]}')
        self.assertEqual(observed["temperature"], 0.0)
        self.assertEqual(observed["seed"], 300)


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
        broker._thread = SimpleNamespace(is_alive=lambda: True)
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

    def test_a_finished_baseline_fails_the_wave_at_once(self) -> None:
        """dylan stops early by design; its thread is gone and no wave can ever arrive."""
        with tempfile.TemporaryDirectory() as tmp:
            episode, broker = self._broker(Path(tmp))
            broker._thread = SimpleNamespace(is_alive=lambda: False)
            with patch.object(episode, "HANDSHAKE_TIMEOUT_S", 3600):
                started = time.monotonic()
                with self.assertRaises(RuntimeError) as caught:
                    broker.next_wave()
                self.assertLess(time.monotonic() - started, 5, "must not wait out the timeout")
            self.assertIn("no further wave", str(caught.exception))

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
