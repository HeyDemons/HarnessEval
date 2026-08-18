from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import (
    JsonlTrace,
    RunContext,
    ToolEnvironment,
    ToolSpec,
    extract_json,
    normalize_json_schema,
)
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
    def test_source_schema_aliases_are_normalized_recursively(self) -> None:
        source = {
            "type": "dict",
            "properties": {
                "weight": {"type": "float", "minimum": 0},
                "tags": {"type": "list", "items": {"type": "str"}},
                "flags": {"type": ["bool", "null"]},
            },
            "required": ["weight"],
        }
        self.assertEqual(
            normalize_json_schema(source),
            {
                "type": "object",
                "properties": {
                    "weight": {"type": "number", "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "flags": {"type": ["boolean", "null"]},
                },
                "required": ["weight"],
            },
        )

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

    def test_react_action_without_the_action_input_label(self) -> None:
        """The prompt only describes "JSON Action Input" in prose, and gpt-5.6-luna names
        the tool then emits its JSON with no such label, wrapped in narration and leaked
        channel markers. Rejecting that shape cost a real GAIA run all 20 turns and every
        tool call, so the label is optional and the arguments are the first JSON object
        after the action name."""
        answer, environment = self.run_profile(
            "react",
            [
                'Thought: get alpha.Action: lookup\n{"key":"alpha"}\n{"key":"alpha"} to=lookup code:',
                'Thought: get beta.Action: lookup\n{"key":"beta"}Thought: then I multiply them.',
                'Thought: multiply.Action: multiply\n{"a":6,"b":7}',
                "Thought: done\nFinal Answer: 42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual([call["name"] for call in environment.calls], ["lookup", "lookup", "multiply"])

    def test_react_action_without_any_json_stays_a_protocol_error(self) -> None:
        """Naming a tool is not enough to invoke it. A turn that never supplies arguments
        must go back as a protocol error rather than reach the tool with a guessed object."""
        answer, environment = self.run_profile(
            "react",
            [
                "Thought: I will look it up. to=lookup code:",
                'Thought: get alpha\nAction: lookup\nAction Input: {"key":"alpha"}',
                "Thought: done\nFinal Answer: 42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual([call["name"] for call in environment.calls], ["lookup"])

    def test_plan_execute_uses_agent_executors_for_textual_steps(self) -> None:
        answer, environment = self.run_profile(
            "plan-execute",
            [
                json.dumps(
                    {
                        "steps": [
                            {"id": "s1", "instruction": "retrieve alpha"},
                            {"id": "s2", "instruction": "retrieve beta"},
                            {"id": "s3", "instruction": "multiply the retrieved values"},
                        ]
                    }
                ),
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"final":"alpha is 6"}',
                '{"tool":"lookup","arguments":{"key":"beta"}}',
                '{"final":"beta is 7"}',
                '{"tool":"multiply","arguments":{"a":6,"b":7}}',
                '{"final":"the product is 42"}',
            ],
        )
        self.assertEqual(answer, "the product is 42")
        self.assertEqual(environment.calls[-1]["result"]["result"]["product"], 42)

    def test_plan_execute_matches_original_step_context_and_return_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    '{"steps":[{"id":"s1","instruction":"retrieve alpha"}]}',
                    '{"tool":"lookup","arguments":{"key":"alpha"}}',
                    '{"final":"alpha is 6"}',
                ]
            )
            context = RunContext(
                "plan-execute",
                "retrieve alpha and explain the result",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"max_turns": 8},
            )
            answer = asyncio.run(run_profile(context))

        self.assertEqual(answer, "alpha is 6")
        self.assertEqual(len(client.messages), 3)
        executor_prompt = client.messages[1][1]["content"]
        self.assertIn("Current objective: retrieve alpha", executor_prompt)
        self.assertNotIn(context.prompt, executor_prompt)

    def test_cmas_parallel_wave(self) -> None:
        answer, environment = self.run_profile(
            "cmas",
            [
                json.dumps(
                    {
                        "assignments": [
                            {"id": "w1", "instruction": "retrieve alpha"},
                            {"id": "w2", "instruction": "retrieve beta"},
                        ]
                    }
                ),
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"tool":"lookup","arguments":{"key":"beta"}}',
                '{"final":"alpha is 6"}',
                '{"final":"beta is 7"}',
                '{"final":"42"}',
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(len(environment.calls), 2)

    def test_cmas_workers_receive_only_their_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    '{"assignments":[{"id":"w1","instruction":"retrieve alpha"}]}',
                    '{"tool":"lookup","arguments":{"key":"alpha"}}',
                    '{"final":"alpha is 6"}',
                    '{"final":"6"}',
                ]
            )
            context = RunContext(
                "cmas",
                "retrieve alpha and beta, then multiply them",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"max_turns": 8},
            )
            answer = asyncio.run(run_profile(context))

        self.assertEqual(answer, "6")
        worker_prompt = client.messages[1][1]["content"]
        synthesis_prompt = client.messages[-1][0]["content"]
        self.assertIn("Assignment: retrieve alpha", worker_prompt)
        self.assertNotIn(context.prompt, worker_prompt)
        self.assertIn(context.prompt, synthesis_prompt)

    def test_dmas_split_result_only_handoff_and_peer_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    '{"requirements":{"mathematical":1.0,"tool_use":1.0}}',
                    '{"decision":"split","reason":"private decomposition rationale",'
                    '"next_agent_id":"1","executable":"retrieve alpha",'
                    '"remaining":"retrieve beta and multiply alpha by beta","description":null}',
                    "Use lookup to retrieve alpha.",
                    '{"tool":"lookup","arguments":{"key":"alpha"}}',
                    '{"final":"6"}',
                    '{"status":"incompleted","reason":"beta and multiplication remain",'
                    '"next_agent_id":"1","remaining":"retrieve beta and multiply alpha by beta"}',
                    '{"decision":"execute","reason":"I can finish from the completed alpha result",'
                    '"next_agent_id":null,"executable":null,"remaining":null,'
                    '"description":"use the completed alpha result"}',
                    "Use the completed alpha result, retrieve beta, and multiply.",
                    '{"tool":"lookup","arguments":{"key":"beta"}}',
                    '{"tool":"multiply","arguments":{"a":6,"b":7}}',
                    '{"final":"42"}',
                ]
            )
            context = RunContext(
                "dmas",
                "retrieve alpha and beta, multiply them",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"max_turns": 8, "dmas_agent_count": 10, "dmas_seed": 0},
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(answer, "42")
        self.assertEqual([call["name"] for call in context.environment.calls], ["lookup", "lookup", "multiply"])
        later_messages = client.messages[6:]
        self.assertTrue(
            any(
                "retrieve beta and multiply alpha by beta" in message["content"]
                for request in later_messages
                for message in request
            )
        )
        self.assertFalse(
            any(
                "private decomposition rationale" in message["content"]
                for request in later_messages
                for message in request
            )
        )
        self.assertEqual(
            [event["decision"] for event in events if event["event"] == "dmas_route"],
            ["split", "execute"],
        )
        self.assertEqual(len([event for event in events if event["event"] == "dmas_progress"]), 1)
        self.assertFalse(
            any(
                "manager" in event.get("role", "")
                for event in events
                if event["event"] in {"llm_request", "llm_response"}
            )
        )

    def test_dmas_forwarding_preserves_task_and_avoids_visited_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    '{"requirements":{"reasoning":1.0}}',
                    '{"decision":"forward","reason":"seek another agent",'
                    '"next_agent_id":"missing","executable":null,"remaining":null,"description":null}',
                    '{"decision":"forward","reason":"seek another agent",'
                    '"next_agent_id":"missing","executable":null,"remaining":null,"description":null}',
                    '{"decision":"execute","reason":"complete locally",'
                    '"next_agent_id":null,"executable":null,"remaining":null,'
                    '"description":"answer the unchanged task"}',
                    "Answer directly.",
                    '{"final":"ok"}',
                ]
            )
            context = RunContext(
                "dmas",
                "preserve this complete task during forwarding",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"max_turns": 8, "dmas_agent_count": 4, "dmas_seed": 0},
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]

        routes = [event for event in events if event["event"] == "dmas_route"]
        self.assertEqual(answer, "ok")
        self.assertEqual([event["decision"] for event in routes], ["forward", "forward", "execute"])
        self.assertEqual(len({event["agent_id"] for event in routes}), 3)
        self.assertTrue(all(event["current_task"] == context.prompt for event in routes))

    def test_lats_tree_search_executes_rollout_and_value_backpropagation(self) -> None:
        answer, environment = self.run_profile(
            "lats",
            [
                '{"thought":"retrieve alpha","tool":"lookup","arguments":{"key":"alpha"}}',
                '{"score":0.7,"success":false,"feedback":"use the value"}',
                '{"thought":"finish from the observation","final":"6"}',
                '{"score":1.0,"success":true,"feedback":"complete"}',
            ],
            policy={
                "lats_iterations": 1,
                "lats_generate_samples": 1,
                "lats_value_samples": 1,
                "lats_rollout_width": 1,
                "lats_tree_depth": 2,
                "lats_rollout_depth": 2,
            },
        )
        self.assertEqual(answer, "6")
        self.assertEqual([item["name"] for item in environment.calls], ["lookup"])

    def test_lats_rejects_non_snapshotable_mutating_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            tool = ToolSpec(
                "mutate",
                "mutate",
                {"type": "object"},
                (sys.executable, str(ROOT / "examples" / "tools" / "arithmetic.py"), "lookup"),
            )
            context = RunContext(
                "lats",
                "change state",
                ScriptedClient([]),
                ToolEnvironment([tool], trace),
                trace,
                {},
            )
            with self.assertRaisesRegex(ValueError, "branch-isolated environment snapshots"):
                asyncio.run(run_profile(context))

    def test_memgpt_memory_functions_heartbeat_and_benchmark_tool(self) -> None:
        answer, environment = self.run_profile(
            "memgpt",
            [
                '{"thought":"save the task fact","function":"archival_memory_insert",'
                '"arguments":{"content":"alpha must be retrieved","request_heartbeat":true}}',
                '{"thought":"retrieve the saved fact","function":"archival_memory_search",'
                '"arguments":{"query":"alpha","page":0,"request_heartbeat":true}}',
                '{"thought":"use the benchmark tool","function":"lookup",'
                '"arguments":{"key":"alpha","request_heartbeat":true}}',
                '{"thought":"deliver","function":"send_message","arguments":{"message":"6"}}',
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual([item["name"] for item in environment.calls], ["lookup"])

    def test_memgpt_summarizes_only_after_provider_context_error(self) -> None:
        class ContextLimitClient(ScriptedClient):
            async def complete(self, messages, *, temperature=None, json_mode=False):
                self.messages.append(messages)
                response = next(self.responses)
                if isinstance(response, Exception):
                    raise response
                return Completion(response, 1, 1, 0.0, 0, {})

        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ContextLimitClient(
                [
                    '{"thought":"retrieve alpha","function":"lookup",'
                    '"arguments":{"key":"alpha","request_heartbeat":true}}',
                    RuntimeError("maximum context length exceeded"),
                    "alpha was retrieved as 6",
                    '{"thought":"deliver","function":"send_message","arguments":{"message":"6"}}',
                ]
            )
            context = RunContext("memgpt", "retrieve alpha", client, environment, trace, {"max_turns": 4})
            answer = asyncio.run(run_profile(context))
            self.assertEqual(answer, "6")
            events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["event"] == "memgpt_active_memory_summarized" for event in events), 1)

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

    def test_sa_invalidates_speculation_after_state_transition(self) -> None:
        state = {"balance": 1}

        async def read_balance(_arguments):
            return {"balance": state["balance"]}

        async def set_balance(arguments):
            state["balance"] = arguments["value"]
            return {"balance": state["balance"]}

        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(
                [
                    ToolSpec(
                        "read_balance",
                        "read",
                        {"type": "object"},
                        ("unused",),
                        parallel=True,
                        read_only=True,
                    ),
                    ToolSpec(
                        "set_balance",
                        "write",
                        {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                        },
                        ("unused",),
                    ),
                ],
                trace,
                {"read_balance": read_balance, "set_balance": set_balance},
            )
            client = ScriptedClient(
                [
                    '{"tool":"set_balance","arguments":{"value":2}}',
                    '{"actions":[{"tool":"read_balance","arguments":{}}]}',
                    '{"tool":"read_balance","arguments":{}}',
                    '{"final":"2"}',
                ]
            )
            context = RunContext(
                "sa",
                "set the balance to 2, then read it",
                client,
                environment,
                trace,
                {"max_turns": 5},
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text().splitlines()]

        self.assertEqual(answer, "2")
        self.assertEqual(state["balance"], 2)
        self.assertEqual([call["name"] for call in environment.calls], ["read_balance", "set_balance", "read_balance"])
        self.assertEqual(environment.calls[-1]["result"]["result"]["balance"], 2)
        self.assertTrue(any(event["event"] == "sa_cache_invalidated" for event in events))

    def test_concatenated_actions_record_only_the_executed_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ScriptedClient(
                [
                    '{"tool":"lookup","arguments":{"key":"alpha"}}'
                    '{"tool":"lookup","arguments":{"key":"beta"}}',
                    '{"final":"6"}',
                ]
            )
            context = RunContext(
                "actor-only",
                "retrieve alpha",
                client,
                environment,
                trace,
                {"max_turns": 4},
            )
            answer = asyncio.run(run_profile(context))

        assistant_history = [
            message["content"]
            for message in client.messages[-1]
            if message["role"] == "assistant"
        ]
        self.assertEqual(answer, "6")
        self.assertEqual([call["arguments"] for call in environment.calls], [{"key": "alpha"}])
        self.assertEqual(
            assistant_history,
            ['{"tool":"lookup","arguments":{"key":"alpha"}}'],
        )

    def test_json_parser_rejects_non_json_trailing_text(self) -> None:
        with self.assertRaises(ValueError):
            extract_json('{"tool":"lookup","arguments":{}} and then continue')

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
