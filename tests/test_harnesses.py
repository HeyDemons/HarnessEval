from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

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
from benchmark_platform.harnesses.rewoo import parse_rewoo_plan


ROOT = Path(__file__).resolve().parents[1]


class ScriptedClient:
    def __init__(self, responses: list[Any]):
        self.responses = iter(responses)
        self.messages = []
        self.json_modes = []
        self.native_tools = []

    async def complete(self, messages, *, temperature=None, json_mode=False):
        self.messages.append(messages)
        self.json_modes.append(json_mode)
        content = next(self.responses)
        return Completion(content, 1, 1, 0.0, 0, {"choices": [{"message": {"content": content}}]})

    async def complete_native(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        temperature=None,
    ):
        self.messages.append(messages)
        self.json_modes.append(False)
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
            {"choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]},
        )


def magentic_ledger(
    *,
    satisfied: bool,
    speaker: str = "Executor",
    instruction: str = "continue",
    progress: bool = True,
    in_loop: bool = False,
) -> str:
    return json.dumps(
        {
            "is_request_satisfied": {"reason": "test", "answer": satisfied},
            "is_progress_being_made": {"reason": "test", "answer": progress},
            "is_in_loop": {"reason": "test", "answer": in_loop},
            "instruction_or_question": {"reason": "test", "answer": instruction},
            "next_speaker": {"reason": "test", "answer": speaker},
        }
    )


def native_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


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
        responses: list[Any],
        *,
        policy: dict | None = None,
        speculator_responses: list[Any] | None = None,
    ) -> tuple[str, ToolEnvironment]:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            speculator_client = (
                ScriptedClient(speculator_responses)
                if speculator_responses is not None
                else None
            )
            context = RunContext(
                profile,
                "retrieve alpha and beta, multiply them",
                ScriptedClient(responses),
                environment,
                trace,
                {"max_turns": 8, **(policy or {})},
                speculator_client=speculator_client,
            )
            answer = asyncio.run(run_profile(context))
            trace_text = trace.path.read_text(encoding="utf-8")
            self.assertTrue(trace_text)
            self.last_client = context.client
            self.last_speculator_client = speculator_client
            self.last_context = context
            self.last_events = [json.loads(line) for line in trace_text.splitlines()]
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

    def test_react_prompt_reproduces_the_upstream_format_block(self) -> None:
        """A bare answer stays a protocol error and still costs a turn: the contract is
        unchanged, so ReAct is no more forgiving than the actor-only control. What was missing
        is that the contract was never stated. Both upstream sources enumerate the legal
        actions -- the paper as "Action can be three types: (1) Search[entity] ...
        (3) Finish[answer]", LangChain's FORMAT_INSTRUCTIONS as "should be one of
        [{tool_names}]" -- and this profile had replaced the whole block with prose, so
        gpt-5.6-terra narrated its actions and stated answers with no label."""
        answer, environment = self.run_profile(
            "react",
            [
                "42",
                'Thought: get alpha\nAction: lookup\nAction Input: {"key":"alpha"}',
                "Thought: done\nFinal Answer: 42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual([call["name"] for call in environment.calls], ["lookup"])

        # ScriptedClient stores the live message list by reference, so read the final state.
        conversation = self.last_client.messages[-1]
        system = conversation[0]["content"]
        for line in (
            "Thought: you should always think about what to do",
            "Action: the action to take, should be one of [lookup, multiply]",
            "Observation: the result of the action",
            "Final Answer: the final answer to the original input question",
        ):
            self.assertIn(line, system)
        # The bare "42" was rejected rather than returned, and came back as upstream's message.
        self.assertIn(
            "Invalid Format: Missing 'Action:' after 'Thought:'",
            [message["content"] for message in conversation if message["role"] == "user"],
        )

    def test_react_action_glued_to_the_action_input_label(self) -> None:
        """gpt-5.6-terra emits steps with no newline between the labels. Capturing a bare word
        reads "Action: lookupAction Input: {...}" as the tool "lookupAction" and burns the turn
        on unknown_tool; upstream's pattern anchors on the label and reads "lookup"."""
        answer, environment = self.run_profile(
            "react",
            [
                'Thought: get alpha.Action: lookupAction Input: {"key":"alpha"}',
                "Thought: done\nFinal Answer: 42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual([call["name"] for call in environment.calls], ["lookup"])

    def test_react_rejects_a_turn_with_both_action_and_final_answer(self) -> None:
        answer, environment = self.run_profile(
            "react",
            [
                'Thought: ambiguous\nAction: lookup\nAction Input: {"key":"alpha"}\nFinal Answer: 6',
                "Thought: done\nFinal Answer: 42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(environment.calls, [])

    def test_magentic_keeps_topology_and_inherits_the_benchmark_toolset(self) -> None:
        """The method owns routing topology; the benchmark bridge owns available tools."""
        from benchmark_platform.harnesses.magentic_one import _magentic_team

        expected_team = ["FileSurfer", "WebSurfer", "Coder", "Executor"]
        self.assertEqual(
            list(_magentic_team({"list_files", "read_file", "run_command", "web_search"})),
            expected_team,
        )
        self.assertEqual(
            list(_magentic_team({"read_file", "list_files", "write_file", "run_command"})),
            expected_team,
        )
        self.assertEqual(
            list(_magentic_team({"get_order_details", "calculate"})),
            expected_team,
        )

        from benchmark_platform.harnesses.magentic_one import _magentic_worker_tools

        names = {"web_search", "read_file", "list_files", "write_file", "run_command"}
        for role in expected_team:
            with self.subTest(role=role):
                self.assertEqual(_magentic_worker_tools(role, names), sorted(names))

    def test_magentic_orchestrator_prompts_match_pinned_autogen_revision(self) -> None:
        from benchmark_platform.harnesses import magentic_one

        names = (
            "ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT",
            "ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT",
            "ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT",
            "ORCHESTRATOR_PROGRESS_LEDGER_PROMPT",
            "ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT",
            "ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT",
            "ORCHESTRATOR_FINAL_ANSWER_PROMPT",
        )
        payload = json.dumps(
            {name: getattr(magentic_one, name) for name in names},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "569ecea527bd61c0d44cb8420fe395f52bb911e5daf62c21b289a56b428de4d9",
        )

    def test_magentic_single_participant_deterministically_overrides_speaker(self) -> None:
        """The pinned orchestrator replaces next_speaker for a one-participant team."""
        answer, environment = self.run_profile(
            "magentic-one",
            [
                "facts",
                "plan",
                magentic_ledger(
                    satisfied=False,
                    speaker="WebSurfer",
                    instruction="look it up",
                ),
                "no retrieval tool here",
                magentic_ledger(satisfied=True),
                "42",
            ],
        )
        self.assertEqual(answer, "42")

    def test_planned_profiles_survive_a_step_that_omits_keys(self) -> None:
        """A planner that omits "tool", names an unplanned step, or writes "$E1.result[0]"
        used to raise straight out of the arm: status=failed, score=None, the whole baseline
        lost to a protocol slip that ReAct hands back to the model as a tool error. Each of
        these plans has to reach its synthesis call instead."""
        answer, environment = self.run_profile(
            "llmcompiler",
            [
                '{"tasks":[{"id":"1","arguments":{"key":"alpha"}},'
                '{"id":"2","tool":"lookup","arguments":{"key":"$9.nothing"}},'
                '{"id":"3","tool":"lookup","arguments":{"key":"$1.result[0]"}}]}',
                '{"action":"finish","answer":"42"}',
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual([call["name"] for call in environment.calls], ["lookup", "lookup"])

        # sa speculates best-effort; a predictor that returns no actions list is a miss.
        answer, _ = self.run_profile(
            "sa",
            ['{"final":"42"}'],
            speculator_responses=['{"predicted":"nothing"}'],
        )
        self.assertEqual(answer, "42")

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
        self.assertIn("Original objective: " + context.prompt, executor_prompt)
        self.assertIn("Current objective: retrieve alpha", executor_prompt)

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

    def test_lats_commits_only_the_selected_branch_to_standard_tool_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ScriptedClient(
                [
                    '{"thought":"try alpha","tool":"lookup","arguments":{"key":"alpha"}}',
                    '{"thought":"try beta","tool":"lookup","arguments":{"key":"beta"}}',
                    '{"score":0.9,"success":false,"feedback":"finish this branch"}',
                    '{"score":0.1,"success":false,"feedback":"weak branch"}',
                    '{"thought":"finish from alpha","final":"6"}',
                    '{"score":1.0,"success":true,"feedback":"complete"}',
                ]
            )
            context = RunContext(
                "lats",
                "retrieve alpha",
                client,
                environment,
                trace,
                {
                    "lats_iterations": 1,
                    "lats_generate_samples": 2,
                    "lats_value_samples": 1,
                    "lats_rollout_width": 1,
                    "lats_tree_depth": 2,
                    "lats_rollout_depth": 2,
                    "lats_max_parallel": 1,
                    "lats_max_llm_calls": 6,
                },
            )
            answer = asyncio.run(run_profile(context))
            events = [
                json.loads(line)
                for line in trace.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(answer, "6")
        self.assertEqual(
            [call["arguments"] for call in environment.calls], [{"key": "alpha"}]
        )
        self.assertEqual(
            [event["arguments"] for event in events if event["event"] == "tool_request"],
            [{"key": "alpha"}],
        )
        self.assertEqual(
            [
                event["arguments"]
                for event in events
                if event["event"] == "lats_tool_request"
            ],
            [{"key": "alpha"}, {"key": "beta"}],
        )

    def test_lats_commits_winning_path_calls_in_trajectory_order(self) -> None:
        answer, environment = self.run_profile(
            "lats",
            [
                '{"thought":"retrieve alpha","tool":"lookup","arguments":{"key":"alpha"}}',
                '{"score":0.8,"success":false,"feedback":"continue"}',
                '{"thought":"retrieve beta","tool":"lookup","arguments":{"key":"beta"}}',
                '{"score":0.9,"success":false,"feedback":"finish"}',
                '{"thought":"answer","final":"42"}',
                '{"score":1.0,"success":true,"feedback":"complete"}',
            ],
            policy={
                "lats_iterations": 1,
                "lats_generate_samples": 1,
                "lats_value_samples": 1,
                "lats_rollout_width": 1,
                "lats_tree_depth": 3,
                "lats_rollout_depth": 3,
                "lats_max_parallel": 1,
                "lats_max_llm_calls": 6,
            },
        )
        self.assertEqual(answer, "42")
        self.assertEqual(
            [call["arguments"] for call in environment.calls],
            [{"key": "alpha"}, {"key": "beta"}],
        )

    def test_lats_samples_proposals_sequentially(self) -> None:
        class ConcurrencyClient:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self.proposals = 0

            async def complete(self, messages, *, temperature=None, json_mode=False):
                prompt = messages[-1]["content"]
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if "Generate one next LATS" in prompt:
                    self.proposals += 1
                    content = json.dumps(
                        {"thought": f"candidate {self.proposals}", "final": str(self.proposals)}
                    )
                else:
                    content = '{"score":1.0,"success":true,"feedback":"complete"}'
                await asyncio.sleep(0.01)
                self.active -= 1
                return Completion(
                    content,
                    1,
                    1,
                    0.01,
                    0,
                    {"choices": [{"message": {"content": content}}]},
                )

        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ConcurrencyClient()
            context = RunContext(
                "lats",
                "choose an answer",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {
                    "lats_iterations": 1,
                    "lats_generate_samples": 5,
                    "lats_value_samples": 1,
                    "lats_rollout_width": 1,
                    "lats_tree_depth": 1,
                    "lats_rollout_depth": 1,
                },
            )
            answer = asyncio.run(run_profile(context))

        # The source samples n choices in one request; a one-choice API samples them one at
        # a time so tree width never silently becomes provider concurrency.
        self.assertEqual(answer, "1")
        self.assertEqual(client.max_active, 1)
        self.assertEqual(context.llm_calls, 10)

    def test_lats_declaration_only_rejected_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace, declaration_only=True)
            context = RunContext("lats", "declare calls", ScriptedClient([]), environment, trace, {})
            with self.assertRaisesRegex(ValueError, "multi-response"):
                asyncio.run(run_profile(context))
            self.assertEqual(context.llm_calls, 0)
            self.assertEqual(environment.calls, [])

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

    def test_memgpt_exposes_upstream_heartbeat_schemas(self) -> None:
        answer, _ = self.run_profile(
            "memgpt",
            ['{"thought":"deliver","function":"send_message","arguments":{"message":"done"}}'],
        )
        self.assertEqual(answer, "done")
        system = self.last_client.messages[0][0]["content"]
        lines = system.splitlines()
        memory = json.loads(next(line.removeprefix("Memory functions: ") for line in lines if line.startswith("Memory functions: ")))
        benchmark = json.loads(next(line.removeprefix("Benchmark functions: ") for line in lines if line.startswith("Benchmark functions: ")))

        for name, schema in memory.items():
            parameters = schema["parameters"]
            if name in {"send_message", "pause_heartbeats"}:
                self.assertNotIn("request_heartbeat", parameters["properties"])
                self.assertNotIn("request_heartbeat", parameters["required"])
            else:
                self.assertEqual(parameters["properties"]["request_heartbeat"]["type"], "boolean")
                self.assertIn("request_heartbeat", parameters["required"])
        for schema in benchmark:
            self.assertEqual(schema["parameters"]["properties"]["request_heartbeat"]["type"], "boolean")
            self.assertIn("request_heartbeat", schema["parameters"]["required"])

    def test_memgpt_declaration_only_tools_keep_schema_and_auto_heartbeat(self) -> None:
        async def record(arguments):
            return {
                "recorded_function_call": "lookup",
                "arguments": arguments,
                "declaration_only": True,
                "execution": "not_run",
            }

        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            tool = ToolSpec(
                "lookup",
                "lookup",
                {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
                (),
                parallel=True,
                read_only=True,
            )
            environment = ToolEnvironment([tool], trace, {"lookup": record})
            client = ScriptedClient(
                [
                    '{"thought":"record the answer call","function":"lookup",'
                    '"arguments":{"key":"alpha"}}',
                    '{"thought":"finish","function":"send_message",'
                    '"arguments":{"message":"done"}}',
                ]
            )
            context = RunContext(
                "memgpt",
                "retrieve alpha",
                client,
                environment,
                trace,
                {"max_turns": 2, "declaration_only_tools": True},
            )
            answer = asyncio.run(run_profile(context))

        self.assertEqual(answer, "done")
        self.assertEqual(environment.calls[0]["arguments"], {"key": "alpha"})
        system = client.messages[0][0]["content"]
        benchmark = json.loads(
            next(
                line.removeprefix("Benchmark functions: ")
                for line in system.splitlines()
                if line.startswith("Benchmark functions: ")
            )
        )
        parameters = benchmark[0]["parameters"]
        self.assertNotIn("request_heartbeat", parameters["properties"])
        self.assertNotIn("request_heartbeat", parameters["required"])
        self.assertIn("do not add request_heartbeat", system)
        self.assertIn(
            "Declaration-only tool call recorded",
            json.dumps(client.messages[1], ensure_ascii=False),
        )

    def test_memgpt_pause_heartbeats_does_not_require_immediate_heartbeat_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            context = RunContext(
                "memgpt",
                "pause timed heartbeats",
                ScriptedClient(
                    ['{"thought":"pause","function":"pause_heartbeats","arguments":{"minutes":5}}']
                ),
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"max_turns": 1},
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(answer, "")
        function_event = next(event for event in events if event["event"] == "memgpt_function")
        self.assertEqual(function_event["function"], "pause_heartbeats")
        self.assertIsNone(function_event["request_heartbeat"])
        yield_event = next(event for event in events if event["event"] == "memgpt_yield")
        self.assertEqual(yield_event["function"], "pause_heartbeats")

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

    def test_aflow_frozen_custom_preserves_plain_text_operator(self) -> None:
        from benchmark_platform.harnesses.aflow import make_artifact
        answer, environment = self.run_profile(
            "aflow",
            ["6"],
            policy={"aflow_artifact": make_artifact(), "aflow_allow_initialization": True},
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 0)
        self.assertEqual(self.last_context.llm_calls, 1)
        self.assertEqual(self.last_client.json_modes, [False])

    def test_dylan_published_text_network_has_no_hidden_tool_loop(self) -> None:
        answer, environment = self.run_profile("dylan", ["42"] * 5)
        self.assertEqual(answer, "42")
        self.assertEqual(environment.calls, [])
        prompts = [messages[0]["content"] for messages in self.last_client.messages]
        self.assertEqual(len(prompts), 5)
        self.assertTrue(all("AI assistant" in prompt for prompt in prompts))

    def test_dylan_preserves_complete_open_ended_candidate(self) -> None:
        answer, _ = self.run_profile("dylan", ["7, 9"] * 5)
        self.assertEqual(answer, "7, 9")

    def test_multi_persona_published_single_model_protocol(self) -> None:
        answer, environment = self.run_profile("multi-persona", ["Final answer: 42"])
        self.assertEqual(answer, "Final answer: 42")
        self.assertEqual(environment.calls, [])
        prompt = self.last_client.messages[0][0]["content"]
        self.assertNotIn("Structural example", prompt)
        self.assertIn("Here are two complete demonstrations", prompt)
        self.assertIn("Example Task 1", prompt)
        self.assertIn("Example Task 2", prompt)
        self.assertIn("Profiles:", prompt)
        self.assertGreaterEqual(prompt.count("Start collaboration!"), 2)
        self.assertIn("Finish collaboration!", prompt)

    def test_magentic_one_ledger_worker_and_delivery(self) -> None:
        answer, environment = self.run_profile(
            "magentic-one",
            [
                "Known facts",
                "Use the executor",
                magentic_ledger(
                    satisfied=False,
                    speaker="Executor",
                    instruction="look up alpha",
                ),
                native_tool_call("lookup", {"key": "alpha"}),
                magentic_ledger(satisfied=True),
                "6",
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 1)
        self.assertEqual(
            self.last_client.native_tools[0][0]["function"]["name"],
            "lookup",
        )

    def test_magentic_one_returns_to_ledger_after_each_participant_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    "Known facts",
                    "Use the executor",
                    magentic_ledger(
                        satisfied=False,
                        instruction="retrieve alpha",
                    ),
                    native_tool_call("lookup", {"key": "alpha"}),
                    magentic_ledger(
                        satisfied=False,
                        instruction="verify prior work",
                    ),
                    "verified alpha is 6",
                    magentic_ledger(satisfied=True),
                    "6",
                ]
            )
            context = RunContext(
                "magentic-one",
                "retrieve alpha",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"max_turns": 8},
            )
            answer = asyncio.run(run_profile(context))
            events = [
                json.loads(line)
                for line in trace.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(answer, "6")
        roles = [event.get("role") for event in events if event["event"] == "llm_request"]
        self.assertEqual(
            roles,
            [
                "orchestrator_facts",
                "orchestrator_plan",
                "orchestrator_ledger",
                "Executor",
                "orchestrator_ledger",
                "Executor",
                "orchestrator_ledger",
                "orchestrator_final",
            ],
        )
        second_ledger = client.messages[4]
        transcript = json.dumps(second_ledger, ensure_ascii=False)
        self.assertIn("lookup", transcript)
        self.assertIn("alpha", transcript)

    def test_magentic_invalid_speaker_retries_the_official_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(
                [
                    ToolSpec(
                        "run_command",
                        "run a command",
                        {"type": "object"},
                        ("/bin/false",),
                    )
                ],
                trace,
            )
            client = ScriptedClient(
                [
                    "facts",
                    "plan",
                    magentic_ledger(
                        satisfied=False,
                        speaker="UnknownWorker",
                        instruction="invalid dispatch",
                    ),
                    magentic_ledger(
                        satisfied=False,
                        speaker="Executor",
                        instruction="valid dispatch",
                    ),
                    "completed one participant turn",
                    magentic_ledger(satisfied=True),
                    "done",
                ]
            )
            context = RunContext(
                "magentic-one",
                "complete the task",
                client,
                environment,
                trace,
                {"magentic_json_retries": 2},
            )
            answer = asyncio.run(run_profile(context))
            events = [
                json.loads(line)
                for line in trace.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(answer, "done")
        retries = [event for event in events if event["event"] == "magentic_ledger_retry"]
        self.assertEqual(len(retries), 1)
        self.assertIn("invalid next speaker", retries[0]["error"])
        self.assertFalse(any(event["event"] == "magentic_unknown_worker" for event in events))

    def test_magentic_round_limit_synthesizes_instead_of_failing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    "facts",
                    "plan",
                    magentic_ledger(satisfied=False, instruction="first"),
                    "first result",
                    magentic_ledger(satisfied=False, instruction="second"),
                    "second result",
                    "bounded final answer",
                ]
            )
            context = RunContext(
                "magentic-one",
                "complete the task",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"magentic_max_rounds": 2},
            )
            answer = asyncio.run(run_profile(context))
            events = [
                json.loads(line)
                for line in trace.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(answer, "bounded final answer")
        termination = [event for event in events if event["event"] == "magentic_termination"]
        self.assertEqual(termination[-1]["reason"], "Max rounds reached.")
        self.assertEqual(
            sum(event["event"] == "magentic_dispatch" for event in events),
            2,
        )

    def test_magentic_replan_updates_ledgers_and_resets_message_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    "initial facts",
                    "initial plan",
                    magentic_ledger(
                        satisfied=False,
                        instruction="first attempt",
                        progress=False,
                        in_loop=True,
                    ),
                    "stalled participant result",
                    magentic_ledger(
                        satisfied=False,
                        instruction="would repeat",
                        progress=False,
                        in_loop=True,
                    ),
                    "updated facts",
                    "updated plan",
                    magentic_ledger(
                        satisfied=False,
                        instruction="new attempt",
                        progress=True,
                    ),
                    "new participant result",
                    magentic_ledger(satisfied=True),
                    "done",
                ]
            )
            context = RunContext(
                "magentic-one",
                "complete the task",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"magentic_max_rounds": 4, "magentic_max_stalls": 2},
            )
            answer = asyncio.run(run_profile(context))
            events = [
                json.loads(line)
                for line in trace.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(answer, "done")
        replans = [event for event in events if event["event"] == "magentic_replan"]
        self.assertEqual(len(replans), 1)
        self.assertEqual(replans[0]["facts"], "updated facts")
        self.assertEqual(replans[0]["plan"], "updated plan")
        post_replan_ledger = json.dumps(client.messages[7], ensure_ascii=False)
        self.assertIn("updated facts", post_replan_ledger)
        self.assertIn("updated plan", post_replan_ledger)
        self.assertNotIn("stalled participant result", post_replan_ledger)
        stall_states = [event for event in events if event["event"] == "magentic_stall_state"]
        self.assertEqual(
            [(item["previous"], item["current"]) for item in stall_states],
            [(0, 1), (1, 2), (2, 1)],
        )

    def test_magentic_invalid_native_arguments_become_a_tool_summary(self) -> None:
        malformed_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "bad-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{bad"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ScriptedClient(
                [
                    "facts",
                    "plan",
                    magentic_ledger(satisfied=False, instruction="look up alpha"),
                    malformed_call,
                    magentic_ledger(satisfied=True),
                    "done",
                ]
            )
            context = RunContext(
                "magentic-one",
                "complete the task",
                client,
                environment,
                trace,
                {},
            )
            answer = asyncio.run(run_profile(context))

        self.assertEqual(answer, "done")
        self.assertEqual(environment.calls, [])
        second_ledger = json.dumps(client.messages[4], ensure_ascii=False)
        self.assertIn("invalid_tool_arguments_json", second_ledger)

    def test_magentic_rejects_mixed_user_message_batch_before_side_effects(self) -> None:
        async def handler(arguments):
            return {"ok": True, "result": arguments}

        message_schema = {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        }
        mixed_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "message-call",
                    "type": "function",
                    "function": {
                        "name": "send_message_to_user",
                        "arguments": '{"content":"question"}',
                    },
                },
                {
                    "id": "lookup-call",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key":"alpha"}',
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            tools = [
                ToolSpec(
                    "send_message_to_user",
                    "send a visible message",
                    message_schema,
                    ("/bin/false",),
                ),
                tool_specs()[0],
            ]
            environment = ToolEnvironment(
                tools,
                trace,
                {"send_message_to_user": handler, "lookup": handler},
            )
            client = ScriptedClient(
                [
                    "facts",
                    "plan",
                    magentic_ledger(satisfied=False, instruction="ask and look up"),
                    mixed_call,
                    magentic_ledger(satisfied=True),
                    "done",
                ]
            )
            context = RunContext(
                "magentic-one",
                "complete the task",
                client,
                environment,
                trace,
                {},
            )
            answer = asyncio.run(run_profile(context))

        self.assertEqual(answer, "done")
        self.assertEqual(environment.calls, [])
        second_ledger = json.dumps(client.messages[4], ensure_ascii=False)
        self.assertIn("must be the only tool call", second_ledger)

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
                '{"action":"finish","answer":"42"}',
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(environment.calls[-1]["result"]["result"]["product"], 42)
        self.assertEqual(self.last_client.json_modes, [True, True])
        for messages, json_mode in zip(
            self.last_client.messages, self.last_client.json_modes, strict=True
        ):
            if json_mode:
                prompt = "\n".join(str(message.get("content", "")) for message in messages)
                self.assertIn("json", prompt.casefold())

    def test_llmcompiler_scans_past_incidental_json_for_the_plan(self) -> None:
        answer, environment = self.run_profile(
            "llmcompiler",
            [
                '{"query":"retrieve alpha"}\n'
                '{"tasks":[{"id":"1","tool":"lookup","arguments":{"key":"alpha"},"dependencies":[]}]}',
                '{"action":"finish","answer":"6"}',
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual([call["name"] for call in environment.calls], ["lookup"])

    def test_llmcompiler_joiner_can_replan_from_failed_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ScriptedClient(
                [
                    '{"tasks":[{"id":"1","tool":"missing","arguments":{},"dependencies":[]}]}',
                    '{"action":"replan","feedback":"use the declared lookup tool"}',
                    '{"tasks":[{"id":"1","tool":"lookup","arguments":{"key":"alpha"},"dependencies":[]}]}',
                    '{"action":"finish","answer":"6"}',
                ]
            )
            context = RunContext(
                "llmcompiler",
                "retrieve alpha",
                client,
                environment,
                trace,
                {"max_turns": 8, "llmcompiler_max_replans": 2},
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text().splitlines()]

        self.assertEqual(answer, "6")
        self.assertEqual([call["name"] for call in environment.calls], ["missing", "lookup"])
        self.assertIn("use the declared lookup tool", client.messages[2][0]["content"])
        self.assertEqual(
            [event["cycle"] for event in events if event["event"] == "llmcompiler_join"],
            [0, 1],
        )

    def test_llmcompiler_final_pass_forces_a_replan_payload_to_finish(self) -> None:
        answer, _ = self.run_profile(
            "llmcompiler",
            [
                '{"tasks":[]}',
                '{"action":"replan","feedback":"best available answer"}',
            ],
        )
        self.assertEqual(answer, "best available answer")
        final_prompt = self.last_client.messages[-1][0]["content"]
        self.assertIn("final planning pass", final_prompt)
        self.assertIn("Do not request another replan", final_prompt)

    def test_rewoo_plan_evidence_solver(self) -> None:
        answer, environment = self.run_profile(
            "rewoo",
            [
                "\n".join(
                    [
                        "Plan: retrieve alpha",
                        '#E1 = lookup[{"key":"alpha"}]',
                        "Plan: retrieve beta",
                        '#E2 = lookup[{"key":"beta"}]',
                        "Plan: multiply the retrieved values",
                        '#E3 = multiply[{"a":#E1.value,"b":#E2.value}]',
                    ]
                ),
                "42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(len(environment.calls), 3)
        self.assertEqual(environment.calls[-1]["arguments"], {"a": 6, "b": 7})
        self.assertEqual(environment.calls[-1]["result"]["result"]["product"], 42)

    def test_rewoo_llm_worker_is_an_explicit_work_phase(self) -> None:
        answer, environment = self.run_profile(
            "rewoo",
            [
                "Plan: derive the evidence directly\n#E1 = LLM[Return the product of six and seven]",
                "42",
                "42",
            ],
        )
        self.assertEqual(answer, "42")
        self.assertEqual(environment.calls, [])

    def test_rewoo_balanced_parser_preserves_nested_worker_input(self) -> None:
        steps = parse_rewoo_plan(
            'Plan: inspect nested content\n#E1 = LLM[Compare [alpha] with {"literal": "]"}]'
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].worker, "LLM")
        self.assertEqual(steps[0].worker_input, 'Compare [alpha] with {"literal": "]"}')

    def test_rewoo_worker_failure_is_visible_to_solver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ScriptedClient(
                [
                    "Plan: request unavailable evidence\n#E1 = MissingWorker[anything]",
                    "cannot determine",
                ]
            )
            context = RunContext(
                "rewoo",
                "answer from the available evidence",
                client,
                environment,
                trace,
                {"max_turns": 8},
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(answer, "cannot determine")
        self.assertEqual(environment.calls, [])
        failures = [event for event in events if event.get("event") == "rewoo_worker_result"]
        self.assertEqual(len(failures), 1)
        self.assertFalse(failures[0]["ok"])
        self.assertEqual(failures[0]["output"]["error"], "unknown_worker")
        self.assertIn("unknown_worker", client.messages[-1][0]["content"])

    def test_rewoo_llm_worker_transforms_interpolated_evidence_for_later_tools(
        self,
    ) -> None:
        answer, environment = self.run_profile(
            "rewoo",
            [
                "\n".join(
                    [
                        "Plan: retrieve alpha",
                        '#E1 = lookup[{"key":"alpha"}]',
                        "Plan: transform the first result",
                        "#E2 = LLM[Return the word beta only. The alpha evidence was #E1.value.]",
                        "Plan: retrieve the transformed key",
                        '#E3 = lookup[{"key":#E2}]',
                    ]
                ),
                "beta",
                "42",
            ],
        )

        self.assertEqual(answer, "42")
        self.assertEqual(
            [call["arguments"] for call in environment.calls],
            [{"key": "alpha"}, {"key": "beta"}],
        )
        self.assertIn(
            "alpha evidence was 6", self.last_client.messages[1][-1]["content"]
        )
        planner_prompt = self.last_client.messages[0][0]["content"]
        self.assertIn("LLM[plain-text instruction]", planner_prompt)
        self.assertIn("#E1 = Worker[input]", planner_prompt)

    def test_rewoo_rejects_concatenated_planner_drafts(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected E2, got E1"):
            parse_rewoo_plan(
                "Plan: retrieve alpha\n"
                '#E1 = lookup[{"key":"alpha"}]\n'
                "Plan: retrieve beta\n"
                '#E1 = lookup[{"key":"beta"}]'
            )

    def test_rewoo_repairs_concatenated_planner_drafts_before_execution(self) -> None:
        malformed = (
            "Plan: retrieve alpha\n"
            '#E1 = lookup[{"key":"alpha"}]'
            "Plan: retrieve alpha again\n"
            '#E1 = lookup[{"key":"alpha"}]'
        )
        answer, environment = self.run_profile(
            "rewoo",
            [
                malformed,
                'Plan: retrieve alpha\n#E1 = lookup[{"key":"alpha"}]',
                "6",
            ],
        )

        self.assertEqual(answer, "6")
        self.assertEqual([call["name"] for call in environment.calls], ["lookup"])
        self.assertEqual(len(self.last_client.messages), 3)
        repair_prompt = self.last_client.messages[1][-1]["content"]
        self.assertIn("Protocol error:", repair_prompt)
        self.assertIn("Do not include commentary", repair_prompt)

    def test_sa_exact_action_cache_hit(self) -> None:
        answer, environment = self.run_profile(
            "sa",
            [
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"final":"6"}',
            ],
            speculator_responses=[
                '{"actions":[{"tool":"lookup","arguments":{"key":"alpha"}}]}',
                '{"actions":[]}',
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 1)
        self.assertEqual(len(self.last_speculator_client.messages), 2)
        actor_response = next(
            event["response_id"]
            for event in self.last_events
            if event.get("event") == "llm_response" and event.get("role") == "sa_actor"
        )
        speculator_response = next(
            event["response_id"]
            for event in self.last_events
            if event.get("event") == "llm_response"
            and event.get("role") == "sa_speculator"
        )
        self.assertEqual(environment.calls[0]["assistant_response_id"], actor_response)
        self.assertNotEqual(environment.calls[0]["assistant_response_id"], speculator_response)

    def test_sa_actor_and_independent_speculator_are_in_flight_together(self) -> None:
        gate = asyncio.Event()
        started = 0

        class GatedClient(ScriptedClient):
            async def complete(self, messages, *, temperature=None, json_mode=False):
                nonlocal started
                self.messages.append(messages)
                self.json_modes.append(json_mode)
                started += 1
                if started == 2:
                    gate.set()
                await asyncio.wait_for(gate.wait(), timeout=1)
                content = next(self.responses)
                return Completion(
                    content,
                    1,
                    1,
                    0.0,
                    0,
                    {"choices": [{"message": {"content": content}}]},
                )

        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            actor = GatedClient(['{"final":"6"}'])
            speculator = GatedClient(['{"actions":[]}'])
            context = RunContext(
                "sa",
                "retrieve alpha",
                actor,
                environment,
                trace,
                {"max_turns": 2},
                speculator_client=speculator,
            )
            answer = asyncio.run(run_profile(context))

        self.assertEqual(answer, "6")
        self.assertEqual(started, 2)
        self.assertEqual(context.actor_llm_calls, 1)
        self.assertEqual(context.speculator_llm_calls, 1)

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
                    '{"tool":"read_balance","arguments":{}}',
                    '{"final":"2"}',
                ]
            )
            speculator_client = ScriptedClient(
                [
                    '{"actions":[{"tool":"read_balance","arguments":{}}]}',
                    '{"actions":[{"tool":"read_balance","arguments":{}}]}',
                    '{"actions":[]}',
                ]
            )
            context = RunContext(
                "sa",
                "set the balance to 2, then read it",
                client,
                environment,
                trace,
                {"max_turns": 5},
                speculator_client=speculator_client,
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text().splitlines()]

        self.assertEqual(answer, "2")
        self.assertEqual(state["balance"], 2)
        self.assertEqual([call["name"] for call in environment.calls], ["set_balance", "read_balance"])
        self.assertEqual(environment.calls[-1]["result"]["result"]["balance"], 2)
        self.assertTrue(
            any(event["event"] in {"sa_cache_invalidated", "sa_cache_stale"} for event in events)
        )

    def test_sa_speculative_miss_is_not_an_authoritative_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ScriptedClient(
                [
                    '{"tool":"lookup","arguments":{"key":"alpha"}}',
                    '{"final":"6"}',
                ]
            )
            speculator_client = ScriptedClient(
                [
                    '{"actions":[{"tool":"lookup","arguments":{"key":"beta"}}]}',
                    '{"actions":[]}',
                ]
            )
            context = RunContext(
                "sa",
                "retrieve alpha",
                client,
                environment,
                trace,
                {"max_turns": 4},
                speculator_client=speculator_client,
            )
            answer = asyncio.run(run_profile(context))
            events = [json.loads(line) for line in trace.path.read_text().splitlines()]

        self.assertEqual(answer, "6")
        self.assertEqual([call["arguments"] for call in environment.calls], [{"key": "alpha"}])
        self.assertTrue(any(event["event"] == "sa_speculative_tool_request" for event in events))
        self.assertFalse(
            any(
                event["event"] == "tool_request"
                and event.get("arguments") == {"key": "beta"}
                for event in events
            )
        )
        self.assertEqual(context.actor_llm_calls, 2)
        self.assertEqual(context.speculator_llm_calls, 2)
        self.assertEqual(context.prompt_tokens, 4)
        self.assertEqual(context.actor_prompt_tokens, 2)
        self.assertEqual(context.speculator_prompt_tokens, 2)

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

    def test_json_parser_uses_first_action_before_non_json_trailing_text(self) -> None:
        action = {"tool": "lookup", "arguments": {}}
        self.assertEqual(
            extract_json('{"tool":"lookup","arguments":{}} and then continue'),
            action,
        )

    def test_json_parser_never_skips_a_tool_for_a_later_final(self) -> None:
        response = (
            '<think>use the lookup first</think>\n'
            '{"tool":"lookup","arguments":{"key":"alpha"}}\n'
            '<think>the task should then be complete</think>\n'
            '{"final":"fabricated without an observation"}'
        )
        self.assertEqual(
            extract_json(response, expected_type=dict),
            {"tool": "lookup", "arguments": {"key": "alpha"}},
        )

    def test_tool_loop_executes_first_mixed_tool_before_accepting_a_later_turn_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            environment = ToolEnvironment(tool_specs(), trace)
            client = ScriptedClient(
                [
                    (
                        '<think>use the lookup first</think>\n'
                        '{"tool":"lookup","arguments":{"key":"alpha"}}\n'
                        '<think>then answer</think>\n'
                        '{"final":"fabricated"}'
                    ),
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

        self.assertEqual(answer, "6")
        self.assertEqual([call["arguments"] for call in environment.calls], [{"key": "alpha"}])
        self.assertFalse(
            any("Protocol error" in str(message.get("content")) for messages in client.messages for message in messages)
        )

    def test_structured_manager_repairs_mixed_plan_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    '{"assignments":[{"id":"w1","instruction":"inspect"}]}\n'
                    '<think>skip execution</think>\n'
                    '{"final":"fabricated"}',
                ]
            )
            context = RunContext(
                "cmas",
                "inspect",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"protocol_repairs": 1},
            )
            plan = asyncio.run(context.complete_json("manager", [{"role": "user", "content": "plan"}]))

        self.assertEqual(plan["assignments"][0]["id"], "w1")
        self.assertEqual(len(client.messages), 1)

    def test_strict_json_accepts_one_reasoning_wrapped_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                ['<think>check the DAG</think> {"action":"replan","feedback":"fix it"}']
            )
            context = RunContext(
                "llmcompiler",
                "compile",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"protocol_repairs": 0},
            )
            decision = asyncio.run(
                context.complete_json(
                    "joiner",
                    [{"role": "user", "content": "join"}],
                    required_root_key="action",
                    strict_single_object=True,
                )
            )

        self.assertEqual(decision["action"], "replan")

    def test_strict_json_rejects_two_reasoning_wrapped_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    '<think>drafts</think> {"action":"replan","feedback":"one"}'
                    '<think>second</think> {"action":"finish","answer":"two"}'
                ]
            )
            context = RunContext(
                "llmcompiler",
                "compile",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {"protocol_repairs": 0},
            )
            with self.assertRaisesRegex(ValueError, "multiple complete JSON objects"):
                asyncio.run(
                    context.complete_json(
                        "joiner",
                        [{"role": "user", "content": "join"}],
                        required_root_key="action",
                        strict_single_object=True,
                    )
                )

    def test_json_parser_preserves_large_complete_value(self) -> None:
        value = {"text": "x" * 200_000, "tail": [1, 2, 3]}
        self.assertEqual(extract_json(json.dumps(value)), value)

    def test_json_parser_accepts_one_complete_fenced_object(self) -> None:
        value = {"tool": "lookup", "arguments": {"key": "alpha"}}
        self.assertEqual(extract_json(f"```json\n{json.dumps(value)}\n```"), value)

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
