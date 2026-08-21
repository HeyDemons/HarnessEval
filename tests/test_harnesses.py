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
from benchmark_platform.harnesses.rewoo import parse_rewoo_plan


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
            self.last_client = context.client
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

    def test_magentic_team_is_staffed_from_the_tools_that_exist(self) -> None:
        """Upstream fills the orchestrator's team from the participants the caller assembled.
        Hardcoding all four roles sent 19 of 32 terminal-bench dispatches to a web_surfer with
        no retrieval tool behind it -- two calls at reasoning_effort=high each, for a worker
        holding the same toolset as everyone else. GAIA keeps the full roster, so its recorded
        arms stay comparable."""
        from benchmark_platform.harnesses.paper_methods import _magentic_team

        self.assertEqual(
            list(_magentic_team({"list_files", "read_file", "run_command", "web_search"})),
            ["web_surfer", "file_surfer", "coder", "executor"],
        )
        self.assertEqual(
            list(_magentic_team({"read_file", "list_files", "write_file", "run_command"})),
            ["file_surfer", "coder", "executor"],
        )
        # Domain-action benchmarks staff the catch-all alone rather than an empty team.
        self.assertEqual(list(_magentic_team({"get_order_details", "calculate"})), ["executor"])

    def test_magentic_off_roster_speaker_falls_back_instead_of_killing_the_arm(self) -> None:
        """A smaller roster makes the model likelier to name the canonical team anyway, and
        `selected unknown worker` used to end the arm with score None."""
        answer, environment = self.run_profile(
            "magentic-one",
            [
                "facts",
                "plan",
                '{"satisfied":false,"in_loop":false,"progress":true,"next_speaker":"web_surfer",'
                '"instruction":"look it up"}',
                '{"report":"no retrieval tool here"}',
                '{"satisfied":true,"in_loop":false,"progress":true,"next_speaker":"executor",'
                '"instruction":"finish"}',
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
        answer, _ = self.run_profile("sa", ['{"predicted":"nothing"}', '{"final":"42"}'])
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

    def test_lats_limits_proposal_parallelism(self) -> None:
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
                    "lats_max_parallel": 2,
                    "lats_max_llm_calls": 10,
                },
            )
            asyncio.run(run_profile(context))

        self.assertEqual(client.max_active, 2)
        self.assertEqual(context.llm_calls, 10)

    def test_lats_reserves_sampling_wave_before_spending_call_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient([])
            context = RunContext(
                "lats",
                "do not start a partial wave",
                client,
                ToolEnvironment(tool_specs(), trace),
                trace,
                {
                    "lats_iterations": 1,
                    "lats_generate_samples": 2,
                    "lats_value_samples": 1,
                    "lats_rollout_width": 1,
                    "lats_tree_depth": 1,
                    "lats_rollout_depth": 1,
                    "lats_max_parallel": 1,
                    "lats_max_llm_calls": 1,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "no terminal answer within 0/1"):
                asyncio.run(run_profile(context))

        self.assertEqual(context.llm_calls, 0)
        self.assertEqual(client.messages, [])

    def test_lats_stops_dispatching_at_the_total_llm_call_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    '{"thought":"alpha","tool":"lookup","arguments":{"key":"alpha"}}',
                    '{"thought":"beta","tool":"lookup","arguments":{"key":"beta"}}',
                    '{"score":0.8,"success":false,"feedback":"continue"}',
                    '{"score":0.7,"success":false,"feedback":"continue"}',
                ]
            )
            environment = ToolEnvironment(tool_specs(), trace)
            context = RunContext(
                "lats",
                "stay within budget",
                client,
                environment,
                trace,
                {
                    "lats_iterations": 3,
                    "lats_generate_samples": 2,
                    "lats_value_samples": 1,
                    "lats_rollout_width": 1,
                    "lats_tree_depth": 3,
                    "lats_rollout_depth": 3,
                    "lats_max_parallel": 1,
                    "lats_max_llm_calls": 4,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "no terminal answer within 4/4"):
                asyncio.run(run_profile(context))
            events = [
                json.loads(line)
                for line in trace.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(context.llm_calls, 4)
        self.assertEqual(len(client.messages), 4)
        self.assertEqual(environment.calls, [])
        self.assertEqual(
            len([event for event in events if event["event"] == "llm_request"]), 4
        )
        self.assertEqual(
            len([event for event in events if event["event"] == "lats_budget_exhausted"]),
            1,
        )

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

    def test_aflow_custom_initialization_control(self) -> None:
        answer, environment = self.run_profile(
            "aflow-custom-init",
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
        prompts = [messages[0]["content"] for messages in self.last_client.messages]
        self.assertEqual(len(set(prompts)), 3)
        self.assertTrue(any("Logical Solver" in prompt for prompt in prompts))
        self.assertTrue(any("Critical Reviewer" in prompt for prompt in prompts))
        self.assertTrue(any("Alternative Solver" in prompt for prompt in prompts))

    def test_dylan_preserves_complete_open_ended_candidate(self) -> None:
        answer, _ = self.run_profile("dylan", ["7, 9", "7, 9", "7, 9"])
        self.assertEqual(answer, "7, 9")

    def test_multi_persona_published_single_model_protocol(self) -> None:
        answer, environment = self.run_profile("multi-persona", ["Final answer: 42"])
        self.assertEqual(answer, "Final answer: 42")
        self.assertEqual(environment.calls, [])
        prompt = self.last_client.messages[0][0]["content"]
        self.assertIn("Structural example", prompt)
        self.assertIn("Skeptical Verifier", prompt)
        self.assertIn("Finish collaboration!", prompt)

    def test_magentic_one_ledger_worker_and_delivery(self) -> None:
        answer, environment = self.run_profile(
            "magentic-one",
            [
                "Known facts",
                "Use the executor",
                '{"satisfied":false,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"look up alpha"}',
                '{"tool":"lookup","arguments":{"key":"alpha"}}',
                '{"report":"alpha is 6"}',
                '{"satisfied":true,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"deliver"}',
                "6",
            ],
        )
        self.assertEqual(answer, "6")
        self.assertEqual(len(environment.calls), 1)

    def test_magentic_one_worker_keeps_private_and_group_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = JsonlTrace(Path(directory) / "trace.jsonl")
            client = ScriptedClient(
                [
                    "Known facts",
                    "Use the executor",
                    '{"satisfied":false,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"retrieve alpha"}',
                    '{"tool":"lookup","arguments":{"key":"alpha"}}',
                    '{"report":"alpha is 6"}',
                    '{"satisfied":false,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"verify prior work"}',
                    '{"report":"verified alpha is 6"}',
                    '{"satisfied":true,"in_loop":false,"progress":true,"next_speaker":"executor","instruction":"finish"}',
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

        self.assertEqual(answer, "6")
        second_dispatch = client.messages[6][0]["content"]
        self.assertIn("alpha is 6", second_dispatch)
        self.assertIn("Your private history", second_dispatch)
        self.assertIn("Group message thread", second_dispatch)

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
                {"max_turns": 8, "llmcompiler_max_replans": 1},
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
                    '{"actions":[{"tool":"lookup","arguments":{"key":"beta"}}]}',
                    '{"final":"6"}',
                ]
            )
            context = RunContext(
                "sa", "retrieve alpha", client, environment, trace, {"max_turns": 4}
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
