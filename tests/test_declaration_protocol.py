import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_platform.bridges import runner
from benchmark_platform.budgets import baseline_limits
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.declaration import (
    MULTI_MODEL_PROTOCOL,
    PUBLISHER_PROTOCOL,
    SELECTED_ACTION_CHAIN_PROTOCOL,
    complete_native_declaration,
    declaration_messages,
    parse_native_declarations,
)
from test_bridges import make_case, write_json


class Trace:
    def __init__(self):
        self.events = []

    async def emit(self, event, **data):
        self.events.append({"event": event, **data})


def native_batch(*ids: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {
                    "name": "lookup_item",
                    "arguments": json.dumps({"id": value}),
                },
            }
            for index, value in enumerate(ids, 1)
        ],
    }


class Client:
    def __init__(self, responses):
        self.responses = iter(responses if isinstance(responses, list) else [responses])
        self.requests = []

    @staticmethod
    def _completion(response):
        message = (
            response
            if isinstance(response, dict)
            else {"role": "assistant", "content": str(response)}
        )
        return Completion(
            str(message.get("content") or ""),
            10,
            5,
            0,
            0,
            {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ]
            },
        )

    async def complete(self, messages, **kwargs):
        self.requests.append({"messages": messages, "tools": [], **kwargs})
        return self._completion(next(self.responses))

    async def complete_native(self, messages, **kwargs):
        self.requests.append({"messages": messages, **kwargs})
        return self._completion(next(self.responses))


class DeclarationTests(unittest.IsolatedAsyncioTestCase):
    async def test_actor_runs_complete_native_tool_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            client = Client([
                native_batch("a"),
                native_batch("b"),
                "done",
            ])
            with patch.object(runner, "completion_client_from_env", return_value=client):
                result = await runner.execute(
                    "bfcl", "actor-only", "case", source, job, baseline_limits("bfcl")
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["agent_turns"], 3)
        self.assertEqual(result["internal_llm_calls"], 3)
        self.assertEqual(result["publisher_llm_calls"], 0)
        self.assertEqual(result["declaration_protocol"], PUBLISHER_PROTOCOL)
        self.assertEqual(result["declaration_output_protocol"], SELECTED_ACTION_CHAIN_PROTOCOL)
        self.assertEqual(result["source_response_ids"], [1, 2])
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(
            [message["role"] for message in client.requests[0]["messages"]],
            ["system", "user"],
        )
        self.assertEqual(client.requests[0]["messages"][-1], {"role": "user", "content": "Call the function"})
        native_tools = client.requests[0]["tools"]
        self.assertEqual(native_tools[0]["function"]["name"], "lookup_item")
        self.assertNotIn("parallel", native_tools[0]["function"])
        self.assertNotIn("read_only", native_tools[0]["function"])
        self.assertEqual(
            [(call["name"], call["arguments"]) for call in result["committed_calls"]],
            [("lookup_item", {"id": "a"}), ("lookup_item", {"id": "b"})],
        )

    async def test_source_system_message_is_not_flattened_or_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            write_json(
                source / "case.json",
                {
                    "prompt": "legacy rendering must not be used",
                    "messages": [
                        {"role": "system", "content": "official system"},
                        {"role": "user", "content": "official user"},
                    ],
                    "functions": [
                        {
                            "name": "lookup_item",
                            "description": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                },
            )
            client = Client('{"final":"not relevant"}')
            with patch.object(runner, "completion_client_from_env", return_value=client):
                result = await runner.execute("bfcl", "actor-only", "case", source, job, {})

        self.assertEqual(result["status"], "completed")
        messages = client.requests[0]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "system", "user"])
        self.assertEqual(sum(message["content"] == "official system" for message in messages), 1)
        self.assertEqual(messages[-1], {"role": "user", "content": "official user"})

    async def test_no_native_calls_publishes_an_atomic_empty_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            client = Client('{"final":"No relevant function."}')
            with patch.object(runner, "completion_client_from_env", return_value=client):
                result = await runner.execute("bfcl", "actor-only", "case", source, job, {})

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["committed_calls"], [])
        self.assertEqual(result["external_assistant_responses"], 1)
        self.assertEqual(result["environment_calls"], 0)

    async def test_internal_proposals_are_not_merged_into_final_batch(self):
        async def method(ctx):
            await ctx.complete("planner", [{"role": "user", "content": "plan"}])
            await ctx.environment.call("lookup_item", {"id": "proposal"})
            return await complete_native_declaration(
                ctx,
                role="existing_final_node",
                messages=declaration_messages(ctx, internal_context="finish"),
                protocol=MULTI_MODEL_PROTOCOL,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            client = Client(["planned", native_batch("final-a", "final-a")])
            with (
                patch.object(runner, "completion_client_from_env", return_value=client),
                patch.object(runner, "run_profile", new=method),
            ):
                result = await runner.execute("bfcl", "llmcompiler", "case", source, job, {})

        self.assertEqual(result["internal_llm_calls"], 2)
        self.assertEqual(result["proposal_tool_calls"], 1)
        self.assertEqual(result["proposal_calls"][0]["arguments"], {"id": "proposal"})
        self.assertEqual(
            [call["arguments"]["id"] for call in result["committed_calls"]],
            ["final-a", "final-a"],
        )
        self.assertEqual(result["source_response_ids"], [2])
        self.assertEqual(result["declaration_output_protocol"], MULTI_MODEL_PROTOCOL)

    async def test_react_runs_full_loop_and_publishes_selected_action_chain(self):
        responses = [
            native_batch("a"),
            native_batch("b"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "finish",
                    "type": "function",
                    "function": {
                        "name": "react_finish",
                        "arguments": json.dumps({"answer": "done"}),
                    },
                }],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            with patch.object(runner, "completion_client_from_env", return_value=Client(responses)):
                result = await runner.execute("bfcl", "react", "case", source, job, {})

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["agent_turns"], 3)
        self.assertEqual(result["source_response_ids"], [1, 2])
        self.assertEqual(result["declaration_output_protocol"], SELECTED_ACTION_CHAIN_PROTOCOL)
        self.assertEqual(
            [call["arguments"]["id"] for call in result["committed_calls"]],
            ["a", "b"],
        )

    async def test_sa_runs_full_speculative_loop_and_publishes_only_actor_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            actor = Client([
                native_batch("a"),
                native_batch("b"),
                "done",
            ])
            speculator = Client([
                '{"actions":[{"tool":"lookup_item","arguments":{"id":"a"}}]}',
                '{"actions":[{"tool":"lookup_item","arguments":{"id":"b"}}]}',
                '{"actions":[{"tool":"lookup_item","arguments":{"id":"wrong"}}]}',
            ])
            with (
                patch.object(runner, "completion_client_from_env", return_value=actor),
                patch.object(runner, "sa_speculator_client_from_env", return_value=speculator),
            ):
                result = await runner.execute(
                    "bfcl", "sa", "case", source, job, baseline_limits("bfcl")
                )
            events = [
                json.loads(line)
                for line in (job / "harness_trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["actor_llm_calls"], 3)
        self.assertEqual(result["speculator_llm_calls"], 3)
        self.assertEqual(result["internal_llm_calls"], 6)
        self.assertEqual(len(result["proposal_calls"]), 2)
        self.assertEqual(result["declaration_output_protocol"], SELECTED_ACTION_CHAIN_PROTOCOL)
        self.assertEqual(
            [call["arguments"]["id"] for call in result["committed_calls"]],
            ["a", "b"],
        )
        self.assertEqual(sum(event["event"] == "sa_cache_hit" for event in events), 2)
        self.assertTrue(any(event["event"] == "sa_predictions_discarded" for event in events))

    async def test_memgpt_runs_processor_heartbeat_loop_before_publication(self):
        responses = [
            '{"thought":"first","function":"lookup_item","arguments":{"id":"a"}}',
            '{"thought":"second","function":"lookup_item","arguments":{"id":"b"}}',
            '{"thought":"done","function":"send_message","arguments":{"message":"complete"}}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            with patch.object(runner, "completion_client_from_env", return_value=Client(responses)):
                result = await runner.execute("bfcl", "memgpt", "case", source, job, {})
            events = [
                json.loads(line)
                for line in (job / "harness_trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["agent_turns"], 3)
        self.assertEqual(result["source_response_ids"], [1, 2])
        self.assertEqual(result["declaration_output_protocol"], SELECTED_ACTION_CHAIN_PROTOCOL)
        self.assertEqual(
            [call["arguments"]["id"] for call in result["committed_calls"]],
            ["a", "b"],
        )
        self.assertEqual(sum(event["event"] == "memgpt_function" for event in events), 2)

    async def test_dmas_split_selects_only_the_terminal_executor_candidate(self):
        responses = [
            '{"requirements":{"reasoning":1.0}}',
            json.dumps({
                "decision": "split",
                "reason": "two parts",
                "next_agent_id": None,
                "executable": "prepare the first call",
                "remaining": "prepare the second call",
                "description": None,
            }),
            "reason about the first subtask",
            native_batch("first"),
            json.dumps({
                "status": "incompleted",
                "reason": "second call remains",
                "next_agent_id": "1",
                "remaining": "prepare the second call",
            }),
            json.dumps({
                "decision": "execute",
                "reason": "finish",
                "next_agent_id": None,
                "executable": None,
                "remaining": None,
                "description": "combine the peer candidate with the remaining call",
            }),
            "reason about the terminal batch",
            native_batch("second"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            client = Client(responses)
            with patch.object(runner, "completion_client_from_env", return_value=client):
                result = await runner.execute("bfcl", "dmas", "case", source, job, {})
            events = [
                json.loads(line)
                for line in (job / "harness_trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["internal_llm_calls"], 8)
        self.assertEqual(result["source_response_ids"], [4, 8])
        self.assertEqual(
            [call["arguments"]["id"] for call in result["committed_calls"]],
            ["first", "second"],
        )
        self.assertEqual(result["proposal_calls"], [])
        self.assertEqual(
            sum(event["event"] == "declaration_candidate_response" for event in events),
            2,
        )
        self.assertEqual(
            sum(event["event"] == "method_declaration_response" for event in events),
            1,
        )
        self.assertEqual(
            sum(event["event"] == "declaration_outputs_aggregated" for event in events),
            1,
        )

    def test_native_parser_preserves_duplicates(self):
        completion = Client._completion(native_batch("a", "a"))
        parsed = parse_native_declarations(completion)
        self.assertEqual(
            [arguments["id"] for _name, arguments, _id in parsed],
            ["a", "a"],
        )

    async def test_malformed_native_arguments_commit_nothing(self):
        malformed = native_batch("a")
        malformed["tool_calls"][0]["function"]["arguments"] = "{bad"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, job = root / "input", root / "job"
            source.mkdir()
            job.mkdir()
            make_case(source, "bfcl")
            with patch.object(runner, "completion_client_from_env", return_value=Client(malformed)):
                result = await runner.execute("bfcl", "actor-only", "case", source, job, {})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["committed_calls"], [])
        self.assertEqual(result["external_assistant_responses"], 0)
