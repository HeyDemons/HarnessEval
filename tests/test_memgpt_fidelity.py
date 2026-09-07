"""MemGPT must preserve failed-function heartbeats across the tool bridge."""
import json
import unittest

from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile


class Trace:
    def __init__(self):
        self.events = []

    async def emit(self, event, **data):
        self.events.append({"event": event, **data})


class Client:
    def __init__(self, first):
        self.responses = iter([first, {"thought": "recover", "function": "send_message", "arguments": {"message": "recovered"}}])
        self.messages = []

    async def complete(self, messages, **kwargs):
        self.messages.append(list(messages))
        return Completion(json.dumps(next(self.responses)), 1, 1, 0, 0, {})


class MemGPTFidelityTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_pressure_uses_last_response_input_plus_output(self):
        class UsageClient:
            def __init__(self, usages):
                self.usages = iter(usages)
                self.messages = []
            async def complete(self, messages, **kwargs):
                self.messages.append(messages)
                prompt, output = next(self.usages)
                value = ({'thought': 'save', 'function': 'core_memory_append', 'arguments':
                          {'name': 'human', 'content': 'fact', 'request_heartbeat': True}}
                         if len(self.messages) == 1 else
                         {'thought': 'done', 'function': 'send_message', 'arguments': {'message': 'done'}})
                return Completion(json.dumps(value), prompt, output, 0, 0, {})
        trace = Trace()
        client = UsageClient([(40, 70), (1, 1)])
        ctx = RunContext('memgpt', 'task', client, ToolEnvironment([], trace), trace,
                         {'memgpt_memory_warning_tokens': 100})
        await run_profile(ctx)
        warning = next(e for e in trace.events if e['event'] == 'memgpt_memory_pressure')
        self.assertEqual(warning['total_tokens'], 110)
        self.assertIn('Warning: the conversation history', str(client.messages[1]))

    async def test_memory_pressure_does_not_sum_json_repair_attempts(self):
        class UsageClient:
            def __init__(self):
                self.index = 0
            async def complete(self, messages, **kwargs):
                self.index += 1
                values = [{}, {'thought': 'save', 'function': 'core_memory_append', 'arguments':
                              {'name': 'human', 'content': 'fact', 'request_heartbeat': True}},
                          {'thought': 'done', 'function': 'send_message', 'arguments': {'message': 'done'}}]
                return Completion(json.dumps(values[self.index - 1]), 60, 10, 0, 0, {})
        trace = Trace()
        ctx = RunContext('memgpt', 'task', UsageClient(), ToolEnvironment([], trace), trace,
                         {'memgpt_memory_warning_tokens': 100})
        await run_profile(ctx)
        self.assertFalse(any(e['event'] == 'memgpt_memory_pressure' for e in trace.events))

    async def test_failed_tool_forces_recovery_with_false_or_missing_heartbeat(self):
        async def raises(args):
            raise ValueError("synthetic failure")

        async def packaged(args):
            return {"ok": False, "error": "tool_process_failed"}

        async def successful_business_result(args):
            return {"ok": True, "result": {"ok": False, "reason": "no matching record"}}

        for handler in (raises, packaged, successful_business_result):
            for heartbeat in (False, None):
                with self.subTest(handler=handler.__name__, heartbeat=heartbeat):
                    arguments = {} if heartbeat is None else {"request_heartbeat": heartbeat}
                    trace = Trace()
                    env = ToolEnvironment([ToolSpec("read", "read", {"type": "object"}, ())], trace, {"read": handler})
                    client = Client({"thought": "read", "function": "read", "arguments": arguments})
                    ctx = RunContext("memgpt", "synthetic task", client, env, trace, {})
                    answer = await run_profile(ctx)
                    if handler is successful_business_result:
                        self.assertEqual(answer, "")
                        self.assertEqual(ctx.llm_calls, 1)
                    else:
                        self.assertEqual(answer, "recovered")
                        self.assertEqual(ctx.llm_calls, 2)
                        self.assertIn("Function call failed", json.dumps(client.messages[1]))
                        self.assertFalse(any(x["event"] == "memgpt_yield" for x in trace.events))

    async def test_schema_error_forces_recovery_without_executing_tool(self):
        async def forbidden(args):
            self.fail("invalid arguments reached tool")

        trace = Trace()
        tool = ToolSpec("read", "read", {"type": "object", "required": ["path"]}, ())
        env = ToolEnvironment([tool], trace, {"read": forbidden})
        client = Client({"thought": "read", "function": "read", "arguments": {"request_heartbeat": False}})
        ctx = RunContext("memgpt", "synthetic task", client, env, trace, {})
        self.assertEqual(await run_profile(ctx), "recovered")
        self.assertEqual(env.calls[0]["result"]["error"], "invalid_arguments")


class MemGPTProtocolDriftTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_object_is_repaired_instead_of_ending_the_episode(self):
        # AutomationBench injects its own agent instructions, which describe a native tool
        # interface. One {"status": "in_progress"} reply used to raise and score the episode 0.
        trace = Trace()
        env = ToolEnvironment([ToolSpec("read", "read", {"type": "object"}, ())], trace, {"read": None})
        client = Client({"status": "in_progress"})
        ctx = RunContext("memgpt", "synthetic task", client, env, trace, {})
        self.assertEqual(await run_profile(ctx), "recovered")
        self.assertEqual(ctx.llm_calls, 2)
        errors = [x for x in trace.events if x["event"] == "json_reply_repair"]
        self.assertEqual([x["role"] for x in errors], ["memgpt_processor"])
        feedback = client.messages[1][-1]["content"]
        self.assertIn("reply.thought is required", feedback)
        self.assertEqual(errors[0]["response_schema"]["required"], ["thought", "function", "arguments"])
