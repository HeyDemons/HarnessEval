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
