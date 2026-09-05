import asyncio
import unittest

from benchmark_platform.budgets import (Deadline, ModelBudgetExceeded, baseline_limits,
                                      native_steps, resolve_budget)
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from test_declaration_protocol import Trace
from test_harnesses import ScriptedClient, native_tool_call


class BudgetTests(unittest.TestCase):
    def test_official_units_are_not_all_agent_turns(self):
        self.assertEqual(baseline_limits("bfcl", {})["model_response_limit"], 1)
        self.assertEqual(baseline_limits("automationbench", {})["model_response_limit"], 50)
        self.assertEqual(baseline_limits("tau2", {})["native_max_steps"], 200)
        self.assertEqual(baseline_limits("vitabench", {})["native_max_steps"], 300)
        for benchmark in ("gaia", "gdpval", "trajectory-bench", "tau2", "vitabench"):
            self.assertEqual(baseline_limits(benchmark, {})["max_turns"], 1000)
        self.assertEqual(native_steps("tau2", {"native_max_steps": 17}), 17)

    def test_wall_and_cancel_grace_do_not_grant_extra_execution_time(self):
        plan = resolve_budget("gaia", env={})
        self.assertEqual((plan.wall_seconds, plan.process_seconds, plan.cancel_grace_seconds), (900, 890, 10))
        self.assertIn("local operator", plan.wall_source)
        self.assertIsNone(plan.official["universal_time_limit"])

    def test_terminal_phases_are_never_clipped_by_shared_arm_limit(self):
        metadata = {"agent": {"timeout_sec": 4000}, "verifier": {"timeout_sec": 3000}}
        plan = resolve_budget("terminal-bench-2", task_metadata=metadata,
                              env={"HARNESS_ARM_TIMEOUT_S": "1", "TERMINAL_BENCH_OUTER_GRACE_S": "0"})
        self.assertEqual((plan.agent_seconds, plan.verifier_seconds), (4000, 3000))
        self.assertEqual((plan.wall_seconds, plan.process_seconds, plan.cancel_grace_seconds), (7000, 7000, 0))
        with self.assertRaises(ValueError):
            resolve_budget("terminal-bench-2", env={})

    def test_invalid_numbers_are_rejected(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_budget("gaia", env={"HARNESS_ARM_TIMEOUT_S": value})
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                baseline_limits("gaia", {"HARNESS_LOOP_SAFETY_LIMIT": value})

    def test_monotonic_remaining_is_shared_across_operations(self):
        now = [10.0]
        deadline = Deadline(30, clock=lambda: now[0])
        now[0] += 12
        self.assertEqual(deadline.remaining, 18)
        now[0] += 9
        self.assertEqual(deadline.remaining, 9)
        now[0] += 12
        self.assertEqual((deadline.remaining, deadline.elapsed), (0, 33))
        prepared = Deadline(30, clock=lambda: 15, started=5)
        self.assertEqual(prepared.remaining, 20)

    def test_external_scorer_is_a_separate_local_budget(self):
        plan = resolve_budget("gdpval", env={"GDPVAL_SCORER_TIMEOUT_S":"123"})
        self.assertEqual(plan.wall_seconds, 1200)
        self.assertEqual(plan.external_scorer_seconds, 123)
        self.assertIsNone(resolve_budget("tau2", env={}).external_scorer_seconds)


class LoopBudgetTests(unittest.IsolatedAsyncioTestCase):
    def context(self, responses, policy):
        trace = Trace()
        async def read(args):
            return "observed"
        return RunContext("react", "Read then answer", ScriptedClient(responses),
                          ToolEnvironment([ToolSpec("read", "read", {"type":"object"}, ())], trace, {"read":read}),
                          trace, {"react_protocol":"native", **policy})

    async def test_valid_work_can_continue_past_twenty_turns(self):
        ctx = self.context([native_tool_call("read", {}) for _ in range(21)] +
                           [native_tool_call("react_finish", {"answer":"done"})], baseline_limits("gaia", {}))
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual(ctx.llm_calls, 22)

    async def test_final_submission_uses_last_slot_and_has_no_environment_tools(self):
        ctx = self.context([native_tool_call("read", {}), native_tool_call("react_finish", {"answer":"done"})],
                           {"max_turns":2, "finalize_on_loop_limit":True})
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual([t["function"]["name"] for t in ctx.client.native_tools[-1]], ["react_finish"])
        self.assertEqual((ctx.llm_calls, len(ctx.environment.calls)), (2, 1))

    async def test_finish_only_slot_cannot_execute_another_environment_action(self):
        ctx = self.context([native_tool_call("read", {})] * 2, {"max_turns":2, "finalize_on_loop_limit":True})
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            await run_profile(ctx)
        self.assertEqual(len(ctx.environment.calls), 1)

    async def test_official_response_limit_is_shared_by_parallel_model_calls(self):
        ctx = self.context(["ok", "ok", "must not run"], {"model_response_limit":2})
        result = await asyncio.gather(*(ctx.complete(str(i), []) for i in range(3)), return_exceptions=True)
        self.assertEqual(sum(isinstance(item, ModelBudgetExceeded) for item in result), 1)
        self.assertEqual((ctx.model_budget.used, ctx.llm_calls), (2, 2))

    async def test_response_cap_reserves_finish_even_before_loop_guard(self):
        ctx = self.context([native_tool_call("read", {}), native_tool_call("react_finish", {"answer":"done"})],
                           {"max_turns":1000, "model_response_limit":2, "finalize_on_loop_limit":True})
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual([t["function"]["name"] for t in ctx.client.native_tools[-1]], ["react_finish"])

    async def test_sa_closing_slot_does_not_launch_more_speculation(self):
        trace = Trace()
        async def read(args):
            return "observed"
        ctx = RunContext("sa", "Read then answer", ScriptedClient([
            '{"tool":"read","arguments":{}}', '{"final":"done"}']),
            ToolEnvironment([ToolSpec("read", "read", {"type":"object"}, (), parallel=True, read_only=True)], trace, {"read":read}),
            trace, {"model_response_limit":2,"finalize_on_loop_limit":True},
            speculator_client=ScriptedClient(['{"actions":[]}']))
        self.assertEqual(await run_profile(ctx), "done")
        self.assertEqual((ctx.model_budget.used, ctx.speculator_llm_calls), (2, 1))
