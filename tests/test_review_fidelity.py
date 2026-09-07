"""Regression tests for source unwrapping and dynamic-protocol boundaries."""
import json
import unittest

from benchmark_platform.harnesses.methods import parse_action_reply, run_profile
from benchmark_platform.harnesses.dylan_policy import parse_action
from benchmark_platform.harnesses.magentic_one import _ledger_error
from benchmark_platform.harnesses.rewoo import parse_rewoo_plan
from test_harnesses import magentic_ledger
from test_reply_contracts import context


class ReviewFidelityTests(unittest.IsolatedAsyncioTestCase):
    async def test_spp_unwrap_matches_source_without_changing_raw_trace(self):
        cases = [
            ("Participants: A\nFinal answer: line one\nline two", "line one\nline two"),
            ("draft\nfinal answer: 42", "42"),
            ("No marker\n  Keep original whitespace  ", "No marker\n  Keep original whitespace  "),
            ("Final answer: first Final answer: second", "first"),
            ("final answer: lower\nFinal answer: upper", "upper"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                ctx, trace = context("multi-persona", [raw])
                self.assertEqual(await run_profile(ctx), expected)
                self.assertEqual(ctx.llm_calls, 1)
                self.assertEqual(ctx.environment.calls, [])
                self.assertIn(raw, [e.get("content") for e in trace.events if e["event"] == "llm_response"])

    async def test_compiler_repairs_bad_ids_before_any_task_runs(self):
        bads = [
            {"tasks": [{"id": value, "tool": "write"}]}
            for value in ("task1", 0, True, 1.5, "-1")
        ] + [
            {"tasks": [{"id": 1}, {"id": "01"}]},
            {"tasks": [{"id": 1, "dependencies": ["task2"]}]},
            {"tasks": [{"id": 1, "dependencies": "2"}]},
            {"tasks": [None]}, {"tasks": {}},
        ]
        for bad in bads:
            with self.subTest(bad=bad):
                calls = []
                async def write(args):
                    calls.append(args)
                    return "observed"
                ctx, trace = context("llmcompiler", [json.dumps(bad),
                    '{"tasks":[{"id":"01","tool":"write","arguments":{"ok":1}}]}',
                    '{"action":"finish","answer":"done"}'], handler=write)
                self.assertEqual(await run_profile(ctx), "done")
                self.assertEqual(calls, [{"ok": 1}])
                self.assertEqual(ctx.llm_calls, 3)
                repair = next(i for i, e in enumerate(trace.events) if e["event"] == "json_reply_repair")
                execution = next(i for i, e in enumerate(trace.events) if e["event"] == "tool_result")
                self.assertLess(repair, execution)

    async def test_compiler_repair_exhaustion_never_executes_partial_plan(self):
        bad = '{"tasks":[{"id":1,"tool":"write"},{"id":"task2","tool":"write"}]}'
        ctx, _ = context("llmcompiler", [bad, bad])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            await run_profile(ctx)
        self.assertEqual(ctx.llm_calls, 2)
        self.assertEqual(ctx.environment.calls, [])

    async def test_rewoo_empty_plan_reaches_solver_with_no_worker_or_repair(self):
        for raw in ("", "This question needs no external evidence."):
            ctx, trace = context("rewoo", [raw, "42"])
            self.assertEqual(await run_profile(ctx), "42")
            roles = [e["role"] for e in trace.events if e["event"] == "llm_request"]
            self.assertEqual(roles, ["rewoo_planner", "rewoo_solver"])
            self.assertEqual(ctx.environment.calls, [])
            self.assertIn("Worker log:\n\n\n", ctx.client.messages[-1][-1]["content"])

    def test_rewoo_malformed_evidence_is_not_empty_plan_or_truncated_input(self):
        for raw in ("Plan: use LLM\n#E1 = LLM[x", "#E1 = invalid", "Plan: missing evidence"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_rewoo_plan(raw)

    async def test_magentic_satisfaction_uses_source_truthiness(self):
        for value in (True, "true", "false", 1, [1]):
            ledger = magentic_ledger(satisfied=value, speaker="not-a-worker")
            self.assertIsNone(_ledger_error(json.loads(ledger), {"Coder": "", "Executor": ""}))
            ctx, _ = context("magentic-one", ["facts", "plan", ledger, "done"])
            self.assertEqual(await run_profile(ctx), "done")
            self.assertEqual(ctx.llm_calls, 4)
            self.assertEqual(ctx.environment.calls, [])
        for value in (False, "", 0, None, []):
            ledger = json.loads(magentic_ledger(satisfied=value, speaker="not-a-worker"))
            self.assertIn("invalid next speaker", _ledger_error(ledger, {"Coder": "", "Executor": ""}))

    async def test_annotations_do_not_consume_repair_rounds(self):
        for method in ("actor-only", "sa", "plan-execute"):
            replies = ['{"tool":"write","arguments":{"x":1},"explanation":"why"}',
                       '{"final":"done","confidence":0.9}']
            if method == "plan-execute":
                replies.insert(0, '{"steps":["Answer"]}')
            ctx, _ = context(method, replies)
            self.assertEqual(await run_profile(ctx), "done")
            self.assertEqual(ctx.llm_calls, len(replies))
            self.assertEqual([c["arguments"] for c in ctx.environment.calls], [{"x": 1}])

    async def test_magentic_stalls_follow_truthiness_without_boolean_coercion(self):
        for progress, in_loop, expected in (("true", "", 0), ("", False, 1), (True, "false", 1)):
            ctx, trace = context("magentic-one", ["facts", "plan",
                magentic_ledger(satisfied=False, speaker="Coder", progress=progress, in_loop=in_loop),
                "worker reply", magentic_ledger(satisfied=True), "done"])
            self.assertEqual(await run_profile(ctx), "done")
            state = next(e for e in trace.events if e["event"] == "magentic_stall_state")
            self.assertEqual(state["current"], expected)

    def test_annotations_do_not_change_dylan_votes(self):
        for annotated in ({"tool": "write", "arguments": {"x": 1}, "explanation": "why"},
                          {"tool": "write", "arguments": {"x": 1}, "confidence": .7}):
            key, ratings = parse_action(json.dumps({"action": annotated, "ratings": [1]}), ["write"])
            self.assertEqual(key, '{"arguments":{"x":1},"tool":"write"}')
            self.assertEqual(ratings, [1])
        for bad in ({"final": "done", "tool": "write", "arguments": {}},
                    {"tool": "write", "arguments": []}, {"tool": ["write"], "arguments": {}}):
            self.assertIsNone(parse_action(json.dumps(bad), ["write"])[0])

    def test_action_errors_explain_required_or_invalid_fields(self):
        with self.assertRaisesRegex(ValueError, r"reply.arguments is required"):
            parse_action_reply('{"tool":"write"}', ["write"])
        with self.assertRaisesRegex(ValueError, r"reply.final.*string"):
            parse_action_reply('{"final":null,"confidence":1}', ["write"])

    async def test_final_slot_forbids_tools_and_repair_calls(self):
        for method in ("actor-only", "sa"):
            for reply in ('{"tool":"write","arguments":{},"explanation":"one more"}',
                          '{"final":"done","tool":"write","arguments":{}}', 'not JSON'):
                ctx, _ = context(method, [reply], {"model_response_limit": 1, "finalize_on_loop_limit": True})
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    await run_profile(ctx)
                self.assertEqual(ctx.llm_calls, 1)
                self.assertEqual(ctx.environment.calls, [])
            ctx, _ = context(method, ['{"final":"done","confidence":0.9}'],
                             {"model_response_limit": 1, "finalize_on_loop_limit": True})
            self.assertEqual(await run_profile(ctx), "done")
            self.assertEqual(ctx.llm_calls, 1)
