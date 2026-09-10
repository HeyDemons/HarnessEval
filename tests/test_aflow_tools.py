import tempfile
from pathlib import Path
import unittest

from benchmark_platform.harnesses import aflow, aflow_tools
from benchmark_platform.harnesses.aflow_search import optimize
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from test_aflow_upstream import Client
from test_dylan_fidelity import Trace


def context(replies, **policy):
    trace = Trace()
    async def write(args):
        return {'saved': args['value']}
    env = ToolEnvironment([ToolSpec('write', 'write', {'type': 'object'}, ())], trace, {'write': write})
    return RunContext('aflow', 'PUBLIC_TASK', Client(replies), env, trace,
        {'aflow_artifact': aflow_tools.make_artifact(), 'aflow_allow_initialization': True, **policy},
        task_messages=[{'role': 'system', 'content': 'PUBLIC_BENCHMARK_POLICY'}])


WRITE = '{"tool":"write","arguments":{"value":"actual"}}'
FINAL = '{"final":"finished"}'


class AFlowToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_executes_tools_observes_result_and_preserves_task_policy(self):
        ctx = context([WRITE, FINAL])
        self.assertEqual(await run_profile(ctx), 'finished')
        self.assertEqual(len(ctx.environment.calls), 1)
        self.assertIn('saved', str(ctx.client.messages[1]))
        self.assertTrue(all('PUBLIC_BENCHMARK_POLICY' in str(m) for m in ctx.client.messages))
        self.assertEqual(ctx.actor_llm_calls, 2)

    async def test_proposals_do_not_mutate_and_only_one_current_proposal_can_commit(self):
        ctx = context([WRITE, WRITE])
        llm = aflow.OperatorLLM(ctx)
        session = aflow_tools.ToolSession(llm, ctx.prompt)
        decide = aflow_tools.ToolDecision(llm)
        first, second = await decide(session), await decide(session)
        self.assertEqual(ctx.environment.calls, [])
        await session.commit(first)
        for stale in (first, second):
            with self.assertRaisesRegex(ValueError, 'stale'):
                await session.commit(stale)
        self.assertEqual(len(ctx.environment.calls), 1)

    async def test_final_model_slot_allows_answer_but_no_write(self):
        ctx = context([WRITE, FINAL], model_response_limit=2, finalize_on_loop_limit=True)
        self.assertEqual(await run_profile(ctx), 'finished')
        self.assertEqual(len(ctx.environment.calls), 1)
        ctx = context([WRITE], model_response_limit=1, finalize_on_loop_limit=True)
        with self.assertRaises(ValueError):
            await run_profile(ctx)
        self.assertEqual(ctx.environment.calls, [])

    async def test_auxiliary_operator_cannot_bypass_final_response_boundary(self):
        ctx = context([WRITE, 'criticism'], model_response_limit=2, finalize_on_loop_limit=True)
        llm = aflow.OperatorLLM(ctx)
        session = aflow_tools.ToolSession(llm, ctx.prompt)
        proposal = await aflow_tools.ToolDecision(llm)(session)
        await aflow.Custom(llm)('consider', '')
        with self.assertRaisesRegex(RuntimeError, 'final response'):
            await session.commit(proposal)
        self.assertEqual(ctx.environment.calls, [])
        self.assertEqual(ctx.model_budget.used, 2)

    def test_qa_and_tool_artifacts_cannot_be_interchanged(self):
        with self.assertRaises(ValueError):
            aflow.validate_artifact(aflow_tools.make_artifact(), allow_initialization=True)
        with self.assertRaises(ValueError):
            aflow_tools.validate_artifact(aflow.make_artifact(), allow_initialization=True)

    def test_generated_code_cannot_access_native_controller_or_python_reflection(self):
        for expression in ('self.llm.ctx', 'self.llm.__dict__', '__import__("os")',
                           'open("/hidden/gold")', '"{0.ctx}".format(self.llm)',
                           'getattr(self.llm, "ctx")'):
            graph = aflow_tools.INITIAL_GRAPH.replace('session = operator.ToolSession',
                f'leak = {expression}\n        session = operator.ToolSession')
            with self.assertRaises(ValueError, msg=expression):
                aflow_tools.validate_artifact(aflow_tools.make_artifact(graph), allow_initialization=True)
        graph = aflow_tools.INITIAL_GRAPH.replace('self.llm =', 'self.ctx = None\n        self.llm =')
        graph = graph.replace('session = operator.ToolSession', 'leak = self.llm.ctx\n        session = operator.ToolSession')
        with self.assertRaises(ValueError):
            aflow_tools.validate_artifact(aflow_tools.make_artifact(graph), allow_initialization=True)
        with self.assertRaises(ValueError):
            aflow_tools.validate_artifact(aflow_tools.make_artifact(prompt='INSTRUCTION = open("/hidden/gold").read()'), allow_initialization=True)
        for injected in ('prompt_custom = self.llm\n        leak = prompt_custom.ctx',
                         'self = self.llm\n        leak = self.ctx'):
            graph = aflow_tools.INITIAL_GRAPH.replace('session = operator.ToolSession',
                injected + '\n        session = operator.ToolSession')
            with self.assertRaises(ValueError):
                aflow_tools.validate_artifact(aflow_tools.make_artifact(graph, aflow_tools.INITIAL_PROMPT + 'ctx = ""'),
                                              allow_initialization=True)

    async def test_aflow_search_freezes_best_and_never_exposes_evaluation_ids(self):
        split = {'benchmark': 'synthetic', 'optimization_case_ids': ['opt'], 'evaluation_case_ids': ['secret-eval-id']}
        graph = aflow_tools.INITIAL_GRAPH.replace('instruction=prompt_custom.INSTRUCTION', 'instruction=prompt_custom.INSTRUCTION + " Be careful."')
        client = Client([f'<graph>{graph}</graph><prompt>{aflow_tools.INITIAL_PROMPT}</prompt><modification>review instruction</modification>'])
        seen = []
        async def evaluate(artifact):
            seen.append(artifact)
            aflow_tools.validate_artifact(artifact, allow_initialization=True)
            return {'score': len(seen) / 2}
        with tempfile.TemporaryDirectory() as directory:
            result = await optimize(client, evaluate, split, Path(directory) / 'search',
                                    rounds=1, validation_rounds=1,
                                    operator_profile='benchmark-tools')
        aflow_tools.validate_artifact(result, benchmark='synthetic', case_id='secret-eval-id')
        self.assertEqual(result['provenance']['selected_round'], 2)
        self.assertEqual(result['provenance']['operator_profile'], 'benchmark-tools')
        self.assertTrue(result['provenance']['benchmark_adapter'])
        self.assertNotIn('secret-eval-id', str(client.messages))
        self.assertIn('ToolSession', str(client.messages))
        with self.assertRaisesRegex(ValueError, 'evaluation manifest'):
            aflow_tools.validate_artifact(result, benchmark='synthetic', case_id='opt')
