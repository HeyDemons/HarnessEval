import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_platform.budgets import ModelBudgetExceeded
from benchmark_platform.bridges.episode import EpisodeBroker, FinalResponse, NativeTool
from benchmark_platform.catalog import Catalog
from benchmark_platform.compatibility import compatibility_rows
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from benchmark_platform.harnesses.profiles import get_profile
from benchmark_platform.harnesses.rewoo import parse_rewoo_plan
from benchmark_platform.harnesses.magentic_one import _progress_ledger
from test_compiler_fidelity import Trace, Client
from test_harnesses import magentic_ledger


class NativeIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_isolated_read_and_adoption_cannot_reach_native_handler(self):
        calls = []
        async def native(args):
            calls.append(args)
            return 'visible'
        env = ToolEnvironment([ToolSpec('lookup','lookup',{'type':'object'},(),read_only=True,parallel=True)],
                              Trace(), {'lookup':native}, isolated_calls_supported=False)
        with self.assertRaisesRegex(RuntimeError, 'no isolated'):
            await env.call_isolated('lookup', {'id':'WRONG'})
        with self.assertRaisesRegex(RuntimeError, 'no isolated'):
            await env.commit_isolated_calls([{'name':'lookup','arguments':{},'result':{'ok':True}}])
        self.assertEqual(calls, [])
        self.assertEqual(env.calls, [])
        await env.call('lookup', {'id':'RIGHT'})
        self.assertEqual(calls, [{'id':'RIGHT'}])

    def test_native_broker_rejects_sa_before_model_or_native_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            broker = EpisodeBroker(profile='sa', prompt='read RIGHT', tools=[
                NativeTool('lookup','lookup',{'type':'object'},read_only=True,parallel=True)],
                trace_path=Path(directory)/'trace.jsonl', policy={},
                client=Client([]), speculator_client=Client([]))
            broker.start()
            with self.assertRaisesRegex(RuntimeError, 'isolated tool execution/commit'):
                broker.next_wave()
            broker._thread.join(2)
            self.assertFalse(broker._thread.is_alive())
            self.assertEqual(broker.context.llm_calls, 0)
            self.assertEqual(broker.context.environment.calls, [])
            self.assertEqual(broker._pending, {})

    def test_native_broker_publishes_actor_hit_and_adopts_shadow_result(self):
        shadow_calls, adopted = [], []
        async def shadow(name, arguments):
            shadow_calls.append((name, arguments))
            return {"ok": True, "result": {"value": "cached"}}
        def commit(name, arguments, request_id):
            adopted.append((name, arguments, request_id))
        actor = Client([{"tool": "lookup", "arguments": {"id": "RIGHT"}}, {"final": "done"}])
        speculator = Client([{"actions": [{"tool": "lookup", "arguments": {"id": "RIGHT"}}]}])
        with tempfile.TemporaryDirectory() as directory:
            broker = EpisodeBroker(profile="sa", prompt="read RIGHT", tools=[
                NativeTool("lookup", "lookup", {"type": "object"}, read_only=True, parallel=True)],
                trace_path=Path(directory) / "trace.jsonl", policy={"max_turns": 3},
                client=actor, speculator_client=speculator,
                speculative_executor=shadow, isolated_commit=commit)
            broker.start()
            wave = broker.next_wave()
            self.assertEqual(len(wave), 1)
            self.assertEqual(wave[0].name, "lookup")
            self.assertEqual(wave[0].arguments, {"id": "RIGHT"})
            self.assertEqual(shadow_calls, [("lookup", {"id": "RIGHT"})])
            self.assertEqual(adopted, [("lookup", {"id": "RIGHT"}, wave[0].id)])
            final = broker.next_wave(tool_results={wave[0].id: ({"value": "cached"}, False)})
            self.assertIsInstance(final, FinalResponse)
            self.assertEqual(final.answer, "done")
            self.assertEqual(len(broker.context.environment.calls), 1)

    def test_tau_matrix_enables_sa_after_native_shadow_adapter(self):
        root = Path(__file__).resolve().parents[1]
        catalog = Catalog(root/'catalog/benchmarks.json',root,root)
        rows = compatibility_rows([get_profile('sa')], [catalog.get(name) for name in ('tau2','automationbench','bfcl')])
        self.assertEqual([r['runnable'] for r in rows], [True,True,True])

    async def test_sa_and_json_actor_receive_identical_requests_including_final_slot(self):
        class RecordingClient(Client):
            def __init__(self, responses):
                super().__init__(responses)
                self.requests = []
            async def complete(self,messages,**kwargs):
                self.requests.append((copy.deepcopy(messages),kwargs))
                return await super().complete(messages,**kwargs)
        histories = []
        for method in ('actor-only','sa'):
            trace = Trace()
            async def read(args): return {'value':42}
            env = ToolEnvironment([ToolSpec('lookup','lookup',{'type':'object'},(),read_only=True,parallel=True)],trace,{'lookup':read})
            actor = RecordingClient([{'tool':'lookup','arguments':{}},{'final':'42'}])
            ctx = RunContext(method,'Read value',actor,env,trace,{'model_response_limit':2,'finalize_on_loop_limit':True},
                             speculator_client=Client([{'actions':[{'tool':'lookup','arguments':{}}]}]))
            self.assertEqual(await run_profile(ctx), '42')
            histories.append(actor.requests)
        self.assertEqual(histories[0],histories[1])


class SourceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def test_rewoo_anchors_remain_relative_to_original_lines(self):
        self.assertEqual(len(parse_rewoo_plan('Plan: a\n#E1 = LLM[x] Plan: leftover note')),1)
        for text in ('Plan:', 'Plan: a\n#E1 = LLM[x]\nPlan:', 'Plan: a\n#E1 = LLM[x]\n  Plan: orphan'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'no evidence call'):
                parse_rewoo_plan(text)

    async def test_magentic_nonhashable_speakers_use_existing_repair(self):
        for speaker in ({'a':1},['Coder']):
            trace = Trace()
            ctx = RunContext('magentic-one','task',Client([
                magentic_ledger(satisfied=False,speaker=speaker), magentic_ledger(satisfied=False,speaker='Coder')]),
                ToolEnvironment([],trace,{}),trace,{})
            ledger = await _progress_ledger(ctx,{'Coder':'code','Executor':'execute'},'team',[])
            self.assertEqual(ledger['next_speaker']['answer'],'Coder')
            self.assertEqual(ctx.llm_calls,2)
            self.assertEqual(sum(e['event']=='magentic_ledger_retry' for e in trace.events),1)

    async def test_dmas_own_limit_returns_last_result_without_synthesis(self):
        for limit in (1,2):
            responses = [{'requirements':{'reasoning':1}}]
            for index in range(limit):
                responses.extend([{'decision':'split','executable':'part','remaining':'rest'},'reason',
                                  {'final':f'result {index}'},{'status':'incompleted','next_agent_id':'B','remaining':'rest'}])
            trace = Trace()
            ctx = RunContext('dmas','task',Client(responses),ToolEnvironment([],trace,{}),trace,
                {'dmas_max_execution_times':limit,'dmas_agents':[{'id':'A','abilities':{'reasoning':1}}, {'id':'B','abilities':{'reasoning':1}}]})
            self.assertEqual(await run_profile(ctx),f'result {limit-1}')
            self.assertEqual(ctx.llm_calls,1+4*limit)
            self.assertEqual(trace.events[-1]['event'],'dmas_execution_limit')
            # An official response limit must still fail before the post-split router.
            ctx = RunContext('dmas','task',Client(responses),ToolEnvironment([],trace,{}),trace,
                             {**ctx.policy,'model_response_limit':4})
            with self.assertRaises(ModelBudgetExceeded):
                await run_profile(ctx)
