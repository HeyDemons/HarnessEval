import unittest

from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.methods import run_profile
from test_compiler_fidelity import Trace, Client


class SpeculativeValidationTests(unittest.IsolatedAsyncioTestCase):
    def environment(self, validate):
        calls = []
        async def lookup(args):
            calls.append(args)
            # A native handler may accept omissions despite a strict API schema.
            return {'query': args['query'], 'limit': args.get('limit', 20)}
        tool = ToolSpec('lookup', 'lookup', {'type': 'object', 'properties': {
            'query': {'type': 'string'}, 'limit': {'type': 'integer'}},
            'required': ['query', 'limit']}, (), parallel=True, read_only=True)
        return ToolEnvironment([tool], Trace(), {'lookup': lookup}, validate_schema=validate), calls

    async def test_native_validation_policy_is_the_same_for_direct_and_isolated_calls(self):
        for validate in (True, False):
            env, calls = self.environment(validate)
            args = {'query': 'document'}
            direct = await env.call('lookup', args)
            isolated, record = await env.call_isolated('lookup', args)
            self.assertEqual(direct, isolated)
            self.assertEqual(direct['ok'], not validate)
            self.assertEqual(len(calls), 0 if validate else 2)

    async def test_sa_cache_hit_preserves_native_defaults_without_reexecution(self):
        env, calls = self.environment(False)
        ctx = RunContext('sa', 'Find document', Client([
            {'tool': 'lookup', 'arguments': {'query': 'document'}}, {'final': 'done'}]), env, env.trace, {},
            speculator_client=Client([{'actions': [{'tool': 'lookup', 'arguments': {'query': 'document'}}]},
                                      {'actions': []}]))
        self.assertEqual(await run_profile(ctx), 'done')
        self.assertEqual(calls, [{'query': 'document'}])
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(env.calls[0]['result']['result']['limit'], 20)
        self.assertTrue(any(e['event'] == 'sa_cache_hit' for e in env.trace.events))
