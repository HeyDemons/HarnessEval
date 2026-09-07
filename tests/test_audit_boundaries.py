import unittest

from benchmark_platform.harnesses.core import ToolSpec
from benchmark_platform.harnesses.dmas import _bounded_score


class AuditBoundaryTests(unittest.TestCase):
    def test_native_schema_keeps_parameters_but_not_controller_flags(self):
        spec = ToolSpec('read', 'Read', {'type': 'object', 'required': ['path']}, (),
                        parallel=True, read_only=True)
        native = spec.native_schema()
        self.assertEqual(set(native), {'name', 'description', 'parameters'})
        self.assertEqual(native['parameters'], spec.prompt_schema()['parameters'])
        self.assertTrue(spec.prompt_schema()['parallel'])
        self.assertTrue(spec.prompt_schema()['read_only'])

    def test_dmas_capability_scores_reject_nonfinite_values(self):
        for value in ('NaN', float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                _bounded_score(value, field='ability')
        self.assertEqual(_bounded_score(.6, field='ability'), .6)
