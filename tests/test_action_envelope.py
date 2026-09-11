"""A model that echoes the function transport's envelope still made a real call.

Samples are verbatim from the 2026-09-08 automationbench sweep, where 24 of 36
aflow-tools arms died on "reply.tool is required" while the payload inside the
{"response": ...} wrapper was a valid tool call.
"""
import unittest

from benchmark_platform.harnesses.methods import parse_action_reply

NAMES = ["api_search", "api_fetch", "base64_encode"]


class ActionEnvelopeTests(unittest.TestCase):
    def test_envelope_is_unwrapped(self):
        cases = [
            ('{"response":{"tool":"api_search","arguments":{"query":"x","top_k":10}}}\n'
             '{"response":{"final":"unable to proceed"}}',
             {"tool": "api_search", "arguments": {"query": "x", "top_k": 10}}),
            ('{"response":{"final":"done"}}', {"final": "done"}),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(parse_action_reply(source, NAMES), expected)

    def test_falsy_final_annotation_is_dropped(self):
        self.assertEqual(
            parse_action_reply('{"tool":"api_search","arguments":{},"final":false}', NAMES),
            {"tool": "api_search", "arguments": {}})

    def test_the_contract_is_not_loosened(self):
        for source in (
            '{"response":"just text"}',
            '{"tool":"api_search","arguments":{},"final":"done"}',
            '{"response":{"tool":"nope","arguments":{}}}',
            '{"response":{"tool":"api_search","arguments":{}},"note":"x"}',
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    parse_action_reply(source, NAMES)


if __name__ == "__main__":
    unittest.main()
