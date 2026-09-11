"""The relay leaks the model's tool-call channel into text; only that prefix is dropped.

Samples are verbatim from harness_trace.jsonl of the 2026-09-07 automationbench and
terminal-bench-2 runs. Legitimate Chinese prose and Chinese JSON arguments must survive
byte-identical, because the strip runs on every completion, including final answers.
"""
import unittest

from benchmark_platform.harnesses.core import strip_channel_leak


class ChannelLeakTests(unittest.TestCase):
    def test_leaked_channel_prefix_is_removed(self):
        cases = [
            ('#E6 = api_search[\u10d7\u10d0\u10d5\u10d0\u10d6json\n{"query":"x"}]',
             '#E6 = api_search[{"query":"x"}]'),
            ('#E8 = api_search[\u0410\u049e\u04d8\u0410{"query":"y"}]',
             '#E8 = api_search[{"query":"y"}]'),
            ('#E6 = api_search[\u4eba\u4eba\u78b0\n{"query":"z"}]',
             '#E6 = api_search[{"query":"z"}]'),
            ('to=send_message_to_user  \u5929\u5929\u8d2d\u5f69\u7968json\n{"message":"hi"}',
             '{"message":"hi"}'),
            ('#E12 = run_command[ \u5927\u53d1\u65f6\u65f6\u5f69\u662fjson\n{"argv":["sh"]}]',
             '#E12 = run_command[{"argv":["sh"]}]'),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                cleaned, removed = strip_channel_leak(source)
                self.assertEqual(cleaned, expected)
                self.assertTrue(removed)

    def test_legitimate_text_is_untouched(self):
        for text in [
            '{"tool":"read_file","arguments":{"path":"/app/x"}}',
            'Plan: read the file.\n#E1 = read_file[{"path":"/app/x"}]',
            'The answer is 42.',
            'Return the following JSON:\n{"a":1}',
            '\u8bf7\u8fd4\u56de\u5982\u4e0b JSON:\n{"a":1}',
            'Final answer: \u5317\u4eac',
            '#E1 = api_search[{"query":"\u7528\u6237\u5217\u8868","top_k":5}]',
        ]:
            with self.subTest(text=text):
                cleaned, removed = strip_channel_leak(text)
                self.assertEqual(cleaned, text)
                self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
