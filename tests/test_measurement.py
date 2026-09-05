import unittest

from benchmark_platform.measurement import normalize_usage, product_actor_metrics


class MeasurementTests(unittest.TestCase):
    def test_openai_cache_is_removed_from_input_once(self):
        usage, known = normalize_usage({"usage": {"prompt_tokens": 1000, "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 50},
            "completion_tokens_details": {"reasoning_tokens": 75}}})
        self.assertTrue(known)
        self.assertEqual(usage, dict(input=350, output=100, cache_read=600, cache_write=50, total=450, all_tokens=1100))

    def test_anthropic_input_excludes_cache(self):
        usage, known = normalize_usage({"anthropic_usage": {"input_tokens": 350, "output_tokens": 100,
            "cache_read_input_tokens": 600, "cache_creation_input_tokens": 50}})
        self.assertTrue(known)
        self.assertEqual(usage["input"], 350)
        self.assertEqual(usage["all_tokens"], 1100)

    def test_responses_and_deepseek_cache(self):
        for raw in ({"responses_usage": {"input_tokens": 1000, "output_tokens": 100,
                          "input_tokens_details": {"cached_tokens": 600}}},
                    {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 600}}):
            usage, known = normalize_usage(raw)
            self.assertTrue(known)
            self.assertEqual(usage["total"], 500)

    def test_product_cache_not_subtracted_twice(self):
        usage, known = normalize_usage(dict(input=350, output=100, cacheRead=600, cacheWrite=50, totalTokens=1100), product=True)
        self.assertTrue(known)
        self.assertEqual(usage["total"], 450)

    def test_zero_initialized_usage_is_unknown(self):
        self.assertFalse(normalize_usage({}, product=False)[1])
        self.assertFalse(normalize_usage(dict(input=0, output=0, cacheRead=0, cacheWrite=0), product=True)[1])
        old_anthropic = {"stop_reason": "end_turn", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
        self.assertFalse(normalize_usage(old_anthropic)[1])

    def test_actor_turns_not_event_count_or_tool_count(self):
        msg = {"role": "assistant", "stopReason": "toolUse", "usage": {"input": 10, "output": 5},
               "content": [{"type": "toolCall", "id": str(i), "name": "lookup", "arguments": {}} for i in range(3)]}
        events = [{"type": "turn_start"}, {"type": "message_end", "message": msg},
                  {"type": "turn_end", "message": msg},
                  {"type": "turn_start"}, {"type": "message_end", "message": {"role": "assistant", "stopReason": "error"}},
                  {"type": "turn_start"}, {"type": "message_end", "message": {"role": "assistant", "stopReason": "aborted"}},
                  {"type": "turn_start"}]
        metrics = product_actor_metrics(events)
        self.assertEqual(metrics["agent_turns"], 1)
        self.assertEqual(len(metrics["committed_calls"]), 3)
        self.assertEqual(metrics["usage"]["total"], 15)
        self.assertEqual(metrics["usage_missing_requests"], 3)
        self.assertFalse(metrics["usage_complete"])

    def test_thinking_only_is_not_an_outward_decision(self):
        events = [{"type": "message_end", "message": {"role": "assistant", "stopReason": "stop",
                  "content": [{"type": "thinking", "thinking": "internal planning"}], "usage": {"input": 10, "output": 2}}},
                  {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop",
                  "content": [{"type": "text", "text": "The answer is 42"}], "usage": {"input": 10, "output": 2}}}]
        metrics = product_actor_metrics(events)
        self.assertEqual(metrics["model_responses"], 2)
        self.assertEqual(metrics["agent_turns"], 1)
