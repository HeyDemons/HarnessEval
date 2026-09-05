"""Reporting-only accounting. Never use these counters as benchmark budgets."""
from __future__ import annotations

METRICS_VERSION = 2
TURN_DEFINITION = "completed-actor-model-response-v2"
TOKEN_DEFINITION = "uncached-input-plus-output-v2"
TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "total", "all_tokens")


def zero_tokens() -> dict:
    return dict.fromkeys(TOKEN_FIELDS, 0)


def count(value) -> int:
    return max(0, int(value)) if isinstance(value, (int, float)) else 0


def normalize_usage(raw: dict, *, product: bool = False) -> tuple[dict, bool]:
    """Normalize disjoint usage buckets; missing usage is unknown, not free.

    OpenAI prompt/input totals include cache; Anthropic input_tokens excludes it.
    Pi/Product input is already uncached. Output includes any billed reasoning.
    Optional cache fields omitted from a reported API usage object default to zero.
    """
    raw = raw if isinstance(raw, dict) else {}
    result = zero_tokens()
    if product:
        mapping = {"input": "input", "output": "output", "cacheRead": "cache_read", "cacheWrite": "cache_write"}
        for source, target in mapping.items():
            result[target] = count(raw.get(source))
        # Older Pi providers initialize a zero usage object before receiving usage.
        # Without an explicit marker an all-zero object cannot prove reported usage.
        known = raw.get("reported") is True or any(result.values())
    else:
        usage = raw.get("anthropic_usage") or raw.get("responses_usage") or raw.get("usage") or {}
        legacy_anthropic = "stop_reason" in raw and "choices" in raw and "anthropic_usage" not in raw
        anthropic = legacy_anthropic or "anthropic_usage" in raw or "cache_creation_input_tokens" in usage or "cache_read_input_tokens" in usage
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        read = count(usage.get("cache_read_input_tokens")) if anthropic else count(details.get("cached_tokens", usage.get("prompt_cache_hit_tokens")))
        write = count(usage.get("cache_creation_input_tokens")) if anthropic else count(details.get("cache_write_tokens"))
        prompt = count(usage.get("prompt_tokens", usage.get("input_tokens")))
        result.update(input=prompt if anthropic else max(0, prompt-read-write),
                      output=count(usage.get("completion_tokens", usage.get("output_tokens"))),
                      cache_read=read, cache_write=write)
        known = any(k in usage for k in ("prompt_tokens", "input_tokens")) and any(k in usage for k in ("completion_tokens", "output_tokens"))
        known = known and not legacy_anthropic
    result["total"] = result["input"] + result["output"]
    result["all_tokens"] = result["total"] + result["cache_read"] + result["cache_write"]
    return result, known


def add_tokens(target: dict, source: dict) -> None:
    for key in TOKEN_FIELDS:
        target[key] = target.get(key, 0) + count(source.get(key))


def product_actor_metrics(events: list[dict]) -> dict:
    """Count completed Actor responses once, independently of tool fanout.

    message_end and turn_end contain the same message: never sum both. Error and
    aborted placeholders are request outcomes, not completed model turns.
    """
    usage = zero_tokens()
    messages = [e["message"] for e in events if e.get("type") == "message_end"
                and isinstance(e.get("message"), dict) and e["message"].get("role") == "assistant"]
    missing = turns = 0
    calls = []
    for message in messages:
        tokens, known = normalize_usage(message.get("usage"), product=True)
        add_tokens(usage, tokens)
        missing += not known
        if message.get("stopReason") not in {"error", "aborted"}:
            turns += 1
        # Preserve the runner's declaration evidence, even on a failed message;
        # execution/scoring eligibility is decided by the bridge, not accounting.
        calls.extend({"id": str(c.get("id") or ""), "name": str(c.get("name") or ""),
                      "arguments": c.get("arguments") if isinstance(c.get("arguments"), dict) else {}}
                     for c in message.get("content") or [] if isinstance(c, dict) and c.get("type") == "toolCall")
    starts = sum(e.get("type") == "turn_start" for e in events)
    missing += max(0, starts-len(messages))
    return {"rounds": turns, "agent_turns": turns, "agent_turns_definition": TURN_DEFINITION,
            "model_responses": turns, "model_attempts": max(starts, len(messages)),
            "committed_calls": calls, "usage": usage,
            "usage_missing_requests": missing, "usage_complete": missing == 0,
            "last_stop_reason": messages[-1].get("stopReason") if messages else None,
            "last_error": messages[-1].get("errorMessage") if messages else None}
