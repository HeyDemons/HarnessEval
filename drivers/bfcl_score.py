#!/usr/bin/env python3
"""Score one BFCL case with the benchmark's own checker, inside the benchmark's own image.

The bridge hands the model BFCL's function declarations and records what it calls without
executing anything, so a case is graded entirely from the call list. Three families of
grader exist and the category name selects between them, exactly as eval_runner.py does:

  irrelevance / live_irrelevance   pass iff the model called nothing
  live_relevance                   pass iff the model called something
  everything else with an answer   the official ast_checker

Only the 65 light cases that ship a "function" field are gradable this way. multi_turn,
memory and web_search evolve real state through executed calls, which the recording bridge
never provides, so they are reported unscorable rather than graded as a zero -- a zero here
would read as "the model got it wrong" when nothing was ever asked of it.

Run inside orch-bench/bfcl:current. Prints one JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# convert_func_name() rewrites a dotted answer-key name to its underscored form only for
# models whose config sets underscore_to_dot, which is what adapters.load_bfcl emulates when
# it declares the tools. Naming an OpenAI-family entry keeps both halves in agreement.
MODEL_NAME = "gpt-4.1-2025-04-14-FC"

# Categories the recording bridge cannot stage. Kept explicit rather than derived so a new
# BFCL category shows up as a loud KeyError instead of being silently graded by the wrong one.
NEEDS_EXECUTION = ("multi_turn", "memory", "web_search")


def language_for(category: str):
    from bfcl_eval.constants.enums import Language

    if category.endswith("_java"):
        return Language.JAVA
    if category.endswith("_javascript"):
        return Language.JAVASCRIPT
    return Language.PYTHON


def grade(case: dict, gold: dict, calls: list[dict]) -> dict:
    category = str(gold.get("test_category") or "")
    if not category:
        return {"score": None, "reason": "gold.json carries no test_category"}
    if category.startswith(NEEDS_EXECUTION):
        return {"score": None, "category": category,
                "reason": "category needs executed tool state; the bridge only records calls"}

    # Order matters: "live_relevance" contains "relevance" and "live_irrelevance" contains
    # both, so the irrelevance test has to come first.
    if "irrelevance" in category:
        return {"score": 1.0 if not calls else 0.0, "category": category,
                "authority": "bfcl_irrelevance_no_function_call", "calls": len(calls)}
    if "relevance" in category:
        return {"score": 1.0 if calls else 0.0, "category": category,
                "authority": "bfcl_relevance_function_call_present", "calls": len(calls)}

    ground_truth = gold.get("ground_truth")
    if not ground_truth:
        return {"score": None, "category": category, "reason": "no ground_truth in gold.json"}

    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    func_description = [item.get("function", item) for item in case.get("functions") or []]
    model_output = [{str(call["name"]): call.get("arguments") or {}} for call in calls]
    try:
        result = ast_checker(func_description, model_output, ground_truth,
                             language_for(category), category, MODEL_NAME)
    except Exception as exc:  # a malformed call must score 0, not abort the sweep
        return {"score": 0.0, "category": category, "authority": "bfcl_official_ast_checker",
                "error_type": f"{type(exc).__name__}", "error": [str(exc)[:300]]}
    verdict = {"score": 1.0 if result.get("valid") else 0.0, "category": category,
               "authority": "bfcl_official_ast_checker"}
    if not result.get("valid"):
        # error_type carries the checker's "...:unclear" initial value even on a pass, which
        # reads as a defect on a case that scored 1.0. Only report it when it explains a zero.
        verdict["error_type"] = result.get("error_type")
        verdict["error"] = result.get("error")
    return verdict


def self_check() -> int:
    """No BFCL import needed: these paths are decided before the checker is reached."""
    gold = lambda c: {"test_category": c}
    assert grade({}, gold("irrelevance"), [])["score"] == 1.0
    assert grade({}, gold("irrelevance"), [{"name": "f", "arguments": {}}])["score"] == 0.0
    assert grade({}, gold("live_irrelevance"), [])["score"] == 1.0
    # live_relevance is the exact opposite and must not be caught by the irrelevance branch.
    assert grade({}, gold("live_relevance"), [])["score"] == 0.0
    assert grade({}, gold("live_relevance"), [{"name": "f", "arguments": {}}])["score"] == 1.0
    for category in ("multi_turn_base", "memory", "web_search"):
        assert grade({}, gold(category), [])["score"] is None
    assert grade({}, gold("simple_python"), [])["score"] is None  # no ground_truth
    print("OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True, help="dir holding input/ and authority/")
    parser.add_argument("--calls", type=Path, required=True, help="JSON list of {name, arguments}")
    args = parser.parse_args()
    case = json.loads((args.prepared / "input" / "case.json").read_text(encoding="utf-8"))
    gold = json.loads((args.prepared / "authority" / "gold.json").read_text(encoding="utf-8"))
    calls = json.loads(args.calls.read_text(encoding="utf-8"))
    print(json.dumps(grade(case, gold, calls), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(self_check())
    raise SystemExit(main())
