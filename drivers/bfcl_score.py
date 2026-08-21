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


def _java_scalar_literal(value, source_type: str) -> str:
    if source_type == "long":
        return f"{value}L"
    if source_type == "float":
        return f"{value}f"
    if source_type == "boolean":
        return str(value).lower()
    if source_type == "char":
        return repr(value)
    return str(value)


def _java_collection_item(value, source_type: str, *, array: bool = False) -> str:
    if source_type == "String":
        # BFCL's Array converter passes entries directly to java_type_converter(String),
        # which does not remove quotes; ArrayList has a separate quote-stripping path.
        return str(value) if array else json.dumps(str(value), ensure_ascii=False)
    return _java_scalar_literal(value, source_type)


def _java_literal(value, schema: dict) -> str:
    source_type = str(schema.get("type") or "any")
    if isinstance(value, str):
        return value
    nested_type = str((schema.get("items") or {}).get("type") or "any")
    if source_type == "Array" and isinstance(value, list):
        elements = ", ".join(
            _java_collection_item(item, nested_type, array=True) for item in value
        )
        component = {"integer": "int", "boolean": "boolean"}.get(
            nested_type, nested_type
        )
        return f"new {component}[]{{{elements}}}"
    if source_type == "ArrayList" and isinstance(value, list):
        elements = ", ".join(
            _java_collection_item(item, nested_type) for item in value
        )
        boxed = {
            "integer": "Integer",
            "long": "Long",
            "float": "Float",
            "double": "Double",
            "boolean": "Boolean",
        }.get(nested_type, nested_type)
        return f"new ArrayList<{boxed}>(Arrays.asList({elements}))"
    if source_type == "HashMap" and isinstance(value, dict):
        entries = " ".join(
            f"put({json.dumps(str(key), ensure_ascii=False)}, "
            f"{json.dumps(item, ensure_ascii=False)});"
            for key, item in value.items()
        )
        return f"new HashMap<String, Object>() {{{{ {entries} }}}}"
    return _java_scalar_literal(value, source_type)


def _javascript_literal(value, schema: dict) -> str:
    source_type = str(schema.get("type") or "any")
    if isinstance(value, str):
        return value
    if source_type == "Bigint":
        return f"{value}n"
    if source_type == "Boolean":
        return str(value).lower()
    if source_type in {"array", "dict"}:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def source_language_model_output(
    func_description: list[dict], calls: list[dict], category: str
) -> list[dict]:
    """Adapt native JSON arguments to the source literals BFCL's AST checker expects.

    BFCL's official FC compiler exposes Java/JavaScript values as OpenAPI JSON types, but
    its AST checker accepts those categories only as strings containing source-language
    literals. This boundary adapter reconciles the two official contracts without changing
    Python/live behavior or teaching a model answer-key values.
    """
    if not category.endswith(("_java", "_javascript")):
        return [{str(call["name"]): call.get("arguments") or {}} for call in calls]

    descriptions = {
        str(function.get("name") or "").replace(".", "_"): function
        for function in func_description
    }
    convert = _java_literal if category.endswith("_java") else _javascript_literal
    output = []
    for call in calls:
        name = str(call["name"])
        arguments = call.get("arguments") or {}
        function = descriptions.get(name)
        properties = (
            ((function or {}).get("parameters") or {}).get("properties") or {}
        )
        normalized = {
            str(parameter): (
                convert(value, properties[parameter])
                if isinstance(properties.get(parameter), dict)
                else value
            )
            for parameter, value in arguments.items()
        }
        output.append({name: normalized})
    return output


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
    model_output = source_language_model_output(func_description, calls, category)
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
    java = [{"name": "f", "parameters": {"properties": {
        "enabled": {"type": "boolean"},
        "ids": {"type": "ArrayList", "items": {"type": "long"}},
    }}}]
    converted = source_language_model_output(
        java,
        [{"name": "f", "arguments": {"enabled": True, "ids": [1, 2]}}],
        "simple_java",
    )
    assert converted == [{"f": {
        "enabled": "true",
        "ids": "new ArrayList<Long>(Arrays.asList(1L, 2L))",
    }}]
    javascript = [{"name": "f", "parameters": {"properties": {
        "count": {"type": "integer"}, "options": {"type": "dict"},
    }}}]
    assert source_language_model_output(
        javascript,
        [{"name": "f", "arguments": {"count": 3, "options": {}}}],
        "simple_javascript",
    ) == [{"f": {"count": "3", "options": "{}"}}]
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
