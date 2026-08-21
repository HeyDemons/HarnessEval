#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def tool_name(original: str) -> str:
    import re

    stem = re.sub(r"[^A-Za-z0-9_]+", "_", original).strip("_").lower() or "tool"
    return f"{stem[:48]}_{hashlib.sha256(original.encode()).hexdigest()[:8]}"


def grade(case: dict[str, Any], gold: dict[str, Any], calls: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    expected = [str(item.get("tool name") or "") for item in gold.get("tool_list") or []]
    mapping = {tool_name(str(item.get("tool name") or "")): str(item.get("tool name") or "") for item in case.get("tools") or []}
    observed = [mapping.get(str(item.get("name") or ""), str(item.get("name") or "")) for item in calls]
    expected_set, observed_set = set(expected), set(observed)
    target = str(gold.get("final_answer") or "").strip()
    answer_exact = bool(target) and answer.strip() == target
    trajectory_exact = expected == observed
    tool_set_exact = expected_set == observed_set
    inclusion = len(expected_set & observed_set) / len(expected_set) if expected_set else None
    # TRAJECT's central measurement is whether the required tool trajectory was recovered.
    # Keep answer exactness and set inclusion alongside it instead of blending unlike metrics.
    return {
        "authority": "traject_pinned_tool_trajectory_metrics",
        "score": 1.0 if trajectory_exact else 0.0,
        "answer_exact": answer_exact,
        "trajectory_exact": trajectory_exact,
        "tool_set_exact": tool_set_exact,
        "tool_inclusion": inclusion,
        "expected_tools": expected,
        "observed_tools": observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads((args.prepared / "input" / "case.json").read_text())
    gold = json.loads((args.prepared / "authority" / "gold.json").read_text())
    calls = json.loads(args.calls.read_text())
    print(json.dumps(grade(case, gold, calls, args.answer.read_text()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
