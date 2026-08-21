#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_COUNTRY_MAPPINGS = {
    "FR": "France", "US": "United States", "USA": "United States",
    "UK": "United Kingdom", "GB": "United Kingdom", "DE": "Germany",
    "IT": "Italy", "ES": "Spain", "JP": "Japan", "CN": "China",
    "AU": "Australia", "CA": "Canada", "IN": "India", "BR": "Brazil",
    "MX": "Mexico", "RU": "Russia", "KR": "South Korea", "NL": "Netherlands",
    "SE": "Sweden", "NO": "Norway", "DK": "Denmark", "FI": "Finland",
    "CH": "Switzerland", "AT": "Austria", "BE": "Belgium", "PT": "Portugal",
    "GR": "Greece", "TR": "Turkey", "PL": "Poland", "CZ": "Czech Republic",
    "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia",
    "SI": "Slovenia", "SK": "Slovakia", "LT": "Lithuania", "LV": "Latvia",
    "EE": "Estonia",
}


def tool_name(original: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", original).strip("_").lower() or "tool"
    return f"{stem[:48]}_{hashlib.sha256(original.encode()).hexdigest()[:8]}"


def _parameter_rows(tool: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    spaced = f"{kind} parameters"
    underscored = f"{kind}_parameters"
    rows = tool.get(spaced)
    if rows is None:
        rows = tool.get(underscored)
    return [item for item in rows or [] if isinstance(item, dict)]


def _normalize_value(value: Any) -> Any:
    """Match the pinned TRAJECT value normalization used by tool_traj_usage."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            value = stripped
    if isinstance(value, str):
        value = re.sub(r"\s*,\s*", ",", value)
        value = re.sub(r"\s+", " ", value).strip()
        if "," in value:
            canonical = {name.casefold(): name for name in _COUNTRY_MAPPINGS.values()}
            value = ",".join(
                _COUNTRY_MAPPINGS.get(part.upper(), canonical.get(part.casefold(), part))
                for part in value.split(",")
            )
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
            try:
                value = datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                pass
        lowered = value.casefold()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        try:
            return int(value) if re.fullmatch(r"[+-]?\d+", value) else float(value)
        except ValueError:
            return value
    return value


def _normalized_parameters(tool: dict[str, Any], kind: str) -> list[tuple[str, Any]]:
    normalized = []
    for row in _parameter_rows(tool, kind):
        value = row.get("value")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        normalized.append((str(row.get("name") or "").strip(), _normalize_value(value)))
    return sorted(normalized, key=lambda item: item[0])


def _predicted_tools(
    case_tools: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_wire_name = {
        tool_name(str(tool.get("tool name") or "")): tool for tool in case_tools
    }
    predicted = []
    for call in calls:
        wire_name = str(call.get("name") or "")
        source = by_wire_name.get(wire_name)
        original_name = str((source or {}).get("tool name") or wire_name)
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        required_names = {
            str(row.get("name") or "") for row in _parameter_rows(source or {}, "required")
        }
        predicted.append(
            {
                "tool name": original_name,
                "required parameters": [
                    {"name": name, "value": value}
                    for name, value in arguments.items()
                    if name in required_names
                ],
                "optional parameters": [
                    {"name": name, "value": value}
                    for name, value in arguments.items()
                    if name not in required_names
                ],
            }
        )
    return predicted


def _tool_usage(expected: list[dict[str, Any]], observed: list[dict[str, Any]]) -> list[bool]:
    expected_names = [str(item.get("tool name") or "") for item in expected]
    observed_names = [str(item.get("tool name") or "") for item in observed]
    usage = []
    for name in sorted(set(expected_names) & set(observed_names)):
        target = next(item for item in expected if str(item.get("tool name") or "") == name)
        prediction = next(item for item in observed if str(item.get("tool name") or "") == name)
        usage.append(
            _normalized_parameters(target, "required")
            == _normalized_parameters(prediction, "required")
            and _normalized_parameters(target, "optional")
            == _normalized_parameters(prediction, "optional")
        )
    return usage


def grade(case: dict[str, Any], gold: dict[str, Any], calls: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    expected_tools = [item for item in gold.get("tool_list") or [] if isinstance(item, dict)]
    predicted_tools = _predicted_tools(
        [item for item in case.get("tools") or [] if isinstance(item, dict)],
        calls,
    )
    expected = [str(item.get("tool name") or "") for item in expected_tools]
    observed = [str(item.get("tool name") or "") for item in predicted_tools]
    expected_set, observed_set = set(expected), set(observed)
    trajectory_type = str(
        gold.get("trajectory_type") or case.get("trajectory_type") or "sequential"
    ).casefold()
    target = str(gold.get("final_answer") or "").strip()
    answer_exact = bool(target) and answer.strip() == target
    trajectory_exact = (
        expected_set == observed_set
        if trajectory_type == "parallel"
        else expected == observed
    )
    tool_set_exact = expected_set == observed_set
    inclusion = len(expected_set & observed_set) / max(1, len(expected))
    usage = _tool_usage(expected_tools, predicted_tools)
    # TRAJECT's central measurement is whether the required tool trajectory was recovered.
    # Keep answer exactness and set inclusion alongside it instead of blending unlike metrics.
    return {
        "authority": "traject_pinned_tool_trajectory_metrics",
        "score": 1.0 if trajectory_exact else 0.0,
        "trajectory_type": trajectory_type,
        "answer_exact": answer_exact,
        "trajectory_exact": trajectory_exact,
        "tool_set_exact": tool_set_exact,
        "tool_inclusion": inclusion,
        "tool_traj_usage": usage,
        "tool_usage": sum(usage) / len(usage) if usage else 0.0,
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
