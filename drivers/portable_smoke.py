from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def parquet_probe(root: Path, preferred: tuple[str, ...] = ()) -> tuple[Path, Any]:
    import pandas as pd

    preferred_files = [root / relative for relative in preferred if (root / relative).is_file()]
    files = preferred_files or sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data under {root}")
    frame = pd.read_parquet(files[0])
    if frame.empty:
        raise ValueError(f"Dataset is empty: {files[0]}")
    return files[0], frame


def gaia(root: Path) -> dict[str, Any]:
    path, frame = parquet_probe(
        root,
        (
            "2023/validation/metadata.parquet",
            "2023/validation/metadata.level1.parquet",
        ),
    )
    answer_column = next((name for name in ("Final answer", "answer", "target") if name in frame.columns), None)
    level_column = next((name for name in ("Level", "level") if name in frame.columns), None)
    if not answer_column:
        raise ValueError(f"GAIA answer column is absent: {list(frame.columns)}")
    target = str(frame.iloc[0][answer_column])
    import sys

    sys.path.insert(0, "/opt/platform")
    from gaia_scorer import question_score

    levels = frame[level_column].value_counts().to_dict() if level_column else {}
    return {
        "dataset": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "levels": {str(key): int(value) for key, value in levels.items()},
        "scores": {"official_scorer_oracle_self_test": float(question_score(target, target))},
        "oracle_smoke": True,
    }


def _listed(value: Any, field: str) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"GDPVal {field} must be a list")
    return [str(item) for item in value]


def gdpval(root: Path, suite_path: Path | None = None) -> dict[str, Any]:
    path, frame = parquet_probe(root)
    lower = {str(name).lower() for name in frame.columns}
    required_columns = {
        "task_id",
        "prompt",
        "reference_files",
        "deliverable_files",
        "rubric_json",
    }
    coverage = {
        "task": bool({"prompt", "task"} & lower),
        "rubric": any(name.startswith("rubric") for name in lower),
        "reference": bool({"reference_files", "gold_reference"} & lower),
        "deliverable": "deliverable_files" in lower,
    }
    records = frame.to_dict(orient="records")
    rows = {str(record.get("task_id")): record for record in records}
    missing_assets: list[str] = []
    invalid_rubrics: list[str] = []
    unsafe_assets: list[str] = []
    for task_id, record in rows.items():
        try:
            references = _listed(record.get("reference_files"), "reference_files")
            deliverables = _listed(record.get("deliverable_files"), "deliverable_files")
            rubric = json.loads(str(record.get("rubric_json") or ""))
            if not isinstance(rubric, list) or not rubric:
                raise ValueError("rubric must be a non-empty list")
        except (SyntaxError, ValueError, json.JSONDecodeError):
            invalid_rubrics.append(task_id)
            continue
        for relative in [*references, *deliverables]:
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                unsafe_assets.append(relative)
            elif not (root / relative_path).is_file():
                missing_assets.append(relative)
    suite = json.loads(suite_path.read_text(encoding="utf-8")) if suite_path else None
    expected_sha256 = ((suite or {}).get("source") or {}).get("parquet_sha256")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    light_missing: list[str] = []
    reference_count_mismatches: list[str] = []
    if suite:
        for case in suite.get("cases") or []:
            task_id = str(case["id"])
            record = rows.get(task_id)
            if record is None:
                light_missing.append(task_id)
                continue
            references = _listed(record.get("reference_files"), "reference_files")
            if len(references) != int(case.get("reference_count", -1)):
                reference_count_mismatches.append(task_id)
    complete = (
        required_columns.issubset(lower)
        and all(coverage.values())
        and not missing_assets
        and not invalid_rubrics
        and not unsafe_assets
        and not light_missing
        and not reference_count_mismatches
        and (expected_sha256 is None or actual_sha256 == expected_sha256)
    )
    return {
        "dataset": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "field_coverage": coverage,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "missing_assets": missing_assets[:20],
        "missing_asset_count": len(missing_assets),
        "invalid_rubric_tasks": invalid_rubrics[:20],
        "invalid_rubric_count": len(invalid_rubrics),
        "unsafe_assets": unsafe_assets[:20],
        "light_missing_cases": light_missing,
        "reference_count_mismatches": reference_count_mismatches,
        "scores": {"dataset_integrity": float(complete)},
        "comparability": "Infrastructure smoke only; official GDPval is expert pairwise grading.",
        "oracle_smoke": True,
    }


def trajectory(root: Path) -> dict[str, Any]:
    data = root / "public_data"
    files = sorted((data / "parallel").rglob("*.json")) + sorted((data / "sequential").rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"TRAJECT data is absent under {data}")
    parsed = []
    total = 0
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        count = len(value) if isinstance(value, list) else len(value.get("data", [])) if isinstance(value, dict) else 0
        total += count
        parsed.append({"path": str(path.relative_to(root)), "records": count})
    tools = json.loads((data / "tools" / "all_tools.json").read_text(encoding="utf-8"))
    return {
        "files": len(files),
        "records": total,
        "tools": len(tools),
        "shards": parsed,
        "scores": {"dataset_integrity": float(total > 0 and len(tools) > 0)},
        "oracle_smoke": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["gaia", "gdpval", "trajectory"])
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args()
    result = (
        gdpval(args.root, args.suite)
        if args.benchmark == "gdpval"
        else globals()[args.benchmark](args.root)
    )
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.benchmark == "gdpval" and result["scores"]["dataset_integrity"] != 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
