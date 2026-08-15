from __future__ import annotations

import argparse
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
    from benchmark_platform.scorers.gaia import question_score

    levels = frame[level_column].value_counts().to_dict() if level_column else {}
    return {
        "dataset": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "levels": {str(key): int(value) for key, value in levels.items()},
        "scores": {"official_scorer_oracle_self_test": float(question_score(target, target))},
        "oracle_smoke": True,
    }


def gdpval(root: Path) -> dict[str, Any]:
    path, frame = parquet_probe(root)
    lower = {str(name).lower() for name in frame.columns}
    coverage = {
        "task": bool({"prompt", "task"} & lower),
        "rubric": any(name.startswith("rubric") for name in lower),
        "reference": bool({"reference_files", "gold_reference"} & lower),
    }
    return {
        "dataset": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "field_coverage": coverage,
        "scores": {"dataset_integrity": float(coverage["task"] and coverage["rubric"])},
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["gaia", "gdpval", "trajectory"])
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = globals()[args.benchmark](args.root)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
