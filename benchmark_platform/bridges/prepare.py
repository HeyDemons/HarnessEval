from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parquet_row(path: Path, case_id: str, id_field: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for row in table.to_pylist():
        if str(row.get(id_field)) == case_id:
            return row
    raise KeyError(f"Case not found: {case_id}")


def prepare_gaia(case_id: str, output: Path) -> None:
    candidates = sorted(Path("/data").rglob("metadata.parquet"))
    row = None
    for path in candidates:
        try:
            row = _parquet_row(path, case_id, "task_id")
            break
        except KeyError:
            continue
    if row is None:
        raise KeyError(f"GAIA case not found: {case_id}")
    workspace = output / "input" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    file_path = str(row.get("file_path") or "")
    attachment = None
    if file_path:
        source = Path(file_path)
        if not source.is_absolute():
            matches = list(Path("/data").rglob(source.name))
            source = matches[0] if matches else Path("/data") / file_path
        if source.is_file():
            attachment = source.name
            shutil.copy2(source, workspace / source.name)
    prompt = str(row["Question"])
    if attachment:
        prompt += f"\n\nAttached file is available in the isolated workspace as: {attachment}"
    _write(output / "input" / "case.json", {"benchmark": "gaia", "case_id": case_id, "prompt": prompt, "attachment": attachment})
    _write(output / "authority" / "gold.json", {"answer": row.get("Final answer"), "level": row.get("Level")})


def prepare_gdpval(case_id: str, output: Path) -> None:
    row = _parquet_row(Path("/data/data/train-00000-of-00001.parquet"), case_id, "task_id")
    workspace = output / "input" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    raw_references = row.get("reference_files") or []
    references = ast.literal_eval(raw_references) if isinstance(raw_references, str) else raw_references
    copied = []
    for relative in references:
        source = Path("/data") / relative
        if source.is_file():
            target = workspace / source.name
            shutil.copy2(source, target)
            copied.append(target.name)
    prompt = str(row["prompt"])
    if copied:
        prompt += "\n\nReference files in the isolated workspace: " + ", ".join(copied)
    _write(output / "input" / "case.json", {"benchmark": "gdpval", "case_id": case_id, "prompt": prompt, "reference_files": copied})
    _write(output / "authority" / "gold.json", {"rubric_json": row.get("rubric_json"), "deliverable_files": row.get("deliverable_files")})


def prepare_trajectory(case_id: str, output: Path) -> None:
    if ":" not in case_id:
        raise ValueError("TRAJECT case id must be RELATIVE_JSON_PATH:INDEX")
    relative, raw_index = case_id.rsplit(":", 1)
    path = (Path("/opt/trajectory/public_data") / relative).resolve()
    root = Path("/opt/trajectory/public_data").resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("TRAJECT case path is outside public_data")
    record = _records(path)[int(raw_index)]
    tools = []
    for tool in record.get("tool list", []):
        tools.append({key: value for key, value in tool.items() if key not in {"executed_output", "execution_status"}})
    _write(output / "input" / "case.json", {"benchmark": "trajectory-bench", "case_id": case_id, "prompt": record["query"], "tools": tools})
    _write(output / "authority" / "gold.json", {"final_answer": record.get("final_answer"), "tool_list": record.get("tool list"), "trajectory_type": record.get("trajectory_type")})


def prepare_bfcl(case_id: str, output: Path) -> None:
    root = Path("/opt/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    selected = None
    source = None
    for path in sorted(root.rglob("*.json")):
        for record in _records(path):
            if str(record.get("id")) == case_id:
                selected, source = record, path.name
                break
        if selected:
            break
    if selected is None:
        raise KeyError(f"BFCL case not found: {case_id}")
    messages = selected.get("question") or []
    prompt = json.dumps(messages, ensure_ascii=False)
    _write(output / "input" / "case.json", {"benchmark": "bfcl", "case_id": case_id, "prompt": prompt, "functions": selected.get("function") or [], "source": source})
    _write(output / "authority" / "gold.json", {key: value for key, value in selected.items() if key not in {"question", "function"}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["gaia", "gdpval", "trajectory-bench", "bfcl"])
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, default=Path("/prepared"))
    args = parser.parse_args()
    if args.output.exists():
        for child in args.output.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        args.output.mkdir(parents=True)
    handlers = {"gaia": prepare_gaia, "gdpval": prepare_gdpval, "trajectory-bench": prepare_trajectory, "bfcl": prepare_bfcl}
    handlers[args.benchmark](args.case, args.output)
    print(json.dumps({"benchmark": args.benchmark, "case_id": args.case, "prepared": True}))


if __name__ == "__main__":
    main()
