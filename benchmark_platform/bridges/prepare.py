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


def _objects(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def _trajectory_tool(tool: dict[str, Any], catalog: Any) -> dict[str, Any]:
    name = str(tool.get("tool name") or "")
    candidates = [
        item
        for item in _objects(catalog)
        if item.get("tool name") == name
        and item.get("domain name")
        and item.get("parent tool name")
        and item.get("API name")
    ]
    signatures = {
        (item["domain name"], item["parent tool name"], item["API name"])
        for item in candidates
    }
    if len(signatures) != 1:
        raise ValueError(f"Expected one executable TRAJECT tool mapping for {name!r}, found {sorted(signatures)!r}")
    definition = max(
        candidates,
        key=lambda item: len(item.get("required_parameters") or []) + len(item.get("optional_parameters") or []),
    )

    def parameters(label: str) -> list[dict[str, Any]]:
        values = {
            str(item.get("name")): item.get("value")
            for item in tool.get(label, []) or []
            if item.get("name")
        }
        schema_label = label.replace(" ", "_")
        merged = []
        for item in definition.get(schema_label, []) or []:
            parameter = dict(item)
            if parameter.get("name") in values:
                parameter["value"] = values[parameter["name"]]
            merged.append(parameter)
        known = {str(item.get("name")) for item in merged}
        merged.extend(dict(item) for item in tool.get(label, []) or [] if str(item.get("name")) not in known)
        return merged

    return {
        "tool name": name,
        "tool description": tool.get("tool description") or definition.get("tool description", ""),
        "domain name": definition["domain name"],
        "parent tool name": definition["parent tool name"],
        "API name": definition["API name"],
        "required parameters": parameters("required parameters"),
        "optional parameters": parameters("optional parameters"),
    }


def _trajectory_source_tools(record: dict[str, Any], case_id: str) -> list[dict[str, Any]]:
    tools = record.get("tool list")
    if tools is None:
        tools = record.get("tool_list")
    if not isinstance(tools, list):
        raise ValueError(f"TRAJECT case has no structured tool list: {case_id}")
    return tools


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
    prompt += (
        "\n\nOutput contract: the final response must contain only the concise answer "
        "that should be passed to the GAIA exact-match scorer."
    )
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
    catalog = json.loads((root / "tools" / "all_tools.json").read_text(encoding="utf-8"))
    source_tools = _trajectory_source_tools(record, case_id)
    tools = [_trajectory_tool(tool, catalog) for tool in source_tools]
    _write(output / "input" / "case.json", {"benchmark": "trajectory-bench", "case_id": case_id, "prompt": record["query"], "tools": tools})
    _write(output / "authority" / "gold.json", {"final_answer": record.get("final_answer"), "tool_list": source_tools, "trajectory_type": record.get("trajectory_type")})


def prepare_bfcl(case_id: str, output: Path) -> None:
    root = Path("/opt/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    selected = None
    source = None
    # Only the top-level category files hold cases. possible_answer/ mirrors their filenames
    # and its records carry the same "id", so rglob let a gold record masquerade as the case
    # itself -- it only ever picked the right one because "B" sorts before "p".
    for path in sorted(root.glob("BFCL_v4_*.json")):
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
    gold = {key: value for key, value in selected.items() if key not in {"question", "function"}}
    # The official scorer dispatches on the category, which is only recoverable from the
    # filename, and grades against an answer key kept in a sibling directory rather than in
    # the case record. Without both, every category except the relevance ones is unscorable
    # -- which is exactly the state this bridge shipped in.
    gold["test_category"] = source[len("BFCL_v4_"):-len(".json")]
    answers = root / "possible_answer" / source
    if answers.is_file():
        for record in _records(answers):
            if str(record.get("id")) == case_id:
                gold.update({key: value for key, value in record.items() if key != "id"})
                break
    _write(output / "authority" / "gold.json", gold)


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
