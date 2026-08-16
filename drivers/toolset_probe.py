from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if ".git" not in path.parts)


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return [value] if isinstance(value, dict) else []
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        return records


def _workspace(kind: str) -> dict[str, Any]:
    modules = ["bs4", "ddgs", "openpyxl", "pandas", "pdfplumber", "PIL", "pyarrow", "pypdf", "docx", "pptx"]
    commands = ["bash", "curl", "file", "ffmpeg", "git", "jq", "pdftotext", "rg", "sqlite3", "tesseract", "unzip"]
    if kind == "gdpval":
        commands.append("libreoffice")
    module_status = {name: importlib.util.find_spec(name) is not None for name in modules}
    command_status = {name: shutil.which(name) is not None for name in commands}
    complete = all(module_status.values()) and all(command_status.values())
    return {
        "toolset": "workspace",
        "modules": module_status,
        "commands": command_status,
        "ddgs_json_stdin": True,
        "tool_count": len(modules) + len(commands),
        "complete": complete,
    }


def _trajectory() -> dict[str, Any]:
    root = Path("/opt/trajectory/public_data")
    tools = json.loads((root / "tools" / "all_tools.json").read_text(encoding="utf-8"))
    task_files = [path for path in _json_files(root) if "tools" not in path.parts and path.name not in {"selected_category.json"}]
    task_count = sum(len(_load_records(path)) for path in task_files)
    invalid = []
    names = []
    for index, tool in enumerate(tools):
        name = tool.get("tool name")
        names.append(name)
        for collection in ("required_parameters", "optional_parameters"):
            if not isinstance(tool.get(collection, []), list):
                invalid.append({"index": index, "field": collection})
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ["openai", "requests", "sklearn"]
    }
    complete = not invalid and all(names) and all(dependencies.values())
    return {
        "toolset": "trajectory-native",
        "tool_count": len(tools),
        "unique_tool_names": len(set(names)),
        "duplicate_tool_records": len(names) - len(set(names)),
        "task_records": task_count,
        "invalid_tools": invalid,
        "dependencies": dependencies,
        "optional_retrieval_backend": {
            name: importlib.util.find_spec(name) is not None
            for name in ["sentence_transformers", "torch"]
        },
        "external_execution_configured": bool(os.getenv("API_URL") and os.getenv("TOOLBENCH_KEY")),
        "complete": complete,
    }


def _vitabench() -> dict[str, Any]:
    from vita.registry import registry

    domains: dict[str, Any] = {}
    task_sets: dict[str, Any] = {}
    all_valid = True
    for name in registry.get_domains():
        environment = registry.get_env_constructor(name)()
        tools = environment.get_tools()
        invalid = [tool.name for tool in tools if not isinstance(tool.openai_schema, dict)]
        all_valid = all_valid and not invalid
        domains[name] = {"tool_count": len(tools), "invalid_schemas": invalid}
    for name in registry.get_task_sets():
        try:
            tasks = registry.get_tasks_loader(name)(language="english")
            task_sets[name] = {"task_count": len(tasks)}
            all_valid = all_valid and bool(tasks)
        except Exception as exc:
            task_sets[name] = {"error": f"{type(exc).__name__}: {exc}"}
            all_valid = False
    return {
        "toolset": "vitabench-native",
        "domains": domains,
        "task_sets": task_sets,
        "tool_count": sum(item["tool_count"] for item in domains.values()),
        "task_count": sum(item.get("task_count", 0) for item in task_sets.values()),
        "complete": all_valid and bool(domains) and bool(task_sets),
    }


def _tau2() -> dict[str, Any]:
    import tau2.registry as registry

    domains: dict[str, Any] = {}
    task_sets: dict[str, Any] = {}
    all_valid = True
    for name in registry.get_domains():
        try:
            if name == "banking_knowledge":
                from tau2.domains.banking_knowledge.environment import get_db
                from tau2.domains.banking_knowledge.retrieval_toolkits import (
                    KnowledgeToolsAllTools,
                )

                # The official alltools constructor eagerly builds a dense index.
                # Schema validation must not call an embedding API or alter its cache.
                toolkit = KnowledgeToolsAllTools(get_db(), object(), object(), object())
                tools = list(toolkit.get_tools().values())
            else:
                environment = registry.get_env_constructor(name)()
                tools = environment.get_tools()
            invalid = [tool.name for tool in tools if not isinstance(tool.openai_schema, dict)]
            domains[name] = {"tool_count": len(tools), "invalid_schemas": invalid}
            if name == "banking_knowledge":
                domains[name]["schema_probe"] = "official_alltools_without_eager_index"
                domains[name]["runtime_requirement"] = (
                    "The default dense retrieval variant requires a configured "
                    "embedding provider when an embeddings cache is absent."
                )
            all_valid = all_valid and not invalid
        except Exception as exc:
            domains[name] = {"error": f"{type(exc).__name__}: {exc}"}
            all_valid = False
    for name in registry.get_task_sets():
        try:
            tasks = registry.get_tasks_loader(name)()
            task_sets[name] = {"task_count": len(tasks)}
            all_valid = all_valid and bool(tasks)
        except Exception as exc:
            task_sets[name] = {"error": f"{type(exc).__name__}: {exc}"}
            all_valid = False
    return {
        "toolset": "tau2-native",
        "domains": domains,
        "task_sets": task_sets,
        "tool_count": sum(item.get("tool_count", 0) for item in domains.values()),
        "task_count": sum(item.get("task_count", 0) for item in task_sets.values()),
        "complete": all_valid and bool(domains) and bool(task_sets),
    }


def _bfcl() -> dict[str, Any]:
    root = Path("/opt/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    files = _json_files(root)
    records = 0
    functions = 0
    invalid = []
    for path in files:
        for row_index, record in enumerate(_load_records(path)):
            records += 1
            declared = record.get("function") or record.get("tools") or []
            if not isinstance(declared, list):
                invalid.append({"file": path.name, "row": row_index, "error": "function list is not an array"})
                continue
            for function in declared:
                function = function.get("function", function) if isinstance(function, dict) else {}
                functions += 1
                if not function.get("name") or not isinstance(function.get("parameters", {}), dict):
                    invalid.append({"file": path.name, "row": row_index, "error": "invalid function schema"})
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ["bfcl_eval", "tree_sitter", "sentence_transformers", "faiss", "rank_bm25"]
    }
    return {
        "toolset": "bfcl-v4",
        "data_files": len(files),
        "records": records,
        "function_schemas": functions,
        "invalid_schemas": invalid,
        "dependencies": dependencies,
        "complete": not invalid and all(dependencies.values()) and records > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["gaia", "gdpval", "trajectory", "vitabench", "tau2", "bfcl"])
    parser.add_argument("--output", type=Path, default=Path("/job/payload.json"))
    args = parser.parse_args()
    probes = {
        "gaia": lambda: _workspace("gaia"),
        "gdpval": lambda: _workspace("gdpval"),
        "trajectory": _trajectory,
        "vitabench": _vitabench,
        "tau2": _tau2,
        "bfcl": _bfcl,
    }
    toolset = probes[args.benchmark]()
    payload = {
        "benchmark": args.benchmark,
        "toolset": toolset,
        "scores": {"complete_toolset_load": 1.0 if toolset["complete"] else 0.0},
        "oracle_smoke": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(f".{args.output.name}.tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not toolset["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
