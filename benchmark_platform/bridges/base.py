from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from benchmark_platform.harnesses.core import ToolHandler, ToolSpec


@dataclass
class BridgeCase:
    benchmark: str
    case_id: str
    prompt: str
    tools: list[ToolSpec]
    handlers: Mapping[str, ToolHandler]
    metadata: dict[str, Any] = field(default_factory=dict)


def native_spec(
    name: str,
    description: str,
    parameters: Mapping[str, Any],
    *,
    parallel: bool,
    read_only: bool,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        command=("/bin/false",),
        parallel=parallel,
        read_only=read_only,
    )


def safe_path(root: Path, raw: str) -> Path:
    root = root.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes the benchmark workspace: {raw}")
    return resolved


def workspace_tools(root: Path, *, writable: bool, include_web: bool) -> tuple[list[ToolSpec], dict[str, ToolHandler]]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs: list[ToolSpec] = []
    handlers: dict[str, ToolHandler] = {}

    async def list_files(arguments: dict[str, Any]) -> Any:
        base = safe_path(root, str(arguments.get("path", ".")))
        pattern = str(arguments.get("pattern", "*"))
        files = [str(path.relative_to(root)) for path in sorted(base.rglob("*")) if path.is_file() and fnmatch.fnmatch(path.name, pattern)]
        return {"root": str(root), "files": files}

    async def read_file(arguments: dict[str, Any]) -> Any:
        path = safe_path(root, str(arguments["path"]))
        payload = path.read_bytes()
        try:
            return {"path": str(path.relative_to(root)), "text": payload.decode("utf-8")}
        except UnicodeDecodeError:
            return {"path": str(path.relative_to(root)), "encoding": "base64", "content": base64.b64encode(payload).decode("ascii")}

    async def write_file(arguments: dict[str, Any]) -> Any:
        path = safe_path(root, str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(arguments["content"])
        path.write_text(content, encoding="utf-8")
        return {"path": str(path.relative_to(root)), "bytes": len(content.encode("utf-8"))}

    async def edit_file(arguments: dict[str, Any]) -> Any:
        path = safe_path(root, str(arguments["path"]))
        old = str(arguments["old"])
        current = path.read_text(encoding="utf-8")
        if current.count(old) != 1:
            raise ValueError("old text must occur exactly once")
        updated = current.replace(old, str(arguments["new"]), 1)
        path.write_text(updated, encoding="utf-8")
        return {"path": str(path.relative_to(root)), "bytes": len(updated.encode("utf-8"))}

    async def run_command(arguments: dict[str, Any]) -> Any:
        argv = [str(item) for item in arguments["argv"]]
        if not argv:
            raise ValueError("argv must not be empty")
        cwd = safe_path(root, str(arguments.get("cwd", ".")))
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH"}
        }
        kwargs: dict[str, Any] = {}
        if os.name == "posix" and os.getuid() == 0:
            kwargs.update({"user": 65534, "group": 65534})
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        stdout, stderr = await process.communicate()
        payload = {
            "argv": argv,
            "cwd": str(cwd.relative_to(root)),
            "returncode": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
        if process.returncode != 0:
            return {"ok": False, "error": "command_failed", **payload}
        return {"ok": True, "result": payload}

    specs.extend(
        [
            native_spec(
                "list_files",
                "List complete relative file paths in the isolated case workspace.",
                {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}}},
                parallel=True,
                read_only=True,
            ),
            native_spec(
                "read_file",
                "Read one complete workspace file; binary files are returned as base64.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                parallel=True,
                read_only=True,
            ),
        ]
    )
    handlers.update({"list_files": list_files, "read_file": read_file})
    specs.append(
        native_spec(
            "run_command",
            "Run one argv command in the isolated case workspace. The process receives no API credentials; stdout and stderr are returned completely.",
            {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
            parallel=False,
            read_only=False,
        )
    )
    handlers["run_command"] = run_command
    if writable:
        specs.extend(
            [
                native_spec(
                    "write_file",
                    "Write complete UTF-8 content in the isolated case workspace.",
                    {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
                    parallel=False,
                    read_only=False,
                ),
                native_spec(
                    "edit_file",
                    "Replace one unique text occurrence in a workspace file.",
                    {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]},
                    parallel=False,
                    read_only=False,
                ),
            ]
        )
        handlers.update({"write_file": write_file, "edit_file": edit_file})
    if include_web:
        async def web_search(arguments: dict[str, Any]) -> Any:
            from ddgs import DDGS

            query = str(arguments["query"])
            results = list(DDGS().text(query, max_results=int(arguments.get("max_results", 10))))
            return {"query": query, "results": results}

        specs.append(
            native_spec(
                "web_search",
                "Search public web sources with DDGS and return complete title, URL, and snippet records.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1}},
                    "required": ["query"],
                },
                parallel=True,
                read_only=True,
            )
        )
        handlers["web_search"] = web_search
    return specs, handlers


def read_case(root: Path) -> dict[str, Any]:
    value = json.loads((root / "case.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
        raise ValueError("Prepared bridge case is invalid")
    return value
