from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import signal
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from benchmark_platform.harnesses.core import SUBPROCESS_ENV, ToolHandler, ToolImage, ToolSpec


MODEL_IMAGE_MIME_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}
# Long enough to pip install a package over the proxy, far shorter than the episode it would
# otherwise be able to consume. HARNESS_COMMAND_TIMEOUT_S overrides it and is on the bridge
# env allowlist, so a benchmark whose commands legitimately run longer can raise it.
COMMAND_TIMEOUT_S = float(os.getenv("HARNESS_COMMAND_TIMEOUT_S", "180"))
# inspect_evals' default GAIA solver uses Inspect's bash/python tools. Inspect caps one
# model-visible tool output at 16 KiB; keep the same default here so a command cannot insert
# several megabytes into the next model request. HARNESS_TASK_OUTPUT_LIMIT already exposes
# this compatibility knob for the task bridge, so workspace and task arms share it.
TOOL_OUTPUT_LIMIT = int(os.getenv("HARNESS_TASK_OUTPUT_LIMIT", str(16 * 1024)))


def clip_tool_output(text: str, stream: str) -> str:
    if len(text) <= TOOL_OUTPUT_LIMIT:
        return text
    half = TOOL_OUTPUT_LIMIT // 2
    truncated = text[:half] + text[-(TOOL_OUTPUT_LIMIT - half) :]
    return (
        f"The {stream} of your command was too long to be displayed.\n"
        f"Here is a truncated version:\n"
        f"<START_TOOL_OUTPUT>\n{truncated}\n<END_TOOL_OUTPUT>"
    )


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
    web_tasks: dict[tuple[str, int], asyncio.Task[list[dict[str, Any]]]] = {}
    web_tasks_lock = asyncio.Lock()

    async def list_files(arguments: dict[str, Any]) -> Any:
        base = safe_path(root, str(arguments.get("path", ".")))
        if "pattern" not in arguments:
            candidates = base.rglob("*")
        else:
            pattern = str(arguments["pattern"])
            pattern_path = Path(pattern)
            if pattern_path.is_absolute() or ".." in pattern_path.parts:
                raise ValueError("list_files pattern must stay relative to the requested path")
            # Match against complete relative paths. Basename-only fnmatch made ``**/*``
            # exclude root files and made directory-bearing patterns impossible.
            candidates = base.glob(pattern)
        files = []
        for path in sorted(candidates):
            safe_path(root, str(path))
            if path.is_file():
                files.append(str(path.relative_to(root)))
        return {"root": str(root), "files": files}

    async def read_file(arguments: dict[str, Any]) -> Any:
        path = safe_path(root, str(arguments["path"]))
        payload = path.read_bytes()
        relative = str(path.relative_to(root))
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if mime_type in MODEL_IMAGE_MIME_TYPES:
                return {
                    "path": relative,
                    "mime_type": mime_type,
                    "bytes": len(payload),
                    "image": ToolImage(mime_type, payload),
                }
            return {
                "path": relative,
                "mime_type": mime_type,
                "bytes": len(payload),
                "binary": True,
                "message": (
                    "Binary content is not inlined. Inspect it with run_command and a format-aware "
                    "program so raw bytes do not consume the model context."
                ),
            }
        return {"path": relative, "text": clip_tool_output(text, "content")}

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
        environment = {name: value for name, value in os.environ.items() if name in SUBPROCESS_ENV}
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
            # A new session so the timeout below can reap the whole tree: a shell that
            # backgrounds a fetch outlives a signal sent to the shell alone.
            start_new_session=True,
            **kwargs,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), COMMAND_TIMEOUT_S)
        except asyncio.TimeoutError:
            # Nothing bounded this call before, and a GAIA case proved what that costs: with
            # no route out, the model's curl hung ~90s per try and the case reached 1678s
            # before the runner cancelled it, scoring nothing. Now that the proxy is passed
            # through, commands really do reach the network, so an unbounded wait is a live
            # way to lose a case. Report it as a tool failure the model can read and retry
            # against rather than letting it consume the whole episode.
            # ponytail: output buffered before the kill is dropped; stream the pipes if a
            # case ever needs the partial transcript of a command that timed out.
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            with suppress(Exception):
                await process.wait()
            # Same keys as the command_failed payload below. plan-execute and rewoo resolve
            # cross-step references like "$s1.stdout" straight off this dict and raise an
            # uncaught KeyError when a key is missing, which kills the whole arm rather than
            # returning an error the model could react to; a timeout must not become a second
            # way to trigger that.
            return {
                "ok": False,
                "error": "command_timeout",
                "argv": argv,
                "cwd": str(cwd.relative_to(root)),
                "returncode": None,
                "stdout": "",
                "stderr": f"command exceeded {COMMAND_TIMEOUT_S}s and was killed",
                "timeout_seconds": COMMAND_TIMEOUT_S,
            }
        payload = {
            "argv": argv,
            "cwd": str(cwd.relative_to(root)),
            "returncode": process.returncode,
            "stdout": clip_tool_output(stdout.decode("utf-8", errors="replace"), "stdout"),
            "stderr": clip_tool_output(stderr.decode("utf-8", errors="replace"), "stderr"),
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
                (
                    "Read one workspace file. UTF-8 files return text with long content middle-truncated; "
                    "supported images are attached as model image input; other binary files return metadata "
                    "only and should be inspected with run_command."
                ),
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
            (
                "Run one argv command in the isolated case workspace. The process receives no API "
                "credentials; long stdout and stderr are middle-truncated to the Inspect tool-output limit."
            ),
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
            max_results = int(arguments.get("max_results", 10))
            key = (query, max_results)
            async with web_tasks_lock:
                task = web_tasks.get(key)
                if task is None:
                    task = asyncio.create_task(
                        asyncio.to_thread(lambda: list(DDGS().text(query, max_results=max_results)))
                    )
                    web_tasks[key] = task
            try:
                results = await asyncio.shield(task)
            except Exception:
                async with web_tasks_lock:
                    if web_tasks.get(key) is task:
                        web_tasks.pop(key, None)
                raise
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
