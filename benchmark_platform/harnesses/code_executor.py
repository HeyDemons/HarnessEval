"""Magentic-One's non-LLM code participant, using the benchmark task container."""
from __future__ import annotations

import json
import re

from .core import RunContext, tool_result_content

CODE_BLOCK = re.compile(r"```(?:\s*([\w+\-]+))?\n([\s\S]*?)```")

# Executed THROUGH run_command in the benchmark workspace/container. Never run
# this wrapper in the host scorer. Scripts persist as they do in AutoGen's local
# executor, but no requested filename may escape the workspace (including links).
EXECUTOR_SCRIPT = r'''
import hashlib, json, pathlib, re, subprocess, sys
root = pathlib.Path.cwd().resolve()
for language, code in json.loads(sys.argv[1]):
    interpreters = {"python": ("python3", ".py"), "py": ("python3", ".py"),
                    "sh": ("sh", ".sh"), "shell": ("sh", ".sh"), "bash": ("bash", ".sh")}
    if language not in interpreters:
        print("Unsupported code language: " + language, file=sys.stderr)
        sys.exit(1)
    interpreter, extension = interpreters[language]
    match = re.match(r"\s*#\s*filename:\s*(.+)", code.splitlines()[0] if code.splitlines() else "")
    filename = match.group(1).strip() if match else ".magentic-code/" + hashlib.sha256(code.encode()).hexdigest() + extension
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or path == root:
        print("Code filename escapes the task workspace", file=sys.stderr)
        sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    completed = subprocess.run([interpreter, str(path)], cwd=root)
    if completed.returncode:
        sys.exit(completed.returncode)
'''


async def execute_code(ctx: RunContext, messages: list[dict[str, str]]) -> str:
    blocks = [(language.strip().lower(), code) for message in messages
              for language, code in CODE_BLOCK.findall(message["content"])]
    if not blocks:
        return "No code blocks found. Provide a fenced python or sh script for the Executor."
    if "run_command" not in ctx.environment.tools:
        return "Code execution is unavailable: the benchmark exposes no run_command tool."
    result = await ctx.environment.call("run_command", {
        "argv": ["python3", "-c", EXECUTOR_SCRIPT, json.dumps(blocks, ensure_ascii=False)]})
    await ctx.trace.emit("magentic_code_execution", code_blocks=len(blocks), llm_calls=0,
                         source_messages=[message["source"] for message in messages])
    if result.get("ok") is not True:
        return "Code execution failed: " + tool_result_content(result)
    payload = result.get("result")
    if not isinstance(payload, dict):
        return tool_result_content(result)
    output = str(payload.get("stdout") or "") + str(payload.get("stderr") or "")
    exit_code = payload.get("returncode", payload.get("exit_code"))
    if exit_code is None:
        return tool_result_content(result)
    if not output.strip():
        return f"Code execution produced no output; exit code {exit_code}."
    if exit_code != 0:
        return f"Code execution exited with code {exit_code}.\n{output}"
    return output
