import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec
from benchmark_platform.harnesses.code_executor import execute_code
from benchmark_platform.harnesses.magentic_one import _participant_turn
from benchmark_platform.harnesses.methods import run_profile
from test_declaration_protocol import Trace
from test_harnesses import ScriptedClient, magentic_ledger, native_tool_call


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    def context(self, root, responses=()):
        trace = Trace()
        async def command(args):
            process = await asyncio.create_subprocess_exec(*args["argv"], cwd=root,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            return {"returncode": process.returncode, "stdout": stdout.decode(), "stderr": stderr.decode()}
        tools = [ToolSpec(name, name, {"type": "object"}, (), read_only=name == "read_file")
                 for name in ("run_command", "read_file", "write_file")]
        return RunContext("magentic-one", "Create a file", ScriptedClient(list(responses)),
            ToolEnvironment(tools, trace, {"run_command": command}), trace, {})

    async def test_executor_runs_python_and_shell_in_same_workspace_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.context(directory)
            response = await _participant_turn(ctx, "Executor", [{"source": "Coder", "content":
                '```python\n# filename: created.py\nfrom pathlib import Path\nPath("evidence.txt").write_text("local-result")\n```\n'
                '```sh\ncat evidence.txt\n```'}])
            self.assertEqual(response, "local-result")
            self.assertTrue((Path(directory) / "created.py").is_file())
            self.assertEqual(ctx.llm_calls, 0)
            self.assertEqual(len(ctx.environment.calls), 1)

    async def test_failure_stops_later_blocks_and_returns_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.context(directory)
            response = await execute_code(ctx, [{"source": "Coder", "content":
                '```sh\necho failed; exit 7\n```\n```sh\ntouch must-not-exist\n```'}])
            self.assertIn("code 7", response)
            self.assertFalse((Path(directory) / "must-not-exist").exists())

    async def test_filename_escape_and_unsupported_language_fail_in_task_container(self):
        with tempfile.TemporaryDirectory() as directory:
            for content in ('```python\n# filename: ../escape.py\nprint(1)\n```',
                            '```unsupported\nprint(1)\n```'):
                response = await execute_code(self.context(directory), [{"source": "Coder", "content": content}])
                self.assertIn("code 1", response)

    async def test_empty_output_and_no_code_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.context(directory)
            response = await execute_code(ctx, [{"source": "Coder", "content": "```python\npass\n```"}])
            self.assertIn("no output; exit code 0", response)
            response = await execute_code(ctx, [{"source": "Coder", "content": "No script"}])
            self.assertIn("No code blocks", response)
            self.assertEqual(len(ctx.environment.calls), 1)

    async def test_file_surfer_cannot_execute_code_or_mutate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.context(directory, [native_tool_call("write_file", {"path": "x"})])
            response = await _participant_turn(ctx, "FileSurfer", [])
            self.assertIn("specialist_tool_not_available", response)
            self.assertEqual(ctx.environment.calls, [])
            self.assertEqual([t["function"]["name"] for t in ctx.client.native_tools[0]], ["read_file"])

    async def test_repeated_executor_dispatch_does_not_rerun_old_code(self):
        code = '```python\nfrom pathlib import Path\np=Path("count")\np.write_text(str(int(p.read_text())+1) if p.exists() else "1")\nprint("done")\n```'
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.context(directory, ["facts", "plan",
                magentic_ledger(satisfied=False, speaker="Coder"), code,
                magentic_ledger(satisfied=False, speaker="Executor"),
                magentic_ledger(satisfied=False, speaker="Executor"),
                magentic_ledger(satisfied=True), "done"])
            self.assertEqual(await run_profile(ctx), "done")
            self.assertEqual((Path(directory) / "count").read_text(), "1")
            self.assertEqual(len(ctx.environment.calls), 1)
            self.assertFalse(ctx.client.native_tools[0])  # Coder writes text/code only.
            self.assertFalse(any(e.get("role") == "Executor" for e in ctx.trace.events if e["event"] == "llm_request"))
