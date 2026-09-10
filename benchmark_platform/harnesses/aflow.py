"""Execute pinned official AFlow workflows, then explicit benchmark adapters.

Source: FoundationAgents/AFlow 3f457218, HotpotQA/workflows/template and
scripts/formatter.py. Artifacts contain code and must run in the benchmark's
agent sandbox, never in a scorer process holding evaluation labels. The public
``aflow`` profile accepts official QA/math/code artifacts and capability-limited
tool-workflow artifacts; their operator profiles and formats cannot be confused.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import re
from types import SimpleNamespace
from typing import Any, Literal

from . import aflow_official as official
from .core import RunContext

REVISION = official.REVISION
FORMAT = "aflow-official-python-v2"
INITIAL_GRAPH = official.INITIAL_GRAPH

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_artifact(artifact: Any, *, allow_initialization: bool = False,
                      benchmark: str | None = None, case_id: str | None = None) -> dict:
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError("AFlow requires a frozen aflow-official-python-v2 artifact")
    profile = artifact.get("operator_profile")
    if profile not in {*official.OPERATOR_PROFILES, "benchmark-tools"}:
        raise ValueError("AFlow artifact requires an explicit official or benchmark-adapter operator profile")
    if not isinstance(artifact.get("graph"), str) or not artifact["graph"].strip():
        raise ValueError("AFlow artifact must contain Python graph source")
    if not isinstance(artifact.get("prompt"), str):
        raise ValueError("AFlow artifact must contain prompt source")
    for name in ("graph", "prompt"):
        ast.parse(artifact[name])
    if artifact.get("code_sha256") != digest({key: artifact[key] for key in ("graph", "prompt")}):
        raise ValueError("AFlow graph/prompt checksum mismatch")
    provenance = artifact.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("source_revision") != REVISION:
        raise ValueError("AFlow artifact must pin the supported operator revision")
    if provenance.get("kind") == "initialization" and allow_initialization:
        return artifact
    if provenance.get("kind") != "optimized":
        raise ValueError("AFlow evaluation requires an optimized artifact, not an initialization control")
    if provenance.get("operator_profile") != profile:
        raise ValueError("AFlow artifact operator profile does not match its search provenance")
    if (profile == "benchmark-tools") != (provenance.get("benchmark_adapter") is True):
        raise ValueError("AFlow benchmark-adapter provenance does not match its operator profile")
    for key in ("optimization_case_ids", "evaluation_case_ids"):
        values = provenance.get(key)
        if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"AFlow provenance requires nonempty {key}")
        if len(values) != len(set(values)):
            raise ValueError(f"AFlow provenance contains duplicate {key}")
    training, evaluation = set(provenance["optimization_case_ids"]), set(provenance["evaluation_case_ids"])
    if training & evaluation:
        raise ValueError("AFlow optimization and evaluation cases overlap")
    if benchmark is not None and provenance.get("benchmark") != benchmark:
        raise ValueError("AFlow artifact benchmark mismatch")
    if case_id is not None and case_id not in evaluation:
        raise ValueError("AFlow case is not in the artifact's frozen evaluation manifest")
    score = provenance.get("validation_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("AFlow artifact requires a finite validation score")
    if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("search_history_sha256", ""))):
        raise ValueError("AFlow artifact requires search history provenance")
    return artifact


def validate_runtime_artifact(artifact: Any, *, allow_initialization: bool = False,
                              benchmark: str | None = None, case_id: str | None = None) -> dict:
    """Validate either artifact family accepted by the public ``aflow`` method."""
    if isinstance(artifact, dict) and artifact.get("format") == "aflow-benchmark-tools-python-v2":
        from .aflow_tools import validate_artifact as validate_tool_artifact

        return validate_tool_artifact(
            artifact,
            allow_initialization=allow_initialization,
            benchmark=benchmark,
            case_id=case_id,
        )
    return validate_artifact(
        artifact,
        allow_initialization=allow_initialization,
        benchmark=benchmark,
        case_id=case_id,
    )


def make_artifact(graph: str | None = None, prompt: str = "", *, operator_profile: str = "qa",
                  provenance: dict | None = None) -> dict:
    if operator_profile not in {*official.OPERATOR_PROFILES, "benchmark-tools"}:
        raise ValueError(f"Unknown AFlow operator profile: {operator_profile}")
    if graph is None:
        if operator_profile == "benchmark-tools":
            raise ValueError("Benchmark-tool AFlow requires an adapter graph")
        graph = official.initial_graph(operator_profile)
    return {"format": FORMAT, "operator_profile": operator_profile, "graph": graph, "prompt": prompt,
            "code_sha256": digest({"graph": graph, "prompt": prompt}),
            "provenance": provenance or {"kind": "initialization", "source_revision": REVISION}}


class OperatorLLM:
    def __init__(self, ctx: RunContext, operator_profile: str = "qa"):
        self.ctx = ctx
        self.operator_profile = operator_profile
        self.calls = 0

    async def generate(self, name: str, prompt: str, fields: dict[str, str] | None = None):
        if fields:
            examples = "\n".join(f"<{key}>{description}</{key}>" for key, description in fields.items())
            prompt += ("\n# Response format (must be strictly followed) (do not include any other formats except "
                       f"for the given XML format):\n{examples}")
        self.calls += 1
        call = self.calls
        if call > int(self.ctx.policy.get("aflow_max_operator_calls", 100)):
            raise RuntimeError("AFlow operator call budget exhausted")
        raw = await self.ctx.complete(f"aflow_{name}_{call}", [{"role": "user", "content": prompt}])
        if fields is None:
            return {"response": raw}
        # Pinned XmlFormatter returns the parsed fields without filling defaults
        # or requiring the optional model fields. A graph may use only `answer`
        # or `solution_letter`; missing `thought` must not fail that graph.
        return {key: value.strip() for key, value in re.findall(r"<(\w+)>(.*?)</\1>", raw, re.DOTALL)}

    async def generate_code(self, name: str, prompt: str, function_name: str | None = None):
        suffix = (
            "\n\nPlease write your code solution in Python. Return ONLY the complete, runnable code "
            "without explanations. Use proper Python syntax and formatting."
        )
        if function_name:
            suffix += f"\nMake sure to include a function named '{function_name}' in your solution."
        value = await self.generate(name, prompt + suffix)
        raw = value["response"]
        match = re.search(r"```python\s*(.*?)\s*```", raw, re.DOTALL) or re.search(
            r"```\s*(.*?)\s*```", raw, re.DOTALL
        )
        code = (match.group(1) if match else raw).strip()
        ast.parse(code)
        if function_name and not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
            for node in ast.parse(code).body
        ):
            raise ValueError(f"AFlow generated code omitted entry point {function_name}")
        return {"response": code}

    def get_usage_summary(self):
        # The public SDK returns cost with the answer. We have no provider price
        # table here; token counters remain authoritative and monetary cost unknown.
        return {"total_cost": None, "prompt_tokens": self.ctx.prompt_tokens,
                "completion_tokens": self.ctx.completion_tokens}


# Public names remain import-compatible while their semantics come from the
# pinned official operator module.
Custom = official.Custom
AnswerGenerate = official.AnswerGenerate
ScEnsemble = official.ScEnsemble
Programmer = official.Programmer
CustomCodeGenerate = official.CustomCodeGenerate
Test = official.Test


def graph_namespace(artifact: dict, llm: OperatorLLM, *, operators: dict | None = None,
                    builtins_override: dict | None = None) -> dict:
    """Redirect only pinned AFlow infrastructure imports, preserving graph code.

    This is a compatibility shim, NOT a security sandbox. Importing generated
    Python on the host scorer is forbidden; the caller owns process isolation.
    """
    prompts: dict[str, Any] = {} if builtins_override is None else {"__builtins__": builtins_override}
    exec(compile(artifact["prompt"], "<aflow-prompts>", "exec"), prompts)
    profile = artifact.get("operator_profile", "qa")
    official_profile = "qa" if profile == "benchmark-tools" else profile
    namespace = {"operator": SimpleNamespace(**official.namespace(official_profile), **(operators or {})),
                 "prompt_custom": SimpleNamespace(**{k: v for k, v in prompts.items() if not k.startswith("__")}),
                 "create_llm_instance": lambda config: llm, "DatasetType": str, "Literal": Literal}
    if builtins_override is not None:
        namespace.update(__builtins__=builtins_override, __name__="aflow_operator_graph")
    tree = ast.parse(artifact["graph"])
    keep = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            aliases = []
            for alias in statement.names:
                if alias.name.startswith("workspace."):
                    dataset = official.DATASET_BY_PROFILE.get(official_profile, "HotpotQA")
                    if not alias.name.startswith(f"workspace.{dataset}.workflows.") or alias.asname not in {"operator", "prompt_custom"}:
                        raise ValueError("AFlow graph imports a dataset/operator profile inconsistent with its artifact")
                else:
                    aliases.append(alias)
            if aliases:
                statement.names = aliases
                keep.append(statement)
        elif isinstance(statement, ast.ImportFrom) and statement.module in {"scripts.async_llm", "scripts.evaluator"}:
            expected = "create_llm_instance" if statement.module == "scripts.async_llm" else "DatasetType"
            if any(alias.name != expected or alias.asname for alias in statement.names):
                raise ValueError("Unsupported AFlow infrastructure import")
        else:
            keep.append(statement)
    tree.body = keep
    exec(compile(tree, "<aflow-graph>", "exec"), namespace)
    return namespace


async def run_aflow(ctx: RunContext) -> str:
    raw_artifact = ctx.policy.get("aflow_artifact")
    if isinstance(raw_artifact, dict) and raw_artifact.get("format") == "aflow-benchmark-tools-python-v2":
        from .aflow_tools import run_aflow_tools

        return await run_aflow_tools(ctx)
    artifact = validate_artifact(raw_artifact,
                                 allow_initialization=ctx.policy.get("aflow_allow_initialization") is True,
                                 benchmark=ctx.policy.get("aflow_benchmark"), case_id=ctx.policy.get("aflow_case_id"))
    profile = artifact["operator_profile"]
    await ctx.trace.emit("aflow_artifact", code_sha256=artifact["code_sha256"],
                         provenance=artifact["provenance"], implementation="official-python-v3",
                         operator_profile=profile, source_revision=REVISION)
    namespace = graph_namespace(artifact, OperatorLLM(ctx, profile))
    workflow_type = namespace.get("Workflow")
    if not inspect.isclass(workflow_type):
        raise ValueError("AFlow graph must define Workflow")
    dataset = official.DATASET_BY_PROFILE[profile]
    workflow = workflow_type(name="AFlow", llm_config={}, dataset=dataset)
    if profile == "code":
        entry_point = ctx.policy.get("aflow_entry_point")
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError("Official AFlow code workflow requires policy.aflow_entry_point")
        result = await workflow(ctx.prompt, entry_point)
    else:
        result = await workflow(ctx.prompt)
    answer = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    if not isinstance(answer, str):
        raise ValueError("AFlow Workflow must return an answer string (optionally with cost)")
    return answer
