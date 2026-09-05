"""Execute frozen Python workflows with the pinned AFlow QA operator semantics.

Source: FoundationAgents/AFlow 3f457218, HotpotQA/workflows/template and
scripts/formatter.py. Artifacts contain code and must run in the benchmark's
agent sandbox, never in a scorer process holding evaluation labels.
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

from .core import RunContext

REVISION = "3f457218fc716093fe53f6df8a5d5e6379d66346"
FORMAT = "aflow-python-v1"
INITIAL_GRAPH = '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem):
        solution = await self.custom(input=problem, instruction="")
        return solution["response"], self.llm.get_usage_summary()["total_cost"]
'''

ANSWER_PROMPT = '''
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
'''
ENSEMBLE_PROMPT = '''
Several answers have been generated to a same question. They are as follows:
{solutions}

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.
In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
'''


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_artifact(artifact: Any, *, allow_initialization: bool = False,
                      benchmark: str | None = None, case_id: str | None = None) -> dict:
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError("AFlow requires a frozen aflow-python-v1 artifact; operator lists are not optimized graphs")
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


def make_artifact(graph: str = INITIAL_GRAPH, prompt: str = "", *, provenance: dict | None = None) -> dict:
    return {"format": FORMAT, "graph": graph, "prompt": prompt,
            "code_sha256": digest({"graph": graph, "prompt": prompt}),
            "provenance": provenance or {"kind": "initialization", "source_revision": REVISION}}


class OperatorLLM:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
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

    def get_usage_summary(self):
        # The public SDK returns cost with the answer. We have no provider price
        # table here; token counters remain authoritative and monetary cost unknown.
        return {"total_cost": None, "prompt_tokens": self.ctx.prompt_tokens,
                "completion_tokens": self.ctx.completion_tokens}


class Custom:
    def __init__(self, llm: OperatorLLM, name: str = "Custom"):
        self.llm, self.name = llm, name

    async def __call__(self, input: str, instruction: str):
        return await self.llm.generate(self.name, instruction + input)


class AnswerGenerate:
    def __init__(self, llm: OperatorLLM, name: str = "AnswerGenerate"):
        self.llm, self.name = llm, name

    async def __call__(self, input: str, mode: str | None = None):
        return await self.llm.generate(self.name, ANSWER_PROMPT.format(input=input),
                                      {"thought": "The step by step thinking process", "answer": "The final answer to the question"})


class ScEnsemble:
    def __init__(self, llm: OperatorLLM, name: str = "ScEnsemble"):
        self.llm, self.name = llm, name

    async def __call__(self, solutions: list[str]):
        if not solutions or len(solutions) > 26:
            raise ValueError("AFlow ScEnsemble requires 1 to 26 candidates")
        text = "".join(f"{chr(65 + index)}: \n{solution}\n\n\n" for index, solution in enumerate(solutions))
        value = await self.llm.generate(self.name, ENSEMBLE_PROMPT.format(solutions=text),
                                       {"thought": "The thought of the most consistent solution.",
                                        "solution_letter": "The letter of most consistent solution."})
        letter = value.get("solution_letter", "").strip().upper()
        mapping = {chr(65 + index): solution for index, solution in enumerate(solutions)}
        if letter not in mapping:
            raise ValueError("AFlow ScEnsemble returned an invalid candidate identifier")
        return {"response": mapping[letter]}


def graph_namespace(artifact: dict, llm: OperatorLLM) -> dict:
    """Redirect only pinned AFlow infrastructure imports, preserving graph code.

    This is a compatibility shim, NOT a security sandbox. Importing generated
    Python on the host scorer is forbidden; the caller owns process isolation.
    """
    prompts: dict[str, Any] = {}
    exec(compile(artifact["prompt"], "<aflow-prompts>", "exec"), prompts)
    namespace = {"operator": SimpleNamespace(Custom=Custom, AnswerGenerate=AnswerGenerate, ScEnsemble=ScEnsemble),
                 "prompt_custom": SimpleNamespace(**{k: v for k, v in prompts.items() if not k.startswith("__")}),
                 "create_llm_instance": lambda config: llm, "DatasetType": str, "Literal": Literal}
    tree = ast.parse(artifact["graph"])
    keep = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            aliases = []
            for alias in statement.names:
                if alias.name.startswith("workspace."):
                    if not alias.name.startswith("workspace.HotpotQA.workflows.") or alias.asname not in {"operator", "prompt_custom"}:
                        raise ValueError("Unsupported AFlow dataset/operator import; this adapter pins HotpotQA operators")
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
    artifact = validate_artifact(ctx.policy.get("aflow_artifact"),
                                 allow_initialization=ctx.policy.get("aflow_allow_initialization") is True,
                                 benchmark=ctx.policy.get("aflow_benchmark"), case_id=ctx.policy.get("aflow_case_id"))
    await ctx.trace.emit("aflow_artifact", code_sha256=artifact["code_sha256"],
                         provenance=artifact["provenance"], implementation="qa-python-v2")
    namespace = graph_namespace(artifact, OperatorLLM(ctx))
    workflow_type = namespace.get("Workflow")
    if not inspect.isclass(workflow_type):
        raise ValueError("AFlow graph must define Workflow")
    workflow = workflow_type(name="AFlow", llm_config={}, dataset="HotpotQA")
    result = await workflow(ctx.prompt)
    answer = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    if not isinstance(answer, str):
        raise ValueError("AFlow Workflow must return an answer string (optionally with cost)")
    return answer
