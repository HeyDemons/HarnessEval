from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .core import RunContext


PLAN_RE = re.compile(r"^\s*Plan\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
EVIDENCE_RE = re.compile(
    r"^\s*#E(\d+)\s*=\s*([A-Za-z_][\w.-]*)\s*\[",
    re.IGNORECASE | re.MULTILINE,
)
REFERENCE_RE = re.compile(r"#E\d+(?:\.(?:[A-Za-z_][\w]*|\d+))*", re.IGNORECASE)


@dataclass(frozen=True)
class ReWOOStep:
    evidence_id: str
    plan: str
    worker: str
    worker_input: str


@dataclass(frozen=True)
class ReWOOEvidence:
    evidence_id: str
    plan: str
    worker: str
    worker_input: Any
    ok: bool
    output: Any

    def as_log_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "plan": self.plan,
            "worker": self.worker,
            "worker_input": self.worker_input,
            "ok": self.ok,
            "output": self.output,
        }


def _balanced_bracket(text: str, open_index: int) -> tuple[str, int]:
    if open_index >= len(text) or text[open_index] != "[":
        raise ValueError("ReWOO evidence call is missing its opening bracket")
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index + 1
            if depth < 0:
                break
    raise ValueError("ReWOO evidence call has an unterminated bracket")


def parse_rewoo_plan(text: str) -> list[ReWOOStep]:
    """Parse the paper's alternating `Plan:` / `#E = Worker[input]` protocol."""
    steps: list[ReWOOStep] = []
    cursor = 0
    while match := EVIDENCE_RE.search(text, cursor):
        plans = list(PLAN_RE.finditer(text, cursor, match.start()))
        if len(plans) != 1:
            evidence_id = f"E{match.group(1)}"
            raise ValueError(
                f"ReWOO {evidence_id} must be preceded by exactly one Plan line; found {len(plans)}"
            )
        number = int(match.group(1))
        expected = len(steps) + 1
        if number != expected:
            raise ValueError(f"ReWOO evidence ids must be sequential: expected E{expected}, got E{number}")
        worker_input, cursor = _balanced_bracket(text, match.end() - 1)
        steps.append(
            ReWOOStep(
                evidence_id=f"E{number}",
                plan=plans[0].group(1).strip(),
                worker=match.group(2),
                worker_input=worker_input.strip(),
            )
        )
    if not steps:
        raise ValueError("ReWOO Planner produced no `Plan:` / `#E = Worker[input]` steps")
    if PLAN_RE.search(text, cursor):
        raise ValueError("ReWOO Planner ended with a Plan that has no evidence call")
    return steps


def _select_reference(reference: str, evidence: dict[str, Any]) -> Any:
    parts = reference[1:].split(".")
    evidence_id = parts[0].upper()
    if evidence_id not in evidence:
        raise ValueError(f"ReWOO reference {reference} is not available")
    selected = evidence[evidence_id]
    for part in parts[1:]:
        try:
            if isinstance(selected, list):
                selected = selected[int(part)]
            elif isinstance(selected, dict):
                selected = selected[part]
            else:
                raise TypeError
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"ReWOO reference {reference} does not resolve") from exc
    return selected


def _reference_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _resolve_value(value: Any, evidence: dict[str, Any]) -> Any:
    if isinstance(value, str):
        exact = REFERENCE_RE.fullmatch(value.strip())
        if exact:
            return _select_reference(exact.group(0), evidence)
        return REFERENCE_RE.sub(
            lambda match: _reference_text(_select_reference(match.group(0), evidence)),
            value,
        )
    if isinstance(value, list):
        return [_resolve_value(item, evidence) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item, evidence) for key, item in value.items()}
    return value


def _quote_bare_references(raw_input: str) -> str:
    """Make source-style bare #E references JSON-decodable without touching strings."""
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(raw_input):
        character = raw_input[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        reference = REFERENCE_RE.match(raw_input, index)
        if reference:
            output.append(json.dumps(reference.group(0)))
            index = reference.end()
            continue
        output.append(character)
        index += 1
    return "".join(output)


def _tool_arguments(raw_input: str, evidence: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(_quote_bare_references(raw_input))
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark-tool input must be one complete JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark-tool input must be one complete JSON object")
    return _resolve_value(value, evidence)


class EvidenceWorker:
    """Execute ReWOO evidence calls using the benchmark workers plus the paper's LLM worker."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    async def run(self, step: ReWOOStep, evidence: dict[str, Any]) -> ReWOOEvidence:
        await self.ctx.trace.emit(
            "rewoo_worker_start",
            evidence_id=step.evidence_id,
            plan=step.plan,
            worker=step.worker,
            worker_input=step.worker_input,
        )
        worker_name = step.worker
        if worker_name.lower() == "llm":
            try:
                request = _resolve_value(step.worker_input, evidence)
                output = await self.ctx.complete(
                    f"rewoo_worker_{step.evidence_id}",
                    [
                        {
                            "role": "user",
                            "content": "Respond directly and briefly with no extra words.\n\n" + str(request),
                        }
                    ],
                    temperature=0.0,
                )
                result = ReWOOEvidence(
                    step.evidence_id,
                    step.plan,
                    "LLM",
                    request,
                    True,
                    output,
                )
            except ValueError as exc:
                result = self._failure(step, step.worker_input, "invalid_reference", str(exc))
        elif worker_name in self.ctx.environment.tools:
            try:
                arguments = _tool_arguments(step.worker_input, evidence)
                tool_result = await self.ctx.environment.call(worker_name, arguments)
                if tool_result.get("ok") is True:
                    result = ReWOOEvidence(
                        step.evidence_id,
                        step.plan,
                        worker_name,
                        arguments,
                        True,
                        tool_result.get("result"),
                    )
                else:
                    result = ReWOOEvidence(
                        step.evidence_id,
                        step.plan,
                        worker_name,
                        arguments,
                        False,
                        tool_result,
                    )
            except ValueError as exc:
                result = self._failure(step, step.worker_input, "invalid_worker_input", str(exc))
        else:
            result = self._failure(
                step,
                step.worker_input,
                "unknown_worker",
                f"available workers: {[*self.ctx.environment.names, 'LLM']}",
            )
        await self.ctx.trace.emit("rewoo_worker_result", **result.as_log_record())
        return result

    @staticmethod
    def _failure(step: ReWOOStep, worker_input: Any, error: str, detail: str) -> ReWOOEvidence:
        return ReWOOEvidence(
            step.evidence_id,
            step.plan,
            step.worker,
            worker_input,
            False,
            {"ok": False, "error": error, "detail": detail},
        )


def _worker_descriptions(ctx: RunContext) -> str:
    workers = [
        f"{tool.name}[JSON object]: {tool.description}; parameters={json.dumps(tool.parameters, ensure_ascii=False, sort_keys=True)}"
        for tool in ctx.environment.tools.values()
    ]
    workers.append(
        "LLM[plain-text instruction]: a pretrained language-model worker for general knowledge, "
        "comparison, and reasoning over prior #E evidence."
    )
    return "\n".join(workers)


def _worker_log(records: list[ReWOOEvidence]) -> str:
    blocks = []
    for record in records:
        blocks.append(
            "\n".join(
                [
                    f"Plan: {record.plan}",
                    "Evidence:",
                    _reference_text(record.output),
                ]
            )
        )
    return "\n\n".join(blocks)


async def run_rewoo(ctx: RunContext) -> str:
    planner_output = await ctx.complete(
        "rewoo_planner",
        [
            {
                "role": "user",
                "content": (
                    "For the following task, make plans that solve it step by step. For each Plan, select one "
                    "worker and provide its complete input to retrieve evidence. Store evidence in sequential "
                    "variables #E1, #E2, ... that later workers may reference. Plan every worker call before any "
                    "worker executes. Each Plan must be followed by exactly one evidence assignment in this format:\n"
                    "Plan: rich description of this step\n"
                    "#E1 = Worker[input]\n\n"
                    "For a benchmark worker, input must be one complete JSON object matching its parameter schema. "
                    "References may be bare #E variables; append a field path such as #E1.value when a structured "
                    "evidence object must supply one scalar JSON field. For LLM, input is a plain-text instruction. "
                    "Do not solve the task or invent evidence in the plan.\n\n"
                    f"Workers:\n{_worker_descriptions(ctx)}\n\n"
                    f"Task: {ctx.prompt}"
                ),
            }
        ],
        temperature=0.0,
    )
    steps = parse_rewoo_plan(planner_output)
    await ctx.trace.emit(
        "rewoo_plan_parsed",
        planner_output=planner_output,
        steps=[step.__dict__ for step in steps],
    )

    worker = EvidenceWorker(ctx)
    values: dict[str, Any] = {}
    records: list[ReWOOEvidence] = []
    for step in steps:
        record = await worker.run(step, values)
        records.append(record)
        values[step.evidence_id] = record.output

    worker_log = _worker_log(records)
    return await ctx.complete(
        "rewoo_solver",
        [
            {
                "role": "user",
                "content": (
                    "Solve the task using the plans and corresponding evidence below. Some evidence may contain "
                    "noise or an explicit worker failure, so assess it cautiously. Respond with the answer directly "
                    "with no extra words.\n\n"
                    f"Task: {ctx.prompt}\n\n"
                    f"Worker log:\n{worker_log}\n\n"
                    f"Task: {ctx.prompt}"
                ),
            }
        ],
        temperature=0.0,
    )
