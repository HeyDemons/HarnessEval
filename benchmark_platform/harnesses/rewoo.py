from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .core import RunContext


PLAN_RE = re.compile(r"^\s*Plan\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
PLAN_HEADER_RE = re.compile(r"^[ \t]*Plan\s*:", re.IGNORECASE | re.MULTILINE)
EVIDENCE_RE = re.compile(
    r"^\s*#E(\d+)\s*=\s*([A-Za-z_][\w.-]*)\s*\[",
    re.IGNORECASE | re.MULTILINE,
)
REFERENCE_RE = re.compile(r"#E\d+(?:\.(?:[A-Za-z_][\w]*|\d+)|\[\d+\])*", re.IGNORECASE)


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
    # Pinned PWS runs Solver with an empty worker log for a zero-step plan.
    # A malformed evidence assignment is not a zero-step plan, however.
    if not steps and re.search(r"^\s*#E\d+", text, re.IGNORECASE | re.MULTILINE):
        raise ValueError("ReWOO Planner contains a malformed evidence call")
    if PLAN_HEADER_RE.search(text, cursor):
        raise ValueError("ReWOO Planner ended with a Plan that has no evidence call")
    return steps


def _select_reference(reference: str, evidence: dict[str, Any]) -> Any:
    # `[0]` alongside `.0`: the planner prompt describes a field path, and a model writing a
    # path into a JSON array reaches for the subscript it would use in any other language.
    parts = [part for part in re.split(r"\.|\[(\d+)\]", reference[1:]) if part]
    evidence_id = parts[0].upper()
    if evidence_id not in evidence:
        raise ValueError(f"ReWOO reference {reference} is not available")
    selected = evidence[evidence_id]
    for part in parts[1:]:
        # A benchmark tool answers with a JSON *string*, so the observation that a planner is
        # told it may select a field from arrives as text. Walking it as a string used to fail
        # every path reference into a real API result while the identical path over a Python
        # dict succeeded -- an execution-boundary artefact, not a wrong plan.
        if isinstance(selected, str):
            try:
                selected = json.loads(selected)
            except json.JSONDecodeError:
                pass
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
    # Resolve before demanding an object. The published protocol lets a worker input *be* an
    # earlier evidence variable -- `api_fetch[#E3]` where #E3 is the request the LLM worker
    # built -- and checking the shape first rejected that plan while it was still literally
    # the string "#E3". Upstream substitutes into the input text and only then hands it to
    # the worker; this is the same order, with the object requirement this adapter adds.
    value = _resolve_value(value, evidence)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if not isinstance(value, dict):
        raise ValueError("benchmark-tool input must be one complete JSON object")
    return value


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
        # The paper's LLM worker is prompted with the instruction alone. Saying so changes
        # nothing about the worker; it stops the planner delegating "build a request
        # matching the api_fetch schema" to a worker that was never shown that schema.
        + (
            " It receives only your instruction text with #E references replaced by their evidence; "
            "it does not see this task, the worker list or any parameter schema."
            if ctx.policy.get("rewoo_llm_worker_scope_note") is True else ""
        )
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


# Planning without observation, spelled out. Each line answers a failure measured on
# automationbench (2026-09-18, 36 cases): the planner took search result 0 unseen and hit the
# wrong service (401/404 on 74 of 160 fetches), asked the LLM worker for "a request matching
# the schema" it was never shown (20% malformed), and had no second source when one failed.
# Nothing here names a benchmark tool; upstream ships hand-written few-shots per benchmark.
UNOBSERVED_PLANNING_GUIDANCE = (
    "You will not see any evidence until the whole plan has run, so plan for what you cannot know:\n"
    "- When a later input depends on choosing among alternatives in earlier evidence (which search "
    "result, which record), add an LLM step that makes the choice from that evidence. Do not take a "
    "fixed position such as the first result.\n"
    "- When an LLM step must produce a benchmark worker's input, write that worker's exact parameter "
    "names, types and required fields into the instruction, and ask for only the finished JSON object "
    "with every value filled in and no placeholders left.\n"
    "- A worker call can fail and you cannot react to it. Where more than one source could hold what "
    "you need, gather evidence from each and let a later LLM step use whichever succeeded.\n\n"
)


async def run_rewoo(ctx: RunContext) -> str:
    guided = ctx.policy.get("rewoo_planner_guidance") == "unobserved-planning-v1"
    # The stock path example has the exact shape of an endpoint search result and reads as
    # an instruction to take result 0; the guided prompt shows the syntax on neutral fields.
    path_example = (
        "#E1.value, #E1.items.0.id or #E1.items[0].id" if guided
        else "#E1.value, #E1.results.0.url or #E1.results[0].url"
    )
    planner_conversation = [
        {
            "role": "user",
            "content": (
                "For the following task, make plans that solve it step by step. For each Plan, select one "
                "worker and provide its complete input to retrieve evidence. Store evidence in sequential "
                "variables #E1, #E2, ... that later workers may reference. Plan every worker call before any "
                "worker executes. Each Plan must be followed by exactly one evidence assignment in this format:\n"
                "Plan: rich description of this step\n"
                "#E1 = Worker[input]\n\n"
                "For a benchmark worker, input must resolve to one complete JSON object matching its parameter "
                "schema: either write the object and reference evidence inside it, or name a single #E variable "
                f"whose evidence is already that object. Append a path such as {path_example} "
                "to select one field from structured evidence. For LLM, input is a plain-text "
                "instruction. "
                "Do not solve the task or invent evidence in the plan.\n\n"
                + (UNOBSERVED_PLANNING_GUIDANCE if guided else "")
                +
                f"Workers:\n{_worker_descriptions(ctx)}\n\n"
                f"Task: {ctx.prompt}"
            ),
        }
    ]
    protocol_repairs = int(ctx.policy.get("protocol_repairs", 1))
    if protocol_repairs < 0:
        raise ValueError("protocol_repairs must be non-negative")
    for attempt in range(protocol_repairs + 1):
        planner_output = await ctx.complete(
            "rewoo_planner",
            planner_conversation,
            temperature=0.0,
        )
        try:
            steps = parse_rewoo_plan(planner_output)
            break
        except ValueError as exc:
            if attempt >= protocol_repairs:
                raise
            await ctx.trace.emit(
                "rewoo_plan_protocol_repair",
                attempt=attempt + 1,
                error=str(exc),
            )
            planner_conversation.extend(
                [
                    {"role": "assistant", "content": planner_output},
                    {
                        "role": "user",
                        "content": (
                            f"Protocol error: {exc}. Return one corrected ReWOO plan only. "
                            "Start at #E1, use each evidence id exactly once in sequential order, and place "
                            "exactly one complete `#E = Worker[input]` line immediately after each `Plan:` "
                            "line. Do not include commentary, tool results, or a second draft."
                        ),
                    },
                ]
            )
    else:
        raise AssertionError("unreachable")
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
    solver_context = (
        "Solve the task using the plans and corresponding evidence below. Some evidence may contain "
        "noise or an explicit worker failure, so assess it cautiously. Respond with the answer directly "
        "with no extra words.\n\n"
        f"Task: {ctx.prompt}\n\n"
        f"Worker log:\n{worker_log}\n\n"
        f"Task: {ctx.prompt}"
    )
    if ctx.policy.get("bfcl_declaration_mode") is True:
        from .declaration import (
            complete_native_declaration,
            declaration_messages,
        )
        return await complete_native_declaration(
            ctx,
            role="rewoo_solver",
            messages=declaration_messages(
                ctx,
                method_instruction=(
                    "You are ReWOO's existing Solver. Use the complete plan and worker records to "
                    "publish the full BFCL native call batch in this response. Worker function "
                    "proposals were not executed and supply no observations."
                ),
                internal_context=solver_context,
            ),
        )
    return await ctx.complete(
        "rewoo_solver",
        [{"role": "user", "content": solver_context}],
        temperature=0.0,
    )
