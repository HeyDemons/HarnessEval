"""AFlow workflow search with explicitly adapted, serial benchmark tool operators.

The original QA profile is unchanged. Generated workflows run only in the agent
sandbox; the optimizer and scorer must never import them. Proposals have no tool
side effects, and only a current, unconsumed proposal may be committed.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

from . import aflow
from .core import RunContext, tool_result_content
from .methods import ACTION_SYSTEM, parse_action_reply
from .reply_contracts import action_schema

FORMAT = "aflow-tools-python-v1"
IMPLEMENTATION = "dynamic-tools-python-v1"
INITIAL_GRAPH = '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.decide = operator.ToolDecision(self.llm)

    async def __call__(self, problem):
        session = operator.ToolSession(self.llm, problem)
        while session.active:
            proposal = await self.decide(session, instruction=prompt_custom.INSTRUCTION)
            await session.commit(proposal)
        return session.answer, self.llm.get_usage_summary()["total_cost"]
'''
INITIAL_PROMPT = 'INSTRUCTION = "Use actual tool observations to complete the task. Verify the requested outcome before finishing."\n'
OPERATOR_DESCRIPTION = (
    "ToolSession(llm, problem) owns the real conversation and tools; active, answer and observation are readable. "
    "ToolDecision(llm)(session, instruction='') proposes one action without executing it. "
    "await session.commit(proposal) executes it exactly once, appends the real observation, or accepts a final answer. "
    "Keep a single session per invocation; never construct observations, mutate its state, or call tools outside commit. "
    "You may use Custom/AnswerGenerate/ScEnsemble for planning or criticism and pass their outputs to the next "
    "ToolDecision instruction. Only select a proposal from the current session state. All model calls share the "
    "benchmark budget. Do not catch budget/cancellation errors, extend limits, import host data, or embed task answers. "
    "Preserve the session loop and return session.answer. No WebShop-specific tools are available. "
)


def make_artifact(graph=INITIAL_GRAPH, prompt=INITIAL_PROMPT, *, provenance=None):
    artifact = aflow.make_artifact(graph, prompt, provenance=provenance)
    artifact["format"] = FORMAT
    return artifact


def validate_artifact(artifact, **kwargs):
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError("aflow-tools requires an aflow-tools-python-v1 workflow; QA artifacts are not tool workflows")
    aflow.validate_artifact({**artifact, "format": aflow.FORMAT}, **kwargs)
    return artifact


@dataclass(frozen=True)
class Proposal:
    session: object
    version: int
    raw: str
    action: dict
    finalizing: bool


class ToolSession:
    def __init__(self, llm, problem: str):
        if getattr(llm, "tool_session", None) is not None:
            raise ValueError("An AFlow invocation must use one shared tool session")
        llm.tool_session = self
        self.llm = llm
        self.messages = [{"role": "system", "content": ACTION_SYSTEM.format(tools=llm.ctx.environment.schema)},
                         {"role": "user", "content": problem}]
        self.version = 0
        self.active = True
        self.answer = ""
        self.observation = ""

    async def commit(self, proposal: Proposal):
        if not self.active or proposal.session is not self or proposal.version != self.version:
            raise ValueError("AFlow cannot commit a stale, foreign or already consumed proposal")
        ctx = self.llm.ctx
        self.version += 1
        self.messages.append({"role": "assistant", "content": proposal.raw})
        if "final" in proposal.action:
            self.answer = proposal.action["final"]
            self.active = False
            return
        if proposal.finalizing or ctx.last_response_used_final_slot or (
                ctx.model_budget.limit is not None and ctx.model_budget.used >= ctx.model_budget.limit):
            raise RuntimeError("AFlow cannot commit an action after the final response boundary")
        await ctx.trace.emit("aflow_tool_commit", step=self.version, action=proposal.action)
        result = await ctx.environment.call(proposal.action["tool"], proposal.action["arguments"])
        self.observation = tool_result_content(result)
        self.messages.append({"role": "user", "content": self.observation})


class ToolDecision:
    def __init__(self, llm, name="ToolDecision"):
        self.llm, self.name = llm, name

    async def __call__(self, session: ToolSession, instruction: str = ""):
        if session.llm is not self.llm or not session.active:
            raise ValueError("ToolDecision requires its own active session")
        ctx = self.llm.ctx
        if session.version >= ctx.max_turns:
            raise RuntimeError("AFlow tool loop exhausted without a final answer")
        self.llm.calls += 1
        if self.llm.calls > int(ctx.policy.get("aflow_max_operator_calls", 100)):
            raise RuntimeError("AFlow operator call budget exhausted")
        finalizing = ctx.should_finalize(session.version) or ctx.model_budget.final_response
        version = session.version
        messages = [*session.messages, {"role": "user", "content": instruction}]
        if finalizing:
            messages.append({"role": "user", "content": "Return a final answer using existing observations. No new tool actions."})
        raw = await ctx.complete(f"aflow_tools_{self.name}_{self.llm.calls}", messages, json_mode=True,
                                 response_schema=action_schema(ctx.environment.names, finalizing=finalizing))
        action = parse_action_reply(raw, ctx.environment.names, finalizing=finalizing)
        await ctx.trace.emit("aflow_tool_proposal", step=session.version + 1, action=action)
        return Proposal(session, version, raw, action, finalizing)


async def run_aflow_tools(ctx: RunContext) -> str:
    artifact = validate_artifact(ctx.policy.get("aflow_artifact"),
        allow_initialization=ctx.policy.get("aflow_allow_initialization") is True,
        benchmark=ctx.policy.get("aflow_benchmark"), case_id=ctx.policy.get("aflow_case_id"))
    await ctx.trace.emit("aflow_artifact", code_sha256=artifact["code_sha256"],
                         provenance=artifact["provenance"], implementation=IMPLEMENTATION,
                         benchmark_adapter=True)
    llm = aflow.OperatorLLM(ctx)
    namespace = aflow.graph_namespace(artifact, llm, operators={"ToolSession": ToolSession, "ToolDecision": ToolDecision})
    workflow = namespace.get("Workflow")
    if not inspect.isclass(workflow):
        raise ValueError("AFlow graph must define Workflow")
    result = await workflow(name="AFlow tools", llm_config={}, dataset="Tools")(ctx.prompt)
    answer = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    session = getattr(llm, "tool_session", None)
    if not isinstance(answer, str) or session is None or session.active or answer != session.answer:
        raise ValueError("AFlow tool workflow must return its completed session answer")
    return answer
