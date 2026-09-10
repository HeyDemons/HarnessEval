"""Secondary benchmark-tool adapter for the pinned official AFlow core.

Official QA/math/code operator profiles remain in ``aflow_official``. Generated
adapter workflows run only in the agent sandbox; the optimizer and scorer never
import them. Proposals have no tool side effects, and only a current, unconsumed
proposal may be committed.
"""
from __future__ import annotations

import ast
import builtins
import inspect
from dataclasses import dataclass

from . import aflow
from .core import RunContext, tool_result_content
from .methods import ACTION_SYSTEM, parse_action_reply
from .reply_contracts import action_schema

FORMAT = "aflow-benchmark-tools-python-v2"
IMPLEMENTATION = "aflow-official-core-benchmark-adapter-v4"
SAFE_BUILTINS = {name: getattr(builtins, name) for name in
                 ("__build_class__", "str", "int", "float", "bool", "list", "dict", "tuple",
                  "range", "len", "enumerate", "zip", "min", "max", "sum", "sorted")}
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
    "Use only the supplied operators and ordinary assignments, if/while/for, lists/dicts, indexing, "
    "string concatenation and f-strings. Define only Workflow.__init__ and Workflow.__call__. "
    "No imports, helper functions, exceptions, reflection, .format(), filesystem or direct llm/context access. "
    "Prompt source must contain only literal constants. "
)


def make_artifact(graph=INITIAL_GRAPH, prompt=INITIAL_PROMPT, *, provenance=None):
    artifact = aflow.make_artifact(
        graph, prompt, operator_profile="benchmark-tools", provenance=provenance
    )
    artifact["format"] = FORMAT
    return artifact


def validate_artifact(artifact, **kwargs):
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError("AFlow tool execution requires an aflow-benchmark-tools-python-v2 artifact")
    aflow.validate_artifact({**artifact, "format": aflow.FORMAT}, **kwargs)
    validate_graph(artifact['graph'], artifact['prompt'])
    return artifact


def validate_graph(graph: str, prompt: str):
    """Accept a capability-limited operator graph, never arbitrary Python.

    Native bridges hold hidden state in the controller process. Restrict the
    generated language before compiling it: no imports/reflection, arbitrary
    calls, custom functions, state mutation or access to llm.ctx/handlers.
    Together with minimal builtins this confines code to public operator APIs.
    The ordinary process deadline still bounds pure-computation infinite loops.
    """
    tree = ast.parse(graph)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise ValueError('Tool graph must define only class Workflow')
    cls = tree.body[0]
    if cls.name != 'Workflow' or cls.bases or cls.keywords or cls.decorator_list:
        raise ValueError('Tool graph cannot customize class construction')
    if len(cls.body) != 2 or {getattr(n, 'name', None) for n in cls.body} != {'__init__', '__call__'}:
        raise ValueError('Tool graph requires only __init__ and async __call__')
    fields = {'llm'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if not isinstance(node.value, ast.Name) or node.value.id != 'self' or node.attr.startswith('_'):
                raise ValueError('Tool graph may assign only its own operator fields')
            fields.add(node.attr)
    operators = {'ToolSession', 'ToolDecision', 'Custom', 'AnswerGenerate', 'ScEnsemble'}
    attributes = {'active', 'answer', 'observation', 'commit', 'get_usage_summary',
                  'get', 'append', 'join', 'replace', 'strip'}
    prompt_names = set()
    prompts = ast.parse(prompt)
    for node in prompts.body:
        if not isinstance(node, ast.Assign) or any(not isinstance(t, ast.Name) or t.id.startswith('_') for t in node.targets):
            raise ValueError('Tool workflow prompts must be literal constants')
        try:
            ast.literal_eval(node.value)
        except (ValueError, TypeError) as exc:
            raise ValueError('Tool workflow prompts must be literal constants') from exc
        prompt_names.update(t.id for t in node.targets)
    allowed = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.arguments, ast.arg,
               ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return, ast.If, ast.While, ast.For,
               ast.Break, ast.Continue, ast.Pass, ast.Await, ast.Call, ast.keyword, ast.Name, ast.Attribute,
               ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
               ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.JoinedStr, ast.FormattedValue,
               ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f'Unsupported tool workflow syntax: {type(node).__name__}')
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node not in cls.body or node.decorator_list or node.args.defaults or node.args.kw_defaults:
                raise ValueError('Tool workflow cannot define helper functions or decorators')
            expected = ['self', 'name', 'llm_config', 'dataset'] if node.name == '__init__' else ['self', 'problem']
            if ([a.arg for a in node.args.args] != expected or node.args.posonlyargs or node.args.kwonlyargs
                    or node.args.vararg or node.args.kwarg or
                    (node.name == '__call__') != isinstance(node, ast.AsyncFunctionDef)):
                raise ValueError('Tool workflow must preserve the operator interface signature')
        if isinstance(node, (ast.Name, ast.arg)) and getattr(node, 'id', getattr(node, 'arg', '')).startswith('_'):
            raise ValueError('Tool workflow cannot access private names')
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in {
                'self', 'operator', 'prompt_custom', 'create_llm_instance', *SAFE_BUILTINS}:
            raise ValueError('Tool workflow cannot rebind capability names')
        if isinstance(node, ast.Attribute):
            root = node.value.id if isinstance(node.value, ast.Name) else None
            permitted = fields if root == 'self' else operators if root == 'operator' else prompt_names if root == 'prompt_custom' else attributes
            if node.attr.startswith('_') or node.attr not in permitted:
                raise ValueError(f'Tool workflow cannot access attribute {node.attr}')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in (set(SAFE_BUILTINS) - {'__build_class__'}) | {'create_llm_instance'}:
                    raise ValueError('Tool workflow cannot call arbitrary functions')
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError('Tool workflow cannot call computed values')


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
        if not isinstance(proposal, Proposal) or not self.active or proposal.session is not self or proposal.version != self.version:
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
    llm = aflow.OperatorLLM(ctx, "benchmark-tools")
    namespace = aflow.graph_namespace(artifact, llm, operators={"ToolSession": ToolSession, "ToolDecision": ToolDecision},
                                      builtins_override=SAFE_BUILTINS)
    workflow = namespace.get("Workflow")
    if not inspect.isclass(workflow):
        raise ValueError("AFlow graph must define Workflow")
    result = await workflow(name="AFlow tools", llm_config={}, dataset="Tools")(ctx.prompt)
    answer = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    session = getattr(llm, "tool_session", None)
    if not isinstance(answer, str) or session is None or session.active or answer != session.answer:
        raise ValueError("AFlow tool workflow must return its completed session answer")
    return answer
