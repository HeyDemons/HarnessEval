"""AFlow QA operator and frozen Python graph fidelity checks."""
import copy
import unittest

from benchmark_platform.harnesses.aflow import make_artifact, run_aflow, validate_artifact, REVISION
from benchmark_platform.harnesses.api import Completion
from benchmark_platform.harnesses.core import RunContext, ToolEnvironment, ToolSpec


class Trace:
    async def emit(self, event, **data):
        pass


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    async def complete(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        return Completion(next(self.replies), 1, 1, 0, 0, {})


def context(replies, graph=None, prompt=""):
    trace = Trace()
    artifact = make_artifact(prompt=prompt, **({"graph": graph} if graph else {}))
    env = ToolEnvironment([ToolSpec("lookup", "lookup", {"type": "object"}, (), read_only=True)], trace)
    return RunContext("aflow", "question", Client(replies), env, trace,
                      {"aflow_artifact": artifact, "aflow_allow_initialization": True})


GRAPH = '''import workspace.HotpotQA.workflows.template.operator as operator
import workspace.HotpotQA.workflows.round_2.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType
class Workflow:
    def __init__(self, name, llm_config, dataset):
        llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(llm)
        self.generate = operator.AnswerGenerate(llm)
        self.ensemble = operator.ScEnsemble(llm)

    async def __call__(self, problem):
        first = await self.custom(problem, prompt_custom.INSTRUCTION)
        second = await self.generate(problem + first["response"])
        candidates = [first["response"], second["answer"]]
        if len(candidates) == 2:
            result = await self.ensemble(candidates)
            return result["response"], None
'''


class AFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_is_one_plain_generation_and_never_executes_tools(self):
        ctx = context(['{"tool":"lookup","arguments":{}}'])
        self.assertEqual(await run_aflow(ctx), '{"tool":"lookup","arguments":{}}')
        self.assertEqual(ctx.client.messages, [[{"role": "user", "content": "question"}]])
        self.assertFalse(ctx.environment.calls)

    async def test_python_dependencies_custom_prompt_and_candidate_selection(self):
        ctx = context(["alpha", "<thought>reason</thought><answer>beta</answer>",
                       "<thought>vote</thought><solution_letter>B</solution_letter>"],
                      GRAPH, 'INSTRUCTION = "prefix:"')
        self.assertEqual(await run_aflow(ctx), "beta")
        self.assertEqual(ctx.client.messages[0][0]["content"], "prefix:question")
        self.assertIn("questionalpha", ctx.client.messages[1][0]["content"])
        self.assertIn("<answer>", ctx.client.messages[1][0]["content"])
        self.assertEqual(ctx.llm_calls, 3)

    async def test_ensemble_cannot_invent_an_answer(self):
        ctx = context(["alpha", "<thought>r</thought><answer>beta</answer>",
                       "<thought>v</thought><solution_letter>outside-candidates</solution_letter>"],
                      GRAPH, 'INSTRUCTION = ""')
        with self.assertRaisesRegex(ValueError, "invalid candidate"):
            await run_aflow(ctx)

    async def test_legacy_workflow_and_missing_artifact_fail_before_calls(self):
        for policy in ({}, {"aflow_workflow": ["Custom"]}, {"aflow_artifact": make_artifact()}):
            ctx = context([])
            ctx.policy = policy
            with self.assertRaises(ValueError):
                await run_aflow(ctx)
            self.assertEqual(ctx.llm_calls, 0)

    def test_artifact_identity_and_split_validation(self):
        artifact = make_artifact(provenance={"kind": "optimized", "source_revision": REVISION,
            "benchmark": "gaia", "optimization_case_ids": ["train"], "evaluation_case_ids": ["test"],
            "validation_score": .5, "search_history_sha256": "a" * 64})
        validate_artifact(artifact, benchmark="gaia", case_id="test")
        for key, value in (("benchmark", "bfcl"), ("case_id", "train")):
            with self.assertRaises(ValueError):
                validate_artifact(artifact, **{key: value})
        artifact["provenance"]["optimization_case_ids"] = ["test"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_artifact(artifact)
        artifact["graph"] += "\n# changed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_artifact(artifact)
