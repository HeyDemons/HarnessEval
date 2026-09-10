import hashlib
import unittest

from benchmark_platform.harnesses import aflow, aflow_official
from test_aflow_upstream import context


class OfficialAFlowOperatorTests(unittest.IsolatedAsyncioTestCase):
    def test_pinned_prompt_bytes_match_upstream_revision(self):
        expected = {
            "ANSWER_GENERATION_PROMPT": "7876fc30beeeab708e53f9f28327f42b286e4214bc5b43116a437674c454bc87",
            "SC_ENSEMBLE_PROMPT": "00866525fd303498fe04f064294bdbe5f1cb0de467c7d70f87da42add89aff84",
            "PYTHON_CODE_VERIFIER_PROMPT": "76ddbcba53ba4aeea383e32acba53b94f0afb898bac0aa86514417f58c686863",
            "REFLECTION_ON_PUBLIC_TEST_PROMPT": "35ba35a4217720afaff39f1d85828ae1023fa0a345c2742443c0c599f95e27a0",
            "WORKFLOW_OPTIMIZE_PROMPT": "e9985226398ed1251dbcc16d0fc1a741ae80d8134ba74a319d54859bba2ed1d3",
            "WORKFLOW_INPUT": "86409dea5266911750f7b3f68ade9e7a8893c38b3deacd3cb6e51980f361f110",
            "WORKFLOW_CUSTOM_USE": "12ea1f4625edd3fe2b498e281c9c440d84569f9a184e8705eb71a8bd70be22c1",
        }
        for name, digest in expected.items():
            with self.subTest(prompt=name):
                self.assertEqual(hashlib.sha256(getattr(aflow_official, name).encode()).hexdigest(), digest)

    def test_pinned_profiles_expose_the_six_configured_operator_classes(self):
        self.assertEqual(
            aflow_official.OPERATOR_PROFILES,
            {
                "qa": ("Custom", "AnswerGenerate", "ScEnsemble"),
                "math": ("Custom", "Programmer", "ScEnsemble"),
                "code": ("Custom", "CustomCodeGenerate", "ScEnsemble", "Test"),
            },
        )
        self.assertEqual(
            set().union(*map(set, aflow_official.OPERATOR_PROFILES.values())),
            {"Custom", "AnswerGenerate", "ScEnsemble", "Programmer", "CustomCodeGenerate", "Test"},
        )

    def test_namespace_is_dataset_profiled_not_an_unofficial_six_operator_union(self):
        self.assertEqual(set(aflow_official.namespace("qa")), {"Custom", "AnswerGenerate", "ScEnsemble"})
        self.assertEqual(set(aflow_official.namespace("math")), {"Custom", "Programmer", "ScEnsemble"})
        self.assertEqual(set(aflow_official.namespace("code")),
                         {"Custom", "CustomCodeGenerate", "ScEnsemble", "Test"})

    async def test_custom_code_generate_uses_the_pinned_code_interface(self):
        ctx = context(["```python\ndef solve_item(x):\n    return x + 1\n```"])
        llm = aflow.OperatorLLM(ctx, "code")
        result = await aflow_official.CustomCodeGenerate(llm)(
            "increment x", "solve_item", "Write the function.\n"
        )
        self.assertIn("def solve_item", result["response"])
        self.assertIn("Write the function.\nincrement x", ctx.client.messages[0][0]["content"])

    async def test_programmer_is_gated_but_preserves_official_solve_execution(self):
        self.assertEqual(
            aflow_official.run_code("def solve():\n    return 42\n"),
            ("Success", "42"),
        )
        ctx = context([])
        llm = aflow.OperatorLLM(ctx, "math")
        programmer = aflow_official.Programmer(llm)
        with self.assertRaisesRegex(RuntimeError, "isolated public-test sandbox"):
            await programmer.exec_code("def solve(): return 42")

    async def test_test_operator_uses_only_explicit_public_tests(self):
        ctx = context([])
        ctx.policy.update(
            aflow_official_code_operators=True,
            aflow_public_tests=["assert add(2, 3) == 5"],
        )
        llm = aflow.OperatorLLM(ctx, "code")
        result = await aflow_official.Test(llm)(
            "add two numbers", "def add(a, b): return a + b", "add"
        )
        self.assertEqual(result, {"result": True, "solution": "def add(a, b): return a + b"})
        self.assertEqual(ctx.llm_calls, 0)

    def test_artifacts_pin_operator_profile_and_reject_legacy_format(self):
        for profile in aflow_official.OPERATOR_PROFILES:
            artifact = aflow.make_artifact(operator_profile=profile)
            self.assertEqual(aflow.validate_artifact(artifact, allow_initialization=True), artifact)
            self.assertEqual(artifact["operator_profile"], profile)
        legacy = {**aflow.make_artifact(), "format": "aflow-python-v1"}
        with self.assertRaisesRegex(ValueError, "aflow-official-python-v2"):
            aflow.validate_artifact(legacy, allow_initialization=True)


if __name__ == "__main__":
    unittest.main()
