"""Pinned FoundationAgents/AFlow operator profiles and operator semantics.

Source revision: 3f457218fc716093fe53f6df8a5d5e6379d66346.
The upstream repository does not expose all operators to every dataset.  Its
configured search spaces are QA, math, and code; their union is six classes.
Benchmark-specific tool operators live in ``aflow_tools`` and are not part of
this module.
"""
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import re
import sys
import traceback
from typing import Any


REVISION = "3f457218fc716093fe53f6df8a5d5e6379d66346"
OPERATOR_PROFILES = {
    "qa": ("Custom", "AnswerGenerate", "ScEnsemble"),
    "math": ("Custom", "Programmer", "ScEnsemble"),
    "code": ("Custom", "CustomCodeGenerate", "ScEnsemble", "Test"),
}
DATASET_BY_PROFILE = {"qa": "HotpotQA", "math": "MATH", "code": "MBPP"}
OPERATOR_DESCRIPTIONS = {
    "Custom": (
        "Generates anything based on customized input and instruction.",
        "custom(input: str, instruction: str) -> dict with key 'response' of type str",
    ),
    "AnswerGenerate": (
        "Generate step by step based on the input. The thought process is in 'thought' and the final answer in 'answer'.",
        "answer_generate(input: str) -> dict with keys 'thought' and 'answer' of type str",
    ),
    "ScEnsemble": (
        "Uses self-consistency to select the solution that appears most frequently in the solution list.",
        "sc_ensemble(solutions: list[str], problem: str = optional for QA) -> dict with key 'response'",
    ),
    "Programmer": (
        "Writes and executes Python code, retrying generation with execution feedback.",
        "programmer(problem: str, analysis: str = 'None') -> dict with keys 'code' and 'output'",
    ),
    "CustomCodeGenerate": (
        "Generates code based on customized input and instruction.",
        "custom_code_generate(problem: str, entry_point: str, instruction: str) -> dict with key 'response'",
    ),
    "Test": (
        "Runs public tests and revises a failing code solution from the observed failures.",
        "test(problem: str, solution: str, entry_point: str) -> dict with keys 'result' and 'solution'",
    ),
}

INITIAL_GRAPH = '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem):
        solution = await self.custom(input=problem, instruction="")
        return solution["response"], self.llm.get_usage_summary()["total_cost"]
'''
CODE_INITIAL_GRAPH = '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem, entry_point):
        solution = await self.custom_code_generate(
            problem=problem, entry_point=entry_point, instruction=""
        )
        return solution["response"], self.llm.get_usage_summary()["total_cost"]
'''

ANSWER_GENERATION_PROMPT = '''
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
'''
QA_SC_ENSEMBLE_PROMPT = '''
Several answers have been generated to a same question. They are as follows:
{solutions}

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.
In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
'''
SC_ENSEMBLE_PROMPT = '''
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
'''
PYTHON_CODE_VERIFIER_PROMPT = '''
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
'''
REFLECTION_ON_PUBLIC_TEST_PROMPT = '''
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.:\x20
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
'''

WORKFLOW_OPTIMIZE_PROMPT = '''You are building a Graph and corresponding Prompt to jointly solve {type} problems.\x20
Referring to the given graph and prompt, which forms a basic example of a {type} solution approach,\x20
please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your\x20
single modification in XML tags in your reply. Ensure they are complete and correct to avoid runtime failures. When\x20
optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider\x20
Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators),\x20
or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph\x20
complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical\x20
representation.Ensure that all the prompts required by the current graph from prompt_custom are included.Exclude any other prompts.
Output the modified graph and all the necessary Prompts in prompt_custom (if needed).
The prompt you need to generate is only the one used in `prompt_custom.XXX` within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in `prompt_custom`; please remove any unused prompts in prompt_custom.
the generated prompt must not contain any placeholders.
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It's crucial to include necessary context during the process.'''

WORKFLOW_INPUT = '''
Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify/ ...)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

First, provide optimization ideas. **Only one detail point can be modified at a time**, and no more than 5 lines of code may be changed per modification—extensive modifications are strictly prohibited to maintain project focus!
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
You do not need to manually import prompt_custom or operator to use them; they are already included in the execution environment.
'''

WORKFLOW_CUSTOM_USE = '''
Here's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n

**Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators are not yet used in the graph, try incorporating them.**
'''

GRAPH_RESPONSE_FORMAT = '''
# Response format (must be strictly followed) (do not include any other formats except for the given XML format):
<modification>modification</modification>
<graph>graph</graph>
<prompt>prompt</prompt>'''


def initial_graph(profile: str) -> str:
    if profile not in OPERATOR_PROFILES:
        raise ValueError(f"Unknown official AFlow operator profile: {profile}")
    return CODE_INITIAL_GRAPH if profile == "code" else INITIAL_GRAPH


def operator_description(profile: str) -> str:
    if profile not in OPERATOR_PROFILES:
        raise ValueError(f"Unknown official AFlow operator profile: {profile}")
    rows = []
    for index, name in enumerate(OPERATOR_PROFILES[profile], 1):
        description, interface = OPERATOR_DESCRIPTIONS[name]
        rows.append(f"{index}. {name}: {description}, with interface {interface}).")
    return "\n".join(rows)


class Custom:
    def __init__(self, llm, name: str = "Custom"):
        self.llm, self.name = llm, name

    async def __call__(self, input: str, instruction: str):
        return await self.llm.generate(self.name, instruction + input)


class AnswerGenerate:
    def __init__(self, llm, name: str = "AnswerGenerate"):
        self.llm, self.name = llm, name

    async def __call__(self, input: str, mode: str | None = None):
        return await self.llm.generate(
            self.name,
            ANSWER_GENERATION_PROMPT.format(input=input),
            {"thought": "The step by step thinking process", "answer": "The final answer to the question"},
        )


class CustomCodeGenerate:
    def __init__(self, llm, name: str = "CustomCodeGenerate"):
        self.llm, self.name = llm, name

    async def __call__(self, problem: str, entry_point: str, instruction: str):
        return await self.llm.generate_code(self.name, instruction + problem, entry_point)


class ScEnsemble:
    def __init__(self, llm, name: str = "ScEnsemble"):
        self.llm, self.name = llm, name

    async def __call__(self, solutions: list[str], problem: str | None = None):
        if not solutions or len(solutions) > 26:
            raise ValueError("AFlow ScEnsemble requires 1 to 26 candidates")
        text = "".join(f"{chr(65 + index)}: \n{solution}\n\n\n" for index, solution in enumerate(solutions))
        if getattr(self.llm, "operator_profile", "qa") in {"qa", "benchmark-tools"}:
            prompt = QA_SC_ENSEMBLE_PROMPT.format(solutions=text)
        else:
            if not isinstance(problem, str):
                raise ValueError("AFlow math/code ScEnsemble requires the problem")
            prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=text)
        value = await self.llm.generate(
            self.name,
            prompt,
            {"thought": "The thought of the most consistent solution.",
             "solution_letter": "The letter of most consistent solution."},
        )
        letter = value.get("solution_letter", "").strip().upper()
        mapping = {chr(65 + index): solution for index, solution in enumerate(solutions)}
        if letter not in mapping:
            raise ValueError("AFlow ScEnsemble returned an invalid candidate identifier")
        return {"response": mapping[letter]}


DISALLOWED_IMPORTS = (
    "os", "sys", "subprocess", "multiprocessing", "matplotlib", "seaborn",
    "plotly", "bokeh", "ggplot", "pylab", "tkinter", "PyQt5", "wx", "pyglet",
)


def run_code(code: str) -> tuple[str, str]:
    try:
        for library in DISALLOWED_IMPORTS:
            if f"import {library}" in code or f"from {library}" in code:
                return "Error", f"Prohibited import: {library} and graphing functionalities"
        namespace: dict[str, Any] = {}
        exec(compile(code, "<aflow-programmer>", "exec"), namespace)
        solve = namespace.get("solve")
        if not callable(solve):
            return "Error", "Function 'solve' not found"
        return "Success", str(solve())
    except Exception as error:
        details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        return "Error", f"Execution error: {error}\n{details}"


class Programmer:
    def __init__(self, llm, name: str = "Programmer"):
        self.llm, self.name = llm, name
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    def __del__(self):
        pool = getattr(self, "process_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    async def exec_code(self, code: str, timeout: int = 30):
        if self.llm.ctx.policy.get("aflow_official_code_operators") is not True:
            raise RuntimeError("Official AFlow code execution is disabled outside an isolated public-test sandbox")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.process_pool, run_code, code)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            future.cancel()
            return "Error", "Code execution timed out"

    async def __call__(self, problem: str, analysis: str = "None"):
        code = None
        output = None
        feedback = ""
        for _ in range(3):
            prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
                problem=problem, analysis=analysis, feedback=feedback
            )
            generated = await self.llm.generate_code(self.name, prompt, "solve")
            code = generated.get("response")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            feedback = (
                "\nThe result of the error from the code you wrote in the previous round:\n"
                f"Code: {code}\n\nStatus: {status}, {output}"
            )
        return {"code": code, "output": output}


def _public_test_result(solution: str, tests: list[str]) -> str | list[dict]:
    failures = []
    for test in tests:
        namespace: dict[str, Any] = {}
        try:
            exec(compile(solution + "\n" + test, "<aflow-public-test>", "exec"), namespace)
        except Exception as error:
            failures.append({"test_fail_case": {
                "test_case": test,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }})
    return "no error" if not failures else failures


class Test:
    def __init__(self, llm, name: str = "Test"):
        self.llm, self.name = llm, name

    def exec_code(self, solution: str, entry_point: str):
        if self.llm.ctx.policy.get("aflow_official_code_operators") is not True:
            raise RuntimeError("Official AFlow public tests are disabled outside an isolated public-test sandbox")
        tests = self.llm.ctx.policy.get("aflow_public_tests")
        if not isinstance(tests, list) or not tests or any(not isinstance(item, str) for item in tests):
            raise ValueError("Official AFlow Test requires explicit public tests")
        return _public_test_result(solution, tests)

    async def __call__(self, problem: str, solution: str, entry_point: str, test_loop: int = 3):
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                problem=problem,
                solution=solution,
                exec_pass="executed successfully",
                test_fail=result,
            )
            revised = await self.llm.generate_code(self.name, prompt, entry_point)
            solution = revised.get("response", solution)
        return {"result": self.exec_code(solution, entry_point) == "no error", "solution": solution}


def namespace(profile: str) -> dict[str, type]:
    classes = {
        "Custom": Custom,
        "AnswerGenerate": AnswerGenerate,
        "ScEnsemble": ScEnsemble,
        "Programmer": Programmer,
        "CustomCodeGenerate": CustomCodeGenerate,
        "Test": Test,
    }
    return {name: classes[name] for name in OPERATOR_PROFILES[profile]}
