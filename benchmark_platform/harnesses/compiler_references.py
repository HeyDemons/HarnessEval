"""Pinned LLMCompiler text substitution over dynamic JSON tool arguments.

References: a00c9d3, output_parser.default_dependency_rule and
task_fetching_unit._replace_arg_mask_with_real_value.
Recursing into dict values is the dynamic-tool transport adaptation; each
string follows the upstream dependency and str(observation) rules.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable


ID_PATTERN = re.compile(r"\$\{?(\d+)\}?")


def infer_dependencies(task_id: str | int, arguments: Any) -> list[str]:
    """Match the pinned parser's numeric references to positive earlier IDs.

    Upstream scans raw argument text before parsing it. Our planner returns
    JSON objects, so scan their serialized text, including nested values.
    Explicit dependency edges are a separate JSON-planner adaptation.
    """
    index = int(task_id)
    if index < 1:
        raise ValueError("LLMCompiler task IDs must be positive integers")
    text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    references = {int(match) for match in ID_PATTERN.findall(text)}
    return [str(dependency) for dependency in sorted(references) if 0 < dependency < index]


def resolve_arguments(value: Any, dependencies: Iterable[str | int], results: dict[str, Any]) -> Any:
    dependencies = sorted({int(item) for item in dependencies}, reverse=True)
    if isinstance(value, str):
        for dependency in dependencies:
            observation = results.get(str(dependency))
            if observation is not None:
                for placeholder in ("${" + str(dependency) + "}", "$" + str(dependency)):
                    value = value.replace(placeholder, str(observation))
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(resolve_arguments(item, dependencies, results) for item in value)
    if isinstance(value, dict):
        return {key: resolve_arguments(item, dependencies, results) for key, item in value.items()}
    return value
