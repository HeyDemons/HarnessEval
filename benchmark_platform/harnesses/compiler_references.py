"""Pinned LLMCompiler text substitution over dynamic JSON tool arguments.

Reference: a00c9d3, task_fetching_unit._replace_arg_mask_with_real_value.
Recursing into dict values is the dynamic-tool transport adaptation; each
string follows the upstream declared-dependency and str(observation) rules.
"""
from __future__ import annotations

from typing import Any, Iterable


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
