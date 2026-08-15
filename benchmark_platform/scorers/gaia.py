from __future__ import annotations

import re
import string


def _normalize_text(value: str, *, punctuation: bool = True) -> str:
    normalized = re.sub(r"\s", "", value).lower()
    return normalized.translate(str.maketrans("", "", string.punctuation)) if punctuation else normalized


def _number(value: str) -> float:
    cleaned = value
    for marker in ("$", "%", ","):
        cleaned = cleaned.replace(marker, "")
    try:
        return float(cleaned)
    except ValueError:
        return float("inf")


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def question_score(answer: str | None, target: str) -> bool:
    candidate = "None" if answer is None else answer
    if _is_float(target):
        return _number(candidate) == float(target)
    if "," in target or ";" in target:
        expected = re.split(r"[,;]", target)
        actual = re.split(r"[,;]", candidate)
        if len(expected) != len(actual):
            return False
        return all(
            _number(left) == float(right)
            if _is_float(right)
            else _normalize_text(left, punctuation=False) == _normalize_text(right, punctuation=False)
            for left, right in zip(actual, expected)
        )
    return _normalize_text(candidate) == _normalize_text(target)
