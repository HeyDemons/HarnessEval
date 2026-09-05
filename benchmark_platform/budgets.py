"""Budget authority and resolution, shared by host runners and native bridges.

Official limits are separate from operator-imposed wall envelopes. No secrets,
model prompts or scoring data belong in a budget record. Resolve after dotenv
loading; do not freeze environment-derived settings at module import time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import time
from typing import Callable, Mapping, Any


BUDGET_VERSION = "benchmark-budgets-v1"
DEFAULT_LOOP_SAFETY_LIMIT = 1000
DEFAULT_TERMINAL_AGENT_SECONDS = 600.0
DEFAULT_TERMINAL_VERIFIER_SECONDS = 600.0
TAU2_MAX_STEPS = 200
VITA_MAX_STEPS = 300
NATIVE_MAX_ERRORS = 10
AUTOMATION_MAX_RESPONSES = 50

OFFICIAL = {
    "gaia": {"kind": "workspace", "source": "gaia-benchmark/GAIA", "universal_time_limit": None},
    "gdpval": {"kind": "artifact-workspace", "source": "openai/gdpval", "universal_time_limit": None},
    "trajectory-bench": {"kind": "remote-tools", "source": "PengfeiHePower/TRAJECT-Bench@2723fd8", "universal_time_limit": None},
    "bfcl": {"kind": "single-turn-declaration", "model_responses": 1, "source": "ShishirPatil/gorilla@6ea5797"},
    "tau2": {"kind": "native-episode", "simulation_steps": TAU2_MAX_STEPS, "max_errors": NATIVE_MAX_ERRORS,
             "source": "sierra-research/tau2-bench@79975ac:src/tau2/config.py"},
    "vitabench": {"kind": "native-episode", "simulation_steps": VITA_MAX_STEPS, "max_errors": NATIVE_MAX_ERRORS,
                  "source": "meituan-longcat/vitabench@742e240:src/vita/config.py"},
    "automationbench": {"kind": "native-workflow", "model_responses": AUTOMATION_MAX_RESPONSES,
                        "source": "zapier/AutomationBench@4a8e106:CLI max-steps default"},
    "terminal-bench-2": {"kind": "task-phases", "source": "harbor-framework/terminal-bench-2@2fd12b8:task.toml"},
    "swe-bench-verified": {"kind": "task-container", "source": "SWE-bench/SWE-bench@726c546", "universal_time_limit": None},
}


def seconds(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite seconds")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite")
    return number


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def terminal_phase_seconds(metadata: Mapping[str, Any]) -> tuple[float, float]:
    return (
        seconds((metadata.get("agent") or {}).get("timeout_sec", DEFAULT_TERMINAL_AGENT_SECONDS), "agent.timeout_sec"),
        seconds((metadata.get("verifier") or {}).get("timeout_sec", DEFAULT_TERMINAL_VERIFIER_SECONDS), "verifier.timeout_sec"),
    )


def loop_limit(benchmark: str, env: Mapping[str, str]) -> int:
    key = {"gdpval": "GDPVAL_MAX_TURNS", "terminal-bench-2": "TERMINAL_BENCH_MAX_TURNS"}.get(benchmark)
    value = env[key] if key and key in env else env.get("HARNESS_LOOP_SAFETY_LIMIT", DEFAULT_LOOP_SAFETY_LIMIT)
    return positive_int(value, key if key and key in env else "HARNESS_LOOP_SAFETY_LIMIT")


def baseline_limits(benchmark: str, env: Mapping[str, str] | None = None) -> dict:
    env = os.environ if env is None else env
    official = OFFICIAL[benchmark]
    policy = {"max_turns": loop_limit(benchmark, env), "finalize_on_loop_limit": True, "budget_version": BUDGET_VERSION}
    if "simulation_steps" in official:
        policy.update(native_max_steps=official["simulation_steps"], native_max_errors=official["max_errors"])
    if "model_responses" in official:
        policy["model_response_limit"] = official["model_responses"]
    return policy


def native_steps(benchmark: str, policy: Mapping[str, Any]) -> int:
    return positive_int(policy.get("native_max_steps", OFFICIAL[benchmark]["simulation_steps"]), "native_max_steps")


def native_errors(benchmark: str, policy: Mapping[str, Any]) -> int:
    return positive_int(policy.get("native_max_errors", OFFICIAL[benchmark]["max_errors"]), "native_max_errors")


def external_scorer_seconds(benchmark: str, env: Mapping[str, str]) -> float | None:
    if benchmark not in {"bfcl", "trajectory-bench", "gdpval"}:
        return None
    key = "GDPVAL_SCORER_TIMEOUT_S" if benchmark == "gdpval" else "HARNESS_SCORER_TIMEOUT_S"
    default = "600" if benchmark == "gdpval" else "900"
    return seconds(env.get(key, env.get("HARNESS_SCORER_TIMEOUT_S", default)), key)


@dataclass(frozen=True)
class BudgetPlan:
    benchmark: str
    official: dict[str, Any]
    wall_seconds: float
    wall_source: str
    cancel_grace_seconds: float
    overhead_seconds: float | None
    baseline_loop_safety_limit: int
    agent_seconds: float | None = None
    verifier_seconds: float | None = None
    external_scorer_seconds: float | None = None
    version: str = BUDGET_VERSION

    @property
    def process_seconds(self) -> float:
        """Time until cancellation starts, with grace INSIDE the wall envelope."""
        return self.wall_seconds - self.cancel_grace_seconds

    def as_dict(self) -> dict:
        return {**asdict(self), "process_seconds": self.process_seconds,
                "wall_scope": "prepare + agent + in-process native evaluation + cancellation",
                "external_host_scoring_included": False}

    def baseline_policy(self) -> dict:
        return baseline_limits(self.benchmark, {"HARNESS_LOOP_SAFETY_LIMIT": str(self.baseline_loop_safety_limit)})


def resolve_budget(benchmark: str, *, env: Mapping[str, str] | None = None,
                   task_metadata: Mapping[str, Any] | None = None) -> BudgetPlan:
    if benchmark not in OFFICIAL:
        raise ValueError(f"Unknown benchmark budget: {benchmark}")
    env = os.environ if env is None else env
    official = dict(OFFICIAL[benchmark])
    loop = loop_limit(benchmark, env)
    grace = seconds(env.get("HARNESS_ARM_CANCEL_GRACE_S", "10"), "HARNESS_ARM_CANCEL_GRACE_S", allow_zero=True)
    if benchmark == "terminal-bench-2":
        if task_metadata is None:
            raise ValueError("Terminal-Bench budget requires task.toml metadata")
        agent, verifier = terminal_phase_seconds(task_metadata)
        overhead = seconds(env.get("TERMINAL_BENCH_OUTER_GRACE_S", "120"), "TERMINAL_BENCH_OUTER_GRACE_S", allow_zero=True)
        official.update(agent_seconds=agent, verifier_seconds=verifier)
        return BudgetPlan(benchmark, official, agent + verifier + overhead,
                          "official phases + local overhead envelope", min(grace, overhead), overhead, loop, agent, verifier)
    key, default = {"gdpval": ("GDPVAL_ARM_TIMEOUT_S", "1200"),
                    "vitabench": ("VITABENCH_ARM_TIMEOUT_S", "1200")}.get(benchmark, ("HARNESS_ARM_TIMEOUT_S", "900"))
    wall = seconds(env.get(key, default), key)
    if grace >= wall:
        raise ValueError(f"{key} must exceed HARNESS_ARM_CANCEL_GRACE_S")
    scorer = external_scorer_seconds(benchmark, env)
    if scorer is not None and scorer <= grace:
        raise ValueError("External scorer timeout must exceed cancellation grace")
    return BudgetPlan(benchmark, official, wall, f"local operator setting {key}", grace, None, loop,
                      external_scorer_seconds=scorer)


class Deadline:
    """One monotonic deadline, reused across retries; never reset per operation."""
    def __init__(self, duration: float, *, clock: Callable[[], float] | None = None, started: float | None = None):
        self.clock = clock or time.monotonic
        self.duration = seconds(duration, "deadline duration")
        self.started = self.clock() if started is None else started
        self.expires = self.started + self.duration

    @property
    def remaining(self) -> float:
        return max(0.0, self.expires - self.clock())

    @property
    def elapsed(self) -> float:
        return max(0.0, self.clock() - self.started)


class ModelBudgetExceeded(RuntimeError):
    """The declared model-response budget has been spent, not an infra failure."""


class ModelResponseBudget:
    def __init__(self, limit: Any = None):
        self.limit = positive_int(limit, "model_response_limit") if limit is not None else None
        self.used = 0

    def reserve(self) -> None:
        # No await between checking and reserving: parallel workers sharing one
        # RunContext cannot each independently spend the final response slot.
        if self.limit is not None and self.used >= self.limit:
            raise ModelBudgetExceeded("Model response budget exhausted")
        self.used += 1

    @property
    def final_response(self) -> bool:
        return self.limit is not None and self.used == self.limit - 1
