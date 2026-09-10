"""Pinned official AFlow MCTS-variant search with isolated evaluation.

Implements FoundationAgents/AFlow@3f45721 score-mixture selection, official
optimizer prompts, dataset-profiled operators, LLM code expansion, repeated
validation, parent-indexed experience and convergence. ``benchmark-tools`` is
an explicit secondary operator adapter. The CLI never executes generated graph
code and never loads benchmark answer keys.
An evaluation command receives a candidate artifact path and the optimization
case manifest path; it must run agents in sandboxes and score only after exit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import random
import re
from typing import Awaitable, Callable

from . import aflow_official as official
from .aflow import REVISION, digest
from .api import aflow_optimizer_client_from_env
from .artifact_provenance import provider_identity


def selection_probabilities(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("AFlow selection requires evaluated candidates")
    # data_utils.py: raw [0,1] scores are multiplied by 100 before alpha=.2.
    maximum = max(scores)
    weights = [math.exp(.2 * 100 * (score - maximum)) for score in scores]
    total = sum(weights)
    return [.3 / len(scores) + .7 * weight / total for weight in weights]


def convergence(history: list[dict], top_k: int = 3, consecutive_rounds: int = 5) -> dict:
    """Pinned ConvergenceUtils default (z=0): five unchanged top-k means.

    Failed, unscored expansions are absent from upstream results.json, so they
    cannot count toward convergence here either. Report actual round IDs rather
    than the source utility's zero-based positions in its scored-round array.
    """
    scored = [row for row in history if row.get("score") is not None]
    result = {"converged": False, "start_round": None, "final_round": None}
    if len(scored) < top_k + 1:
        return result
    previous = None
    unchanged = 0
    for index, row in enumerate(scored):
        top = sorted((item["score"] for item in scored[:index + 1]), reverse=True)[:top_k]
        current = sum(top) / len(top)
        unchanged = unchanged + 1 if previous is not None and current == previous else 0
        if unchanged >= consecutive_rounds:
            return {"converged": True, "start_round": scored[index - consecutive_rounds + 1]["round"],
                    "final_round": row["round"]}
        previous = current
    return result


def validate_split(split: dict):
    if not isinstance(split.get("benchmark"), str) or not split["benchmark"]:
        raise ValueError("AFlow split manifest requires a benchmark")
    for field in ("optimization_case_ids", "evaluation_case_ids"):
        values = split.get(field)
        if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"AFlow requires {field}")
        if len(values) != len(set(values)):
            raise ValueError(f"AFlow duplicate ids in {field}")
    if set(split["optimization_case_ids"]) & set(split["evaluation_case_ids"]):
        raise ValueError("AFlow optimization and evaluation cases overlap")


def _format_experience(parent: dict, history: list[dict]) -> str:
    children = [row for row in history if row.get("parent") == parent["round"]]
    if not children:
        return f"No experience data found for round {parent['round']}."
    lines = [f"Original Score: {parent['score']}", "These are some conclusions drawn from experience:", ""]
    for row in children:
        if row.get("score") is not None and row["score"] > parent["score"]:
            continue
        lines.append(f"-Absolutely prohibit {row['modification']} (Score: {row.get('score')})")
    # The pinned source uses the same 'Absolutely prohibit' wording for successful
    # children too. Preserve it as algorithm behavior rather than silently fixing it.
    for row in children:
        if row.get("score") is not None and row["score"] > parent["score"]:
            lines.append(f"-Absolutely prohibit {row['modification']}")
    lines.extend(["", "Note: Take into account past failures and avoid repeating the same mistakes, as these failures indicate that these approaches are ineffective. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax like for, if, else, etc., or modifying the prompt."])
    return "\n".join(lines)


def _operator_description(profile: str) -> str:
    if profile in official.OPERATOR_PROFILES:
        return official.operator_description(profile)
    if profile == "benchmark-tools":
        from .aflow_tools import OPERATOR_DESCRIPTION

        return official.operator_description("qa") + "\n" + OPERATOR_DESCRIPTION
    raise ValueError(f"Unknown AFlow operator profile: {profile}")


def expansion_prompt(parent: dict, history: list[dict], problem_type: str,
                     operator_profile: str = "qa") -> str:
    adapter = ""
    if operator_profile == "benchmark-tools":
        adapter = (
            "\nThis operator profile is an explicit benchmark adapter, not an official AFlow dataset. "
            "Preserve exactly one ToolSession and its real observation/commit boundary. Generated code runs "
            "with restricted builtins: no imports, reflection, filesystem access, direct context access, or "
            "invented observations. Return session.answer after the session becomes inactive."
        )
    graph_input = official.WORKFLOW_INPUT.format(
        experience=_format_experience(parent, history),
        score=parent["score"],
        graph=parent["artifact"]["graph"],
        prompt=parent["artifact"]["prompt"],
        operator_description=_operator_description(operator_profile),
        type=problem_type,
        log=json.dumps(parent.get("feedback", [])),
    )
    return (
        graph_input
        + official.WORKFLOW_CUSTOM_USE
        + official.WORKFLOW_OPTIMIZE_PROMPT.format(type=problem_type)
        + adapter
        + official.GRAPH_RESPONSE_FORMAT
    )


async def optimize(client, evaluate: Callable[[dict], Awaitable[dict]], split: dict, output: Path, *,
                   rounds: int = 20, validation_rounds: int = 5, sample: int = 4,
                   seed: int = 0, problem_type: str = "question answering",
                   check_convergence: bool = True, max_generation_attempts: int | None = None,
                   resume: bool = False, operator_profile: str = "qa") -> dict:
    if operator_profile == "benchmark-tools":
        from .aflow_tools import make_artifact, validate_artifact
    elif operator_profile in official.OPERATOR_PROFILES:
        from .aflow import make_artifact as official_make_artifact, validate_artifact

        def make_artifact(graph=None, prompt="", *, provenance=None):
            return official_make_artifact(
                graph, prompt, operator_profile=operator_profile, provenance=provenance
            )
    else:
        raise ValueError(f"Unknown AFlow operator profile: {operator_profile}")
    validate_split(split)
    if min(rounds, validation_rounds, sample) < 1:
        raise ValueError("AFlow search budgets must be positive")
    if not isinstance(check_convergence, bool):
        raise ValueError("check_convergence must be a boolean")
    if max_generation_attempts is not None and max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be positive or None")
    identity = {
        "split": split,
        "rounds": rounds,
        "validation_rounds": validation_rounds,
        "sample": sample,
        "seed": seed,
        "problem_type": problem_type,
        "check_convergence": check_convergence,
        "max_generation_attempts": max_generation_attempts,
        "operator_profile": operator_profile,
    }
    state_path = output / "search.json"
    if resume:
        if not output.is_dir() or not (output / "history.json").is_file():
            raise ValueError("AFlow resume requires an existing search history")
        history = json.loads((output / "history.json").read_text())
        generations = json.loads((output / "generations.json").read_text()) if (output / "generations.json").is_file() else []
        if (not isinstance(history, list) or not history
                or [row.get("round") for row in history] != list(range(1, len(history) + 1))):
            raise ValueError("AFlow resume history must contain contiguous rounds starting at one")
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get("identity") != identity:
                raise ValueError("AFlow resume configuration differs from the original search")
        else:
            # Search directories created before resumable metadata existed are
            # accepted only through the explicit flag; main_async still checks
            # their optimization-only manifest against the requested split.
            state = {"identity": identity, "legacy_resume": True, "resume_count": 0}
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    else:
        output.mkdir(parents=True, exist_ok=False)
        history: list[dict] = []
        generations: list[dict] = []
        state = {"identity": identity, "legacy_resume": False, "resume_count": 0}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    rng = random.Random(seed)
    if resume:
        # Weighted random selection consumes one random draw per generation
        # attempt. Replay them so future parent choices remain deterministic.
        for _ in generations:
            rng.random()
    stopped = convergence(history)

    async def assess(artifact: dict) -> tuple[float, list]:
        scores, feedback = [], []
        for _ in range(validation_rounds):
            result = await evaluate(artifact)
            score = result.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("AFlow evaluator must return a finite score in [0,1]")
            scores.append(score)
            # The evaluator decides which optimization-only feedback is public.
            if "feedback" in result:
                feedback.append(result["feedback"])
        return sum(scores) / len(scores), feedback

    if not resume:
        artifact = make_artifact()
        score, feedback = await assess(artifact)
        history.append({"round": 1, "parent": None, "modification": "initialization", "artifact": artifact,
                        "score": score, "feedback": feedback})
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    start_round = history[-1]["round"] + 1
    pending_convergence = check_convergence and stopped["converged"]
    for round_id in (() if pending_convergence else range(start_round, rounds + 2)):
        accepted = [item for item in generations
                    if item.get("round") == round_id and item.get("rejection") is None]
        if len(accepted) > 1:
            raise ValueError(f"AFlow round {round_id} has multiple accepted expansions")
        if accepted:
            pending = output / f"expansion-{round_id}.txt"
            if not pending.is_file():
                raise ValueError(f"AFlow round {round_id} is missing its accepted expansion")
            content = pending.read_text()
            fields = {key: match.group(1).strip() for key in ("graph", "prompt", "modification")
                      if (match := re.search(fr"<{key}>(.*?)</{key}>", content, re.DOTALL))}
            parent = next((item for item in history if item["round"] == accepted[0]["parent"]), None)
            if parent is None:
                raise ValueError(f"AFlow round {round_id} refers to a missing parent")
            row = {"round": round_id, "parent": parent["round"], "modification": "", "score": None,
                   "resumed_pending_evaluation": True}
        else:
            prior_attempts = [item for item in generations if item.get("round") == round_id]
            attempt = max((item.get("attempt", 0) for item in prior_attempts), default=0)
            while True:
                if max_generation_attempts is not None and attempt >= max_generation_attempts:
                    raise RuntimeError("AFlow same-round generation budget exhausted")
                attempt += 1
                # The pinned optimizer reselects the parent inside the regeneration loop.
                candidates = sorted((item for item in history if item.get("score") is not None),
                                    key=lambda item: (-item["score"], item["round"]))[:sample]
                parent = rng.choices(candidates, selection_probabilities([item["score"] for item in candidates]))[0]
                row = {"round": round_id, "parent": parent["round"], "modification": "", "score": None}
                # Provider failures remain missing measurements, never failed candidates.
                reply = await client.complete([{
                    "role": "user",
                    "content": expansion_prompt(parent, history, problem_type, operator_profile),
                }])
                (output / f"expansion-{round_id}-attempt-{attempt}.txt").write_text(reply.content)
                fields = {key: match.group(1).strip() for key in ("graph", "prompt", "modification")
                          if (match := re.search(fr"<{key}>(.*?)</{key}>", reply.content, re.DOTALL))}
                modification = fields.get("modification", "")
                repeated = any(item.get("parent") == parent["round"] and item["modification"] == modification
                               for item in history)
                retry = "empty_modification" if not modification else "repeated_modification" if repeated else None
                generations.append({"round": round_id, "attempt": attempt, "parent": parent["round"],
                                    "modification": modification, "rejection": retry,
                                    "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
                                    "elapsed_seconds": reply.elapsed_seconds, "transport_retries": reply.transport_retries})
                (output / "generations.json").write_text(json.dumps(generations, indent=2) + "\n")
                if retry is None:
                    break
            (output / f"expansion-{round_id}.txt").write_text(reply.content)
        try:
            for key in ("graph", "prompt", "modification"):
                if key not in fields:
                    raise ValueError(f"Missing expansion field {key}")
            row["modification"] = fields["modification"]
            artifact = make_artifact(fields["graph"], fields["prompt"])
            validate_artifact(artifact, allow_initialization=True)
            if artifact["code_sha256"] == parent["artifact"]["code_sha256"]:
                raise ValueError("Expansion did not change graph or prompts")
            row.update(artifact=artifact, modification=fields["modification"])
        except (ValueError, SyntaxError) as exc:
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
        else:
            row["score"], row["feedback"] = await assess(artifact)
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        stopped = convergence(history)
        if check_convergence and stopped["converged"]:
            break
    best = max((row for row in history if row.get("score") is not None), key=lambda row: row["score"])
    frozen = make_artifact(best["artifact"]["graph"], best["artifact"]["prompt"], provenance={
        "kind": "optimized", "source_revision": REVISION, **split,
        "validation_score": best["score"], "selected_round": best["round"],
        "search_history_sha256": digest(history), "optimizer": "foundationagents-aflow-mcts-v1-port",
        "optimizer_source_revision": REVISION,
        "generations_sha256": digest(generations), "generation_calls": len(generations),
        "optimization_config": provider_identity(client),
        "seed": seed, "rounds": rounds, "validation_rounds": validation_rounds, "sample": sample,
        "completed_rounds": len(history) - 1, "check_convergence": check_convergence,
        "stop_reason": "converged" if check_convergence and stopped["converged"] else "round_budget",
        "convergence": stopped, "max_generation_attempts": max_generation_attempts,
        "operator_profile": operator_profile,
        "benchmark_adapter": operator_profile == "benchmark-tools",
        "resume_count": state["resume_count"],
    })
    validate_artifact(frozen)
    (output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


async def main_async(args):
    split = json.loads(args.split_manifest.read_text())
    validate_split(split)
    command = json.loads(args.evaluate_command)
    if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
        raise ValueError("--evaluate-command must be a JSON argv list")
    counter = 0
    # Do not pass evaluation IDs to the evaluator or optimizer prompts.
    evaluations = args.output.with_name(args.output.name + "-evaluations")
    evaluations.mkdir(parents=True, exist_ok=args.resume)
    opt_manifest = evaluations / "optimization-cases.json"
    expected_manifest = {"benchmark": split["benchmark"], "case_ids": split["optimization_case_ids"]}
    if args.resume:
        if not opt_manifest.is_file() or json.loads(opt_manifest.read_text()) != expected_manifest:
            raise ValueError("AFlow resume split differs from the existing optimization manifest")
        existing = [int(match.group(1)) for path in evaluations.glob("candidate-*.json")
                    if (match := re.fullmatch(r"candidate-(\d+)\.json", path.name))]
        counter = max(existing, default=0)
    else:
        opt_manifest.write_text(json.dumps(expected_manifest))

    async def evaluate(artifact):
        nonlocal counter
        counter += 1
        path = evaluations / f"candidate-{counter}.json"
        path.write_text(json.dumps(artifact))
        process = await asyncio.create_subprocess_exec(*command, str(path), str(opt_manifest),
                                                       stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), args.evaluation_timeout)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        (evaluations / f"evaluation-{counter}.json").write_bytes(stdout)
        (evaluations / f"evaluation-{counter}.log").write_bytes(stderr)
        if process.returncode:
            raise RuntimeError("AFlow evaluator failed; inspect isolated evaluation artifacts")
        return json.loads(stdout)

    await optimize(aflow_optimizer_client_from_env(), evaluate, split, args.output, rounds=args.rounds,
                   validation_rounds=args.validation_rounds, sample=args.sample, seed=args.seed,
                   problem_type=args.problem_type, check_convergence=args.check_convergence,
                   max_generation_attempts=args.max_generation_attempts, resume=args.resume,
                   operator_profile=args.operator_profile)
    print(str(args.output / "frozen.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evaluate-command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--validation-rounds", type=int, default=5)
    parser.add_argument("--sample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problem-type", default="question answering")
    parser.add_argument("--operator-profile", choices=(*official.OPERATOR_PROFILES, "benchmark-tools"),
                        default="qa")
    parser.add_argument("--check-convergence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-generation-attempts", type=int,
                        help="Optional per-round regeneration cap; unset matches the upstream unbounded loop")
    parser.add_argument("--evaluation-timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
