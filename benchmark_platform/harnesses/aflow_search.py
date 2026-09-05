"""Offline AFlow code search; evaluation is delegated to an isolated process.

Implements the pinned optimizer's score-mixture selection, LLM code expansion,
repeated validation, and parent-indexed success/failure experience. The CLI
never executes generated graph code and never loads benchmark answer keys.
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

from .aflow import REVISION, digest, make_artifact, validate_artifact
from .api import completion_client_from_env
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


def expansion_prompt(parent: dict, history: list[dict], problem_type: str) -> str:
    experience = [{"modification": row["modification"], "before": parent["score"],
                   "after": row.get("score"), "succeed": row.get("score") is not None and row["score"] > parent["score"]}
                  for row in history if row.get("parent") == parent["round"]]
    return (
        f"Optimize this Python AFlow workflow for {problem_type} tasks. Change one detail at a time, at most five "
        "graph lines. You may add/remove operators, change their parameters or custom prompts, and use Python "
        "loops and conditions. Keep at most ten logical nodes. Preserve Workflow(name,llm_config,dataset) "
        "and its async __call__(problem) returning (answer, cost).\n"
        "Available operators: Custom(llm)(input,instruction) -> {'response':str}; "
        "AnswerGenerate(llm)(input) -> {'thought':str,'answer':str}; "
        "ScEnsemble(llm)(solutions) -> {'response':str}, selecting one original candidate. "
        "create_llm_instance(llm_config), operator and prompt_custom are supplied. "
        "All custom prompt constants must be defined in the prompt source. Do not redefine built-in operator prompts. "
        "Custom concatenates instruction+input literally; pass prior results explicitly. Do not put answers in code. "
        "Avoid modifications already attempted from this parent. Output complete Python sources in "
        "<graph>...</graph> and <prompt>...</prompt>, and describe the single change in <modification>...</modification>.\n"
        f"Validation score: {parent['score']}\nGraph:\n{parent['artifact']['graph']}\n"
        f"Prompt:\n{parent['artifact']['prompt']}\nExperience:\n{json.dumps(experience)}\n"
        f"Optimization-split feedback:\n{json.dumps(parent.get('feedback', []))}"
    )


async def optimize(client, evaluate: Callable[[dict], Awaitable[dict]], split: dict, output: Path, *,
                   rounds: int = 20, validation_rounds: int = 5, sample: int = 4,
                   seed: int = 0, problem_type: str = "question answering",
                   check_convergence: bool = True, max_generation_attempts: int | None = None) -> dict:
    validate_split(split)
    if min(rounds, validation_rounds, sample) < 1:
        raise ValueError("AFlow search budgets must be positive")
    if not isinstance(check_convergence, bool):
        raise ValueError("check_convergence must be a boolean")
    if max_generation_attempts is not None and max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be positive or None")
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    history: list[dict] = []
    generations: list[dict] = []
    stopped = {"converged": False, "start_round": None, "final_round": None}

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

    artifact = make_artifact()
    score, feedback = await assess(artifact)
    history.append({"round": 1, "parent": None, "modification": "initialization", "artifact": artifact,
                    "score": score, "feedback": feedback})
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    for round_id in range(2, rounds + 2):
        attempt = 0
        while True:
            if max_generation_attempts is not None and attempt >= max_generation_attempts:
                raise RuntimeError("AFlow same-round generation budget exhausted")
            attempt += 1
            # The pinned optimizer reselects the parent inside the regeneration loop.
            candidates = sorted((row for row in history if row.get("score") is not None),
                                key=lambda row: (-row["score"], row["round"]))[:sample]
            parent = rng.choices(candidates, selection_probabilities([row["score"] for row in candidates]))[0]
            row = {"round": round_id, "parent": parent["round"], "modification": "", "score": None}
            # Provider failures remain missing measurements, never failed candidates.
            reply = await client.complete([{"role": "user", "content": expansion_prompt(parent, history, problem_type)}])
            (output / f"expansion-{round_id}-attempt-{attempt}.txt").write_text(reply.content)
            fields = {key: match.group(1).strip() for key in ("graph", "prompt", "modification")
                      if (match := re.search(fr"<{key}>(.*?)</{key}>", reply.content, re.DOTALL))}
            modification = fields.get("modification", "")
            repeated = any(r.get("parent") == parent["round"] and r["modification"] == modification
                           for r in history)
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
        "search_history_sha256": digest(history), "optimizer": "aflow-score-mixture-python-v2",
        "generations_sha256": digest(generations), "generation_calls": len(generations),
        "optimization_config": provider_identity(client),
        "seed": seed, "rounds": rounds, "validation_rounds": validation_rounds, "sample": sample,
        "completed_rounds": len(history) - 1, "check_convergence": check_convergence,
        "stop_reason": "converged" if check_convergence and stopped["converged"] else "round_budget",
        "convergence": stopped, "max_generation_attempts": max_generation_attempts,
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
    evaluations.mkdir(parents=True, exist_ok=False)
    opt_manifest = evaluations / "optimization-cases.json"
    opt_manifest.write_text(json.dumps({"benchmark": split["benchmark"], "case_ids": split["optimization_case_ids"]}))

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

    await optimize(completion_client_from_env(), evaluate, split, args.output, rounds=args.rounds,
                   validation_rounds=args.validation_rounds, sample=args.sample, seed=args.seed,
                   problem_type=args.problem_type, check_convergence=args.check_convergence,
                   max_generation_attempts=args.max_generation_attempts)
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
    parser.add_argument("--check-convergence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-generation-attempts", type=int,
                        help="Optional per-round regeneration cap; unset matches the upstream unbounded loop")
    parser.add_argument("--evaluation-timeout", type=int, default=900)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
