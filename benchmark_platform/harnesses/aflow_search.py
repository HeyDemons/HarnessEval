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


def selection_probabilities(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("AFlow selection requires evaluated candidates")
    # data_utils.py: raw [0,1] scores are multiplied by 100 before alpha=.2.
    maximum = max(scores)
    weights = [math.exp(.2 * 100 * (score - maximum)) for score in scores]
    total = sum(weights)
    return [.3 / len(scores) + .7 * weight / total for weight in weights]


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
                   seed: int = 0, problem_type: str = "question answering") -> dict:
    validate_split(split)
    if min(rounds, validation_rounds, sample) < 1:
        raise ValueError("AFlow search budgets must be positive")
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    history: list[dict] = []

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
        candidates = sorted((row for row in history if row.get("score") is not None),
                            key=lambda row: (-row["score"], row["round"]))[:sample]
        parent = rng.choices(candidates, selection_probabilities([row["score"] for row in candidates]))[0]
        row = {"round": round_id, "parent": parent["round"], "modification": "", "score": None}
        # Transport/evaluator errors must propagate; they are missing measurements,
        # not low-scoring workflows. Only malformed candidate source is retryable.
        reply = await client.complete([{"role": "user", "content": expansion_prompt(parent, history, problem_type)}])
        (output / f"expansion-{round_id}.txt").write_text(reply.content)
        try:
            fields = {}
            for key in ("graph", "prompt", "modification"):
                match = re.search(fr"<{key}>(.*?)</{key}>", reply.content, re.DOTALL)
                if match is None:
                    raise ValueError(f"Missing expansion field {key}")
                fields[key] = match.group(1).strip()
            if not fields["modification"] or any(r.get("parent") == parent["round"] and r["modification"] == fields["modification"] for r in history):
                raise ValueError("Empty or repeated modification")
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
    best = max((row for row in history if row.get("score") is not None), key=lambda row: row["score"])
    frozen = make_artifact(best["artifact"]["graph"], best["artifact"]["prompt"], provenance={
        "kind": "optimized", "source_revision": REVISION, **split,
        "validation_score": best["score"], "selected_round": best["round"],
        "search_history_sha256": digest(history), "optimizer": "aflow-score-mixture-python-v1",
        "seed": seed, "rounds": rounds, "validation_rounds": validation_rounds, "sample": sample,
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
                   problem_type=args.problem_type)
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
    parser.add_argument("--evaluation-timeout", type=int, default=900)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
