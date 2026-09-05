"""Optimize a text DyLAN team on public questions, then freeze it for evaluation.

The pinned MMLU runner aggregates importance across queries. This adapter uses
the open-ended demo network and selects top individual agents by mean summed
layer importance; it does not reproduce MMLU subject/subset selection scripts.
No evaluator, answer key, or held-out prompt is loaded by this optimizer.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import random

from .aflow import digest
from .aflow_search import validate_split
from .artifact_provenance import provider_identity
from .core import JsonlTrace, RunContext, ToolEnvironment
from .dylan import ROLE_PROMPTS, backward, forward

REVISION = "006e440a519f7cf21e2826f3b8033d84ae9bf07c"
FORMAT = "dylan-team-v1"


def validate_config(roles: list[str], team_size: int, rounds: int, seed: int) -> None:
    if not isinstance(roles, list) or not roles or any(not isinstance(role, str) or role not in ROLE_PROMPTS for role in roles):
        raise ValueError("DyLAN candidates must be a nonempty list of pinned role names")
    if type(team_size) is not int or not 1 <= team_size <= len(roles):
        raise ValueError("DyLAN team size must be within the candidate population")
    if type(rounds) is not int or rounds < 1 or type(seed) is not int:
        raise ValueError("DyLAN requires positive rounds and an integer seed")


def freeze_team(records: list[dict], split: dict, *, roles: list[str], team_size: int = 2,
                rounds: int = 3, seed: int = 0, optimization_config: dict | None = None) -> dict:
    validate_split(split)
    validate_config(roles, team_size, rounds, seed)
    if not isinstance(records, list) or len(records) != len(split["optimization_case_ids"]):
        raise ValueError("DyLAN requires one importance record per optimization case")
    ids = [row.get("case_id") for row in records]
    if len(set(ids)) != len(ids) or set(ids) != set(split["optimization_case_ids"]):
        raise ValueError("DyLAN importance records do not match the optimization split")
    for row in records:
        scores = row.get("importance")
        if (not isinstance(scores, list) or len(scores) != len(roles)
                or any(type(score) not in (int, float) or not math.isfinite(score) or score < 0 for score in scores)
                or not sum(scores)):
            raise ValueError("DyLAN importance must contain finite nonnegative candidate scores with positive mass")
    scores = [sum(row["importance"][i] for row in records) / len(records) for i in range(len(roles))]
    selected = sorted(range(len(roles)), key=lambda index: (-scores[index], index))[:team_size]
    artifact = {"format": FORMAT, "source_revision": REVISION,
                "importance_protocol": "reply-hypothesis-v2",
                **{key: split[key] for key in ("benchmark", "optimization_case_ids", "evaluation_case_ids")},
                "candidate_roles": roles, "selected_agents": selected, "importance_scores": scores,
                "rounds": rounds, "seed": seed, "trials_sha256": digest(records),
                "optimization_config": optimization_config or {"kind": "external-importance-records"},
                "selection": "mean-layer-importance-top-k-stable-id"}
    artifact["artifact_sha256"] = digest(artifact)
    return artifact


def validate_team(artifact: dict, *, benchmark: str | None = None, case_id: str | None = None) -> dict:
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT or artifact.get("source_revision") != REVISION:
        raise ValueError("DyLAN requires a frozen team artifact at the pinned source revision")
    if artifact.get("artifact_sha256") != digest({k: v for k, v in artifact.items() if k != "artifact_sha256"}):
        raise ValueError("DyLAN frozen team checksum mismatch")
    validate_split(artifact)
    selected = artifact.get("selected_agents")
    if not isinstance(selected, list) or any(type(index) is not int for index in selected) or len(set(selected)) != len(selected):
        raise ValueError("DyLAN selected agents must be distinct integer candidate IDs")
    roles = artifact.get("candidate_roles")
    validate_config(roles, len(selected), artifact.get("rounds"), artifact.get("seed"))
    if any(not 0 <= index < len(roles) for index in selected):
        raise ValueError("DyLAN selected agent is outside candidate population")
    scores = artifact.get("importance_scores")
    if (not isinstance(scores, list) or len(scores) != len(roles)
            or any(type(score) not in (int, float) or not math.isfinite(score) or score < 0 for score in scores)
            or not sum(scores)):
        raise ValueError("DyLAN frozen team has invalid importance scores")
    expected = sorted(range(len(roles)), key=lambda index: (-scores[index], index))[:len(selected)]
    if selected != expected or artifact.get("selection") != "mean-layer-importance-top-k-stable-id":
        raise ValueError("DyLAN selected team does not match its declared selection rule")
    if artifact.get("importance_protocol") != "reply-hypothesis-v2":
        raise ValueError("DyLAN team uses an unsupported importance protocol")
    checksum = artifact.get("trials_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("DyLAN frozen team requires trial provenance")
    if benchmark is not None and benchmark != artifact["benchmark"]:
        raise ValueError("DyLAN frozen team benchmark mismatch")
    if case_id is not None and case_id not in artifact["evaluation_case_ids"]:
        raise ValueError("DyLAN case is outside the frozen evaluation split")
    return artifact


async def optimize_team(client, cases: list[dict], split: dict, output: Path, *, roles: list[str],
                        team_size: int = 2, rounds: int = 3, seed: int = 0) -> dict:
    validate_split(split)
    validate_config(roles, team_size, rounds, seed)
    if not isinstance(cases, list) or any(not isinstance(row, dict) for row in cases):
        raise ValueError("DyLAN optimization cases must be public id/prompt objects")
    ids = [row.get("id") for row in cases]
    if (any(not isinstance(case_id, str) for case_id in ids) or len(set(ids)) != len(ids)
            or set(ids) != set(split["optimization_case_ids"])):
        raise ValueError("DyLAN public questions must exactly match the optimization split")
    if any(set(row) != {"id", "prompt"} or not isinstance(row["prompt"], str) or not row["prompt"].strip() for row in cases):
        raise ValueError("DyLAN optimization cases accept only id and nonempty public prompt, never answer fields")
    output.mkdir(parents=True, exist_ok=False)
    by_id = {row["id"]: row for row in cases}
    records = []
    # Use one seeded stream across training queries, as the pinned MMLU entry does.
    rng = random.Random(seed)
    for index, case_id in enumerate(split["optimization_case_ids"]):
        prompt = by_id[case_id]["prompt"]
        trace = JsonlTrace(output / f"trial-{index + 1}.jsonl")
        ctx = RunContext("dylan-optimization", prompt, client, ToolEnvironment([], trace), trace, {})
        answer, layers = await forward(ctx, [ROLE_PROMPTS[role] for role in roles], list(range(len(roles))),
                                       rounds, rng, "optimization")
        scores = backward(layers, answer, len(roles))
        records.append({"case_id": case_id, "importance": scores, "llm_calls": ctx.llm_calls,
                        "prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        await trace.emit("dylan_optimization_importance", **records[-1])
        (output / "trials.json").write_text(json.dumps(records, indent=2) + "\n")
    artifact = freeze_team(records, split, roles=roles, team_size=team_size, rounds=rounds, seed=seed,
                           optimization_config=provider_identity(client))
    validate_team(artifact)
    (output / "team.json").write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--optimization-cases", type=Path, required=True, help="JSON array of public {id,prompt} objects")
    parser.add_argument("--roles", nargs="+", choices=list(ROLE_PROMPTS), default=["Assistant"] * 4)
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .api import completion_client_from_env
    asyncio.run(optimize_team(completion_client_from_env(), json.loads(args.optimization_cases.read_text()),
                             json.loads(args.split_manifest.read_text()), args.output, roles=args.roles,
                             team_size=args.team_size, rounds=args.rounds, seed=args.seed))
    print(args.output / "team.json")


if __name__ == "__main__":
    main()
