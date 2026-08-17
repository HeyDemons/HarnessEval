#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SEED = "harnesseval-light-v1"
TRAJECT_REVISION = "2723fd890778dbfb6af9e3aa8ee1c22272979468"
VITA_REVISION = "742e240855bf8686a0842360749d5ea970ea3987"
TAU_REVISION = "79975ac5741e23fbb1d2ac44262d62398a6d87bd"
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
TERMINAL_REVISION = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"


def rank(benchmark: str, stratum: str, case_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{benchmark}:{stratum}:{case_id}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_pick(rows: Iterable[dict[str, Any]], count: int, benchmark: str, stratum: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: rank(benchmark, stratum, str(row["id"])))
    if len(ordered) < count:
        raise ValueError(f"{benchmark}/{stratum} has {len(ordered)} cases; {count} requested")
    return ordered[:count]


def balanced_pick(
    rows: list[dict[str, Any]],
    count: int,
    benchmark: str,
    stratum: str,
    dimensions: tuple[str, ...],
) -> list[dict[str, Any]]:
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    seen = {dimension: Counter() for dimension in dimensions}
    while len(selected) < count:
        if not remaining:
            raise ValueError(f"{benchmark}/{stratum} has fewer than {count} eligible cases")

        def key(row: dict[str, Any]) -> tuple[float, str]:
            coverage = sum(1.0 / (1 + seen[dimension][str(row[dimension])]) for dimension in dimensions)
            return (-coverage, rank(benchmark, stratum, str(row["id"])))

        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        selected.append(chosen)
        for dimension in dimensions:
            seen[dimension][str(chosen[dimension])] += 1
    return selected


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{manifest['benchmark']}.json"
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, target)


def base_manifest(
    benchmark: str,
    source: dict[str, Any],
    cases: list[dict[str, Any]],
    strata_summary: dict[str, int],
    method: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "mode": "light",
        "status": "ready",
        "declared_count": len(cases),
        "source": source,
        "selection_policy": {
            "frozen": True,
            "seed": SEED,
            "method": method,
            "model_outcomes_used": False,
            "historical_runs_used": False,
        },
        "strata_summary": strata_summary,
        "cases": cases,
    }


def generate_gaia(orch_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    data_root = orch_root / "gaia_data" / "2023"
    cases: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    for level, count in ((1, 10), (2, 20), (3, 26)):
        path = data_root / "validation" / f"metadata.level{level}.parquet"
        file_hashes[str(path.relative_to(data_root))] = file_sha256(path)
        rows = [
            {
                "id": str(row["task_id"]),
                "split": "validation",
                "level": level,
                "attachment": str(row.get("file_name") or ""),
                "scoreability": "local_official",
            }
            for row in pq.read_table(path).to_pylist()
        ]
        cases.extend(stable_pick(rows, count, "gaia", f"validation-l{level}"))

    manifest = base_manifest(
        "gaia",
        {"dataset": "GAIA 2023", "metadata_sha256": file_hashes},
        cases,
        {"level_1": 10, "level_2": 20, "level_3": 26},
        "SHA-256 rank within the public validation split; all 26 locally scoreable validation L3 cases are retained",
    )
    manifest["locally_scoreable_count"] = 56
    manifest["scoring_note"] = "Every selected case has a public validation answer and participates in the local score denominator."
    return manifest


def deliverable_type(files: list[str]) -> str:
    suffixes = sorted({Path(name).suffix.lower() or "none" for name in files})
    return "+".join(suffixes) or "none"


def generate_gdpval(orch_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = orch_root / "gdpval" / "data" / "train-00000-of-00001.parquet"
    rows: list[dict[str, Any]] = []
    for row in pq.read_table(path).to_pylist():
        rubric_count = len(json.loads(row["rubric_json"]))
        reference_count = len(row.get("reference_files") or [])
        rows.append(
            {
                "id": str(row["task_id"]),
                "sector": str(row["sector"]),
                "occupation": str(row["occupation"]),
                "deliverable_type": deliverable_type(row.get("deliverable_files") or []),
                "reference_count": reference_count,
                "reference_bin": "none" if reference_count == 0 else "one_two" if reference_count <= 2 else "three_plus",
                "rubric_item_count": rubric_count,
                "rubric_bin": "low" if rubric_count < 30 else "medium" if rubric_count < 60 else "high",
            }
        )

    cases: list[dict[str, Any]] = []
    sectors = sorted({row["sector"] for row in rows})
    if len(sectors) != 9:
        raise ValueError(f"Expected 9 GDPval sectors, found {len(sectors)}")
    for sector in sectors:
        pool = [row for row in rows if row["sector"] == sector]
        cases.extend(
            balanced_pick(
                pool,
                3,
                "gdpval",
                sector,
                ("deliverable_type", "reference_bin", "rubric_bin", "occupation"),
            )
        )
    return base_manifest(
        "gdpval",
        {"dataset": "local GDPval snapshot", "parquet_sha256": file_sha256(path)},
        cases,
        {sector: 3 for sector in sectors},
        "Three cases per sector, greedily balancing public deliverable type, reference count, rubric size, and occupation metadata",
    )


def generate_trajectory(orch_root: Path, inventory_path: Path) -> dict[str, Any]:
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    if inventory.get("schema_version") != 1:
        raise ValueError(f"Unsupported TRAJECT inventory schema: {inventory_path}")
    cases = inventory.get("cases")
    if not isinstance(cases, list) or len(cases) != 100:
        raise ValueError("TRAJECT light inventory must contain exactly 100 cases")

    ids = [str(case.get("id") or "") for case in cases]
    audit_ids = [str(case.get("audit_id") or "") for case in cases]
    if not all(ids) or len(set(ids)) != len(ids) or not all(audit_ids) or len(set(audit_ids)) != len(audit_ids):
        raise ValueError("TRAJECT inventory case ids and audit ids must be non-empty and unique")

    domains = Counter(str(case.get("domain")) for case in cases)
    strata = Counter(str(case.get("sample_stratum")) for case in cases)
    if len(domains) != 10 or set(domains.values()) != {10}:
        raise ValueError(f"TRAJECT domain allocation is not 10 x 10: {domains}")
    if strata != Counter({"parallel_hard": 25, "parallel_simple": 25, "sequential_hard": 25, "sequential_simple": 25}):
        raise ValueError(f"TRAJECT topology/difficulty allocation mismatch: {strata}")
    if any(case.get("live_status") != "all_strict_success" for case in cases):
        raise ValueError("TRAJECT light inventory contains an endpoint that did not pass the strict live probe")

    data_root = (orch_root / "TRAJECT-Bench" / "public_data").resolve()
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    for case in cases:
        relative, raw_index = case["id"].rsplit(":", 1)
        source_path = (data_root / relative).resolve()
        if data_root not in source_path.parents or not source_path.is_file():
            raise ValueError(f"TRAJECT case path is outside the pinned public data: {case['id']}")
        records = source_cache.setdefault(source_path, json.loads(source_path.read_text(encoding="utf-8")))
        record = records[int(raw_index)]
        tools = record.get("tool list")
        if tools is None:
            tools = record.get("tool_list")
        if not record.get("query") or not isinstance(tools, list):
            raise ValueError(f"TRAJECT source record is incomplete: {case['id']}")
        if len(tools) != case["tool_calls"]:
            raise ValueError(f"TRAJECT tool count differs from audited inventory: {case['id']}")

    manifest = base_manifest(
        "trajectory-bench",
        {
            "revision": TRAJECT_REVISION,
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "source_workbook": inventory["source_workbook"],
            "inventory_summary": inventory["inventory_summary"],
            "source_case_records_verified": len(cases),
        },
        cases,
        {
            **{f"domain:{name}": count for name, count in sorted(domains.items())},
            **{f"stratum:{name}": count for name, count in sorted(strata.items())},
        },
        "Workbook-audited endpoint-eligible cases with fixed 10-per-domain and 25-per-topology/difficulty-cell allocation",
    )
    manifest["selection_policy"] = {
        "frozen": True,
        "method": manifest["selection_policy"]["method"],
        "model_outcomes_used": False,
        "baseline_runs_used": False,
        "endpoint_probe_used": True,
        "dataset_historical_tool_outputs_used": True,
    }
    manifest["eligibility_note"] = (
        "Every declared endpoint passed the live ToolBench probe. Exact historical argument combinations were not rerun, "
        "so endpoint eligibility is not a guarantee of immutable external responses."
    )
    return manifest


def vita_registry() -> dict[str, list[str]]:
    code = (
        "import json; from vita.registry import registry; "
        "print(json.dumps({name:[str(task.id) for task in registry.get_tasks_loader(name)()] "
        "for name in registry.get_task_sets()}))"
    )
    completed = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", "orch-bench/vitabench:742e240", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def generate_vitabench() -> dict[str, Any]:
    registry = vita_registry()
    expected = ("delivery", "ota", "instore", "cross_domain")
    if set(registry) != set(expected):
        raise ValueError(f"Unexpected VitaBench task sets: {sorted(registry)}")
    cases: list[dict[str, Any]] = []
    for domain in expected:
        rows = [{"id": case_id, "domain": domain} for case_id in registry[domain]]
        cases.extend(stable_pick(rows, 15, "vitabench", domain))
    return base_manifest(
        "vitabench",
        {"revision": VITA_REVISION, "registry_counts": {name: len(ids) for name, ids in registry.items()}},
        cases,
        {domain: 15 for domain in expected},
        "SHA-256 rank within each pinned official domain; independent of the historical result-informed VitaBench-60 set",
    )


def action_bin(count: int) -> str:
    if count == 0:
        return "zero"
    if count <= 3:
        return "one_three"
    if count <= 6:
        return "four_six"
    return "seven_plus"


def generate_tau2(orch_root: Path) -> dict[str, Any]:
    root = orch_root / "rcg" / ".external" / "tau2-bench" / "data" / "tau2" / "domains"
    cases: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for domain in ("airline", "retail", "telecom"):
        path = root / domain / "tasks.json"
        source_hashes[domain] = file_sha256(path)
        rows = []
        for task in json.loads(path.read_text(encoding="utf-8")):
            criteria = task.get("evaluation_criteria") or {}
            actions = len(criteria.get("actions") or [])
            communication = len(criteria.get("communicate_info") or []) + len(criteria.get("nl_assertions") or [])
            rows.append(
                {
                    "id": f"{domain}:{task['id']}",
                    "domain": domain,
                    "action_count": actions,
                    "action_bin": action_bin(actions),
                    "communication_count": communication,
                    "communication_bin": action_bin(communication),
                }
            )
        cases.extend(
            balanced_pick(rows, 10, "tau2", domain, ("action_bin", "communication_bin"))
        )
    return base_manifest(
        "tau2",
        {"revision": TAU_REVISION, "task_file_sha256": source_hashes},
        cases,
        {"airline": 10, "retail": 10, "telecom": 10},
        "Ten cases per main domain, balancing public action and communication criterion counts",
    )


def generate_bfcl(platform_root: Path) -> dict[str, Any]:
    data_root = platform_root / ".sources" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    paths = sorted(data_root.glob("BFCL_v4_*.json"))
    if len(paths) != 20:
        raise ValueError(f"Expected 20 BFCL V4 files, found {len(paths)}")
    format_path = data_root / "BFCL_v4_format_sensitivity.json"
    format_groups = json.loads(format_path.read_text(encoding="utf-8"))
    format_sensitive_ids = {case_id for case_ids in format_groups.values() for case_id in case_ids}
    task_paths = [path for path in paths if path != format_path]
    cases: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {format_path.name: file_sha256(format_path)}
    summary: dict[str, int] = {}
    for path in task_paths:
        category = path.stem.removeprefix("BFCL_v4_")
        source_hashes[path.name] = file_sha256(path)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                case_id = str(row["id"])
                rows.append(
                    {
                        "id": case_id,
                        "category": category,
                        "format_sensitive": case_id in format_sensitive_ids,
                    }
                )
        selected = balanced_pick(rows, 5, "bfcl", category, ("format_sensitive",))
        cases.extend(selected)
        summary[category] = len(selected)
    return base_manifest(
        "bfcl",
        {"revision": BFCL_REVISION, "category_file_sha256": source_hashes},
        cases,
        summary,
        "Five cases per BFCL V4 task category, balancing membership in the auxiliary format-sensitivity map; report category-macro and pooled scores",
    )


def generate_terminal(orch_root: Path) -> dict[str, Any]:
    root = orch_root / "rcg" / ".external" / "terminal-bench-2"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/task.toml")):
        task = tomllib.loads(path.read_text(encoding="utf-8"))
        metadata = task.get("metadata") or {}
        environment = task.get("environment") or {}
        rows.append(
            {
                "id": path.parent.name,
                "category": str(metadata.get("category") or "unspecified"),
                "difficulty": str(metadata.get("difficulty") or "unspecified"),
                "internet_allowed": bool(environment.get("allow_internet", False)),
                "docker_image": str(environment.get("docker_image") or ""),
            }
        )
    cases = balanced_pick(rows, 20, "terminal-bench-2", "official-tasks", ("category", "difficulty", "internet_allowed"))
    summary = dict(sorted(Counter(case["difficulty"] for case in cases).items()))
    manifest = base_manifest(
        "terminal-bench-2",
        {"revision": TERMINAL_REVISION, "official_task_count": len(rows)},
        cases,
        {f"difficulty:{key}": value for key, value in summary.items()},
        "Greedy coverage over public category, difficulty, and internet metadata with SHA-256 tie-breaking",
    )
    manifest["runner_note"] = (
        "The cases are frozen and source-valid. The current catalog adapter executes regex-log only; "
        "suite execution requires the generic per-task adapter expansion before a batch claim is valid."
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate frozen light-suite manifests from pinned public metadata.")
    platform_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--platform-root", type=Path, default=platform_root)
    parser.add_argument("--orch-root", type=Path, default=platform_root.parent)
    parser.add_argument("--output-dir", type=Path, default=platform_root / "catalog" / "suites" / "light")
    parser.add_argument(
        "--trajectory-inventory",
        type=Path,
        default=platform_root / "catalog" / "suites" / "sources" / "trajectory_online_reproducible_100.json",
    )
    args = parser.parse_args()

    manifests = [
        generate_gaia(args.orch_root.resolve()),
        generate_gdpval(args.orch_root.resolve()),
        generate_trajectory(args.orch_root.resolve(), args.trajectory_inventory.resolve()),
        generate_vitabench(),
        generate_tau2(args.orch_root.resolve()),
        generate_bfcl(args.platform_root.resolve()),
        generate_terminal(args.orch_root.resolve()),
    ]
    for manifest in manifests:
        write_manifest(args.output_dir.resolve(), manifest)
        print(f"{manifest['benchmark']}: {manifest['declared_count']}")


if __name__ == "__main__":
    main()
