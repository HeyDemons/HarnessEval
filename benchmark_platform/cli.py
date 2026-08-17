from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .engine import Platform
from .compatibility import compatibility_rows
from .harnesses import PROFILES
from .suites import SUITE_MODES, SuiteCatalog
from .util import atomic_json, select


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORCH_ROOT = ROOT.parent
DEFAULT_CATALOG = ROOT / "catalog" / "benchmarks.json"
DEFAULT_SUITES = ROOT / "catalog" / "suites.json"


def _platform(args: argparse.Namespace) -> Platform:
    return Platform(ROOT, args.orch_root, args.catalog)


def _ids(args: argparse.Namespace, platform: Platform) -> list[str]:
    return select(args.benchmarks, platform.catalog.ids())


def _default_run_dir(kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "runs" / f"{kind}_{stamp}"


def _mounts(values: list[str], parser: argparse.ArgumentParser) -> list[dict[str, str]]:
    mounts = []
    for value in values:
        parts = value.rsplit(":", 2)
        if len(parts) == 2:
            host, container = parts
            mode = "ro"
        elif len(parts) == 3 and parts[2] in {"ro", "rw"}:
            host, container, mode = parts
        else:
            parser.error(f"Invalid mount: {value}")
        mounts.append({"host": host, "container": container, "mode": mode})
    return mounts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docker-native benchmark control plane (no Inspect dependency).")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    parser.add_argument("--orch-root", type=Path, default=DEFAULT_ORCH_ROOT)
    sub = parser.add_subparsers(dest="action", required=True)

    list_parser = sub.add_parser("list", help="List registered benchmarks and fidelity status.")
    list_parser.add_argument("--json", action="store_true")

    harnesses = sub.add_parser("harnesses", help="List built-in theory harness profiles.")
    harnesses.add_argument("--json", action="store_true")

    matrix = sub.add_parser("matrix", help="Report baseline x benchmark bridge and tool-contract status.")
    matrix.add_argument("--json", action="store_true")

    suite = sub.add_parser("suite", help="Resolve frozen light subsets or benchmark-owned full suites.")
    suite.add_argument("benchmarks", nargs="*", default=["all"])
    suite.add_argument("--mode", choices=SUITE_MODES, default="light")
    suite.add_argument("--json", action="store_true")
    suite.add_argument("--ids-only", action="store_true")

    doctor = sub.add_parser("doctor", help="Probe local Docker, data, images, and provider blockers.")
    doctor.add_argument("benchmarks", nargs="*", default=["all"])
    doctor.add_argument("--json", action="store_true")

    build = sub.add_parser("build", help="Build self-packaged benchmark images.")
    build.add_argument("benchmarks", nargs="*", default=["all"])
    build.add_argument("--pull", action="store_true")

    smoke = sub.add_parser("smoke", help="Run benchmark-faithful infrastructure/oracle smokes.")
    smoke.add_argument("benchmarks", nargs="*", default=["all"])
    smoke.add_argument("--run-dir", type=Path)
    smoke.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    smoke.add_argument("--retry-failed", action="store_true")
    smoke.add_argument("--no-build", action="store_true")

    run = sub.add_parser("run", help="Run one isolated case; command after -- executes inside its benchmark image.")
    run.add_argument("benchmark")
    run.add_argument("--case", required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--no-build", action="store_true")
    run.add_argument("--pass-env", action="append", default=[])
    run.add_argument(
        "--mount",
        action="append",
        default=[],
        metavar="HOST:CONTAINER[:ro|rw]",
        help="Add an explicit harness or artifact mount without changing the benchmark catalog.",
    )

    harness_run = sub.add_parser(
        "harness-run",
        help="Run one built-in harness against a request contract inside Docker.",
    )
    harness_run.add_argument("profile", choices=[profile.id for profile in PROFILES])
    harness_run.add_argument("--request", type=Path, required=True)
    harness_run.add_argument("--case", required=True)
    harness_run.add_argument("--run-dir", type=Path, required=True)
    harness_run.add_argument("--image", default="python:3.12-slim")
    harness_run.add_argument("--network", default="bridge")
    harness_run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    harness_run.add_argument("--retry-failed", action="store_true")
    harness_run.add_argument("--no-pull", action="store_true")
    harness_run.add_argument("--pass-env", action="append", default=[])
    harness_run.add_argument("--mount", action="append", default=[], metavar="HOST:CONTAINER[:ro|rw]")

    bridge_run = sub.add_parser(
        "bridge-run",
        help="Run a built-in baseline through a benchmark-owned isolated tool bridge.",
    )
    bridge_run.add_argument("profile", choices=[profile.id for profile in PROFILES])
    bridge_run.add_argument("benchmark")
    bridge_run.add_argument("--case", required=True)
    bridge_run.add_argument("--run-dir", type=Path, required=True)
    bridge_run.add_argument("--network", default="bridge")
    bridge_run.add_argument("--policy", default="{}", help="JSON object with explicit baseline policy parameters")
    bridge_run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    bridge_run.add_argument("--retry-failed", action="store_true")
    bridge_run.add_argument("--no-build", action="store_true")
    bridge_run.add_argument("--pass-env", action="append", default=[])
    return parser


def _main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    inner_command: list[str] = []
    if "--" in argv:
        separator = argv.index("--")
        inner_command = argv[separator + 1 :]
        argv = argv[:separator]
    args = parser.parse_args(argv)
    if inner_command and args.action != "run":
        parser.error("A command after -- is only valid for the run action")
    platform = _platform(args)
    if args.action == "list":
        rows = [benchmark.raw for benchmark in platform.catalog]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for benchmark in platform.catalog:
                scoring = benchmark.raw["scoring"]
                print(f"{benchmark.id:24} {benchmark.adapter['kind']:16} {scoring['comparability']:12} {benchmark.name}")
        return
    if args.action == "harnesses":
        rows = [profile.__dict__ for profile in PROFILES]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for profile in PROFILES:
                print(f"{profile.id:16} {profile.provenance:22} {profile.name}")
        return
    if args.action == "matrix":
        rows = compatibility_rows(PROFILES, platform.catalog)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                print(
                    f"{row['baseline']:16} {row['benchmark']:22} "
                    f"{row['bridge_status']:45} {row['tool_contract']}"
                )
        return
    if args.action == "suite":
        suites = SuiteCatalog(args.suites, ROOT, platform.catalog.ids())
        selected = select(args.benchmarks, suites.ids(args.mode))
        rows = [suites.get(benchmark_id, args.mode) for benchmark_id in selected]
        if args.ids_only:
            if len(rows) != 1:
                parser.error("--ids-only requires exactly one benchmark")
            if rows[0].get("status") != "ready":
                parser.error(f"suite is not materialized: {rows[0]['benchmark']} {args.mode}")
            if rows[0].get("declared_count") is None:
                parser.error("--ids-only is unavailable when enumeration is owned by the benchmark runner")
            for case in rows[0]["cases"]:
                print(case["id"])
        elif args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                count = row.get("declared_count")
                if count is not None:
                    count_text = str(count)
                elif row["mode"] == "full":
                    count_text = "official-full"
                else:
                    count_text = "unmaterialized"
                local = row.get("locally_scoreable_count")
                score_text = f", {local} locally scoreable" if local is not None and local != count else ""
                print(f"{row['status'].upper():7} {row['benchmark']:22} {row['mode']:5} {count_text}{score_text}")
                if row.get("reason"):
                    print(f"  {row['reason']}")
        return
    if args.action == "doctor":
        report = [platform.doctor(platform.catalog.get(item)) for item in _ids(args, platform)]
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            for item in report:
                state = "READY" if item["ready"] else "BUILD" if item["buildable"] else "BLOCKED"
                print(f"{state:7} {item['benchmark']}: {item['name']}")
                for check in item["checks"]:
                    marker = "ok" if check["ok"] else "missing"
                    print(f"  {marker:7} {check['name']}" + (f" - {check.get('reason')}" if check.get("reason") else ""))
        return
    if args.action == "build":
        failed = False
        for item in _ids(args, platform):
            result = platform.build(platform.catalog.get(item), pull=args.pull)
            print(json.dumps(result, ensure_ascii=False))
            failed = failed or result["status"] not in {"completed", "not_buildable"}
        raise SystemExit(1 if failed else 0)
    if args.action == "smoke":
        run_dir = (args.run_dir or _default_run_dir("smoke")).resolve()
        selected = _ids(args, platform)
        atomic_json(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "kind": "smoke",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "benchmarks": selected,
                "oracle_results_are_not_model_scores": True,
            },
        )
        results = []
        for item in selected:
            benchmark = platform.catalog.get(item)
            if not benchmark.smoke and benchmark.adapter["kind"] not in {"terminal-task", "external-vm"}:
                continue
            result = platform.run(
                benchmark,
                case_id=benchmark.smoke.get("case_id", "infrastructure") if benchmark.smoke else "oracle",
                run_dir=run_dir,
                smoke=True,
                resume=args.resume,
                retry_failed=args.retry_failed,
                build_missing=not args.no_build,
            )
            results.append(result)
            atomic_json(run_dir / "summary.json", {"schema_version": 1, "results": results})
        print(f"Run artifacts: {run_dir}")
        raise SystemExit(1 if any(result["status"] != "completed" for result in results) else 0)
    if args.action == "run":
        mounts = _mounts(args.mount, parser)
        result = platform.run(
            platform.catalog.get(args.benchmark),
            case_id=args.case,
            run_dir=args.run_dir,
            command_override=inner_command or None,
            resume=args.resume,
            retry_failed=args.retry_failed,
            build_missing=not args.no_build,
            pass_env=args.pass_env,
            extra_mounts=mounts,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "completed" else 1)
    if args.action == "harness-run":
        profile = next(item for item in PROFILES if item.id == args.profile)
        result = platform.run_harness(
            profile=profile.__dict__,
            request_path=args.request,
            case_id=args.case,
            run_dir=args.run_dir,
            image=args.image,
            network=args.network,
            resume=args.resume,
            retry_failed=args.retry_failed,
            pull_missing=not args.no_pull,
            pass_env=args.pass_env,
            extra_mounts=_mounts(args.mount, parser),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "completed" else 1)
    if args.action == "bridge-run":
        profile = next(item for item in PROFILES if item.id == args.profile)
        try:
            policy = json.loads(args.policy)
        except json.JSONDecodeError as exc:
            parser.error(f"--policy must be one complete JSON object: {exc}")
        if not isinstance(policy, dict):
            parser.error("--policy must decode to an object")
        result = platform.run_bridge_harness(
            benchmark=platform.catalog.get(args.benchmark),
            profile=profile.__dict__,
            case_id=args.case,
            run_dir=args.run_dir,
            network=args.network,
            policy=policy,
            resume=args.resume,
            retry_failed=args.retry_failed,
            build_missing=not args.no_build,
            pass_env=args.pass_env,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "completed" else 1)


def main() -> None:
    try:
        _main()
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
