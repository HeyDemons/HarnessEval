"""Build and validate the frozen Terminal-Bench verifier images.

The PERSEUS integration grades Terminal-Bench in a *separate* container rather than in
the live task container, so it needs one prebuilt image per task:

    harnesseval/terminal-bench-2-verifier:<task-id>

The image is the task's own source image plus ``/opt/harnesseval/verifier.sh``. Five
labels pin it to exactly one source image and one ``tests/test.sh``, so a stale cache
can never be mistaken for a fresh one; the consumer re-checks those labels and refuses
to grade against a mismatch.

The grading semantics live in ``terminal_cache_verifier.sh`` and mirror
``terminal_runtime.run_shared_verifier``. Nothing here interprets a task's tests or
computes a score: the reward is whatever ``test.sh`` writes to ``/logs/verifier``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPOSITORY = "harnesseval/terminal-bench-2-verifier"
LABEL_PREFIX = "org.harnesseval.terminal-cache"
SCHEMA = "2"
VERIFIER_SCRIPT = Path(__file__).with_name("terminal_cache_verifier.sh")

DockerCommand = Callable[..., list[str]]


def _default_docker(*args: str) -> list[str]:
    return ["docker", *args]


def verifier_image(task_id: str) -> str:
    return f"{REPOSITORY}:{task_id}"


def test_digest(task_dir: Path) -> str:
    return hashlib.sha256((task_dir / "tests" / "test.sh").read_bytes()).hexdigest()


def inspect(docker: DockerCommand, image: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        docker("image", "inspect", image), text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        return None
    values = json.loads(completed.stdout)
    return values[0] if values else None


def expected_labels(
    *, task_id: str, task_dir: Path, source_image: str, source_id: str
) -> dict[str, str]:
    return {
        f"{LABEL_PREFIX}.schema": SCHEMA,
        f"{LABEL_PREFIX}.task": task_id,
        f"{LABEL_PREFIX}.source-image": source_image,
        f"{LABEL_PREFIX}.source-id": source_id,
        f"{LABEL_PREFIX}.test-sha256": test_digest(task_dir),
    }


def _mismatches(labels: dict[str, str], expected: dict[str, str]) -> dict[str, Any]:
    return {
        name: {"expected": value, "observed": labels.get(name)}
        for name, value in expected.items()
        if value is None or labels.get(name) != value
    }


def build(
    docker: DockerCommand,
    *,
    task_id: str,
    task_dir: Path,
    source_image: str,
    source_id: str,
    platform: str | None = None,
    log: Any = None,
) -> str:
    """Layer the verifier script onto the task's own image and label the result."""
    image = verifier_image(task_id)
    labels = expected_labels(
        task_id=task_id, task_dir=task_dir, source_image=source_image, source_id=source_id
    )
    with tempfile.TemporaryDirectory(prefix="terminal-cache-") as context:
        root = Path(context)
        script = root / "verifier.sh"
        shutil.copyfile(VERIFIER_SCRIPT, script)
        # COPY preserves the mode, so no RUN layer is needed to make it executable.
        script.chmod(0o755)
        (root / "Dockerfile").write_text(
            f"FROM {source_image}\nCOPY verifier.sh /opt/harnesseval/verifier.sh\n",
            encoding="utf-8",
        )
        command = docker("build", "-t", image)
        if platform:
            command.extend(["--platform", platform])
        for name, value in labels.items():
            command.extend(["--label", f"{name}={value}"])
        command.append(str(root))
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if log is not None:
        log.write(completed.stdout or "")
        log.write(completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Unable to build the verifier cache for {task_id}: "
            f"{(completed.stderr or completed.stdout or '').strip()[-2000:]}"
        )
    return image


def require_cache(
    docker: DockerCommand = _default_docker,
    *,
    task_id: str,
    task_dir: Path,
    source_image: str,
    build_if_missing: bool = True,
    platform: str | None = None,
    log: Any = None,
) -> str:
    """Return a verifier image whose labels match this task and source image exactly."""
    source = inspect(docker, source_image)
    if source is None:
        raise RuntimeError(
            f"Terminal-Bench source image is missing for {task_id}: {source_image}"
        )
    source_id = source.get("Id")
    expected = expected_labels(
        task_id=task_id, task_dir=task_dir, source_image=source_image, source_id=source_id
    )
    cached = inspect(docker, verifier_image(task_id))
    labels = ((cached or {}).get("Config", {}).get("Labels", {}) or {})
    mismatches = _mismatches(labels, expected)
    if cached is not None and not mismatches:
        return verifier_image(task_id)
    if not build_if_missing:
        raise RuntimeError(
            f"Terminal verifier cache is missing or stale for {task_id}. "
            f"Details: {json.dumps(mismatches, sort_keys=True)}"
        )
    return build(
        docker,
        task_id=task_id,
        task_dir=task_dir,
        source_image=source_image,
        source_id=source_id,
        platform=platform,
        log=log,
    )


def _suite_cases(suite_path: Path) -> list[dict[str, Any]]:
    return json.loads(suite_path.read_text(encoding="utf-8"))["cases"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orch-root", default=os.environ.get("ORCH_ROOT", ""), help="benchmark data root"
    )
    parser.add_argument(
        "--suite",
        default=str(
            Path(__file__).resolve().parents[1]
            / "catalog/suites/light/terminal-bench-2.json"
        ),
    )
    parser.add_argument("--task", action="append", default=[], help="task id, repeatable")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if current")
    args = parser.parse_args(argv)
    if not args.orch_root:
        parser.error("--orch-root or ORCH_ROOT is required")

    task_root = Path(args.orch_root) / "rcg/.external/terminal-bench-2"
    cases = _suite_cases(Path(args.suite))
    if args.task:
        wanted = set(args.task)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            parser.error(f"tasks are not in the suite: {sorted(missing)}")

    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        task_id = str(case["id"])
        task_dir = task_root / task_id
        source_image = str(case["docker_image"])
        print(f"[{index}/{len(cases)}] {task_id}", flush=True)
        try:
            if args.rebuild:
                source = inspect(_default_docker, source_image)
                if source is None:
                    raise RuntimeError(f"source image is missing: {source_image}")
                image = build(
                    _default_docker,
                    task_id=task_id,
                    task_dir=task_dir,
                    source_image=source_image,
                    source_id=source.get("Id"),
                    platform=args.platform,
                )
            else:
                image = require_cache(
                    task_id=task_id,
                    task_dir=task_dir,
                    source_image=source_image,
                    platform=args.platform,
                )
            print(f"    ok {image}", flush=True)
        except Exception as error:  # noqa: BLE001 - report every task, fail at the end
            failures.append(task_id)
            print(f"    FAILED {type(error).__name__}: {error}", flush=True)
    if failures:
        print(f"\n{len(failures)}/{len(cases)} failed: {', '.join(failures)}")
        return 1
    print(f"\nall {len(cases)} verifier caches are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
