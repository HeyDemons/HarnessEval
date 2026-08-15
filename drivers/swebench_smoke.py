from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from datasets import load_dataset
from swebench.harness import reporting, run_evaluation
from swebench.harness.test_spec.test_spec import make_test_spec as official_make_test_spec


INSTANCE_ID = "django__django-11790"
RUN_ID = "platform-smoke"
DATASET_ID = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
TASK_IMAGE = "ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11790:latest"
TASK_IMAGE_DIGEST = (
    "ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11790@"
    "sha256:c627df716cfe70d7ffe2d41939ced83ad7858a5968820ff6b5a81dfe59ffa422"
)
TASK_IMAGE_ALIAS = "platform-arm64/sweb.eval.arm64.django_1776_django-11790:latest"


def make_arm64_test_spec(instance, *args, **kwargs):
    kwargs["arch"] = "arm64"
    return official_make_test_spec(instance, *args, **kwargs)


def main() -> None:
    dataset = load_dataset(DATASET_ID, split="test", revision=DATASET_REVISION)
    matches = [dict(row) for row in dataset if row["instance_id"] == INSTANCE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one pinned dataset row for {INSTANCE_ID}, found {len(matches)}")
    case_dataset = Path("/job/pinned-case.json")
    case_dataset.write_text(json.dumps(matches, ensure_ascii=False) + "\n", encoding="utf-8")

    machine = platform.machine().lower()
    arm64 = machine in {"arm64", "aarch64"}
    image_digests: list[str] = []
    labels: dict[str, str] = {}
    image_valid = True
    namespace = "swebench"
    architecture_adapter = "official make_test_spec default arch=x86_64"
    task_image_name = "official swebench namespace image"
    if arm64:
        inspect = subprocess.run(
            ["docker", "image", "inspect", TASK_IMAGE],
            text=True,
            capture_output=True,
            check=False,
        )
        if inspect.returncode != 0:
            subprocess.run(["docker", "pull", TASK_IMAGE_DIGEST], check=True)
            subprocess.run(["docker", "image", "tag", TASK_IMAGE_DIGEST, TASK_IMAGE], check=True)
            inspect = subprocess.run(
                ["docker", "image", "inspect", TASK_IMAGE],
                text=True,
                capture_output=True,
                check=True,
            )
        image_metadata = json.loads(inspect.stdout)[0]
        image_digests = image_metadata.get("RepoDigests", [])
        labels = image_metadata.get("Config", {}).get("Labels", {}) or {}
        image_valid = TASK_IMAGE_DIGEST in image_digests and labels.get("swe-bench.instance_id") == INSTANCE_ID
        if image_valid:
            subprocess.run(["docker", "image", "tag", TASK_IMAGE, TASK_IMAGE_ALIAS], check=True)
        namespace = "platform-arm64"
        architecture_adapter = "official make_test_spec(arch='arm64')"
        task_image_name = TASK_IMAGE

    report_dir = Path("/job/official-report")
    if arm64:
        run_evaluation.make_test_spec = make_arm64_test_spec
        reporting.make_test_spec = make_arm64_test_spec
    run_evaluation.main(
        dataset_name=str(case_dataset),
        split="test",
        instance_ids=[INSTANCE_ID],
        predictions_path="gold",
        max_workers=1,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=RUN_ID,
        timeout=1800,
        namespace=namespace,
        rewrite_reports=False,
        modal=False,
        report_dir=str(report_dir),
    )
    report_name = f"gold.{RUN_ID}.json"
    report_candidates = [report_dir / report_name, Path("/job") / report_name]
    report_path = next((path for path in report_candidates if path.is_file()), report_candidates[0])
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    resolved = report.get("resolved_ids") == [INSTANCE_ID]
    payload = {
        "official_report": report,
        "official_report_path": str(report_path),
        "harness_returncode": 0,
        "architecture_adapter": architecture_adapter,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "task_image": {
            "name": task_image_name,
            "local_alias": TASK_IMAGE_ALIAS if arm64 else None,
            "expected_digest": TASK_IMAGE_DIGEST if arm64 else None,
            "observed_digests": image_digests,
            "labels": labels,
            "valid": image_valid,
            "provenance": "epoch-research ARM64 image" if arm64 else "official SWE-bench namespace",
        },
        "scores": {"resolved": float(resolved)},
        "oracle_smoke": True,
    }
    output = Path("/job/payload.json")
    pending = output.with_name(".payload.json.tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(output)
    if not resolved or not image_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
