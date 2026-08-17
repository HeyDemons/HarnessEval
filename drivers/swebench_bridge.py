from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from datasets import load_dataset
from swebench.harness import reporting, run_evaluation
from swebench.harness.test_spec.test_spec import make_test_spec as official_make_test_spec


DATASET_ID = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
ARM64_CASE = "django__django-11790"
ARM64_IMAGE = "ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11790:latest"
ARM64_DIGEST = (
    "ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11790@"
    "sha256:c627df716cfe70d7ffe2d41939ced83ad7858a5968820ff6b5a81dfe59ffa422"
)
ARM64_ALIAS = "platform-arm64/sweb.eval.arm64.django_1776_django-11790:latest"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _case(instance_id: str) -> dict[str, Any]:
    dataset = load_dataset(DATASET_ID, split="test", revision=DATASET_REVISION)
    matches = [dict(row) for row in dataset if row["instance_id"] == instance_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one pinned SWE-bench case for {instance_id}, found {len(matches)}")
    return matches[0]


def _inspect(image: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        text=True,
        capture_output=True,
        check=False,
    )
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def _task_image(row: dict[str, Any]) -> dict[str, Any]:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        if row["instance_id"] != ARM64_CASE:
            raise RuntimeError(
                "This macOS release has a digest-pinned ARM64 task image only for "
                f"{ARM64_CASE}; no architecture-faithful image is configured for {row['instance_id']}"
            )
        metadata = _inspect(ARM64_IMAGE)
        if metadata is None:
            subprocess.run(["docker", "pull", ARM64_DIGEST], check=True)
            subprocess.run(["docker", "image", "tag", ARM64_DIGEST, ARM64_IMAGE], check=True)
            metadata = _inspect(ARM64_IMAGE)
        if metadata is None:
            raise RuntimeError("Unable to inspect the configured ARM64 SWE-bench image")
        digests = metadata.get("RepoDigests", [])
        labels = metadata.get("Config", {}).get("Labels", {}) or {}
        if ARM64_DIGEST not in digests or labels.get("swe-bench.instance_id") != row["instance_id"]:
            raise RuntimeError("ARM64 SWE-bench image provenance check failed")
        subprocess.run(["docker", "image", "tag", ARM64_IMAGE, ARM64_ALIAS], check=True)
        return {
            "name": ARM64_ALIAS,
            "platform": "linux/arm64/v8",
            "namespace": "platform-arm64",
            "architecture_adapter": "official make_test_spec(arch='arm64')",
            "expected_digest": ARM64_DIGEST,
            "observed_digests": digests,
        }
    spec = official_make_test_spec(row, namespace="swebench")
    if _inspect(spec.instance_image_key) is None:
        subprocess.run(["docker", "pull", spec.instance_image_key], check=True)
    metadata = _inspect(spec.instance_image_key) or {}
    return {
        "name": spec.instance_image_key,
        "platform": spec.platform,
        "namespace": "swebench",
        "architecture_adapter": "official make_test_spec default arch=x86_64",
        "expected_digest": None,
        "observed_digests": metadata.get("RepoDigests", []),
    }


def prepare(instance_id: str, output: Path) -> None:
    row = _case(instance_id)
    image = _task_image(row)
    _write(
        output,
        {
            "schema_version": 1,
            "instance_id": instance_id,
            "prompt": row["problem_statement"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "workspace_root": "/testbed",
            "task_image": image,
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "hidden_fields_exposed_to_agent": [],
        },
    )


def evaluate(instance_id: str, patch_path: Path, output: Path) -> None:
    row = _case(instance_id)
    image = _task_image(row)
    model_patch = patch_path.read_text(encoding="utf-8")
    run_id = f"harnesseval-{uuid.uuid4().hex[:12]}"
    predictions = [
        {
            "instance_id": instance_id,
            "model_name_or_path": "harnesseval-profile",
            "model_patch": model_patch,
        }
    ]
    with tempfile.TemporaryDirectory(prefix="harnesseval-swe-") as directory:
        root = Path(directory)
        dataset_path = root / "pinned-case.json"
        predictions_path = root / "predictions.json"
        report_dir = root / "report"
        dataset_path.write_text(json.dumps([row], ensure_ascii=False) + "\n", encoding="utf-8")
        predictions_path.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
        if image["namespace"] == "platform-arm64":
            def make_arm64_test_spec(instance, *args, **kwargs):
                kwargs["arch"] = "arm64"
                return official_make_test_spec(instance, *args, **kwargs)

            run_evaluation.make_test_spec = make_arm64_test_spec
            reporting.make_test_spec = make_arm64_test_spec
        report_reference = run_evaluation.main(
            dataset_name=str(dataset_path),
            split="test",
            instance_ids=[instance_id],
            predictions_path=str(predictions_path),
            max_workers=1,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=None,
            namespace=image["namespace"],
            rewrite_reports=False,
            modal=False,
            report_dir=str(report_dir),
        )
        if isinstance(report_reference, dict):
            report = report_reference
        elif isinstance(report_reference, (str, Path)):
            report_path = Path(report_reference)
            if not report_path.is_absolute():
                report_path = Path.cwd() / report_path
            report = json.loads(report_path.read_text(encoding="utf-8"))
        elif report_reference is None:
            report = {}
        else:
            raise TypeError(
                "Unsupported SWE-bench report reference: "
                f"{type(report_reference).__name__}"
            )
    _write(
        output,
        {
            "schema_version": 1,
            "benchmark": "swe-bench-verified",
            "instance_id": instance_id,
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "task_image": image,
            "patch_bytes": len(model_patch.encode("utf-8")),
            "official_report": report,
            "scores": {"resolved": float(instance_id in report.get("resolved_ids", []))},
            "native_score_status": "completed",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--case", required=True)
    prepare_parser.add_argument("--output", type=Path, default=Path("/job/public_case.json"))
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--case", required=True)
    evaluate_parser.add_argument("--patch", type=Path, default=Path("/job/model.patch"))
    evaluate_parser.add_argument("--output", type=Path, default=Path("/job/payload.json"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.case, args.output)
    else:
        evaluate(args.case, args.patch, args.output)


if __name__ == "__main__":
    main()
