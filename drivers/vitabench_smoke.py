from __future__ import annotations

import json
import subprocess
from pathlib import Path

import vita


def main() -> None:
    root = Path("/opt/vitabench")
    data = root / "data" / "vita" / "domains"
    task_files = sorted(data.glob("*/tasks_en.json"))
    counts = {}
    for path in task_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        counts[path.parent.name] = len(value)
    help_run = subprocess.run(["vita", "--help"], text=True, capture_output=True, check=False)
    payload = {
        "source_root": str(root),
        "task_counts": counts,
        "cli_returncode": help_run.returncode,
        "scores": {"package_and_dataset_integrity": float(help_run.returncode == 0 and sum(counts.values()) > 0)},
        "oracle_smoke": True,
    }
    output = Path("/job/payload.json")
    pending = output.with_name(".payload.json.tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["scores"]["package_and_dataset_integrity"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
