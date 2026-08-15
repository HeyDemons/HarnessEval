from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("--command", action="append", required=True)
    args = parser.parse_args()
    commands = [item.split(":::") for item in args.command]
    checks = []
    for command in commands:
        process = subprocess.run(command, text=True, capture_output=True, check=False)
        checks.append({"command": command, "returncode": process.returncode})
        print(process.stdout, end="")
        print(process.stderr, end="")
    payload = {
        "benchmark": args.benchmark,
        "checks": checks,
        "scores": {"package_cli_integrity": float(all(item["returncode"] == 0 for item in checks))},
        "oracle_smoke": True,
    }
    output = Path("/job/payload.json")
    pending = output.with_name(".payload.json.tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(output)
    if payload["scores"]["package_cli_integrity"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
