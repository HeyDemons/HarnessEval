from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if clean == value and clean:
        return clean
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{clean or 'case'}-{digest}"


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        for name, replacement in variables.items():
            value = value.replace(f"${{{name}}}", replacement)
        return os.path.expanduser(value)
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def stream_process(
    command: list[str],
    terminal: TextIO,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    prefix: str = "",
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            rendered = f"{prefix}{line}" if prefix else line
            sys.stdout.write(rendered)
            sys.stdout.flush()
            terminal.write(line)
            terminal.flush()
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise


def command_exists(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def select(values: Iterable[str], available: Iterable[str]) -> list[str]:
    known = list(available)
    requested = list(values)
    if not requested or requested == ["all"]:
        return known
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    return requested
