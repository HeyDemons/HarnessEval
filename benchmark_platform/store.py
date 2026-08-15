from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import fcntl

from .util import append_jsonl, atomic_json, slug, utc_now


TERMINAL_STATUSES = {"completed", "failed", "blocked", "cancelled"}


@dataclass
class CaseStore:
    run_dir: Path
    benchmark_id: str
    case_id: str

    @property
    def case_dir(self) -> Path:
        return self.run_dir / slug(self.benchmark_id) / slug(self.case_id)

    @property
    def result_path(self) -> Path:
        return self.case_dir / "result.json"

    def existing(self) -> dict[str, Any] | None:
        if not self.result_path.is_file():
            return None
        return json.loads(self.result_path.read_text(encoding="utf-8"))

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.case_dir / ".case.lock"
        with lock_path.open("a+", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                stream.seek(0)
                owner = stream.read().strip() or "unknown owner"
                raise RuntimeError(f"Case is already running: {self.benchmark_id}/{self.case_id} ({owner})") from exc
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()} acquired_at={utc_now()}\n")
            stream.flush()
            os.fsync(stream.fileno())
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def next_attempt(self) -> tuple[int, Path]:
        attempts = self.case_dir / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        numbers = [int(path.name) for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()]
        number = max(numbers, default=0) + 1
        attempt = attempts / f"{number:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        return number, attempt

    def event(self, attempt: Path, event: str, **details: Any) -> None:
        append_jsonl(
            attempt / "events.jsonl",
            {"at": utc_now(), "event": event, **details},
        )

    def start(self, attempt: Path, request: dict[str, Any]) -> None:
        atomic_json(attempt / "request.json", request)
        atomic_json(
            self.case_dir / "running.json",
            {
                "schema_version": 1,
                "benchmark": self.benchmark_id,
                "case_id": self.case_id,
                "attempt": attempt.name,
                "started_at": request["started_at"],
            },
        )
        self.event(attempt, "started")

    def finish(self, attempt: Path, result: dict[str, Any]) -> None:
        atomic_json(attempt / "result.json", result)
        atomic_json(self.result_path, result)
        (self.case_dir / "running.json").unlink(missing_ok=True)
        self.event(attempt, "finished", status=result["status"])
