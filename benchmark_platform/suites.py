from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from .util import expand


SUITE_MODES = ("light", "full")


class SuiteCatalog:
    def __init__(self, path: Path, platform_root: Path, benchmark_ids: list[str]):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError(f"Unsupported suite catalog schema in {path}")

        variables = {
            "PLATFORM_ROOT": str(platform_root.resolve()),
            "SUITE_DIR": str(path.resolve().parent),
        }
        modes = expand(raw.get("modes", {}), variables)
        if set(modes) != set(SUITE_MODES):
            raise ValueError(f"Suite catalog must define exactly {SUITE_MODES}")

        expected = set(benchmark_ids)
        for mode in SUITE_MODES:
            actual = set(modes[mode])
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(f"Suite mode {mode} mismatch: missing={missing}, extra={extra}")
        self._modes: dict[str, dict[str, dict[str, Any]]] = modes

    def modes(self) -> tuple[str, ...]:
        return SUITE_MODES

    def ids(self, mode: str) -> list[str]:
        self._require_mode(mode)
        return list(self._modes[mode])

    def __iter__(self) -> Iterator[str]:
        return iter(self._modes["light"])

    def get(self, benchmark_id: str, mode: str) -> dict[str, Any]:
        self._require_mode(mode)
        try:
            descriptor = deepcopy(self._modes[mode][benchmark_id])
        except KeyError as exc:
            raise ValueError(f"Unknown benchmark suite: {benchmark_id}") from exc

        descriptor.update({"schema_version": 1, "benchmark": benchmark_id, "mode": mode})
        manifest_path = descriptor.pop("manifest", None)
        if manifest_path is None:
            descriptor.setdefault("declared_count", None)
            descriptor.setdefault("cases", [])
            return descriptor

        path = Path(manifest_path)
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
        self._validate_manifest(path, manifest, benchmark_id, mode)
        manifest["manifest_path"] = str(path)
        manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        return manifest

    @staticmethod
    def _require_mode(mode: str) -> None:
        if mode not in SUITE_MODES:
            raise ValueError(f"Unknown suite mode: {mode}")

    @staticmethod
    def _validate_manifest(path: Path, manifest: dict[str, Any], benchmark_id: str, mode: str) -> None:
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported suite manifest schema in {path}")
        if manifest.get("benchmark") != benchmark_id or manifest.get("mode") != mode:
            raise ValueError(f"Suite manifest identity mismatch in {path}")
        if manifest.get("status") != "ready":
            raise ValueError(f"Materialized suite manifest must be ready: {path}")
        cases = manifest.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"Suite manifest cases must be a list: {path}")
        case_ids = [case.get("id") for case in cases if isinstance(case, dict)]
        if len(case_ids) != len(cases) or not all(isinstance(case_id, str) and case_id for case_id in case_ids):
            raise ValueError(f"Every suite case must have a non-empty string id: {path}")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"Duplicate case ids in suite manifest: {path}")
        if manifest.get("declared_count") != len(cases):
            raise ValueError(f"Suite declared_count does not match cases in {path}")
