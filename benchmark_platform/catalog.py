from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .util import expand


@dataclass(frozen=True)
class Benchmark:
    id: str
    name: str
    raw: dict[str, Any]

    @property
    def adapter(self) -> dict[str, Any]:
        return self.raw["adapter"]

    @property
    def source(self) -> dict[str, Any]:
        return self.raw["source"]

    @property
    def smoke(self) -> dict[str, Any] | None:
        return self.raw.get("smoke")


class Catalog:
    def __init__(self, path: Path, platform_root: Path, orch_root: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError(f"Unsupported catalog schema in {path}")
        variables = {
            "PLATFORM_ROOT": str(platform_root.resolve()),
            "ORCH_ROOT": str(orch_root.resolve()),
            "CATALOG_DIR": str(path.resolve().parent),
            "HOME": str(Path.home()),
        }
        entries = expand(raw.get("benchmarks", []), variables)
        self._items: dict[str, Benchmark] = {}
        for entry in entries:
            for required in ("id", "name", "source", "adapter", "scoring"):
                if required not in entry:
                    raise ValueError(f"Catalog entry lacks {required}: {entry}")
            benchmark = Benchmark(entry["id"], entry["name"], entry)
            if benchmark.id in self._items:
                raise ValueError(f"Duplicate benchmark id: {benchmark.id}")
            self._items[benchmark.id] = benchmark

    def __iter__(self) -> Iterator[Benchmark]:
        return iter(self._items.values())

    def ids(self) -> list[str]:
        return list(self._items)

    def get(self, benchmark_id: str) -> Benchmark:
        try:
            return self._items[benchmark_id]
        except KeyError as exc:
            raise ValueError(f"Unknown benchmark: {benchmark_id}") from exc
