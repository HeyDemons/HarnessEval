#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Invocation:
    case_id: str
    domain: str
    tool_name: str
    parent_tool_name: str
    api_name: str
    arguments: dict[str, Any]
    stored_output_present: bool
    stored_output_error: bool

    def tool_key(self) -> str:
        return fingerprint(
            {
                "domain": self.domain,
                "parent_tool_name": self.parent_tool_name,
                "api_name": self.api_name,
            }
        )

    def invocation_key(self) -> str:
        return fingerprint(
            {
                "domain": self.domain,
                "parent_tool_name": self.parent_tool_name,
                "api_name": self.api_name,
                "arguments": self.arguments,
            }
        )


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def iter_task_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if "tools" in path.relative_to(root).parts or path.name.startswith("."):
            continue
        yield path


def iter_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from iter_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_objects(item)


def catalog_definitions(root: Path) -> dict[str, list[dict[str, Any]]]:
    catalog_path = root / "tools" / "all_tools.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    definitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in iter_objects(catalog):
        if item.get("tool name") and item.get("domain name") and item.get("parent tool name") and item.get("API name"):
            definitions[str(item["tool name"])].append(item)
    return definitions


def resolve_tool(tool: dict[str, Any], definitions: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any] | None, str | None]:
    if tool.get("domain name") and tool.get("parent tool name") and tool.get("API name"):
        return tool, None
    name = str(tool.get("tool name") or "")
    candidates = definitions.get(name, [])
    signatures = {
        (str(item["domain name"]), str(item["parent tool name"]), str(item["API name"]))
        for item in candidates
    }
    if len(signatures) != 1:
        return None, f"{name}: expected one executable mapping, found {sorted(signatures)!r}"
    domain, parent, api = next(iter(signatures))
    resolved = dict(tool)
    resolved.update({"domain name": domain, "parent tool name": parent, "API name": api})
    return resolved, None


def parameters(tool: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("required parameters", "optional parameters"):
        for item in tool.get(field) or []:
            if isinstance(item, dict) and item.get("name") and "value" in item:
                result[str(item["name"])] = item["value"]
    return result


def stored_output_is_error(tool: dict[str, Any]) -> bool:
    if str(tool.get("execution_status") or "").lower() in {"error", "failed", "failure"}:
        return True
    output = tool.get("executed_output")
    if not isinstance(output, str):
        return False
    normalized = output.lstrip().lower()
    return normalized.startswith("error") or normalized.startswith("{'error'") or normalized.startswith('{"error"')


def load_dataset(root: Path) -> tuple[list[dict[str, Any]], list[Invocation], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    invocations: list[Invocation] = []
    unreadable: list[str] = []
    mapping_errors: list[str] = []
    records = 0
    raw_calls = 0
    calls_with_output = 0
    calls_with_error = 0
    definitions = catalog_definitions(root)

    for path in iter_task_files(root):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        rows = value if isinstance(value, list) else [value]
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            query = row.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            records += 1
            case_id = f"{path.relative_to(root).as_posix()}:{index}"
            case_invocations: list[Invocation] = []
            case_mapping_errors = []
            tool_rows = row.get("tool list")
            if tool_rows is None:
                tool_rows = row.get("tool_list")
            for tool in tool_rows or []:
                if not isinstance(tool, dict):
                    continue
                raw_calls += 1
                output = tool.get("executed_output")
                calls_with_output += output is not None and str(output).strip() != ""
                calls_with_error += stored_output_is_error(tool)
                resolved, mapping_error = resolve_tool(tool, definitions)
                if resolved is None:
                    detail = f"{case_id}: {mapping_error}"
                    mapping_errors.append(detail)
                    case_mapping_errors.append(detail)
                    continue
                invocation = Invocation(
                    case_id=case_id,
                    domain=str(resolved["domain name"]),
                    tool_name=str(resolved.get("tool name") or ""),
                    parent_tool_name=str(resolved["parent tool name"]),
                    api_name=str(resolved["API name"]),
                    arguments=parameters(resolved),
                    stored_output_present=output is not None and str(output).strip() != "",
                    stored_output_error=stored_output_is_error(resolved),
                )
                case_invocations.append(invocation)
                invocations.append(invocation)
            cases.append(
                {
                    "case_id": case_id,
                    "invocations": case_invocations,
                    "mapping_errors": case_mapping_errors,
                }
            )

    metadata = {
        "data_root": str(root),
        "json_task_files": len(list(iter_task_files(root))),
        "task_records": records,
        "cases_with_tools": sum(bool(case["invocations"]) for case in cases),
        "tool_calls": raw_calls,
        "mapped_tool_calls": len(invocations),
        "unique_tool_names": len({item.tool_name for item in invocations}),
        "unique_tool_endpoints": len({item.tool_key() for item in invocations}),
        "unique_invocations": len({item.invocation_key() for item in invocations}),
        "calls_with_stored_output": calls_with_output,
        "calls_with_stored_error": calls_with_error,
        "mapping_error_calls": len(mapping_errors),
        "mapping_error_examples": mapping_errors[:20],
        "unreadable_files": unreadable,
        "domains": dict(sorted(Counter(item.domain for item in invocations).items())),
    }
    return cases, invocations, metadata


def representative_probes(invocations: list[Invocation], granularity: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Invocation]] = defaultdict(list)
    for invocation in invocations:
        key = invocation.tool_key() if granularity == "tool" else invocation.invocation_key()
        grouped[key].append(invocation)

    probes = []
    for key, rows in grouped.items():
        representative = rows[0]
        probes.append(
            {
                "fingerprint": key,
                "granularity": granularity,
                "domain": representative.domain,
                "tool_name": representative.tool_name,
                "parent_tool_name": representative.parent_tool_name,
                "api_name": representative.api_name,
                "arguments": representative.arguments,
                "representative_case": representative.case_id,
                "affected_cases": len({item.case_id for item in rows}),
                "observed_invocations": len(rows),
            }
        )
    return sorted(probes, key=lambda item: (item["domain"], item["tool_name"], item["fingerprint"]))


def select_probes(probes: list[dict[str, Any]], per_domain: int | None, limit: int | None) -> list[dict[str, Any]]:
    selected = probes
    if per_domain is not None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for probe in probes:
            grouped[probe["domain"]].append(probe)
        selected = [probe for domain in sorted(grouped) for probe in grouped[domain][:per_domain]]
    if limit is not None:
        selected = selected[:limit]
    return selected


def redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "<redacted>") if secret else value
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items() if key != "toolbench_key"}
    return value


def classify(status: int, parsed: Any) -> tuple[str, str | None]:
    if status != 200:
        return "http_error", f"HTTP {status}"
    if not isinstance(parsed, dict):
        return "invalid_response", "response is not a JSON object"
    error = parsed.get("error")
    if error not in (None, "", False, 0):
        return "backend_error", str(error)
    if "response" not in parsed:
        return "missing_response", "response field is absent"
    response = parsed.get("response")
    if response in (None, ""):
        return "empty_response", "response field is empty"
    if isinstance(response, str) and response.lstrip().lower().startswith("error"):
        return "tool_error", response
    return "success", None


def execute_probe(
    probe: dict[str, Any],
    service_url: str,
    toolbench_key: str,
    request_timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    payload = {
        "category": probe["domain"],
        "tool_name": probe["parent_tool_name"],
        "api_name": probe["api_name"],
        "tool_input": probe["arguments"],
        "strip": "none",
        "toolbench_key": toolbench_key,
    }
    started = time.perf_counter()
    attempts = []
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            service_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "toolbench_key": toolbench_key},
            method="POST",
        )
        attempt_started = time.perf_counter()
        status = 0
        body = ""
        parsed: Any = None
        transport_error: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                status = response.status
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            transport_error = f"{type(exc).__name__}: {exc}"

        if body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None

        if transport_error:
            outcome, detail = "transport_error", transport_error
        elif status != 200:
            outcome, detail = "http_error", f"HTTP {status}"
        elif body and parsed is None:
            outcome, detail = "invalid_json", "response body is not valid JSON"
        else:
            outcome, detail = classify(status, parsed)

        attempts.append(
            {
                "attempt": attempt + 1,
                "seconds": time.perf_counter() - attempt_started,
                "http_status": status or None,
                "outcome": outcome,
                "detail": redact(detail, toolbench_key),
                "response_json": redact(parsed, toolbench_key),
                "response_body": redact(body, toolbench_key),
            }
        )
        retryable = outcome == "transport_error" or (status >= 500 and status <= 599)
        if not retryable or attempt >= retries:
            break
        time.sleep(retry_delay * (2**attempt))

    successful = [attempt for attempt in attempts if attempt["outcome"] == "success"]
    responded = [attempt for attempt in attempts if attempt["http_status"] is not None]
    final = successful[-1] if successful else responded[-1] if responded else attempts[-1]

    return {
        **probe,
        "completed_at": time.time(),
        "seconds": time.perf_counter() - started,
        "attempt_count": len(attempts),
        "http_status": final["http_status"],
        "outcome": final["outcome"],
        "detail": final["detail"],
        "request": {
            "category": payload["category"],
            "tool_name": payload["tool_name"],
            "api_name": payload["api_name"],
            "tool_input": payload["tool_input"],
            "strip": payload["strip"],
        },
        "response_json": final["response_json"],
        "response_body": final["response_body"],
        "attempts": attempts,
    }


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("fingerprint"):
            rows.append(value)
    return rows


def case_coverage(cases: list[dict[str, Any]], outcomes: dict[str, str], granularity: str) -> dict[str, int]:
    counts = Counter()
    for case in cases:
        if case.get("mapping_errors"):
            counts["unmapped"] += 1
            continue
        invocations = case["invocations"]
        if not invocations:
            counts["no_tools"] += 1
            continue
        keys = {
            item.tool_key() if granularity == "tool" else item.invocation_key()
            for item in invocations
        }
        values = [outcomes.get(key) for key in keys]
        if all(value == "success" for value in values):
            counts["fully_accessible"] += 1
        elif any(value is not None and value != "success" for value in values):
            counts["blocked"] += 1
        else:
            counts["unknown"] += 1
    return dict(counts)


def build_summary(
    metadata: dict[str, Any],
    cases: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    results: list[dict[str, Any]],
    granularity: str,
) -> dict[str, Any]:
    latest = {row["fingerprint"]: row for row in results}
    outcomes = {key: row.get("outcome", "unknown") for key, row in latest.items()}
    response_lengths = [
        len(response)
        for row in latest.values()
        if isinstance((response := (row.get("response_json") or {}).get("response")), str)
    ]
    max_response_chars = max(response_lengths, default=0)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for row in latest.values():
        by_domain[str(row.get("domain") or "unknown")][str(row.get("outcome") or "unknown")] += 1
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "granularity": granularity,
        "dataset": metadata,
        "probe_inventory": len(probes),
        "completed_probes": len(latest),
        "outcomes": dict(sorted(Counter(outcomes.values()).items())),
        "response_observability": {
            "http_200": sum(row.get("http_status") == 200 for row in latest.values()),
            "nonempty_response": sum(
                (row.get("response_json") or {}).get("response") not in (None, "")
                for row in latest.values()
            ),
            "max_response_chars": max_response_chars,
            "responses_at_max_chars": response_lengths.count(max_response_chars) if response_lengths else 0,
            "note": (
                "Repeated responses at the observed maximum can indicate upstream clipping. "
                "The probe records complete gateway response bodies and sends strip=none."
            ),
        },
        "by_domain": {domain: dict(sorted(counts.items())) for domain, counts in sorted(by_domain.items())},
        "case_coverage": case_coverage(cases, outcomes, granularity),
        "coverage_note": (
            "Tool granularity estimates case coverage from one representative invocation per endpoint. "
            "Use --granularity invocation for parameter-specific coverage."
            if granularity == "tool"
            else "Invocation granularity measures exact parameterized calls present in the dataset."
        ),
    }


def main() -> None:
    default_root = Path(__file__).resolve().parents[2] / "TRAJECT-Bench" / "public_data"
    parser = argparse.ArgumentParser(description="Measure live ToolBench coverage for TRAJECT-Bench")
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=Path("runs/trajectory_backend_probe/probes.jsonl"))
    parser.add_argument("--granularity", choices=("tool", "invocation"), default="tool")
    parser.add_argument("--per-domain", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--service-url", default=os.getenv("API_URL", ""))
    parser.add_argument("--toolbench-key", default=os.getenv("TOOLBENCH_KEY", ""))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--rerun-non-success",
        action="store_true",
        help="append fresh probes for prior non-success results while retaining successful results",
    )
    args = parser.parse_args()

    if args.per_domain is not None and args.per_domain < 1:
        parser.error("--per-domain must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retries < 0:
        parser.error("--retries must be zero or greater")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be zero or greater")

    root = args.data_root.resolve()
    output = args.output.resolve()
    summary_path = output.with_name("summary.json")
    cases, invocations, metadata = load_dataset(root)
    probes = representative_probes(invocations, args.granularity)
    previous = load_results(output)
    summary = build_summary(metadata, cases, probes, previous, args.granularity)
    atomic_json(summary_path, summary)
    if args.metadata_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if not args.service_url or not args.toolbench_key:
        parser.error("API_URL/--service-url and TOOLBENCH_KEY/--toolbench-key are required for live probes")

    latest_previous = {row["fingerprint"]: row for row in previous}
    completed = {
        fingerprint
        for fingerprint, row in latest_previous.items()
        if not args.rerun_non_success or row.get("outcome") == "success"
    }
    selected = [
        probe
        for probe in select_probes(probes, args.per_domain, args.limit)
        if probe["fingerprint"] not in completed
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    execute_probe,
                    probe,
                    args.service_url,
                    args.toolbench_key,
                    args.request_timeout,
                    args.retries,
                    args.retry_delay,
                ): probe
                for probe in selected
            }
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                previous.append(row)
                summary = build_summary(metadata, cases, probes, previous, args.granularity)
                atomic_json(summary_path, summary)
                print(
                    json.dumps(
                        {
                            "domain": row["domain"],
                            "tool": row["tool_name"],
                            "outcome": row["outcome"],
                            "seconds": row["seconds"],
                            "completed": summary["completed_probes"],
                            "inventory": summary["probe_inventory"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    print(json.dumps(build_summary(metadata, cases, probes, previous, args.granularity), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
