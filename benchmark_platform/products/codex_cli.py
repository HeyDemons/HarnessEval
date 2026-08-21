from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from benchmark_platform.catalog import Benchmark
from benchmark_platform.store import CaseStore, TERMINAL_STATUSES
from benchmark_platform.util import atomic_json, utc_now

from .pi_cli import (
    HARNESS_PROVIDER_ENV,
    HOST_ENV,
    NATIVE_EPISODE_BENCHMARKS,
    SUPPORTED_BENCHMARKS,
    TASK_BENCHMARKS,
    ProductServerHandle,
    _close_server,
    _concat,
    _finalize_task,
    _jsonl,
    _request_json,
    _score_result,
    _scorer_answer,
    _start_task_tool_server,
    _start_tool_server,
    _tool_results,
)


DEFAULT_CODEX_CANDIDATES = (
    Path("/Applications/ChatGPT.app/Contents/Resources/codex"),
    Path.home() / ".codex" / "plugins" / ".plugin-appserver" / "codex",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _codex_executable(value: str | None) -> tuple[Path, str]:
    candidates: list[Path] = []
    if value and value != "codex":
        candidate = Path(value).expanduser()
        located = shutil.which(value) if candidate.name == value else str(candidate)
        if located:
            candidates.append(Path(located))
    else:
        candidates.extend(DEFAULT_CODEX_CANDIDATES)
        located = shutil.which("codex")
        if located:
            candidates.append(Path(located))
    failures: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            failures.append(f"{resolved}: not executable")
            continue
        completed = subprocess.run(
            [str(resolved), "--version"], text=True, capture_output=True, check=False
        )
        if completed.returncode == 0:
            version = completed.stdout.strip() or completed.stderr.strip()
            return resolved, version
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        failures.append(f"{resolved}: {detail[-1] if detail else 'version probe failed'}")
    detail = "; ".join(failures) if failures else "no candidate found"
    raise FileNotFoundError(f"Codex CLI is unavailable: {detail}")


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _codex_config(
    *,
    model: str,
    provider: str | None,
    base_url: str | None,
    api_key_env: str | None,
    thinking: str | None,
    manifest_path: Path,
    endpoint: str,
    mcp_bridge: Path,
) -> str:
    if any((provider, base_url, api_key_env)) and not all((provider, base_url, api_key_env)):
        raise ValueError("Custom Codex providers require --provider, --base-url, and --api-key-env together")
    lines = [
        f"model = {_toml_string(model)}",
        'approval_policy = "never"',
        'sandbox_mode = "read-only"',
        'web_search = "disabled"',
    ]
    if provider:
        lines.append(f"model_provider = {_toml_string(provider)}")
    if thinking:
        lines.append(f"model_reasoning_effort = {_toml_string(thinking)}")
    lines.extend(
        [
            "",
            "[features]",
            "plugins = false",
            "remote_plugin = false",
            "multi_agent = false",
            "shell_tool = false",
            "unified_exec = false",
            "goals = false",
            "hooks = false",
            "memories = false",
            "skill_mcp_dependency_install = false",
        ]
    )
    if provider and base_url and api_key_env:
        lines.extend(
            [
                "",
                f"[model_providers.{provider}]",
                f"name = {_toml_string(provider)}",
                f"base_url = {_toml_string(base_url.rstrip('/'))}",
                f"env_key = {_toml_string(api_key_env)}",
                'wire_api = "responses"',
            ]
        )
    lines.extend(
        [
            "",
            "[mcp_servers.harnesseval]",
            f"command = {_toml_string(sys.executable)}",
            f"args = [{_toml_string(str(mcp_bridge))}]",
            "required = true",
            "startup_timeout_sec = 30",
            'default_tools_approval_mode = "approve"',
            "",
            "[mcp_servers.harnesseval.env]",
            f"HARNESSEVAL_TOOL_MANIFEST = {_toml_string(str(manifest_path))}",
            f"HARNESSEVAL_TOOL_ENDPOINT = {_toml_string(endpoint)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _codex_environment(codex_env: list[str], codex_home: Path) -> dict[str, str]:
    environment = {name: os.environ[name] for name in HOST_ENV if name in os.environ}
    for name in codex_env:
        if name in os.environ:
            environment[name] = os.environ[name]
    environment["CODEX_HOME"] = str(codex_home)
    return environment


def _codex_command(
    *, executable: Path, workdir: Path, answer_path: Path, model: str
) -> list[str]:
    return [
        str(executable),
        "exec",
        "--ephemeral",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "--model",
        model,
        "--output-last-message",
        str(answer_path),
        "--cd",
        str(workdir),
        "-",
    ]


def _run_codex_process(
    command: list[str],
    *,
    prompt: str,
    cwd: Path,
    env: dict[str, str],
    events_path: Path,
    stderr_path: Path,
    terminal_path: Path,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    write_lock = threading.Lock()
    with (
        events_path.open("w", encoding="utf-8") as events,
        stderr_path.open("w", encoding="utf-8") as errors,
        terminal_path.open("a", encoding="utf-8") as terminal,
    ):
        def drain_stderr() -> None:
            for line in process.stderr:
                errors.write(line)
                errors.flush()
                with write_lock:
                    terminal.write(f"[stderr] {line}")
                    terminal.flush()
                sys.stderr.write(f"[codex] {line}")
                sys.stderr.flush()

        thread = threading.Thread(target=drain_stderr, name="codex-cli-stderr", daemon=True)
        thread.start()
        try:
            process.stdin.write(prompt)
            process.stdin.close()
            for line in process.stdout:
                events.write(line)
                events.flush()
                with write_lock:
                    terminal.write(line)
                    terminal.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    sys.stdout.write("[codex] non-JSON output recorded\n")
                else:
                    kind = str(event.get("type") or "event") if isinstance(event, dict) else "event"
                    if kind in {"turn.completed", "item.completed", "error"}:
                        sys.stdout.write(f"[codex] {kind}\n")
                sys.stdout.flush()
            returncode = process.wait()
            thread.join()
            return returncode
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            thread.join()
            raise
        finally:
            process.stdout.close()
            process.stderr.close()


def _event_tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") not in {"mcp_tool_call", "tool_call"}:
            continue
        call_id = str(item.get("id") or "")
        if call_id and call_id in seen:
            continue
        seen.add(call_id)
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append(
            {
                "id": call_id,
                "name": str(item.get("tool") or item.get("name") or ""),
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
        )
    return calls


def _codex_metrics(
    events: list[dict[str, Any]], environment_trajectory: list[dict[str, Any]]
) -> dict[str, Any]:
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning_output": 0, "total": 0}
    reported_turns = 0
    errors: list[str] = []
    for event in events:
        if event.get("type") == "turn.completed":
            reported_turns += 1
            raw = event.get("usage") or {}
            values = {
                "input": raw.get("input_tokens"),
                "output": raw.get("output_tokens"),
                "cache_read": raw.get("cached_input_tokens"),
                "cache_write": raw.get("cache_write_input_tokens"),
                "reasoning_output": raw.get("reasoning_output_tokens"),
            }
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    usage[key] += value
            usage["total"] += sum(value for key, value in values.items() if key in {"input", "output"} and isinstance(value, (int, float)))
        elif event.get("type") == "error" and isinstance(event.get("message"), str):
            errors.append(event["message"])
    traced_calls = [
        {
            "id": "",
            "name": str(call.get("name") or ""),
            "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
        }
        for call in environment_trajectory
    ]
    return {
        "rounds": reported_turns,
        "reported_turns": reported_turns,
        "committed_calls": traced_calls or _event_tool_calls(events),
        "usage": usage,
        "errors": errors,
    }


def run_codex_cli(
    *,
    platform: Any,
    benchmark: Benchmark,
    case_id: str,
    run_dir: Path,
    executable: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
    thinking: str | None = None,
    policy: dict[str, Any] | None = None,
    pass_env: list[str] | None = None,
    codex_env: list[str] | None = None,
    resume: bool = True,
    retry_failed: bool = False,
    build_missing: bool = True,
) -> dict[str, Any]:
    if benchmark.id not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Codex CLI product bridge is unavailable for benchmark: {benchmark.id}")
    if not model:
        raise ValueError("Codex CLI product runs require --model")
    policy = dict(policy or {})
    pass_env = list(pass_env or [])
    codex_env = list(codex_env or [])
    if api_key_env and api_key_env not in codex_env:
        codex_env.append(api_key_env)
    allowed_env = set(benchmark.raw.get("env_allowlist", [])) | HARNESS_PROVIDER_ENV
    unknown_env = sorted(set(pass_env) - allowed_env)
    if unknown_env:
        raise ValueError(f"Unsupported Codex product environment variable(s): {', '.join(unknown_env)}")
    present_benchmark_env = sorted(name for name in set(pass_env) if name in os.environ)
    present_codex_env = sorted(name for name in set(codex_env) if name in os.environ)
    if api_key_env and api_key_env not in present_codex_env:
        raise ValueError(f"Codex provider key environment variable is unset: {api_key_env}")
    codex, version = _codex_executable(executable)
    mcp_bridge = Path(__file__).with_name("codex_mcp_bridge.py").resolve()

    if benchmark.id == "terminal-bench-2":
        if not platform.image_exists(benchmark.adapter["image"]):
            if not build_missing:
                raise RuntimeError("Terminal task image is unavailable")
            built = platform.build(benchmark)
            if built["status"] != "completed":
                raise RuntimeError(f"Terminal task image build failed: {built}")
    elif not platform.image_is_current(benchmark.adapter):
        if not build_missing:
            raise RuntimeError("Benchmark image is missing or stale")
        built = platform.build(benchmark)
        if built["status"] != "completed":
            raise RuntimeError(f"Benchmark image build failed: {built}")

    prepared = (
        None
        if benchmark.id in NATIVE_EPISODE_BENCHMARKS | TASK_BENCHMARKS
        else platform._prepare_bridge_case(benchmark, case_id, run_dir)
    )
    identity = {
        "product": "codex-cli",
        "benchmark": benchmark.id,
        "case": case_id,
        "codex_executable": str(codex),
        "codex_executable_sha256": _sha256(codex),
        "codex_version": version,
        "provider": provider,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "model": model,
        "thinking": thinking,
        "policy": policy,
        "benchmark_environment_names": present_benchmark_env,
        "codex_environment_names": present_codex_env,
        "mcp_bridge_sha256": _sha256(mcp_bridge),
        "prepared_case_sha256": (
            _sha256(prepared / "input" / "case.json") if prepared is not None else None
        ),
    }
    store = CaseStore(run_dir.resolve(), f"{benchmark.id}-codex", case_id)
    with store.lock():
        existing = store.existing()
        if existing and resume and existing.get("resume_identity") != identity:
            raise RuntimeError(
                "The existing case was produced by a different Codex executable, model, provider, or policy. "
                "Use a new run directory or --no-resume."
            )
        if existing and resume and existing.get("status") in TERMINAL_STATUSES:
            if existing.get("status") != "failed" or not retry_failed:
                return {**existing, "resume_skipped": True}

        attempt_number, attempt = store.next_attempt()
        request = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": case_id,
            "product": "codex-cli",
            "attempt": attempt_number,
            "started_at": utc_now(),
            "resume_identity": identity,
            "benchmark_environment_names": present_benchmark_env,
            "codex_environment_names": present_codex_env,
        }
        store.start(attempt, request)
        handle: ProductServerHandle | None = None
        started = time.perf_counter()
        agent_execution_seconds = 0.0
        try:
            handle = (
                _start_task_tool_server(
                    platform=platform,
                    benchmark=benchmark,
                    attempt=attempt,
                    case_id=case_id,
                    pass_env=present_benchmark_env,
                )
                if benchmark.id in TASK_BENCHMARKS
                else _start_tool_server(
                    platform=platform,
                    benchmark=benchmark,
                    prepared=prepared,
                    attempt=attempt,
                    case_id=case_id,
                    policy=policy,
                    pass_env=present_benchmark_env,
                )
            )
            store.event(attempt, "product_bridge_ready", endpoint="loopback")
            event_paths: list[Path] = []
            stderr_paths: list[Path] = []
            returncodes: list[int] = []
            bridge_result: dict[str, Any] | None = None
            answer = ""
            turn = 0
            workdir = attempt / "codex_workdir"
            workdir.mkdir()
            while bridge_result is None:
                turn += 1
                manifest = _request_json(f"{handle.url}/manifest")
                manifest_path = attempt / f"tool_manifest-turn-{turn:03d}.json"
                atomic_json(manifest_path, manifest)
                event_path = attempt / f"codex-events-turn-{turn:03d}.jsonl"
                stderr_path = attempt / f"codex-stderr-turn-{turn:03d}.log"
                answer_path = attempt / f"codex-answer-turn-{turn:03d}.txt"
                event_paths.append(event_path)
                stderr_paths.append(stderr_path)
                codex_home = attempt / f"codex_home-turn-{turn:03d}"
                codex_home.mkdir()
                config = _codex_config(
                    model=model,
                    provider=provider,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    thinking=thinking,
                    manifest_path=manifest_path,
                    endpoint=handle.url,
                    mcp_bridge=mcp_bridge,
                )
                (codex_home / "config.toml").write_text(config, encoding="utf-8")
                command = _codex_command(
                    executable=codex, workdir=workdir, answer_path=answer_path, model=model
                )
                store.event(
                    attempt,
                    "codex_turn_started",
                    turn=turn,
                    tool_count=len(manifest.get("tools") or []),
                    task_system_time=str((manifest.get("metadata") or {}).get("system_time") or "") or None,
                )
                actor_started = time.perf_counter()
                try:
                    returncode = _run_codex_process(
                        command,
                        prompt=str(manifest.get("prompt") or ""),
                        cwd=workdir,
                        env=_codex_environment(present_codex_env, codex_home),
                        events_path=event_path,
                        stderr_path=stderr_path,
                        terminal_path=attempt / "terminal.log",
                    )
                finally:
                    agent_execution_seconds += time.perf_counter() - actor_started
                returncodes.append(returncode)
                answer = answer_path.read_text(encoding="utf-8") if answer_path.is_file() else ""
                turn_events, _ = _jsonl(event_path)
                store.event(
                    attempt,
                    "codex_turn_finished",
                    turn=turn,
                    returncode=returncode,
                    answer_produced=bool(answer.strip()),
                )
                if returncode != 0 or not answer.strip():
                    break
                if benchmark.id in NATIVE_EPISODE_BENCHMARKS and "###STOP###" not in answer:
                    continuation = _request_json(f"{handle.url}/turn", {"content": answer})
                    if continuation.get("episode_complete") is not True:
                        continue
                bridge_result = _request_json(
                    f"{handle.url}/final",
                    {"profile": "codex-cli", "answer": answer, "committed_calls": []},
                )

            _concat(event_paths, attempt / "codex-events.jsonl")
            _concat(stderr_paths, attempt / "codex-stderr.log")
            events, malformed_events = _jsonl(attempt / "codex-events.jsonl")
            tool_trace_path = attempt / "benchmark_server" / "tool_trace.jsonl"
            environment_trajectory, malformed_tool_rows = _tool_results(tool_trace_path)
            actor = _codex_metrics(events, environment_trajectory)
            scorer_answer = _scorer_answer(benchmark.id, answer)
            returncode = next((code for code in returncodes if code != 0), returncodes[-1] if returncodes else 1)
            answer_produced = bool(answer.strip())
            if bridge_result is None and answer_produced:
                bridge_result = _request_json(
                    f"{handle.url}/final",
                    {
                        "profile": "codex-cli",
                        "answer": answer,
                        "committed_calls": actor["committed_calls"],
                    },
                )
            bridge_result = bridge_result or {}
            if benchmark.id in TASK_BENCHMARKS and answer_produced:
                bridge_result.update(
                    _finalize_task(
                        platform=platform,
                        benchmark=benchmark,
                        case_id=case_id,
                        attempt=attempt,
                        handle=handle,
                        pass_env=present_benchmark_env,
                    )
                )
            status = "completed" if returncode == 0 and answer_produced else "failed"
            failure_kind = None
            if not answer_produced and actor["errors"]:
                failure_kind = "provider_error"
            elif not answer_produced:
                failure_kind = "no_final_answer"
            elif returncode != 0:
                failure_kind = "codex_cli_error"
            environment_calls = bridge_result.get("environment_tool_calls", len(environment_trajectory))
            reported_trajectory = bridge_result.get("calls")
            committed_trajectory = (
                reported_trajectory
                if isinstance(reported_trajectory, list) and reported_trajectory
                else actor["committed_calls"]
            )
            harness_result = {
                "schema_version": 1,
                "status": status,
                "failure_kind": failure_kind,
                "benchmark": benchmark.id,
                "case_id": case_id,
                "profile": "codex-cli",
                "agent_execution_seconds": agent_execution_seconds,
                "returncode": returncode,
                "answer": answer,
                "scorer_answer": scorer_answer,
                "actor": actor,
                "tools": {
                    "calls": len(committed_trajectory),
                    "trajectory": committed_trajectory,
                    "environment_calls": environment_calls,
                    "environment_trajectory": environment_trajectory,
                },
                "native": bridge_result if benchmark.id in NATIVE_EPISODE_BENCHMARKS | TASK_BENCHMARKS else None,
                "parse_health": {
                    "event_rows": len(events),
                    "malformed_event_rows": malformed_events,
                    "malformed_tool_trace_rows": malformed_tool_rows,
                },
                "product": {
                    "name": "codex",
                    "version": version,
                    "executable": str(codex),
                    "provider": provider,
                    "base_url": base_url,
                    "api_key_env": api_key_env,
                    "model": model,
                    "thinking": thinking,
                    "automatic_resources_disabled": True,
                    "benchmark_tools_via_mcp": True,
                },
                "artifacts": {
                    "events": str(attempt / "codex-events.jsonl"),
                    "stderr": str(attempt / "codex-stderr.log"),
                    "terminal": str(attempt / "terminal.log"),
                    "tool_trace": str(tool_trace_path),
                },
            }
            harness_result["score"] = _score_result(benchmark.id, prepared, harness_result, bridge_result)
            atomic_json(attempt / "harness_result.json", harness_result)
            result = {
                **request,
                "status": status,
                "failure_kind": failure_kind,
                "finished_at": utc_now(),
                "execution_seconds": time.perf_counter() - started,
                "returncode": returncode,
                "harness": harness_result,
            }
            store.finish(attempt, result)
            return result
        except KeyboardInterrupt:
            result = {
                **request,
                "status": "cancelled",
                "finished_at": utc_now(),
                "execution_seconds": time.perf_counter() - started,
            }
            store.finish(attempt, result)
            raise
        except Exception as exc:
            result = {
                **request,
                "status": "failed",
                "failure_kind": "infrastructure_error",
                "finished_at": utc_now(),
                "execution_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            store.finish(attempt, result)
            return result
        finally:
            _close_server(platform, handle)
