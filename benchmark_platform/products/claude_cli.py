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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claude_executable(value: str | None) -> tuple[Path, str]:
    requested = value or "claude"
    located = shutil.which(requested)
    candidate = Path(located or requested).expanduser().resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"Claude Code CLI is unavailable: {candidate}")
    completed = subprocess.run(
        [str(candidate), "--version"], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Claude Code version probe failed: {detail}")
    return candidate, (completed.stdout.strip() or completed.stderr.strip())


def _claude_mcp_config(*, manifest_path: Path, endpoint: str, mcp_bridge: Path) -> dict[str, Any]:
    return {
        "mcpServers": {
            "harnesseval": {
                "type": "stdio",
                "command": "/usr/bin/env",
                "args": [
                    "-i",
                    "PATH=/usr/bin:/bin",
                    "PYTHONIOENCODING=utf-8",
                    f"HARNESSEVAL_TOOL_MANIFEST={manifest_path}",
                    f"HARNESSEVAL_TOOL_ENDPOINT={endpoint}",
                    sys.executable,
                    str(mcp_bridge),
                ],
            }
        }
    }


def _claude_settings(tool_names: list[str]) -> dict[str, Any]:
    return {
        "permissions": {
            "allow": [f"mcp__harnesseval__{name}" for name in tool_names],
        }
    }


def _claude_environment(
    claude_env: list[str],
    *,
    config_dir: Path,
    home: Path,
    api_key_env: str,
    base_url: str,
) -> dict[str, str]:
    environment = {name: os.environ[name] for name in HOST_ENV if name in os.environ}
    for name in claude_env:
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update(
        {
            "HOME": str(home),
            "CLAUDE_CONFIG_DIR": str(config_dir),
            "ANTHROPIC_API_KEY": os.environ[api_key_env],
            "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
            "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1",
            "ENABLE_TOOL_SEARCH": "false",
        }
    )
    return environment


def _claude_command(
    *,
    executable: Path,
    model: str,
    thinking: str | None,
    mcp_config_path: Path,
    settings_path: Path,
    tool_names: list[str],
) -> list[str]:
    command = [
        str(executable),
        "--bare",
        "--print",
        "--input-format",
        "text",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--no-chrome",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config_path),
        "--settings",
        str(settings_path),
        "--tools",
        "",
        "--allow-dangerously-skip-permissions",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
    ]
    if tool_names:
        command.extend(
            ["--allowedTools", ",".join(f"mcp__harnesseval__{name}" for name in tool_names)]
        )
    if thinking and thinking != "off":
        effort = "high" if thinking == "xhigh" else thinking
        if effort in {"low", "medium", "high"}:
            command.extend(["--effort", effort])
    return command


def _run_claude_process(
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
                sys.stderr.write(f"[claude] {line}")
                sys.stderr.flush()

        thread = threading.Thread(target=drain_stderr, name="claude-cli-stderr", daemon=True)
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
                    sys.stdout.write("[claude] non-JSON output recorded\n")
                else:
                    if isinstance(event, dict) and event.get("type") in {"assistant", "result"}:
                        sys.stdout.write(f"[claude] {event['type']}\n")
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


def _event_answer(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            return event["result"]
    return ""


def _event_tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        blocks = message.get("content") if isinstance(message, dict) else None
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            call_id = str(block.get("id") or "")
            if call_id and call_id in seen:
                continue
            seen.add(call_id)
            name = str(block.get("name") or "")
            prefix = "mcp__harnesseval__"
            calls.append(
                {
                    "id": call_id,
                    "name": name[len(prefix) :] if name.startswith(prefix) else name,
                    "arguments": block.get("input") if isinstance(block.get("input"), dict) else {},
                }
            )
    return calls


def _claude_metrics(
    events: list[dict[str, Any]], environment_trajectory: list[dict[str, Any]]
) -> dict[str, Any]:
    usage = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning_output": 0,
        "total": 0,
    }
    reported_turns = 0
    errors: list[str] = []
    result_events = [event for event in events if event.get("type") == "result"]
    for event in result_events:
        reported_turns += int(event.get("num_turns") or 0)
        raw = event.get("usage") or {}
        values = {
            "input": raw.get("input_tokens"),
            "output": raw.get("output_tokens"),
            "cache_read": raw.get("cache_read_input_tokens"),
            "cache_write": raw.get("cache_creation_input_tokens"),
            "reasoning_output": (raw.get("output_tokens_details") or {}).get("thinking_tokens"),
        }
        for key, value in values.items():
            if isinstance(value, (int, float)):
                usage[key] += value
        usage["total"] += sum(
            value
            for key, value in values.items()
            if key in {"input", "output"} and isinstance(value, (int, float))
        )
        if event.get("is_error"):
            detail = event.get("result") or event.get("subtype") or "Claude Code error"
            errors.append(str(detail))
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "permission_denied":
            errors.append(str(event.get("message") or "Claude Code tool permission denied"))
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


def run_claude_cli(
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
    claude_env: list[str] | None = None,
    resume: bool = True,
    retry_failed: bool = False,
    build_missing: bool = True,
) -> dict[str, Any]:
    if benchmark.id not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Claude Code product bridge is unavailable for benchmark: {benchmark.id}")
    if not model or not base_url or not api_key_env:
        raise ValueError("Claude Code product runs require --model, --base-url, and --api-key-env")
    policy = dict(policy or {})
    pass_env = list(pass_env or [])
    claude_env = list(claude_env or [])
    if api_key_env not in claude_env:
        claude_env.append(api_key_env)
    allowed_env = set(benchmark.raw.get("env_allowlist", [])) | HARNESS_PROVIDER_ENV
    unknown_env = sorted(set(pass_env) - allowed_env)
    if unknown_env:
        raise ValueError(f"Unsupported Claude product environment variable(s): {', '.join(unknown_env)}")
    present_benchmark_env = sorted(name for name in set(pass_env) if name in os.environ)
    present_claude_env = sorted(name for name in set(claude_env) if name in os.environ)
    if api_key_env not in present_claude_env:
        raise ValueError(f"Claude provider key environment variable is unset: {api_key_env}")
    claude, version = _claude_executable(executable)
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
        "product": "claude-code",
        "benchmark": benchmark.id,
        "case": case_id,
        "claude_executable": str(claude),
        "claude_executable_sha256": _sha256(claude),
        "claude_version": version,
        "provider": provider,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "model": model,
        "thinking": thinking,
        "policy": policy,
        "benchmark_environment_names": present_benchmark_env,
        "claude_environment_names": present_claude_env,
        "mcp_bridge_sha256": _sha256(mcp_bridge),
        "prepared_case_sha256": (
            _sha256(prepared / "input" / "case.json") if prepared is not None else None
        ),
    }
    store = CaseStore(run_dir.resolve(), f"{benchmark.id}-claude", case_id)
    with store.lock():
        existing = store.existing()
        if existing and resume and existing.get("resume_identity") != identity:
            raise RuntimeError(
                "The existing case was produced by a different Claude Code executable, model, provider, "
                "or policy. Use a new run directory or --no-resume."
            )
        if existing and resume and existing.get("status") in TERMINAL_STATUSES:
            if existing.get("status") != "failed" or not retry_failed:
                return {**existing, "resume_skipped": True}

        attempt_number, attempt = store.next_attempt()
        request = {
            "schema_version": 1,
            "benchmark": benchmark.id,
            "case_id": case_id,
            "product": "claude-code",
            "attempt": attempt_number,
            "started_at": utc_now(),
            "resume_identity": identity,
            "benchmark_environment_names": present_benchmark_env,
            "claude_environment_names": present_claude_env,
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
            workdir = attempt / "claude_workdir"
            workdir.mkdir()
            while bridge_result is None:
                turn += 1
                manifest = _request_json(f"{handle.url}/manifest")
                manifest_path = attempt / f"tool_manifest-turn-{turn:03d}.json"
                atomic_json(manifest_path, manifest)
                event_path = attempt / f"claude-events-turn-{turn:03d}.jsonl"
                stderr_path = attempt / f"claude-stderr-turn-{turn:03d}.log"
                event_paths.append(event_path)
                stderr_paths.append(stderr_path)
                config_dir = attempt / f"claude_config-turn-{turn:03d}"
                home = attempt / f"claude_home-turn-{turn:03d}"
                config_dir.mkdir()
                home.mkdir()
                mcp_config_path = attempt / f"claude-mcp-turn-{turn:03d}.json"
                atomic_json(
                    mcp_config_path,
                    _claude_mcp_config(
                        manifest_path=manifest_path,
                        endpoint=handle.url,
                        mcp_bridge=mcp_bridge,
                    ),
                )
                tool_names = [str(entry["name"]) for entry in manifest.get("tools") or []]
                settings_path = attempt / f"claude-settings-turn-{turn:03d}.json"
                atomic_json(settings_path, _claude_settings(tool_names))
                command = _claude_command(
                    executable=claude,
                    model=model,
                    thinking=thinking,
                    mcp_config_path=mcp_config_path,
                    settings_path=settings_path,
                    tool_names=tool_names,
                )
                store.event(
                    attempt,
                    "claude_turn_started",
                    turn=turn,
                    tool_count=len(tool_names),
                    task_system_time=str((manifest.get("metadata") or {}).get("system_time") or "") or None,
                )
                actor_started = time.perf_counter()
                try:
                    returncode = _run_claude_process(
                        command,
                        prompt=str(manifest.get("prompt") or ""),
                        cwd=workdir,
                        env=_claude_environment(
                            present_claude_env,
                            config_dir=config_dir,
                            home=home,
                            api_key_env=api_key_env,
                            base_url=base_url,
                        ),
                        events_path=event_path,
                        stderr_path=stderr_path,
                        terminal_path=attempt / "terminal.log",
                    )
                finally:
                    agent_execution_seconds += time.perf_counter() - actor_started
                returncodes.append(returncode)
                turn_events, _ = _jsonl(event_path)
                answer = _event_answer(turn_events)
                store.event(
                    attempt,
                    "claude_turn_finished",
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
                    {"profile": "claude-code", "answer": answer, "committed_calls": []},
                )

            _concat(event_paths, attempt / "claude-events.jsonl")
            _concat(stderr_paths, attempt / "claude-stderr.log")
            events, malformed_events = _jsonl(attempt / "claude-events.jsonl")
            tool_trace_path = attempt / "benchmark_server" / "tool_trace.jsonl"
            environment_trajectory, malformed_tool_rows = _tool_results(tool_trace_path)
            actor = _claude_metrics(events, environment_trajectory)
            scorer_answer = _scorer_answer(benchmark.id, answer)
            returncode = next(
                (code for code in returncodes if code != 0),
                returncodes[-1] if returncodes else 1,
            )
            answer_produced = bool(answer.strip())
            if bridge_result is None and answer_produced:
                bridge_result = _request_json(
                    f"{handle.url}/final",
                    {
                        "profile": "claude-code",
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
                failure_kind = "claude_cli_error"
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
                "profile": "claude-code",
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
                "native": (
                    bridge_result
                    if benchmark.id in NATIVE_EPISODE_BENCHMARKS | TASK_BENCHMARKS
                    else None
                ),
                "parse_health": {
                    "event_rows": len(events),
                    "malformed_event_rows": malformed_events,
                    "malformed_tool_trace_rows": malformed_tool_rows,
                },
                "product": {
                    "name": "claude-code",
                    "version": version,
                    "executable": str(claude),
                    "provider": provider,
                    "base_url": base_url,
                    "api_key_env": api_key_env,
                    "model": model,
                    "thinking": thinking,
                    "automatic_resources_disabled": True,
                    "benchmark_tools_via_mcp": True,
                },
                "artifacts": {
                    "events": str(attempt / "claude-events.jsonl"),
                    "stderr": str(attempt / "claude-stderr.log"),
                    "terminal": str(attempt / "terminal.log"),
                    "tool_trace": str(tool_trace_path),
                },
            }
            harness_result["score"] = _score_result(
                benchmark.id, prepared, harness_result, bridge_result
            )
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
