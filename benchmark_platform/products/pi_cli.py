from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark_platform.catalog import Benchmark
from benchmark_platform.store import CaseStore, TERMINAL_STATUSES
from benchmark_platform.util import atomic_json, utc_now


NATIVE_EPISODE_BENCHMARKS = {"tau2", "vitabench"}
TASK_BENCHMARKS = {"terminal-bench-2", "swe-bench-verified"}
SUPPORTED_BENCHMARKS = {
    "gaia",
    "gdpval",
    "trajectory-bench",
    "bfcl",
    *NATIVE_EPISODE_BENCHMARKS,
    *TASK_BENCHMARKS,
}
HARNESS_PROVIDER_ENV = {
    "HARNESS_API_BASE",
    "HARNESS_API_TYPE",
    "HARNESS_API_KEY",
    "HARNESS_MODEL",
    "HARNESS_TEMPERATURE",
    "HARNESS_API_TIMEOUT_S",
    "HARNESS_API_RETRIES",
    "HARNESS_MAX_OUTPUT_TOKENS",
}
HOST_ENV = {
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "PI_AGENT_DIR",
    "PI_CODING_AGENT_DIR",
    "PI_PACKAGE_DIR",
    "NODE_PATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}


@dataclass
class ProductServerHandle:
    process: subprocess.Popen[str]
    log: TextIO
    url: str
    task_container: str | None = None
    context: dict[str, Any] | None = None


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"content-type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=None) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Product bridge HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Product bridge transport failed: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Product bridge connection failed: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Product bridge response must be one JSON object")
    return value


def _wait_manifest(url: str, process: subprocess.Popen[str], log_path: Path) -> dict[str, Any]:
    while process.poll() is None:
        try:
            return _request_json(f"{url}/manifest")
        except RuntimeError as manifest_error:
            try:
                status = _request_json(f"{url}/status")
            except RuntimeError:
                status = {}
            if status.get("state") == "failed":
                detail = str(status.get("error") or manifest_error)
                raise RuntimeError(f"Product bridge failed before publishing its manifest: {detail}")
            time.sleep(0.1)
    detail = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    raise RuntimeError(f"Product bridge exited before publishing its manifest:\n{detail}")


def _docker_host_port(container_name: str, port: int, process: subprocess.Popen[str]) -> int:
    while process.poll() is None:
        completed = subprocess.run(
            ["docker", "port", container_name, f"{port}/tcp"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return int(completed.stdout.strip().rsplit(":", 1)[1])
        time.sleep(0.1)
    raise RuntimeError("Product bridge exited before Docker published its port")


def _record_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(completed.stdout)
        log.write(completed.stderr)
    return completed


def _server_environment(pass_env: list[str]) -> dict[str, str]:
    environment = dict(os.environ)
    for name in pass_env:
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _start_tool_server(
    *,
    platform: Any,
    benchmark: Benchmark,
    prepared: Path | None,
    attempt: Path,
    case_id: str,
    policy: dict[str, Any],
    pass_env: list[str],
) -> ProductServerHandle:
    server_dir = attempt / "benchmark_server"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_dir / "server.log"
    container_name = f"harnesseval-pi-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        container_name,
        "--network",
        "bridge",
        "-p",
        "127.0.0.1::8765",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=1g",
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONUNBUFFERED=1",
        *platform._egress_env("bridge"),
        "-v",
        f"{platform.root}:/opt/harnesseval:ro",
        "-v",
        f"{server_dir}:/job:rw",
        "-w",
        "/opt/harnesseval",
    ]
    if prepared is not None:
        command.extend(["-v", f"{prepared / 'input'}:/bridge:ro"])
    for name in pass_env:
        if name in os.environ:
            command.extend(["-e", name])
    if benchmark.id in NATIVE_EPISODE_BENCHMARKS:
        module = (
            "benchmark_platform.bridges.vita_product_server"
            if benchmark.id == "vitabench"
            else "benchmark_platform.bridges.tau_product_server"
        )
        command.extend(
            [
                benchmark.adapter["image"],
                "python",
                "-m",
                module,
                "--case",
                case_id,
                "--policy",
                json.dumps(policy, ensure_ascii=False, sort_keys=True),
            ]
        )
    else:
        command.extend(
            [
                benchmark.adapter["image"],
                "python",
                "-m",
                "benchmark_platform.bridges.product_server",
                "--benchmark",
                benchmark.id,
                "--case",
                case_id,
            ]
        )
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=_server_environment(pass_env),
    )
    try:
        port = _docker_host_port(container_name, 8765, process)
        url = f"http://127.0.0.1:{port}"
        _wait_manifest(url, process, log_path)
        return ProductServerHandle(process, log, url)
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait()
        log.close()
        raise


def _swe_controller_command(
    platform: Any,
    benchmark: Benchmark,
    job: Path,
    action: str,
    case_id: str,
    pass_env: list[str],
) -> list[str]:
    command = platform._docker("run", "--rm", "--init", "--network", "bridge")
    command.extend(platform._egress_env("bridge"))
    command.extend(["-v", f"{platform._docker_socket()}:/var/run/docker.sock:rw"])
    command.extend(["-e", "DOCKER_HOST=unix:///var/run/docker.sock"])
    for name in pass_env:
        if name == "HF_TOKEN" and name in os.environ:
            command.extend(["-e", name])
    command.extend(
        [
            "-v",
            f"{job.resolve()}:/job:rw",
            benchmark.adapter["image"],
            "python",
            "/opt/platform/swebench_bridge.py",
            action,
            "--case",
            case_id,
        ]
    )
    return command


def _start_task_tool_server(
    *,
    platform: Any,
    benchmark: Benchmark,
    attempt: Path,
    case_id: str,
    pass_env: list[str],
) -> ProductServerHandle:
    server_dir = attempt / "benchmark_server"
    server_dir.mkdir(parents=True, exist_ok=True)
    setup_log = server_dir / "setup.log"
    container = f"harnesseval-pi-task-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    context: dict[str, Any] = {"server_dir": server_dir}
    if benchmark.id == "terminal-bench-2":
        task_dir = Path(benchmark.adapter["task_dir"])
        prompt = (task_dir / "instruction.md").read_text(encoding="utf-8")
        workspace_root = "/app"
        create = platform._docker("create", "--init", "--name", container)
        create.extend(["--label", "orch.benchmark-platform=1", "--label", "orch.product-bridge=1"])
        if docker_platform := benchmark.adapter.get("platform"):
            create.extend(["--platform", docker_platform])
        network = "bridge" if benchmark.adapter.get("allow_internet") else "none"
        create.extend(["--network", network, *platform._egress_env(network)])
        create.extend(
            [
                "-w",
                workspace_root,
                benchmark.adapter["image"],
                "sh",
                "-lc",
                "while :; do sleep 3600; done",
            ]
        )
        context.update({"task_dir": task_dir, "workspace_root": workspace_root})
    else:
        prepare = _swe_controller_command(platform, benchmark, server_dir, "prepare", case_id, pass_env)
        completed = _record_command(prepare, setup_log)
        public_path = server_dir / "public_case.json"
        if completed.returncode != 0 or not public_path.is_file():
            raise RuntimeError("SWE-bench public case preparation failed; see setup.log")
        public_case = json.loads(public_path.read_text(encoding="utf-8"))
        if public_case.get("hidden_fields_exposed_to_agent") != []:
            raise RuntimeError("SWE-bench preparation exposed hidden authority fields")
        prompt = str(public_case["prompt"])
        workspace_root = str(public_case["workspace_root"])
        create = platform._docker("create", "--init", "--name", container)
        create.extend(["--label", "orch.benchmark-platform=1", "--label", "orch.product-bridge=1"])
        create.extend(["--platform", public_case["task_image"]["platform"]])
        create.extend(["--network", "bridge", *platform._egress_env("bridge")])
        create.extend(
            [
                "-w",
                workspace_root,
                public_case["task_image"]["name"],
                "sh",
                "-lc",
                "while :; do sleep 3600; done",
            ]
        )
        context.update({"public_case": public_case, "workspace_root": workspace_root})
    created = _record_command(create, setup_log)
    if created.returncode != 0:
        raise RuntimeError("Task container creation failed; see setup.log")
    started = _record_command(platform._docker("start", container), setup_log)
    if started.returncode != 0:
        subprocess.run(platform._docker("rm", "-f", container), capture_output=True, check=False)
        raise RuntimeError("Task container start failed; see setup.log")

    prompt_path = server_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    endpoint_path = server_dir / "task_product_server.json"
    log_path = server_dir / "server.log"
    log = log_path.open("w", encoding="utf-8")
    environment = dict(os.environ)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(platform.root) + (os.pathsep + python_path if python_path else "")
    command = [
        sys.executable,
        "-m",
        "benchmark_platform.bridges.task_product_server",
        "--benchmark",
        benchmark.id,
        "--case",
        case_id,
        "--prompt-file",
        str(prompt_path),
        "--container",
        container,
        "--workspace-root",
        workspace_root,
        "--job",
        str(server_dir),
    ]
    process = subprocess.Popen(
        command,
        cwd=platform.root,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        while process.poll() is None and not endpoint_path.is_file():
            time.sleep(0.1)
        if not endpoint_path.is_file():
            raise RuntimeError("Task product bridge exited before publishing its endpoint")
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        url = f"http://{endpoint['host']}:{endpoint['port']}"
        _wait_manifest(url, process, log_path)
        return ProductServerHandle(process, log, url, container, context)
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait()
        log.close()
        subprocess.run(platform._docker("rm", "-f", container), capture_output=True, check=False)
        raise


def _finalize_task(
    *, platform: Any, benchmark: Benchmark, case_id: str, attempt: Path, handle: ProductServerHandle, pass_env: list[str]
) -> dict[str, Any]:
    assert handle.task_container is not None
    context = handle.context or {}
    server_dir = Path(context["server_dir"])
    evaluator_log = server_dir / "evaluator.log"
    if benchmark.id == "terminal-bench-2":
        workspace = attempt / "workspace"
        workspace.mkdir()
        copied = _record_command(
            platform._docker("cp", f"{handle.task_container}:/app/.", str(workspace)),
            evaluator_log,
        )
        if copied.returncode != 0:
            raise RuntimeError("Unable to copy Terminal-Bench workspace")
        logs = attempt / "verifier"
        logs.mkdir()
        task_dir = Path(context["task_dir"])
        verifier = f"{handle.task_container}-verifier"
        verify = platform._docker("run", "--rm", "--init", "--name", verifier)
        if docker_platform := benchmark.adapter.get("platform"):
            verify.extend(["--platform", docker_platform])
        verify.extend(platform._egress_env("bridge"))
        verify.extend(
            [
                "--network",
                "bridge",
                "-v",
                f"{workspace.resolve()}:/app:rw",
                "-v",
                f"{task_dir / 'tests'}:/tests:ro",
                "-v",
                f"{logs.resolve()}:/logs/verifier:rw",
                "-w",
                "/app",
                benchmark.adapter["image"],
                "bash",
                "/tests/test.sh",
            ]
        )
        checked = _record_command(verify, evaluator_log)
        reward_path = logs / "reward.txt"
        reward = reward_path.read_text(encoding="utf-8").strip() if reward_path.is_file() else None
        return {
            "native_score_status": "completed" if reward is not None else "failed",
            "native_score": float(reward) if reward is not None else None,
            "native_reward": float(reward) if reward is not None else None,
            "evaluator_returncode": checked.returncode,
            "termination_reason": "official_verifier",
        }

    workspace_root = str(context["workspace_root"])
    staged = _record_command(
        platform._docker("exec", "-w", workspace_root, handle.task_container, "git", "add", "-A"),
        evaluator_log,
    )
    if staged.returncode != 0:
        raise RuntimeError("Unable to stage SWE-bench workspace changes")
    patch_result = _record_command(
        platform._docker(
            "exec",
            "-w",
            workspace_root,
            handle.task_container,
            "git",
            "-c",
            "core.fileMode=false",
            "diff",
            "--cached",
            "--binary",
        ),
        evaluator_log,
    )
    if patch_result.returncode != 0:
        raise RuntimeError("Unable to extract SWE-bench model patch")
    (server_dir / "model.patch").write_text(patch_result.stdout, encoding="utf-8")
    evaluate = _swe_controller_command(platform, benchmark, server_dir, "evaluate", case_id, pass_env)
    checked = _record_command(evaluate, evaluator_log)
    payload_path = server_dir / "payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.is_file() else {}
    resolved = payload.get("scores", {}).get("resolved")
    return {
        "native_score_status": payload.get("native_score_status", "failed"),
        "native_score": resolved,
        "native_reward": resolved,
        "evaluator_returncode": checked.returncode,
        "termination_reason": "official_swebench_evaluator",
        "official_evaluation": payload,
    }


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    if not path.is_file():
        return rows, malformed
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                malformed += 1
    return rows, malformed


def _tool_results(path: Path) -> tuple[list[dict[str, Any]], int]:
    events, malformed = _jsonl(path)
    calls = [
        {
            key: event[key]
            for key in (
                "name",
                "arguments",
                "result",
                "state_version_before",
                "state_version_after",
            )
            if key in event
        }
        for event in events
        if event.get("event") == "tool_result"
    ]
    return calls, malformed


def _assistant_text(events: list[dict[str, Any]]) -> str:
    answers: list[str] = []
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        if parts:
            answers.append("\n".join(parts))
    return answers[-1] if answers else ""


def _scorer_answer(benchmark_id: str, answer: str) -> str:
    if benchmark_id != "gaia":
        return answer
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    for prefix in ("Final answer:", "Final Answer:", "Answer:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
            break
    return candidate.strip("`* ")


def _actor_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    rounds = 0
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0}
    last_stop_reason = None
    last_error = None
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        rounds += 1
        last_stop_reason = message.get("stopReason")
        last_error = message.get("errorMessage")
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "toolCall":
                calls.append(
                    {
                        "id": str(item.get("id") or ""),
                        "name": str(item.get("name") or ""),
                        "arguments": item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
                    }
                )
        raw_usage = message.get("usage") or {}
        for source, target in (
            ("input", "input"),
            ("output", "output"),
            ("cacheRead", "cache_read"),
            ("cacheWrite", "cache_write"),
            ("totalTokens", "total"),
        ):
            value = raw_usage.get(source)
            if isinstance(value, (int, float)):
                usage[target] += value
    return {
        "rounds": rounds,
        "committed_calls": calls,
        "usage": usage,
        "last_stop_reason": last_stop_reason,
        "last_error": last_error,
    }


def _score_result(
    benchmark_id: str, prepared: Path | None, result: dict[str, Any], bridge_result: dict[str, Any]
) -> dict[str, Any]:
    if benchmark_id in NATIVE_EPISODE_BENCHMARKS | TASK_BENCHMARKS:
        return {
            "authority": f"{benchmark_id}_native_evaluator",
            "status": bridge_result.get("native_score_status", "not_requested"),
            "score": bridge_result.get("native_score"),
            "reward": bridge_result.get("native_reward"),
            "termination_reason": bridge_result.get("termination_reason"),
        }
    if prepared is None:
        raise ValueError(f"Prepared authority is required for {benchmark_id}")
    gold = json.loads((prepared / "authority" / "gold.json").read_text(encoding="utf-8"))
    if benchmark_id == "gaia":
        from benchmark_platform.scorers.gaia import question_score

        target = str(gold.get("answer") or "")
        prediction = str(result.get("scorer_answer") or result.get("answer") or "")
        return {
            "authority": "gaia_public_answer_scorer",
            "status": "completed",
            "score": 1.0 if question_score(prediction, target) else 0.0,
            "scorer_input": prediction,
        }
    if benchmark_id == "bfcl" and str(gold.get("id") or "").startswith("irrelevance_"):
        return {
            "authority": "bfcl_irrelevance_no_function_call",
            "status": "completed",
            "score": 1.0 if not result["actor"]["committed_calls"] else 0.0,
        }
    if benchmark_id == "trajectory-bench":
        from benchmark_platform.bridges.adapters import _tool_name

        target = str(gold.get("final_answer") or "").strip()
        public_case = json.loads((prepared / "input" / "case.json").read_text(encoding="utf-8"))
        normalized_to_public = {
            _tool_name(str(item.get("tool name") or "")): str(item.get("tool name") or "")
            for item in public_case.get("tools") or []
        }
        expected_tools = {str(item.get("tool name") or "") for item in gold.get("tool_list") or []}
        observed_tools = {
            normalized_to_public.get(item["name"], item["name"])
            for item in result["actor"]["committed_calls"]
        }
        return {
            "authority": "traject_official_tool_name_metrics_and_answer_diagnostic",
            "status": "completed",
            "answer_exact": bool(target) and result.get("answer", "").strip() == target,
            "trajectory_exact": expected_tools == observed_tools,
            "tool_inclusion": len(expected_tools & observed_tools) / len(expected_tools) if expected_tools else None,
        }
    return {
        "authority": "gdpval_external_rubric" if benchmark_id == "gdpval" else "benchmark_native_scorer",
        "status": "not_run",
    }


def _pi_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    located = shutil.which(value) if candidate.name == value else str(candidate)
    if not located:
        raise FileNotFoundError(f"Pi CLI is unavailable: {value}")
    resolved = Path(located).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"Pi CLI is not executable: {resolved}")
    return resolved


def _pi_version(executable: Path) -> str:
    completed = subprocess.run([str(executable), "--version"], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to query Pi version: {completed.stderr or completed.stdout}")
    return completed.stdout.strip() or completed.stderr.strip()


def _pi_environment(pi_env: list[str], manifest_path: Path, endpoint: str) -> dict[str, str]:
    environment = {name: os.environ[name] for name in HOST_ENV if name in os.environ}
    for name in pi_env:
        if name in os.environ:
            environment[name] = os.environ[name]
    environment["HARNESSEVAL_TOOL_MANIFEST"] = str(manifest_path)
    environment["HARNESSEVAL_TOOL_ENDPOINT"] = endpoint
    environment["PI_OFFLINE"] = "1"
    return environment


def _pi_config_identity() -> dict[str, Any]:
    configured = os.environ.get("PI_CODING_AGENT_DIR") or os.environ.get("PI_AGENT_DIR")
    root = Path(configured).expanduser() if configured else Path.home() / ".pi" / "agent"
    fingerprints: dict[str, str | None] = {}
    for name in ("settings.json", "models.json"):
        path = root / name
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    auth_providers: list[str] = []
    auth_path = root / "auth.json"
    if auth_path.is_file():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            auth = None
        if isinstance(auth, dict):
            auth_providers = sorted(str(name) for name in auth)
    return {
        "directory": str(root.resolve()),
        "settings_sha256": fingerprints["settings.json"],
        "models_sha256": fingerprints["models.json"],
        "auth_provider_names": auth_providers,
    }


def _pi_command(
    *,
    executable: Path,
    extension: Path,
    tools: list[str],
    prompt: str,
    provider: str | None,
    model: str | None,
    thinking: str | None,
) -> list[str]:
    command = [
        str(executable),
        "--mode",
        "json",
        "--print",
        "--no-session",
        "--offline",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-extensions",
        "--no-builtin-tools",
        "--extension",
        str(extension),
        "--tools",
        ",".join(tools),
    ]
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    if thinking:
        command.extend(["--thinking", thinking])
    command.extend(["-p", prompt])
    return command


def _run_pi_process(
    command: list[str], *, cwd: Path, env: dict[str, str], events_path: Path, stderr_path: Path, terminal_path: Path
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
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
                sys.stderr.write(f"[pi] {line}")
                sys.stderr.flush()

        thread = threading.Thread(target=drain_stderr, name="pi-cli-stderr", daemon=True)
        thread.start()
        try:
            for line in process.stdout:
                events.write(line)
                events.flush()
                with write_lock:
                    terminal.write(line)
                    terminal.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    sys.stdout.write("[pi] non-JSON output recorded\n")
                else:
                    kind = str(event.get("type") or "event") if isinstance(event, dict) else "event"
                    if kind in {"message_end", "tool_execution_start", "tool_execution_end"}:
                        sys.stdout.write(f"[pi] {kind}\n")
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


def _concat(paths: list[Path], destination: Path) -> None:
    with destination.open("wb") as output:
        for path in paths:
            if path.is_file():
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output)


def _close_server(platform: Any, handle: ProductServerHandle | None) -> None:
    if handle is None:
        return
    if handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait()
    handle.log.close()
    if handle.task_container:
        subprocess.run(
            platform._docker("rm", "-f", handle.task_container),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def run_pi_cli(
    *,
    platform: Any,
    benchmark: Benchmark,
    case_id: str,
    run_dir: Path,
    executable: str = "pi",
    provider: str | None = None,
    model: str | None = None,
    thinking: str | None = None,
    policy: dict[str, Any] | None = None,
    pass_env: list[str] | None = None,
    pi_env: list[str] | None = None,
    resume: bool = True,
    retry_failed: bool = False,
    build_missing: bool = True,
) -> dict[str, Any]:
    if benchmark.id not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Pi CLI product bridge is unavailable for benchmark: {benchmark.id}")
    policy = dict(policy or {})
    pass_env = list(pass_env or [])
    pi_env = list(pi_env or [])
    allowed_env = set(benchmark.raw.get("env_allowlist", [])) | HARNESS_PROVIDER_ENV
    unknown_env = sorted(set(pass_env) - allowed_env)
    if unknown_env:
        raise ValueError(f"Unsupported Pi product environment variable(s): {', '.join(unknown_env)}")
    present_benchmark_env = sorted(name for name in set(pass_env) if name in os.environ)
    present_pi_env = sorted(name for name in set(pi_env) if name in os.environ)
    pi = _pi_executable(executable)
    version = _pi_version(pi)
    extension = Path(__file__).with_name("pi_tool_bridge.ts").resolve()
    if not extension.is_file():
        raise FileNotFoundError(f"Pi tool bridge extension is missing: {extension}")

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
    executable_digest = hashlib.sha256(pi.read_bytes()).hexdigest()
    identity = {
        "product": "pi-cli",
        "benchmark": benchmark.id,
        "case": case_id,
        "pi_executable": str(pi),
        "pi_executable_sha256": executable_digest,
        "pi_version": version,
        "provider": provider,
        "model": model,
        "thinking": thinking,
        "policy": policy,
        "benchmark_environment_names": present_benchmark_env,
        "pi_environment_names": present_pi_env,
        "pi_config": _pi_config_identity(),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "prepared_case_sha256": (
            hashlib.sha256((prepared / "input" / "case.json").read_bytes()).hexdigest()
            if prepared is not None
            else None
        ),
    }
    store = CaseStore(run_dir.resolve(), f"{benchmark.id}-pi", case_id)
    with store.lock():
        existing = store.existing()
        if existing and resume and existing.get("resume_identity") != identity:
            raise RuntimeError(
                "The existing case was produced by a different Pi executable, model, provider, or policy. "
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
            "product": "pi-cli",
            "attempt": attempt_number,
            "started_at": utc_now(),
            "resume_identity": identity,
            "benchmark_environment_names": present_benchmark_env,
            "pi_environment_names": present_pi_env,
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
            all_events: list[dict[str, Any]] = []
            returncodes: list[int] = []
            bridge_result: dict[str, Any] | None = None
            answer = ""
            turn = 0
            workdir = attempt / "pi_workdir"
            workdir.mkdir()
            while bridge_result is None:
                turn += 1
                manifest = _request_json(f"{handle.url}/manifest")
                manifest_path = attempt / f"tool_manifest-turn-{turn:03d}.json"
                atomic_json(manifest_path, manifest)
                tools = [str(item["name"]) for item in manifest.get("tools") or []]
                event_path = attempt / f"pi-events-turn-{turn:03d}.jsonl"
                stderr_path = attempt / f"pi-stderr-turn-{turn:03d}.log"
                event_paths.append(event_path)
                stderr_paths.append(stderr_path)
                task_time = str((manifest.get("metadata") or {}).get("system_time") or "").strip()
                command = _pi_command(
                    executable=pi,
                    extension=extension,
                    tools=tools,
                    prompt=str(manifest.get("prompt") or ""),
                    provider=provider,
                    model=model,
                    thinking=thinking,
                )
                store.event(
                    attempt,
                    "pi_turn_started",
                    turn=turn,
                    tool_count=len(tools),
                    task_system_time=task_time or None,
                )
                actor_started = time.perf_counter()
                try:
                    returncode = _run_pi_process(
                        command,
                        cwd=workdir,
                        env=_pi_environment(present_pi_env, manifest_path, handle.url),
                        events_path=event_path,
                        stderr_path=stderr_path,
                        terminal_path=attempt / "terminal.log",
                    )
                finally:
                    agent_execution_seconds += time.perf_counter() - actor_started
                returncodes.append(returncode)
                turn_events, _ = _jsonl(event_path)
                all_events.extend(turn_events)
                answer = _assistant_text(turn_events)
                turn_actor = _actor_metrics(turn_events)
                actor = _actor_metrics(all_events)
                store.event(
                    attempt,
                    "pi_turn_finished",
                    turn=turn,
                    returncode=returncode,
                    rounds=turn_actor["rounds"],
                    tool_calls=len(turn_actor["committed_calls"]),
                    cumulative_rounds=actor["rounds"],
                    cumulative_tool_calls=len(actor["committed_calls"]),
                )
                if returncode != 0 or turn_actor["last_stop_reason"] == "error" or not answer.strip():
                    break
                if benchmark.id in NATIVE_EPISODE_BENCHMARKS and "###STOP###" not in answer:
                    continuation = _request_json(f"{handle.url}/turn", {"content": answer})
                    if continuation.get("episode_complete") is not True:
                        continue
                bridge_result = _request_json(
                    f"{handle.url}/final",
                    {
                        "profile": "pi-cli",
                        "answer": answer,
                        "committed_calls": actor["committed_calls"],
                    },
                )

            _concat(event_paths, attempt / "pi-events.jsonl")
            _concat(stderr_paths, attempt / "pi-stderr.log")
            events, malformed_events = _jsonl(attempt / "pi-events.jsonl")
            actor = _actor_metrics(events)
            answer = _assistant_text(events)
            scorer_answer = _scorer_answer(benchmark.id, answer)
            returncode = next((code for code in returncodes if code != 0), returncodes[-1] if returncodes else 1)
            provider_failed = actor["last_stop_reason"] == "error"
            answer_produced = not provider_failed and bool(answer.strip())
            if bridge_result is None and answer_produced:
                bridge_result = _request_json(
                    f"{handle.url}/final",
                    {
                        "profile": "pi-cli",
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
            if provider_failed:
                failure_kind = "provider_error"
            elif not answer_produced:
                failure_kind = "no_final_answer"
            elif returncode != 0:
                failure_kind = "pi_cli_error"
            tool_trace_path = attempt / "benchmark_server" / "tool_trace.jsonl"
            traced_environment_calls, malformed_tool_rows = _tool_results(tool_trace_path)
            environment_trajectory = bridge_result.get("environment_calls")
            if environment_trajectory is None:
                environment_trajectory = traced_environment_calls
            environment_calls = bridge_result.get("environment_tool_calls")
            if environment_calls is None:
                environment_calls = len(environment_trajectory)
            harness_result = {
                "schema_version": 1,
                "status": status,
                "failure_kind": failure_kind,
                "benchmark": benchmark.id,
                "case_id": case_id,
                "profile": "pi-cli",
                "agent_execution_seconds": agent_execution_seconds,
                "returncode": returncode,
                "answer": answer,
                "scorer_answer": scorer_answer,
                "actor": actor,
                "tools": {
                    "calls": bridge_result.get("tool_calls", len(actor["committed_calls"])),
                    "trajectory": bridge_result.get("calls", actor["committed_calls"]),
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
                    "name": "pi",
                    "version": version,
                    "executable": str(pi),
                    "provider": provider,
                    "model": model,
                    "thinking": thinking,
                    "automatic_resources_disabled": True,
                },
                "artifacts": {
                    "events": str(attempt / "pi-events.jsonl"),
                    "stderr": str(attempt / "pi-stderr.log"),
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
