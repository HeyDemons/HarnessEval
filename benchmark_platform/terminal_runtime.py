from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO


DEFAULT_AGENT_TIMEOUT_SEC = 600.0
DEFAULT_VERIFIER_TIMEOUT_SEC = 600.0
VERIFIER_MAX_ATTEMPTS = 3
DOCKER_CLIENT_GRACE_SEC = 15.0
TERMINAL_AGENT_ENVIRONMENT_HINT = (
    "You are running as root inside a Docker container. Do not use sudo - it is "
    "not installed and not needed since you already have root privileges."
)

# inspect_evals retries Harbor test scripts when their unstructured output contains a
# transport failure. Keep that contract here, with the common apt/pip spellings observed in
# Terminal-Bench 2 logs added to the upstream list.
NETWORK_ERROR_PATTERNS = (
    "connection refused",
    "connection reset",
    "network is unreachable",
    "temporary failure in name resolution",
    "temporary failure resolving",
    "could not resolve host",
    "could not connect to",
    "unable to connect to",
    "failed to connect",
    "failed to download",
    "connection timed out",
    "no route to host",
    "the requested url returned error",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
)

# A verifier often tests a service that the agent was asked to start inside the task
# container. A refused localhost connection is then an ordinary failed assertion, not a
# transient failure of the verifier's own network. Keep genuine external transport failures
# retryable while allowing the official reward for a missing local service to be recorded.
LOCAL_SERVICE_NETWORK_PATTERNS = {
    "connection refused",
    "could not connect to",
    "unable to connect to",
    "failed to connect",
}
LOCAL_ENDPOINT_MARKERS = (
    "host='localhost'",
    'host="localhost"',
    "('localhost',",
    '("localhost",',
    "localhost:",
    "localhost port ",
    "127.0.0.1",
    "[::1]",
)


DockerCommand = Callable[..., list[str]]


@dataclass(frozen=True)
class TerminalTaskSettings:
    agent_timeout_sec: float
    verifier_timeout_sec: float
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    network: str
    workdir: str | None
    verifier_mode: str

    @property
    def resource_limits(self) -> dict[str, int | None]:
        return {
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
        }


def terminal_agent_prompt(instruction: str) -> str:
    """Add the environment guidance used by the official Terminal-Bench 2 solver."""
    return f"{TERMINAL_AGENT_ENVIRONMENT_HINT}\n\n{instruction}"


def _positive_float(value: Any, *, default: float, field: str) -> float:
    resolved = default if value is None else float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return resolved


def _positive_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field} must be positive")
    return resolved


def terminal_task_settings(metadata: dict[str, Any]) -> TerminalTaskSettings:
    environment = metadata.get("environment") or {}
    agent = metadata.get("agent") or {}
    verifier = metadata.get("verifier") or {}

    network_mode = str(environment.get("network_mode") or "").strip().lower()
    if environment.get("allow_internet") is not None:
        network = "bridge" if bool(environment["allow_internet"]) else "none"
    elif network_mode in {"no-network", "none"}:
        network = "none"
    else:
        # Harbor's environment baseline is public when no mode is declared. Docker's bridge
        # network is the local provider's equivalent.
        network = "bridge"

    explicit_verifier_mode = verifier.get("environment_mode")
    if explicit_verifier_mode is None:
        verifier_mode = (
            "separate" if verifier.get("environment") is not None else "shared"
        )
    else:
        verifier_mode = str(explicit_verifier_mode).strip().lower()
    if verifier_mode not in {"shared", "separate"}:
        raise ValueError(f"Unsupported verifier.environment_mode: {verifier_mode!r}")

    workdir = environment.get("workdir")
    if workdir is not None and not str(workdir).startswith("/"):
        raise ValueError("environment.workdir must be an absolute container path")

    return TerminalTaskSettings(
        agent_timeout_sec=_positive_float(
            agent.get("timeout_sec"),
            default=DEFAULT_AGENT_TIMEOUT_SEC,
            field="agent.timeout_sec",
        ),
        verifier_timeout_sec=_positive_float(
            verifier.get("timeout_sec"),
            default=DEFAULT_VERIFIER_TIMEOUT_SEC,
            field="verifier.timeout_sec",
        ),
        cpus=_positive_int(environment.get("cpus"), field="environment.cpus"),
        memory_mb=_positive_int(
            environment.get("memory_mb"), field="environment.memory_mb"
        ),
        storage_mb=_positive_int(
            environment.get("storage_mb"), field="environment.storage_mb"
        ),
        network=network,
        workdir=str(workdir) if workdir is not None else None,
        verifier_mode=verifier_mode,
    )


def docker_resource_flags(settings: TerminalTaskSettings) -> list[str]:
    flags: list[str] = []
    if settings.cpus is not None:
        flags.extend(["--cpus", str(settings.cpus)])
    if settings.memory_mb is not None:
        flags.extend(["--memory", f"{settings.memory_mb}m"])
    # Docker has no portable per-container writable-layer quota. `--storage-opt size=` is
    # storage-driver-specific and fails on the rootless overlay provider used by the run
    # server, so storage_mb remains recorded provenance rather than a fake enforced limit.
    return flags


def terminal_outer_timeout_sec(
    metadata: dict[str, Any], *, cleanup_grace_sec: float = 120.0
) -> float:
    settings = terminal_task_settings(metadata)
    return (
        settings.agent_timeout_sec + settings.verifier_timeout_sec + cleanup_grace_sec
    )


def _decoded(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_captured(
    command: list[str], *, timeout_sec: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            _decoded(exc.stdout),
            _decoded(exc.stderr)
            + (
                f"\nDocker client timed out after {timeout_sec:.1f}s"
                if timeout_sec
                else ""
            ),
        )


def docker_exec_command(
    docker: DockerCommand,
    container: str,
    argv: list[str],
    *,
    timeout_sec: float | None = None,
    workdir: str | None = None,
    user: str | int | None = None,
    env: dict[str, str] | None = None,
    interactive: bool = False,
) -> list[str]:
    command = docker("exec")
    if interactive:
        command.append("-i")
    if workdir is not None:
        command.extend(["-w", workdir])
    if user is not None:
        command.extend(["-u", str(user)])
    for name, value in (env or {}).items():
        command.extend(["-e", f"{name}={value}"])
    command.append(container)
    if timeout_sec is not None:
        # The timeout lives inside the task container. Killing only the host-side `docker
        # exec` client leaves its process running in the container and lets it race the
        # verifier; GNU coreutils timeout terminates the actual command while preserving the
        # task container and all other services the agent started.
        command.extend(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=5s",
                f"{max(0.001, timeout_sec):.3f}s",
            ]
        )
    command.extend(argv)
    return command


def container_workdir(docker: DockerCommand, container: str) -> str:
    inspected = run_captured(
        docker("inspect", "-f", "{{.Config.WorkingDir}}", container),
        timeout_sec=30,
    )
    if inspected.returncode != 0:
        raise RuntimeError(
            f"Unable to inspect task container workdir: {inspected.stderr or inspected.stdout}"
        )
    return inspected.stdout.strip() or "/"


def _write_process_output(
    result: subprocess.CompletedProcess[str],
    log: TextIO | None,
    *,
    prefix: str,
) -> None:
    if log is None:
        return
    for stream_name, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        if not value:
            continue
        log.write(f"{prefix}[{stream_name}]\n{value}")
        if not value.endswith("\n"):
            log.write("\n")
    log.flush()


def _network_error(output: str) -> str | None:
    lowered = output.lower()
    local_service = any(marker in lowered for marker in LOCAL_ENDPOINT_MARKERS)
    return next(
        (
            pattern
            for pattern in NETWORK_ERROR_PATTERNS
            if pattern in lowered
            and not (local_service and pattern in LOCAL_SERVICE_NETWORK_PATTERNS)
        ),
        None,
    )


def _read_rewards(logs_dir: Path) -> tuple[dict[str, float], str | None]:
    json_path = logs_dir / "reward.json"
    text_path = logs_dir / "reward.txt"
    try:
        if json_path.is_file():
            value = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not value:
                raise ValueError("reward.json must contain a non-empty object")
            rewards: dict[str, float] = {}
            for name, score in value.items():
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f"reward.json field {name!r} is not numeric")
                rewards[str(name)] = float(score)
            return rewards, None
        if text_path.is_file():
            raw = text_path.read_text(encoding="utf-8").strip()
            if not raw:
                raise ValueError("reward.txt is empty")
            return {"reward": float(raw)}, None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {}, f"Invalid verifier reward: {type(exc).__name__}: {exc}"
    return {}, "Verifier produced neither reward.json nor reward.txt"


def _infrastructure_result(
    *,
    reason: str,
    error: str,
    returncode: int,
    attempts: int,
) -> dict[str, Any]:
    return {
        "status": "infra_failed",
        "scores": {},
        "returncode": returncode,
        "attempts": attempts,
        "termination_reason": reason,
        "error": error,
    }


def run_shared_verifier(
    *,
    docker: DockerCommand,
    container: str,
    task_dir: Path,
    logs_dir: Path,
    timeout_sec: float,
    log: TextIO | None = None,
    prefix: str = "[terminal-bench:verifier] ",
    max_attempts: int = VERIFIER_MAX_ATTEMPTS,
    verifier_env: dict[str, str] | None = None,
    verifier_user: str | int | None = None,
) -> dict[str, Any]:
    """Upload hidden tests after the agent, then grade in the same live container.

    Harbor's default verifier mode is shared. Uploading, rather than bind-mounting, keeps
    tests hidden during the agent phase and writable during verification (several TB2 tests
    compile native extensions in /tests). The task image's configured workdir remains in
    effect because no `docker exec -w` override is supplied.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    tests_dir = task_dir / "tests"
    if not (tests_dir / "test.sh").is_file():
        return _infrastructure_result(
            reason="verifier_setup_error",
            error=f"Terminal task has no tests/test.sh: {tests_dir}",
            returncode=1,
            attempts=0,
        )

    prepared = run_captured(
        docker(
            "exec",
            container,
            "sh",
            "-lc",
            "rm -rf /tests /logs/verifier && mkdir -p /tests /logs/verifier",
        ),
        timeout_sec=30,
    )
    _write_process_output(prepared, log, prefix=f"{prefix}setup ")
    if prepared.returncode != 0:
        return _infrastructure_result(
            reason="verifier_setup_error",
            error="Unable to create /tests and /logs/verifier in the agent container",
            returncode=prepared.returncode,
            attempts=0,
        )

    uploaded = run_captured(
        docker("cp", f"{tests_dir.resolve()}/.", f"{container}:/tests"),
        timeout_sec=60,
    )
    _write_process_output(uploaded, log, prefix=f"{prefix}upload ")
    if uploaded.returncode != 0:
        return _infrastructure_result(
            reason="verifier_setup_error",
            error="Unable to upload hidden tests into the agent container",
            returncode=uploaded.returncode,
            attempts=0,
        )

    deadline = time.monotonic() + timeout_sec
    attempts = 0
    checked = subprocess.CompletedProcess([], 1, "", "Verifier did not run")
    network_pattern: str | None = None
    timed_out = False
    while attempts < max_attempts:
        attempts += 1
        network_pattern = None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            checked = subprocess.CompletedProcess(
                [], 124, "", "Verifier phase timed out"
            )
            break

        # A zero written by a failed dependency install must not survive into a retry and be
        # mistaken for a valid model verdict.
        cleared = run_captured(
            docker(
                "exec",
                container,
                "sh",
                "-lc",
                "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json",
            ),
            timeout_sec=min(30.0, remaining + DOCKER_CLIENT_GRACE_SEC),
        )
        if cleared.returncode != 0:
            checked = cleared
            break

        command = docker_exec_command(
            docker,
            container,
            ["bash", "/tests/test.sh"],
            timeout_sec=remaining,
            user=verifier_user,
            env=verifier_env,
        )
        checked = run_captured(
            command,
            timeout_sec=remaining + DOCKER_CLIENT_GRACE_SEC,
        )
        attempt_prefix = f"{prefix}attempt {attempts}/{max_attempts} "
        _write_process_output(checked, log, prefix=attempt_prefix)
        (logs_dir / f"test-stdout-attempt-{attempts:03d}.txt").write_text(
            checked.stdout or "", encoding="utf-8"
        )
        (logs_dir / f"test-stderr-attempt-{attempts:03d}.txt").write_text(
            checked.stderr or "", encoding="utf-8"
        )

        timed_out = checked.returncode == 124
        network_pattern = _network_error(
            (checked.stdout or "") + "\n" + (checked.stderr or "")
        )
        if timed_out or network_pattern is None:
            break
        if attempts >= max_attempts:
            break
        delay = min(float(2 ** (attempts - 1)), max(0.0, deadline - time.monotonic()))
        if delay <= 0:
            timed_out = True
            break
        if log is not None:
            log.write(
                f"{prefix}transient network error ({network_pattern}); "
                f"retrying after {delay:.1f}s\n"
            )
            log.flush()
        time.sleep(delay)

    copied = run_captured(
        docker("cp", f"{container}:/logs/verifier/.", str(logs_dir.resolve())),
        timeout_sec=60,
    )
    _write_process_output(copied, log, prefix=f"{prefix}download ")
    if copied.returncode != 0:
        return _infrastructure_result(
            reason="verifier_log_download_error",
            error="Unable to copy verifier logs from the task container",
            returncode=copied.returncode,
            attempts=attempts,
        )
    (logs_dir / "test-stdout.txt").write_text(checked.stdout or "", encoding="utf-8")
    (logs_dir / "test-stderr.txt").write_text(checked.stderr or "", encoding="utf-8")

    if timed_out:
        return _infrastructure_result(
            reason="verifier_timeout",
            error=f"Verifier exceeded its {timeout_sec:.1f}s wall-clock timeout",
            returncode=checked.returncode,
            attempts=attempts,
        )
    if network_pattern is not None:
        return _infrastructure_result(
            reason="verifier_network_error",
            error=(
                f"Verifier still reported a transient network error after {attempts} "
                f"attempt(s): {network_pattern}"
            ),
            returncode=checked.returncode,
            attempts=attempts,
        )

    scores, reward_error = _read_rewards(logs_dir)
    if reward_error is not None:
        return _infrastructure_result(
            reason="verifier_output_error",
            error=reward_error,
            returncode=checked.returncode,
            attempts=attempts,
        )
    return {
        "status": "completed",
        "scores": scores,
        "returncode": checked.returncode,
        "attempts": attempts,
        "termination_reason": "official_shared_verifier",
        "error": None,
    }


def copy_container_workdir(
    *,
    docker: DockerCommand,
    container: str,
    workdir: str,
    destination: Path,
) -> subprocess.CompletedProcess[str]:
    destination.mkdir(parents=True, exist_ok=True)
    return run_captured(
        docker("cp", f"{container}:{workdir.rstrip('/') or '/'}/.", str(destination)),
        timeout_sec=120,
    )
