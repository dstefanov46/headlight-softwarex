#!/usr/bin/env python3
"""Run and resume C=4 campaigns and controller-specific seed-104/105 extensions."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
RUN_DATE = "20261004"
SEEDS = (102, 103)
WORKLOADS = ("gradual-ramp", "diurnal", "flash-crowd")
CONTROLLERS = ("headlight-v2", "hpa", "keda")
EXPECTED_TRACES = {
    "gradual-ramp": "b1a3d85fce1cd8a4ce44bef382794c8c12fcb2647f389ae789a284cc19cb00ca",
    "diurnal": "ced8b35581298a0e88cdadc9939033d41372591ed75913aaea118fa99c057ce2",
    "flash-crowd": "015943997cc853524e85ae0aeb965bcdf92da58e4d1be2003a0ae6a485f73116",
}
MEASUREMENT_SECONDS = {"gradual-ramp": 1200, "diurnal": 2400, "flash-crowd": 1200}
WARMUP_SECONDS = 480
NAMESPACE = "trainticket"
CONTROLLER_DEPLOYMENTS = (
    "ai-module",
    "ai-module-v2",
    "lstm-autoscaler",
    "threshold-autoscaler",
    "kubernetes-autoscaler-eval",
)
CANARY_EVIDENCE = (
    REPO_ROOT
    / "results/headlight-v2-live/emergency-disabled-canary-seed-401-20261004/shadow-smoke.ok"
)
CAMPAIGN_STATE_DIR = (
    REPO_ROOT / "results/headlight-v2-live/qualification-seed102-103-20261004"
)
LEGACY_STATE_DIR = CAMPAIGN_STATE_DIR
BASELINE_EXTENSION = False
HEADLIGHT_V2_EXTENSION = False
_memory_guard: subprocess.Popen[str] | None = None
_memory_guard_dir: Path | None = None
_extension_provenance: dict[str, Any] | None = None

_active_process: subprocess.Popen[str] | None = None
_received_signal: int | None = None


class CampaignInterrupted(RuntimeError):
    pass


def extension_campaign() -> bool:
    return BASELINE_EXTENSION or HEADLIGHT_V2_EXTENSION


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_dir(seed: int, workload: str) -> Path:
    if extension_campaign():
        return CAMPAIGN_STATE_DIR / "cells" / f"{workload}-seed-{seed}"
    return (
        REPO_ROOT
        / f"results/headlight-v2-live/qualification-{workload}-c4-seed-{seed}-{RUN_DATE}"
    )


def attempt_dir(seed: int, workload: str, controller: str, attempt: int) -> Path:
    name = controller if attempt == 1 else f"{controller}-attempt-{attempt}"
    return campaign_dir(seed, workload) / name


def status_path(seed: int, workload: str, controller: str, attempt: int) -> Path:
    return campaign_dir(seed, workload) / "status" / f"{controller}-attempt-{attempt}.json"


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def summary_result(
    output: Path, seed: int, expected_trace: str, *,
    controller: str | None = None, workload: str | None = None,
) -> tuple[bool, str]:
    summary = output / f"seed-{seed}/summaries/summary.csv"
    trace = output / f"seed-{seed}/raw/load_profile.csv"
    if not summary.is_file():
        return False, "missing_summary"
    try:
        with summary.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
    except (OSError, StopIteration, csv.Error) as exc:
        return False, f"unreadable_summary:{type(exc).__name__}"
    if (row.get("seed") != str(seed)
            or (controller is not None and row.get("controller") != controller)
            or (workload is not None and row.get("workload") != workload)):
        return False, "summary_identity_mismatch"
    validity = row.get("technical_validity", "")
    outcome = row.get("performance_outcome", "")
    if validity != "valid" or outcome not in {"measured", "poor_but_valid"}:
        return False, f"summary_not_valid:{validity or 'missing'}/{outcome or 'missing'}"
    if not trace.is_file():
        return False, "missing_load_trace"
    actual_trace = sha256(trace)
    if actual_trace != expected_trace:
        return False, f"trace_mismatch:{actual_trace}"
    return True, "completed_valid_bundle"


def existing_attempts(seed: int, workload: str, controller: str) -> list[tuple[int, Path]]:
    root = campaign_dir(seed, workload)
    found: list[tuple[int, Path]] = []
    base = root / controller
    if base.exists():
        found.append((1, base))
    for path in root.glob(f"{controller}-attempt-*"):
        try:
            number = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if number >= 2:
            found.append((number, path))
    return sorted(found)


def attempt_is_valid(
    seed: int,
    workload: str,
    controller: str,
    attempt: int,
    output: Path,
) -> bool:
    path = status_path(seed, workload, controller, attempt)
    artifacts_valid, artifact_reason = summary_result(
        output, seed, EXPECTED_TRACES[workload], controller=controller, workload=workload,
    )
    if path.exists():
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            status = {
                "schema_version": "headlight.campaign-cell-status/v1",
                "cell": {"seed": seed, "workload": workload, "controller": controller},
                "attempt": attempt,
                "output_dir": relative(output),
                "status": "invalid",
                "reason": f"unreadable_status:{type(exc).__name__}",
                "retry_allowed": True,
                "finished_at": utc_now(),
            }
            write_json(path, status)
            return False
        recorded_status = status.get("status")
        if recorded_status == "invalid":
            return False
        if recorded_status == "running":
            status.update(
                {
                    "status": "invalid",
                    "reason": "interrupted_previous_attempt",
                    "retry_allowed": True,
                    "finished_at": utc_now(),
                    "artifact_validation": artifact_reason,
                }
            )
            write_json(path, status)
            return False
        if recorded_status == "valid" and artifacts_valid:
            return True
        status.update(
            {
                "status": "invalid",
                "reason": f"recorded_status_or_artifacts_invalid:{recorded_status}/{artifact_reason}",
                "retry_allowed": True,
                "finished_at": utc_now(),
            }
        )
        write_json(path, status)
        return False
    write_json(
        path,
        {
            "schema_version": "headlight.campaign-cell-status/v1",
            "cell": {"seed": seed, "workload": workload, "controller": controller},
            "attempt": attempt,
            "output_dir": relative(output),
            "status": "valid" if artifacts_valid else "invalid",
            "reason": artifact_reason,
            "retry_allowed": not artifacts_valid,
            "inferred_from_existing_artifacts": True,
            "recorded_at": utc_now(),
        },
    )
    return artifacts_valid


def valid_attempt(seed: int, workload: str, controller: str) -> Path | None:
    valid_outputs: list[Path] = []
    for attempt, output in existing_attempts(seed, workload, controller):
        if attempt_is_valid(seed, workload, controller, attempt, output):
            valid_outputs.append(output)
    if len(valid_outputs) > 1:
        raise RuntimeError(
            f"multiple valid attempts for seed={seed} workload={workload} controller={controller}"
        )
    return valid_outputs[0] if valid_outputs else None


def next_attempt(seed: int, workload: str, controller: str) -> tuple[int, Path]:
    attempts = existing_attempts(seed, workload, controller)
    number = attempts[-1][0] + 1 if attempts else 1
    output = attempt_dir(seed, workload, controller, number)
    if output.exists():
        raise RuntimeError(f"refusing to reuse attempt path: {output}")
    return number, output


def validate_environment() -> None:
    expected_gateway = REPO_ROOT / "experiments/configs/gateway-protocols/research-v4-gateway.yaml"
    checks = {
        "HEADLIGHT_EVALUATION_CAPACITY": "4",
        "HEADLIGHT_GATEWAY_PROTOCOL_PATH": str(expected_gateway),
    }
    if BASELINE_EXTENSION:
        checks.update({
            "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
            "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
            "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
        })
    if not BASELINE_EXTENSION:
        checks["HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY"] = "1"
    for name, expected in checks.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must be {expected!r}")
    if not BASELINE_EXTENSION:
        evidence = Path(os.environ.get("HEADLIGHT_V2_SHADOW_EVIDENCE", str(CANARY_EVIDENCE)))
        if not evidence.is_file() or evidence.stat().st_size == 0:
            raise RuntimeError(f"missing matching Headlight shadow evidence: {evidence}")
        os.environ["HEADLIGHT_V2_SHADOW_EVIDENCE"] = str(evidence)
    kubeconfig = os.environ.get("HEADLIGHT_KUBECONFIG") or os.environ.get("KUBECONFIG")
    if not kubeconfig:
        raise RuntimeError("HEADLIGHT_KUBECONFIG or KUBECONFIG must be set")
    kubeconfig_path = Path(kubeconfig)
    if not kubeconfig_path.is_file():
        raise RuntimeError(f"configured kubeconfig does not exist: {kubeconfig_path}")
    os.environ["KUBECONFIG"] = str(kubeconfig_path)
    if not os.environ.get("HEADLIGHT_PREFLIGHT_PROMETHEUS_URL"):
        raise RuntimeError("HEADLIGHT_PREFLIGHT_PROMETHEUS_URL must be set")


def handle_signal(signum: int, _frame: Any) -> None:
    global _received_signal
    _received_signal = signum
    process = _active_process
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def stream_process(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool]:
    global _active_process
    if resource_guard_failed():
        raise RuntimeError("host resource guard aborted before cell launch")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"started_at={utc_now()}\n$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        _active_process = process

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = False
        try:
            if extension_campaign():
                deadline = time.monotonic() + timeout_seconds
                while True:
                    if resource_guard_failed():
                        raise RuntimeError("host resource guard exited; stopping active cell")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout_seconds)
                    try:
                        return_code = process.wait(timeout=min(2, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            else:
                return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait(timeout=30)
        except RuntimeError:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            return_code = 1
        finally:
            reader.join(timeout=10)
            if process.stdout is not None:
                process.stdout.close()
            _active_process = None
        log.write(
            f"finished_at={utc_now()} exit_code={return_code} timed_out={str(timed_out).lower()}\n"
        )
    return return_code, timed_out


def run_bounded(command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started_at = utc_now()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout.strip()
        return {
            "command": command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "exit_code": completed.returncode,
            "timed_out": False,
            "output_tail": output[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "command": command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "exit_code": None,
            "timed_out": True,
            "output_tail": output[-2000:],
        }


def cleanup_after_failure() -> list[dict[str, Any]]:
    commands: list[list[str]] = [
        [
            "kubectl",
            "--context",
            "cluster1",
            "-n",
            NAMESPACE,
            "delete",
            "job",
            "headlight-paper-locust",
            "--ignore-not-found",
            "--wait=false",
        ],
        [
            "kubectl",
            "--context",
            "cluster1",
            "-n",
            NAMESPACE,
            "delete",
            "configmap",
            "headlight-paper-locust",
            "--ignore-not-found",
        ],
    ]
    for context in ("cluster1", "cluster2", "cluster3"):
        commands.append(
            [
                "kubectl",
                "--context",
                context,
                "-n",
                NAMESPACE,
                "delete",
                "deployment",
                *CONTROLLER_DEPLOYMENTS,
                "--ignore-not-found",
                "--wait=false",
            ]
        )
    for context in ("cluster2", "cluster3"):
        commands.extend(
            [
                [
                    "kubectl",
                    "--context",
                    context,
                    "-n",
                    NAMESPACE,
                    "delete",
                    "scaledobject",
                    "--all",
                    "--ignore-not-found",
                    "--wait=false",
                ],
                [
                    "kubectl",
                    "--context",
                    context,
                    "-n",
                    NAMESPACE,
                    "delete",
                    "hpa",
                    "--all",
                    "--ignore-not-found",
                    "--wait=false",
                ],
            ]
        )
    return [run_bounded(command, timeout_seconds=45) for command in commands]


def check_prometheus() -> dict[str, Any]:
    endpoints = [
        item.strip().rstrip("/")
        for item in os.environ["HEADLIGHT_PREFLIGHT_PROMETHEUS_URL"].split(",")
        if item.strip()
    ]
    results: list[dict[str, Any]] = []
    for endpoint in endpoints:
        started_at = utc_now()
        try:
            with urllib.request.urlopen(f"{endpoint}/-/ready", timeout=10) as response:
                body = response.read().decode(errors="replace")
                ok = 200 <= response.status < 300
            results.append(
                {
                    "endpoint": endpoint,
                    "ok": ok,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "response": body[-500:],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "endpoint": endpoint,
                    "ok": False,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "name": "prometheus",
        "ok": bool(results) and all(item["ok"] for item in results),
        "results": results,
    }


def wait_for_no_resources(
    name: str,
    context: str,
    resource: str,
    resource_names: tuple[str, ...] = (),
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    started_at = utc_now()
    deadline = time.monotonic() + timeout_seconds
    observations: list[dict[str, Any]] = []
    while True:
        command = ["kubectl", "--context", context, "-n", NAMESPACE, "get", resource]
        if resource_names:
            command.extend(resource_names)
            command.append("--ignore-not-found")
        command.extend(["-o", "name"])
        result = run_bounded(command, timeout_seconds=20)
        observations.append(result)
        output = result["output_tail"].strip()
        namespace_absent = "not found" in output.lower() and "namespace" in output.lower()
        resource_type_absent = "doesn't have a resource type" in output.lower()
        if namespace_absent or resource_type_absent or (
            result["exit_code"] == 0 and not result["timed_out"] and not output
        ):
            return {
                "name": name,
                "ok": True,
                "started_at": started_at,
                "finished_at": utc_now(),
                "observations": observations,
            }
        if time.monotonic() >= deadline:
            return {
                "name": name,
                "ok": False,
                "started_at": started_at,
                "finished_at": utc_now(),
                "observations": observations,
            }
        time.sleep(5)


def workload_readiness_check(cluster: str) -> dict[str, Any]:
    started_at = utc_now()
    discovery = run_bounded(
        [
            "kubectl",
            "--context",
            cluster,
            "-n",
            NAMESPACE,
            "get",
            "deployment",
            "-o",
            "name",
        ],
        timeout_seconds=30,
    )
    output = discovery["output_tail"]
    if discovery["exit_code"] != 0:
        namespace_absent = "not found" in output.lower() and "namespace" in output.lower()
        return {
            "name": f"{cluster}-workloads-ready",
            "ok": namespace_absent,
            "skipped": namespace_absent,
            "reason": "namespace_absent" if namespace_absent else "deployment_discovery_failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "discovery": discovery,
        }
    deployments = sorted(
        line.removeprefix("deployment.apps/").removeprefix("deployment/")
        for line in output.splitlines()
        if line.removeprefix("deployment.apps/").removeprefix("deployment/")
        not in CONTROLLER_DEPLOYMENTS
    )
    deployments = [name for name in deployments if name]
    if not deployments:
        return {
            "name": f"{cluster}-workloads-ready",
            "ok": True,
            "skipped": True,
            "reason": "no_workload_deployments_present",
            "started_at": started_at,
            "finished_at": utc_now(),
            "discovery": discovery,
        }
    wait = run_bounded(
        [
            "kubectl",
            "--context",
            cluster,
            "-n",
            NAMESPACE,
            "wait",
            *(f"deployment/{name}" for name in deployments),
            "--for=condition=Available",
            "--timeout=300s",
        ],
        timeout_seconds=330,
    )
    return {
        "name": f"{cluster}-workloads-ready",
        "ok": wait["exit_code"] == 0 and not wait["timed_out"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "deployments": deployments,
        "discovery": discovery,
        "wait": wait,
    }


def refresh_gateway_weights_if_possible() -> dict[str, Any]:
    started_at = utc_now()
    probes: list[dict[str, Any]] = []
    services = ("ts-auth-service", "ts-route-service", "ts-travel-service")
    for cluster in ("cluster2", "cluster3"):
        probe = run_bounded(
            [
                "kubectl",
                "--context",
                cluster,
                "-n",
                NAMESPACE,
                "get",
                "service",
                *services,
            ],
            timeout_seconds=30,
        )
        probes.append(probe)
        if probe["exit_code"] != 0 or probe["timed_out"]:
            return {
                "name": "gateway-weights-refresh",
                "ok": True,
                "skipped": True,
                "reason": "workload_services_not_present",
                "started_at": started_at,
                "finished_at": utc_now(),
                "service_probes": probes,
            }
    refresh = run_bounded(
        [sys.executable, "experiments/scripts/gateway_weights.py", "--once"],
        timeout_seconds=45,
    )
    return {
        "name": "gateway-weights-refresh",
        "ok": refresh["exit_code"] == 0 and not refresh["timed_out"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "service_probes": probes,
        "refresh": refresh,
    }


def recovery_gate() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def command_check(name: str, command: list[str], timeout_seconds: int) -> None:
        result = run_bounded(command, timeout_seconds=timeout_seconds)
        result["name"] = name
        result["ok"] = result["exit_code"] == 0 and not result["timed_out"]
        checks.append(result)

    for context in ("karmada-host", "karmada-apiserver", "cluster1", "cluster2", "cluster3"):
        command_check(
            f"{context}-api-ready",
            ["kubectl", "--context", context, "get", "--raw=/readyz"],
            20,
        )
    for cluster in ("cluster2", "cluster3"):
        command_check(
            f"{cluster}-member-ready",
            [
                "kubectl",
                "--context",
                "karmada-apiserver",
                "wait",
                f"cluster/{cluster}",
                "--for=condition=Ready",
                "--timeout=60s",
            ],
            75,
        )
    command_check(
        "gateway-ready",
        [
            "kubectl",
            "--context",
            "cluster1",
            "-n",
            NAMESPACE,
            "rollout",
            "status",
            "deployment/trainticket-gateway",
            "--timeout=120s",
        ],
        140,
    )
    command_check(
        "crp-controller-ready",
        [
            "kubectl",
            "--context",
            "karmada-host",
            "-n",
            "crp-system",
            "rollout",
            "status",
            "deployment/crp-controller-manager",
            "--timeout=120s",
        ],
        140,
    )
    command_check(
        "actions-handler-ready",
        [
            "kubectl",
            "--context",
            "karmada-host",
            "-n",
            "actions-handler-system",
            "rollout",
            "status",
            "statefulset/actions-handler",
            "--timeout=120s",
        ],
        140,
    )
    for cluster in ("cluster1", "cluster2", "cluster3"):
        checks.append(
            wait_for_no_resources(
                f"{cluster}-no-controller-deployments",
                cluster,
                "deployment",
                CONTROLLER_DEPLOYMENTS,
            )
        )
    for cluster in ("cluster2", "cluster3"):
        checks.append(workload_readiness_check(cluster))
        checks.append(wait_for_no_resources(f"{cluster}-no-hpa", cluster, "hpa"))
        checks.append(
            wait_for_no_resources(f"{cluster}-no-scaledobject", cluster, "scaledobject")
        )
    checks.append(
        wait_for_no_resources(
            "no-locust-job",
            "cluster1",
            "job",
            ("headlight-paper-locust",),
        )
    )
    checks.append(refresh_gateway_weights_if_possible())
    checks.append(check_prometheus())
    failed = [check["name"] for check in checks if not check.get("ok")]
    return {
        "status": "passed" if not failed else "failed",
        "started_at": checks[0]["started_at"] if checks else utc_now(),
        "finished_at": utc_now(),
        "failed_checks": failed,
        "checks": checks,
    }


def command_for_cell(
    seed: int, workload: str, controller: str, output: Path
) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    python_path = REPO_ROOT / ".venv/bin/python"
    python = str(python_path) if python_path.is_file() else sys.executable
    if controller == "headlight-v2":
        config = (
            f"experiments/configs/qualification/"
            f"headlight-v2-emergency-disabled-{workload}-seed-{seed}-{RUN_DATE}.yaml"
        )
        if HEADLIGHT_V2_EXTENSION:
            config = (f"experiments/configs/headlight-v2-extension/"
                      f"headlight-v2-emergency-disabled-{workload}-seed-{seed}.yaml")
        return (
            [
                python,
                "experiments/run_headlight_v2_live.py",
                "--run",
                config,
                "--output-dir",
                relative(output),
            ],
            environment,
        )
    config = f"experiments/configs/runs/{controller}-{workload}-multi-cluster-headlight.yaml"
    environment["HEADLIGHT_EXPERIMENT_OUTPUT_DIR"] = relative(output)
    return (
        [
            python,
            "experiments/run_experiment.py",
            "--run",
            config,
            "--seeds",
            str(seed),
        ],
        environment,
    )


def classify_failure(log_path: Path, exit_code: int, timed_out: bool) -> str:
    if extension_campaign() and _memory_guard_dir is not None and (_memory_guard_dir / "ABORT").exists():
        return "host_resource_guard_abort"
    if resource_guard_failed():
        return "host_resource_guard_failed"
    if _received_signal is not None:
        return f"campaign_interrupted:signal_{_received_signal}"
    if timed_out:
        return "campaign_cell_timeout"
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
        if log_path.is_file()
        else ""
    )
    if "Locust job did not complete within" in text or "timed out waiting for the condition" in text:
        return "infrastructure_api_timeout:locust_completion_wait"
    if "TLS handshake timeout" in text or "connection refused" in text.lower():
        return "kubernetes_api_unavailable"
    return f"runner_exit:{exit_code}"


def run_cell(seed: int, workload: str, controller: str) -> tuple[bool, bool]:
    attempt, output = next_attempt(seed, workload, controller)
    output.mkdir(parents=True)
    cell_log = output / "campaign-cell.log"
    status_file = status_path(seed, workload, controller, attempt)
    command, environment = command_for_cell(seed, workload, controller, output)
    timeout_seconds = int(
        os.environ.get(
            "HEADLIGHT_CAMPAIGN_CELL_TIMEOUT_SECONDS",
            str(WARMUP_SECONDS + MEASUREMENT_SECONDS[workload] + 3600),
        )
    )
    status: dict[str, Any] = {
        "schema_version": "headlight.campaign-cell-status/v1",
        "cell": {"seed": seed, "workload": workload, "controller": controller},
        "attempt": attempt,
        "output_dir": relative(output),
        "status": "running",
        "reason": "",
        "retry_allowed": False,
        "started_at": utc_now(),
        "finished_at": None,
        "command": command,
        "timeout_seconds": timeout_seconds,
        "expected_trace_sha256": EXPECTED_TRACES[workload],
    }
    if extension_campaign():
        status["environment_provenance"] = _extension_provenance
    write_json(status_file, status)
    print(f"Running seed {seed}, {workload}, {controller}, attempt {attempt}", flush=True)
    exit_code, timed_out = stream_process(
        command,
        env=environment,
        log_path=cell_log,
        timeout_seconds=timeout_seconds,
    )
    valid, validation_reason = summary_result(
        output, seed, EXPECTED_TRACES[workload], controller=controller, workload=workload,
    )
    if exit_code == 0 and valid and not resource_guard_failed():
        status.update(
            {
                "status": "valid",
                "reason": validation_reason,
                "retry_allowed": False,
                "finished_at": utc_now(),
                "exit_code": exit_code,
                "timed_out": timed_out,
            }
        )
        write_json(status_file, status)
        print(
            f"CELL_VALID seed={seed} workload={workload} controller={controller} attempt={attempt}",
            flush=True,
        )
        return True, True

    reason = (classify_failure(cell_log, exit_code, timed_out)
              if exit_code != 0 or resource_guard_failed() else validation_reason)
    status.update(
        {
            "status": "invalid",
            "reason": reason,
            "retry_allowed": True,
            "finished_at": utc_now(),
            "exit_code": exit_code,
            "timed_out": timed_out,
        }
    )
    write_json(status_file, status)
    print(
        f"CELL_INVALID seed={seed} workload={workload} controller={controller} "
        f"attempt={attempt} reason={reason}",
        flush=True,
    )
    cleanup = cleanup_after_failure()
    recovery = recovery_gate()
    status["cleanup"] = cleanup
    status["recovery"] = recovery
    write_json(status_file, status)
    recovered = recovery["status"] == "passed"
    if recovered:
        print("RECOVERY_PASSED continuing campaign", flush=True)
    else:
        print(
            f"RECOVERY_FAILED checks={','.join(recovery['failed_checks'])}; stopping campaign",
            flush=True,
        )
    return False, recovered


def verify_controller_group(seed: int, workload: str) -> bool:
    outputs = {controller: valid_attempt(seed, workload, controller) for controller in CONTROLLERS}
    if any(output is None for output in outputs.values()):
        return False
    hashes = {
        controller: sha256(output / f"seed-{seed}/raw/load_profile.csv")
        for controller, output in outputs.items()
        if output is not None
    }
    expected = EXPECTED_TRACES[workload]
    if set(hashes.values()) != {expected}:
        raise RuntimeError(
            f"controller trace mismatch for seed={seed} workload={workload}: {hashes}"
        )
    print(f"TRACE_GROUP_VALID seed={seed} workload={workload} trace={expected}", flush=True)
    return True


def campaign_summary(status: str, reason: str = "") -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        for workload in WORKLOADS:
            for controller in CONTROLLERS:
                output = valid_attempt(seed, workload, controller)
                cells.append(
                    {
                        "seed": seed,
                        "workload": workload,
                        "controller": controller,
                        "status": "valid" if output is not None else "remaining",
                        "valid_output_dir": relative(output) if output is not None else None,
                        "attempt_count": len(existing_attempts(seed, workload, controller)),
                    }
                )
    valid_count = sum(cell["status"] == "valid" for cell in cells)
    return {
        "schema_version": "headlight.campaign-status/v1",
        "status": status,
        "reason": reason,
        "updated_at": utc_now(),
        "expected_cell_count": len(SEEDS) * len(WORKLOADS) * len(CONTROLLERS),
        "valid_cell_count": valid_count,
        "remaining_cell_count": len(cells) - valid_count,
        "cells": cells,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip valid cells and put retries in fresh numbered attempt directories",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--baseline-extension", action="store_true",
                        help="run only HPA/KEDA, seeds 104/105, against the retained C=4 protocol")
    modes.add_argument("--headlight-v2-extension", action="store_true",
                       help="run only Headlight, seeds 104/105, for all three C=4 workloads")
    parser.add_argument("--campaign-dir", type=Path,
                        help="fresh repository results directory; required for an extension")
    parser.add_argument("--dry-run", action="store_true",
                        help="extension only: generate and verify offline bundles without live operations")
    args = parser.parse_args(argv)
    extension = args.baseline_extension or args.headlight_v2_extension
    if (args.campaign_dir is not None or args.dry_run) and not extension:
        parser.error("--campaign-dir and --dry-run require an extension mode")
    if extension and args.campaign_dir is None:
        parser.error("extension mode requires --campaign-dir")
    return args


def configure_campaign(args: argparse.Namespace) -> None:
    global BASELINE_EXTENSION, HEADLIGHT_V2_EXTENSION, SEEDS, CONTROLLERS, CAMPAIGN_STATE_DIR
    if not args.baseline_extension and not args.headlight_v2_extension:
        return
    root = validate_extension_root(args.campaign_dir)
    BASELINE_EXTENSION = args.baseline_extension
    HEADLIGHT_V2_EXTENSION = args.headlight_v2_extension
    SEEDS = (104, 105)
    CONTROLLERS = ("headlight-v2",) if HEADLIGHT_V2_EXTENSION else ("hpa", "keda")
    CAMPAIGN_STATE_DIR = root


def validate_extension_root(path: Path) -> Path:
    root = path.resolve()
    if not root.is_relative_to(REPO_ROOT / "results") or root == REPO_ROOT / "results":
        raise ValueError("campaign directory must be a subdirectory of repository results")
    if any(root.is_relative_to(REPO_ROOT / "results" / name)
           for name in ("headlight-v2-live", "socc26", "capacity-qualification")):
        raise ValueError("refusing to use a historical results directory")
    return root


def headlight_v2_extension_plan(dry_run: bool) -> dict[str, Any]:
    global _extension_provenance
    from experiments.lib.baseline_extension import baseline_environment
    from experiments.validate_headlight_v2_live_configs import validate_extension_config

    reference = REPO_ROOT / "experiments/assets/headlight-v2/execution-environment.json"
    retained = json.loads(reference.read_text())
    configured = baseline_environment(CAMPAIGN_STATE_DIR)
    pinned = {key: value for key, value in retained.items()
              if key.startswith("HEADLIGHT_V2_") or key in ("HEADLIGHT_CODE_REVISION", "HEADLIGHT_IMAGE_REVISION")}
    if not dry_run:
        for key, value in pinned.items():
            if os.environ.get(key) != value:
                raise ValueError(f"{key} differs from retained Headlight settings; regenerate campaign-env.sh")
    configured.update(pinned)
    configured.update({
        "HEADLIGHT_V2_BUNDLE_DIR": str(REPO_ROOT / "results/headlight-v2-live/bundle-full-seed-601"),
        "HEADLIGHT_V2_POLICY_CONFIG": str(REPO_ROOT / "experiments/configs/models/headlight-v2-closed-loop-policy.json"),
        "HEADLIGHT_V2_SHADOW_EVIDENCE": os.environ.get("HEADLIGHT_V2_SHADOW_EVIDENCE",
                                                     str(CAMPAIGN_STATE_DIR / "shadow-smoke.ok")),
    })
    # As with the baseline launcher, remove inherited development overrides.
    for key in list(os.environ):
        if key.startswith("HEADLIGHT_") and key not in configured:
            del os.environ[key]
    os.environ.update(configured)
    configs = []
    cells = []
    for seed in SEEDS:
        for workload in WORKLOADS:
            output = (CAMPAIGN_STATE_DIR / "dry-run" / f"{workload}-seed-{seed}" / "headlight-v2"
                      if dry_run else attempt_dir(seed, workload, "headlight-v2", 1))
            command, _ = command_for_cell(seed, workload, "headlight-v2", output)
            config = REPO_ROOT / command[command.index("--run") + 1]
            spec = validate_extension_config(config)["spec"]
            configs.append({"path": relative(config), "sha256": sha256(config)})
            cells.append({"seed": seed, "workload": workload, "controller": "headlight-v2",
                          "output_dir": relative(output), "command": command,
                          "expected_trace_sha256": spec["expected_comparator_trace_sha256"]})
    _extension_provenance = {"reference_path": relative(reference), "reference_sha256": sha256(reference),
                             "environment": configured, "configs": configs}
    return {"schema_version": "headlight.v2-extension-plan/v1", "cells": cells,
            "seeds": list(SEEDS), "controllers": list(CONTROLLERS), "configs": configs,
            "environment": configured, "environment_provenance": _extension_provenance,
            "minimum_runtime_seconds": 2 * sum(MEASUREMENT_SECONDS.values()) + 6 * (WARMUP_SECONDS + 60),
            "environment_equivalence_established": False,
            "comparability_note": "Follow-up cells preserve the v2 model, policy and workload protocol. "
                                  "Host, storage and image equivalence require separate verification.",
            "dry_run": dry_run}


def extension_plan(dry_run: bool) -> dict[str, Any]:
    if HEADLIGHT_V2_EXTENSION:
        return headlight_v2_extension_plan(dry_run)
    global _extension_provenance
    from experiments.lib.baseline_extension import audit_baselines, baseline_environment

    # Explicitly replace stale experimental overrides; retain OS/tool settings.
    configured = baseline_environment(CAMPAIGN_STATE_DIR)
    for key in list(os.environ):
        if key.startswith("HEADLIGHT_") and key not in configured:
            del os.environ[key]
    os.environ.update(configured)
    audit = audit_baselines()
    _extension_provenance = None
    if not dry_run:
        provenance = json.loads((CAMPAIGN_STATE_DIR / "environment-provenance.json").read_text())
        snapshot = REPO_ROOT / provenance["snapshot_path"]
        if json.loads(snapshot.read_text()) != provenance or provenance["environment"] != configured:
            raise ValueError("prepared environment differs; regenerate campaign environment")
        _extension_provenance = {"path": relative(snapshot), "sha256": sha256(snapshot)}
    cells = []
    for seed in SEEDS:
        for workload in WORKLOADS:
            for controller in CONTROLLERS:
                output = (CAMPAIGN_STATE_DIR / "dry-run" / f"{workload}-seed-{seed}" / controller
                          if dry_run else attempt_dir(seed, workload, controller, 1))
                command, _ = command_for_cell(seed, workload, controller, output)
                cells.append({"seed": seed, "workload": workload, "controller": controller,
                              "output_dir": relative(output), "command": command,
                              "expected_trace_sha256": EXPECTED_TRACES[workload]})
    return {"schema_version": "headlight.baseline-extension-plan/v1", "cells": cells,
            "minimum_runtime_seconds": 4 * sum(MEASUREMENT_SECONDS.values()) + 12 * (WARMUP_SECONDS + 60),
            "baseline_audit": audit, "environment_provenance": _extension_provenance, "dry_run": dry_run}


def dry_run_extension(plan: dict[str, Any]) -> int:
    for cell in plan["cells"]:
        output = REPO_ROOT / cell["output_dir"]
        if output.exists():
            raise RuntimeError(f"refusing to overwrite dry-run bundle: {output}")
        _, environment = command_for_cell(cell["seed"], cell["workload"], cell["controller"], output)
        result = subprocess.run([*cell["command"], "--dry-run"], cwd=REPO_ROOT, env=environment, check=False)
        if result.returncode:
            return result.returncode
        actual = sha256(output / f"seed-{cell['seed']}" / "raw/load_profile.csv")
        if actual != cell["expected_trace_sha256"]:
            raise RuntimeError(f"dry-run trace mismatch: {cell}")
    print(f"DRY_RUN_COMPLETE {len(plan['cells'])} {'Headlight' if HEADLIGHT_V2_EXTENSION else 'HPA/KEDA'} "
          "bundles; no live experiment started", flush=True)
    return 0


def start_memory_guard(stack: ExitStack) -> None:
    """Reuse the qualified host guard; stop it when this launcher exits."""
    global _memory_guard, _memory_guard_dir
    _memory_guard_dir = CAMPAIGN_STATE_DIR / "guards" / str(time.time_ns())
    _memory_guard_dir.mkdir(parents=True)
    log = stack.enter_context((_memory_guard_dir / "guard.log").open("w"))
    python_path = REPO_ROOT / ".venv/bin/python"
    _memory_guard = subprocess.Popen([
        str(python_path) if python_path.is_file() else sys.executable,
        "experiments/scripts/qualification_memory_guard.py",
        "--output-dir", str(_memory_guard_dir), "--mem-limit-gib", "6",
        "--mem-grace-seconds", "30", "--tmp-limit-gib", "2", "--interval-seconds", "2",
    ], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def stop_guard() -> None:
        if _memory_guard is not None and _memory_guard.poll() is None:
            (_memory_guard_dir / "monitor-stop").touch()
            try:
                # ABORT precedes Docker discovery/stop. Keep testbed locks until
                # that bounded shutdown completes, including its child process.
                _memory_guard.wait(timeout=90 if (_memory_guard_dir / "ABORT").exists() else 15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(_memory_guard.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    _memory_guard.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(_memory_guard.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    _memory_guard.wait(timeout=15)
        if _memory_guard is not None:
            # Reap the guard above, then stop any descendant surviving its exit.
            try:
                os.killpg(_memory_guard.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    stack.callback(stop_guard)
    deadline = time.monotonic() + 10
    while not (_memory_guard_dir / "memory-guard-config.json").exists():
        if _memory_guard.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError("host resource guard failed to start")
        time.sleep(0.1)


def resource_guard_failed() -> bool:
    return extension_campaign() and (
        (_memory_guard_dir is not None and (_memory_guard_dir / "ABORT").exists())
        or (_memory_guard is not None and _memory_guard.poll() is not None)
    )


def main() -> int:
    args = parse_args()
    configure_campaign(args)
    plan = extension_plan(args.dry_run) if extension_campaign() else None
    if not args.dry_run:
        validate_environment()
    CAMPAIGN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = CAMPAIGN_STATE_DIR / "campaign.lock"
    with ExitStack() as stack:
        lock = stack.enter_context(lock_path.open("a", encoding="utf-8"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if extension_campaign() and not args.dry_run:
                # Hold the original launcher's lock too: both campaigns share a testbed.
                LEGACY_STATE_DIR.mkdir(parents=True, exist_ok=True)
                shared = stack.enter_context((LEGACY_STATE_DIR / "campaign.lock").open("a", encoding="utf-8"))
                fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another campaign process holds the testbed lock") from exc

        def record_failure(exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
            if exc is not None:
                status = "interrupted" if isinstance(exc, CampaignInterrupted) else "failed"
                summary = ({"status": status, "reason": str(exc), "dry_run": True}
                           if args.dry_run else campaign_summary(status, f"{type(exc).__name__}: {exc}"))
                write_json(CAMPAIGN_STATE_DIR / ("dry-run-status.json" if args.dry_run else "campaign-status.json"), summary)
            return False

        # Runs while our locks are held. A launcher denied a lock writes nothing.
        stack.push(record_failure)

        if plan is not None:
            if HEADLIGHT_V2_EXTENSION and not args.dry_run:
                identity = {key: plan[key] for key in ("environment", "configs", "seeds", "controllers")}
                snapshot = CAMPAIGN_STATE_DIR / "campaign-inputs.json"
                if snapshot.exists() and json.loads(snapshot.read_text()) != identity:
                    raise RuntimeError("v2 campaign inputs changed; use a fresh campaign directory")
                if not snapshot.exists():
                    write_json(snapshot, identity)
            plan_path = CAMPAIGN_STATE_DIR / ("campaign-plan.json" if args.dry_run else f"launch-plan-{time.time_ns()}.json")
            write_json(plan_path, plan)
        if args.dry_run:
            return dry_run_extension(plan)
        if HEADLIGHT_V2_EXTENSION:
            print(f"CAMPAIGN_START controller=headlight-v2 seeds=104,105 cells=6 "
                  f"output={CAMPAIGN_STATE_DIR}", flush=True)

        existing = [
            output
            for seed in SEEDS
            for workload in WORKLOADS
            for controller in CONTROLLERS
            for _attempt, output in existing_attempts(seed, workload, controller)
        ]
        if existing and not args.resume:
            raise RuntimeError("existing campaign outputs found; rerun with --resume")

        for seed in SEEDS:
            for workload in WORKLOADS:
                for controller in CONTROLLERS:
                    if resource_guard_failed():
                        raise RuntimeError("host resource guard aborted between cells")
                    valid_attempt(seed, workload, controller)

        incomplete_existing = any(
            existing_attempts(seed, workload, controller)
            and valid_attempt(seed, workload, controller) is None
            for seed in SEEDS
            for workload in WORKLOADS
            for controller in CONTROLLERS
        )
        if HEADLIGHT_V2_EXTENSION:
            print("Checking testbed cleanup and recovery before the first cell", flush=True)
        initial_cleanup = cleanup_after_failure() if extension_campaign() or (incomplete_existing and args.resume) else []
        initial_recovery = recovery_gate()
        write_json(
            CAMPAIGN_STATE_DIR / "initial-recovery.json",
            {"cleanup": initial_cleanup, "recovery": initial_recovery, "recorded_at": utc_now()},
        )
        if initial_recovery["status"] != "passed":
            summary = campaign_summary(
                "blocked",
                f"initial recovery failed: {','.join(initial_recovery['failed_checks'])}",
            )
            write_json(CAMPAIGN_STATE_DIR / "campaign-status.json", summary)
            print(f"CAMPAIGN_BLOCKED {summary['reason']}", flush=True)
            return 2

        write_json(
            CAMPAIGN_STATE_DIR / "campaign-status.json",
            campaign_summary("running"),
        )

        if extension_campaign():
            start_memory_guard(stack)

        invalid_seen = False
        for seed in SEEDS:
            for workload in WORKLOADS:
                for controller in CONTROLLERS:
                    existing_valid = valid_attempt(seed, workload, controller)
                    if existing_valid is not None:
                        print(
                            f"CELL_SKIPPED_VALID seed={seed} workload={workload} "
                            f"controller={controller} output={relative(existing_valid)}",
                            flush=True,
                        )
                        continue
                    if resource_guard_failed():
                        raise RuntimeError("host resource guard aborted between cells")
                    valid, recovered = run_cell(seed, workload, controller)
                    invalid_seen = invalid_seen or not valid
                    write_json(
                        CAMPAIGN_STATE_DIR / "campaign-status.json",
                        campaign_summary("running"),
                    )
                    if _received_signal is not None:
                        raise CampaignInterrupted(f"received signal {_received_signal}")
                    if extension_campaign() and not valid:
                        summary = campaign_summary("incomplete", "technical failure; inspect evidence before --resume")
                        write_json(CAMPAIGN_STATE_DIR / "campaign-status.json", summary)
                        return 1
                    if not recovered:
                        summary = campaign_summary(
                            "blocked",
                            f"recovery failed after seed={seed} workload={workload} controller={controller}",
                        )
                        write_json(CAMPAIGN_STATE_DIR / "campaign-status.json", summary)
                        return 2
                verify_controller_group(seed, workload)

        if resource_guard_failed():
            raise RuntimeError("host resource guard aborted before campaign completion")
        summary = campaign_summary("incomplete" if invalid_seen else "complete")
        if invalid_seen:
            summary["reason"] = "one or more attempts were invalid; rerun with --resume"
        if summary["remaining_cell_count"] == 0:
            summary["status"] = "complete"
            summary["reason"] = ""
            write_json(CAMPAIGN_STATE_DIR / "campaign-status.json", summary)
            print(f"CAMPAIGN_COMPLETE all {summary['expected_cell_count']} cells are valid", flush=True)
            return 0
        write_json(CAMPAIGN_STATE_DIR / "campaign-status.json", summary)
        print(
            f"CAMPAIGN_INCOMPLETE valid={summary['valid_cell_count']} "
            f"remaining={summary['remaining_cell_count']}; rerun with --resume",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        raise SystemExit(main())
    except CampaignInterrupted as exc:
        print(f"CAMPAIGN_INTERRUPTED {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"CAMPAIGN_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
