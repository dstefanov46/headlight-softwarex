#!/usr/bin/env python3
"""Run selected Threshold evaluation cells as explicitly evaluation evidence."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ai-module/ai-module"))

from experiments.lib.baseline_extension import TRACE_HASHES, baseline_environment

PREPARATION_ROOT = REPO_ROOT / "results/threshold-preparation/20261011T071451Z"
PREPARED_SETTINGS = PREPARATION_ROOT / "prepared-settings.json"
SMOKE_REPORT = PREPARATION_ROOT / "crp-smoke-verified-cleanup/report.json"
DEFAULT_CELLS = (
    (104, "diurnal"),
    (104, "flash-crowd"),
    (105, "gradual-ramp"),
    (105, "diurnal"),
    (105, "flash-crowd"),
)
EVALUATION_SEEDS = (101, 102, 103, 104, 105)
WORKLOADS = tuple(TRACE_HASHES)
RUN_FILES = {name: f"experiments/configs/runs/threshold-{name}-multi-cluster-headlight.yaml"
             for name in TRACE_HASHES}
NAMESPACE = "trainticket"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_cells_for_seeds(seeds: list[int] | tuple[int, ...]) -> tuple[tuple[int, str], ...]:
    selected = tuple(int(seed) for seed in seeds)
    if not selected:
        raise ValueError("at least one evaluation seed is required")
    if len(selected) != len(set(selected)):
        raise ValueError("evaluation seed selection cannot contain duplicates")
    unsupported = sorted(set(selected) - set(EVALUATION_SEEDS))
    if unsupported:
        raise ValueError(f"evaluation seeds must be selected from {EVALUATION_SEEDS}; got {unsupported}")
    return tuple((seed, workload) for seed in sorted(selected) for workload in WORKLOADS)


def campaign_identity(
    settings: dict[str, str], cells: tuple[tuple[int, str], ...] = DEFAULT_CELLS,
) -> dict[str, Any]:
    return {
        "schema_version": "headlight.threshold-evaluation-campaign/v1",
        "scope": "evaluation_not_matched",
        "controller": "threshold",
        "capacity_users": 4,
        "capacity_qualified": False,
        "matched_comparison_eligible": False,
        "historical_gateway_image_verified": False,
        "threshold_image": settings["HEADLIGHT_THRESHOLD_IMAGE"],
        "threshold_image_digest": settings["HEADLIGHT_THRESHOLD_IMAGE_ID"],
        "cells": [
            {"seed": seed, "workload": workload, "trace_sha256": TRACE_HASHES[workload]}
            for seed, workload in cells
        ],
    }


def validate_campaign_root(path: Path, *, resume: bool) -> Path:
    root = path.expanduser().resolve()
    allowed = (REPO_ROOT / "results/threshold-evaluation").resolve()
    if root == allowed or not root.is_relative_to(allowed):
        raise ValueError(f"campaign output must be a new child of {allowed}")
    if root.exists():
        if not resume:
            raise ValueError(f"campaign directory already exists; use --resume to continue: {root}")
        if not (root / "campaign.json").is_file():
            raise ValueError(f"cannot resume without campaign.json: {root}")
    elif resume:
        raise ValueError(f"cannot resume a campaign directory that does not exist: {root}")
    return root


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def initialize_environment(root: Path, prepared_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    settings = json.loads(prepared_path.read_text(encoding="utf-8"))
    required = {
        "HEADLIGHT_THRESHOLD_IMAGE", "HEADLIGHT_THRESHOLD_IMAGE_ID",
        "HEADLIGHT_GATEWAY_PROTOCOL_PATH", "HEADLIGHT_TRAINTICKET_PATH",
        "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE", "HEADLIGHT_TRAINTICKET_MONGO_STORAGE",
        "HEADLIGHT_EVALUATION_CAPACITY", "HEADLIGHT_SHADOW_MODE",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"prepared Threshold settings are missing: {', '.join(missing)}")
    for key, expected in {
        "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
        "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
        "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
        "HEADLIGHT_EVALUATION_CAPACITY": "4",
        "HEADLIGHT_SHADOW_MODE": "0",
    }.items():
        if str(settings[key]) != expected:
            raise ValueError(f"prepared setting {key} must be {expected!r}")
    smoke = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    smoke_pods = json.loads((SMOKE_REPORT.parent / "controller-pods.json").read_text(encoding="utf-8"))
    smoke_images = [
        container
        for pod in smoke_pods.get("items", [])
        for container in pod.get("status", {}).get("containerStatuses", [])
        if container.get("name") == "autoscaler"
    ]
    expected_digest = settings["HEADLIGHT_THRESHOLD_IMAGE_ID"]
    if (
        not smoke.get("passed")
        or smoke.get("image") != settings["HEADLIGHT_THRESHOLD_IMAGE"]
        or smoke.get("evaluation_run") is not False
        or smoke.get("synthetic_metrics") is not True
        or smoke.get("cleanup_errors")
        or not smoke_images
        or any(
            container.get("image", "").removeprefix("docker.io/") != settings["HEADLIGHT_THRESHOLD_IMAGE"]
            or container.get("imageID", "").split("@")[-1] != expected_digest
            for container in smoke_images
        )
    ):
        raise ValueError("Threshold CRP actuation smoke does not match the prepared image")

    inherited = {
        key: os.environ[key]
        for key in ("KUBECONFIG", "HEADLIGHT_KUBECONFIG", "HEADLIGHT_INTERNAL_KUBECONFIG", "HEADLIGHT_KARMADA_DATA")
        if os.environ.get(key)
    }
    for key in list(os.environ):
        if key.startswith("HEADLIGHT_"):
            del os.environ[key]
    os.environ.update(inherited)
    env = baseline_environment(root)
    env.update({key: str(value) for key, value in settings.items()})
    env.update({
        "HEADLIGHT_CAPACITY_CONTRACT": str(root / "capacity-contract.json"),
        "HEADLIGHT_CONTROLLER_CONTEXT": "cluster3",
        "HEADLIGHT_SHUTDOWN_ON_EXIT": "0",
        "HEADLIGHT_STRICT_FINAL_READINESS": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    })
    env["PATH"] = str(REPO_ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.update(env)
    image_check = subprocess.run(
        ["docker", "image", "inspect", settings["HEADLIGHT_THRESHOLD_IMAGE"]],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False, timeout=30,
    )
    if image_check.returncode:
        raise RuntimeError(f"prepared Threshold image is not available locally: {image_check.stderr.strip()}")
    # Importing capacity qualification imports run_experiment, whose gateway
    # protocol is resolved from the environment at module import time.
    from experiments.run_capacity_qualification import preflight_monitoring, preflight_testbed

    preflight = {
        "testbed": preflight_testbed(Path(env["HEADLIGHT_KUBECONFIG"])),
        "monitoring": preflight_monitoring(),
        "threshold_smoke": smoke,
        "prepared_settings_sha256": sha256(prepared_path),
        "prepared_settings_path": str(prepared_path.resolve()),
    }
    verify_testbed_idle_for_campaign()
    return env, preflight


def kubectl(context: str, *args: str, timeout: int = 30) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", "--context", context, "--request-timeout=15s", *args, "-o", "json"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True, timeout=timeout,
    )
    return json.loads(result.stdout)


def verify_testbed_idle_for_campaign() -> None:
    for resource in ("job/headlight-paper-locust", "configmap/headlight-paper-locust"):
        result = subprocess.run(
            ["kubectl", "--context", "cluster1", "-n", NAMESPACE,
             "get", resource, "-o", "json"],
            cwd=REPO_ROOT, text=True, capture_output=True, check=False, timeout=20,
        )
        if result.returncode == 0:
            raise RuntimeError(f"refusing to start while {resource} already exists in cluster1")
        if "not found" not in result.stderr.lower():
            raise RuntimeError(f"cannot verify that {resource} is absent: {result.stderr.strip()}")
    result = subprocess.run(
        ["kubectl", "--context", "cluster3", "-n", NAMESPACE,
         "get", "deployment/threshold-autoscaler", "--ignore-not-found", "-o", "json"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True, timeout=20,
    )
    if result.stdout.strip():
        deployment = json.loads(result.stdout)
        if int(deployment.get("spec", {}).get("replicas", 0)) != 0:
            raise RuntimeError("refusing to start while the Threshold controller deployment is active")


def install_readiness_guard(
    runner, memory_guard: subprocess.Popen[str], guard_dir: Path,
) -> None:
    original_deploy = runner.deploy_workload
    original_health = runner.RunEvidence.check_measurement_phase

    def deploy_verified(manifest: dict[str, Any], seed_dir: Path) -> None:
        original_deploy(manifest, seed_dir)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            bindings = kubectl("karmada-apiserver", "-n", NAMESPACE, "get", "resourcebindings.work.karmada.io")
            deployments = [item for item in bindings["items"]
                           if item.get("spec", {}).get("resource", {}).get("kind") == "Deployment"]
            all_applied = len(deployments) == 18 and all(
                any(condition.get("type") == "FullyApplied" and condition.get("status") == "True"
                    for condition in item.get("status", {}).get("conditions", []))
                for item in deployments
            )
            if all_applied:
                members = {cluster: kubectl(cluster, "-n", NAMESPACE, "get", "deployments")
                           for cluster in ("cluster2", "cluster3")}
                expected_names = {
                    item.get("spec", {}).get("resource", {}).get("name")
                    for item in deployments
                }
                all_ready = all(
                    len(expected_names) == 18
                    and len([item for item in member["items"]
                             if item.get("metadata", {}).get("name") in expected_names]) == 18
                    and all(
                        deployment.get("spec", {}).get("replicas") == 1
                        and deployment.get("status", {}).get("readyReplicas") == 1
                        for deployment in member["items"]
                        if deployment.get("metadata", {}).get("name") in expected_names
                    ) for member in members.values()
                )
                if all_ready:
                    atomic_json(seed_dir / "raw/ownership-readiness.json", {
                        "all18_bindings_fully_applied": True,
                        "all_member_initial_placements_ready": True,
                        "bindings": bindings,
                        "members": members,
                    })
                    return
            time.sleep(2)
        raise RuntimeError("evaluation campaign refused load: Karmada bindings/member placements did not become ready")

    def check_guard(self) -> None:
        if (guard_dir / "ABORT").exists():
            raise RuntimeError("campaign host memory guard aborted")
        if memory_guard.poll() is not None and memory_guard.returncode != 0:
            raise RuntimeError(f"campaign host memory guard exited with {memory_guard.returncode}")
        original_health(self)

    runner.deploy_workload = deploy_verified
    runner.RunEvidence.check_measurement_phase = check_guard


def start_metrics_observer(root: Path) -> tuple[threading.Event, threading.Thread]:
    from headlight.features import BASE_FEATURES, PrometheusMetricCollector
    from headlight.profile import load_service_profile

    stop = threading.Event()

    def observe() -> None:
        profile = load_service_profile(str(REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json"))
        path = root / "metric-observations.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            while not stop.is_set():
                timestamp = int(time.time())
                try:
                    urls = []
                    for cluster in ("cluster2", "cluster3"):
                        node = kubectl(cluster, "get", "nodes")["items"][0]
                        address = next(item["address"] for item in node["status"]["addresses"]
                                       if item["type"] == "InternalIP")
                        service = kubectl(cluster, "-n", "monitoring", "get", "service",
                                          "prometheus-kube-prometheus-prometheus")
                        port = next(item["nodePort"] for item in service["spec"]["ports"] if item["port"] == 9090)
                        urls.append(f"http://{address}:{port}")
                    collector = PrometheusMetricCollector(",".join(urls), profile, namespace=NAMESPACE,
                                                          timeout_seconds=3, capture_observations=True)
                    sample = collector.collect(timestamp)
                    observations = {
                        node.name: {
                            feature: {
                                "value": None if sample.missing[index, BASE_FEATURES.index(feature)]
                                else float(sample.values[index, BASE_FEATURES.index(feature)]),
                                "age_seconds": int(sample.ages_seconds[index, BASE_FEATURES.index(feature)]),
                            }
                            for feature in ("p95_latency_ms", "cpu_utilization")
                        }
                        for index, node in enumerate(profile.nodes) if node.scalable
                    }
                    row = {"timestamp": timestamp, "endpoints": urls, "observations": observations,
                           "query_evidence": collector.observation_evidence}
                except Exception as exc:  # preserve a timestamped telemetry collection failure
                    row = {"timestamp": timestamp, "error": f"{type(exc).__name__}: {exc}"}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                stop.wait(30)

    thread = threading.Thread(target=observe, name="threshold-evaluation-metrics", daemon=True)
    thread.start()
    return stop, thread


def start_memory_guard(guard_dir: Path) -> tuple[subprocess.Popen[str], Any]:
    guard_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    guard_log = (guard_dir / "guard.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "experiments/scripts/qualification_memory_guard.py"),
         "--output-dir", str(guard_dir)],
        cwd=REPO_ROOT, stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    return process, guard_log


def stop_memory_guard(process: subprocess.Popen[str], log) -> None:
    process.poll()
    if process.returncode is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log.close()


def cleanup_load_resources(attempt_root: Path) -> bool:
    result = subprocess.run(
        ["kubectl", "--context", "cluster1", "-n", NAMESPACE,
         "delete", "job/headlight-paper-locust", "configmap/headlight-paper-locust",
         "--ignore-not-found", "--wait=true", "--timeout=45s"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False, timeout=55,
    )
    (attempt_root / "load-cleanup.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode == 0


def acquire_testbed_locks() -> ExitStack:
    paths = (
        REPO_ROOT / "results/hpa-keda-extension/seed104-105-20261010T175756Z/campaign.lock",
        REPO_ROOT / "results/headlight-v2-live/qualification-seed102-103-20261004/campaign.lock",
        REPO_ROOT / "results/threshold-evaluation/.active-campaign.lock",
    )
    stack = ExitStack()
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a", encoding="utf-8"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stack.close()
        raise RuntimeError("another HPA/KEDA, Headlight, or Threshold campaign holds the shared testbed lock") from exc
    return stack


def _attempts(cell_root: Path) -> list[Path]:
    return sorted(cell_root.glob("attempt-*"), key=lambda path: int(path.name.removeprefix("attempt-")))


def _valid_attempts(cell_root: Path, seed: int) -> list[Path]:
    return [attempt for attempt in _attempts(cell_root) if _is_valid_attempt(attempt, seed)]


def _is_valid_attempt(attempt: Path, seed: int) -> bool:
    status_path = attempt / "campaign-cell-status.json"
    output = attempt / "output"
    if not status_path.is_file() or not (output / "manifest.yaml").is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
        if status.get("status") != "valid":
            return False
        from experiments.lib.quality import validate_bundle
        return bool(validate_bundle(output / f"seed-{seed}", manifest, dry_run=False).get("valid"))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def run_cell(
    runner, root: Path, seed: int, workload: str, status: dict[str, Any],
    cells: tuple[tuple[int, str], ...],
) -> dict[str, Any]:
    cell_root = root / "cells" / f"{workload}-seed-{seed}"
    previous = _attempts(cell_root)
    valid = _valid_attempts(cell_root, seed)
    if len(valid) > 1:
        raise RuntimeError(f"multiple valid attempts exist for {workload}/seed-{seed}")
    if valid:
        return {"seed": seed, "workload": workload, "status": "valid",
                "attempt": valid[0].name, "resumed": True}
    attempt_number = (int(previous[-1].name.removeprefix("attempt-")) + 1) if previous else 1
    attempt_root = cell_root / f"attempt-{attempt_number}"
    output = attempt_root / "output"
    attempt_root.mkdir(parents=True, mode=0o700)
    status_row = {"seed": seed, "workload": workload, "attempt": attempt_number,
                  "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(attempt_root / "campaign-cell-status.json", status_row)
    status[(seed, workload)] = status_row
    atomic_json(root / "campaign-status.json", {"cells": [status[key] for key in cells], "updated_at": time.time()})

    old_argv = sys.argv
    old_output = os.environ.get("HEADLIGHT_EXPERIMENT_OUTPUT_DIR")
    old_manifest = os.environ.get("HEADLIGHT_THRESHOLD_RENDERED_MANIFEST")
    os.environ["HEADLIGHT_EXPERIMENT_OUTPUT_DIR"] = str(output)
    os.environ["HEADLIGHT_THRESHOLD_RENDERED_MANIFEST"] = str(attempt_root / "threshold-rendered.yaml")
    sys.argv = ["run_experiment.py", "--run", RUN_FILES[workload], "--seeds", str(seed),
                "--evaluation-trial", "--evaluation-campaign-cell"]
    log_path = attempt_root / "console.log"
    exit_code = 1
    error = None
    try:
        with log_path.open("w", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)):
            exit_code = runner.main()
        if exit_code != 0:
            raise RuntimeError(f"run_experiment returned {exit_code}")
        manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
        from experiments.lib.quality import validate_bundle
        result = validate_bundle(output / f"seed-{seed}", manifest, dry_run=False)
        if not result.get("valid"):
            raise RuntimeError("cell bundle did not pass final validation")
        status_row.update({"status": "valid", "quality": result})
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        status_row.update({"status": "invalid", "error": error})
        raise
    finally:
        sys.argv = old_argv
        if old_output is None:
            os.environ.pop("HEADLIGHT_EXPERIMENT_OUTPUT_DIR", None)
        else:
            os.environ["HEADLIGHT_EXPERIMENT_OUTPUT_DIR"] = old_output
        if old_manifest is None:
            os.environ.pop("HEADLIGHT_THRESHOLD_RENDERED_MANIFEST", None)
        else:
            os.environ["HEADLIGHT_THRESHOLD_RENDERED_MANIFEST"] = old_manifest
        status_row["finished_at"] = datetime.now(timezone.utc).isoformat()
        cleanup_successful = cleanup_load_resources(attempt_root)
        status_row["load_cleanup_successful"] = cleanup_successful
        if not cleanup_successful:
            status_row["status"] = "invalid"
            status_row["cleanup_error"] = "could not remove campaign Locust Job/ConfigMap"
        atomic_json(attempt_root / "campaign-cell-status.json", status_row)
        status[(seed, workload)] = status_row
        atomic_json(root / "campaign-status.json", {"cells": [status[key] for key in cells], "updated_at": time.time()})
    if not cleanup_successful:
        raise RuntimeError(f"load cleanup failed for {workload}/seed-{seed}; see {attempt_root / 'load-cleanup.log'}")
    return status_row


def run_campaign(
    campaign_dir: Path, prepared_path: Path, *, resume: bool,
    cells: tuple[tuple[int, str], ...] = DEFAULT_CELLS,
) -> int:
    locks = acquire_testbed_locks()
    try:
        return _run_campaign_locked(campaign_dir, prepared_path, resume=resume, cells=cells)
    finally:
        locks.close()


def _run_campaign_locked(
    campaign_dir: Path, prepared_path: Path, *, resume: bool,
    cells: tuple[tuple[int, str], ...] = DEFAULT_CELLS,
) -> int:
    root = validate_campaign_root(campaign_dir, resume=resume)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    lock_path = root / "campaign.lock"
    lock = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(f"another process holds campaign lock: {lock_path}") from exc

    env, preflight = initialize_environment(root, prepared_path)
    identity = campaign_identity(env, cells)
    identity["prepared_settings_sha256"] = sha256(prepared_path)
    identity["created_at"] = datetime.now(timezone.utc).isoformat()
    campaign_json = root / "campaign.json"
    if resume:
        existing = json.loads(campaign_json.read_text(encoding="utf-8"))
        for key in ("schema_version", "scope", "controller", "capacity_users", "capacity_qualified",
                    "matched_comparison_eligible", "historical_gateway_image_verified", "threshold_image",
                    "threshold_image_digest", "cells", "prepared_settings_sha256"):
            if existing.get(key) != identity.get(key):
                raise ValueError(f"resume campaign identity changed: {key}")
    else:
        atomic_json(campaign_json, identity)
        atomic_json(root / "preflight.json", preflight)

    status: dict[tuple[int, str], dict[str, Any]] = {}
    for seed, workload in cells:
        status[(seed, workload)] = {"seed": seed, "workload": workload, "status": "pending"}
    status_path = root / "campaign-status.json"
    if resume and status_path.is_file():
        for row in json.loads(status_path.read_text(encoding="utf-8")).get("cells", []):
            key = (int(row["seed"]), row["workload"])
            if key in status:
                status[key] = row

    guard_dir = root / "guards" / str(time.time_ns())
    guard = None
    guard_log = None
    metric_stop = None
    metric_thread = None
    error = None
    try:
        guard, guard_log = start_memory_guard(guard_dir)
        from experiments import run_experiment as runner
        install_readiness_guard(runner, guard, guard_dir)
        metric_stop, metric_thread = start_metrics_observer(root)
        print(f"CAMPAIGN_DIR={root}", flush=True)
        for seed, workload in cells:
            valid_attempts = _valid_attempts(root / "cells" / f"{workload}-seed-{seed}", seed)
            if len(valid_attempts) > 1:
                raise RuntimeError(f"multiple valid attempts exist for {workload}/seed-{seed}")
            if valid_attempts:
                valid_attempt = valid_attempts[0]
                status[(seed, workload)] = {"seed": seed, "workload": workload, "status": "valid",
                                            "attempt": valid_attempt.name, "resumed": True}
                atomic_json(status_path, {"cells": [status[key] for key in cells], "updated_at": time.time()})
                continue
            if (guard_dir / "ABORT").exists():
                raise RuntimeError("campaign host memory guard aborted before next cell")
            run_cell(runner, root, seed, workload, status, cells)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if metric_stop is not None:
            metric_stop.set()
        if metric_thread is not None:
            metric_thread.join(timeout=5)
        if guard is not None:
            (guard_dir / "monitor-stop").touch()
            stop_memory_guard(guard, guard_log)
        result = {
            "status": "complete" if error is None else "interrupted_or_failed",
            "error": error,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "cells": [status[key] for key in cells],
            "matched_comparison_eligible": False,
            "capacity_qualified": False,
        }
        atomic_json(root / "campaign-status.json", result)
        lock.close()
    print(f"CAMPAIGN_STATUS={root / 'campaign-status.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path,
                        help="new output directory; defaults to a timestamped results/threshold-evaluation path")
    parser.add_argument("--prepared-settings", type=Path, default=PREPARED_SETTINGS)
    parser.add_argument("--seeds", type=int, nargs="+", choices=EVALUATION_SEEDS,
                        help="run all three workloads for these seeds; without this, run the original five remaining 104/105 cells")
    parser.add_argument("--resume", action="store_true", help="continue this campaign without replacing prior attempts")
    args = parser.parse_args()
    cells = DEFAULT_CELLS if args.seeds is None else campaign_cells_for_seeds(args.seeds)
    if args.campaign_dir is None:
        if args.seeds is None:
            campaign_name = "seed104-105-five-cells"
        else:
            seed_label = "-".join(str(seed) for seed in sorted(set(args.seeds)))
            campaign_name = f"seed{seed_label}-evaluation"
        campaign_dir = REPO_ROOT / "results/threshold-evaluation" / (
            f"{campaign_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    else:
        campaign_dir = args.campaign_dir
    return run_campaign(campaign_dir, args.prepared_settings, resume=args.resume, cells=cells)


if __name__ == "__main__":
    raise SystemExit(main())
