#!/usr/bin/env python3
"""Run a watchdog-protected serialized redeployment and baseline gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_local_24gb_measurement import (
    GIB,
    REPO_ROOT,
    Sampler,
    WORKLOADS,
    initialize_series,
    preflight,
    snapshot,
)


WATCHDOG = REPO_ROOT / "experiments" / "scripts" / "local_measurement_watchdog.py"
MEASUREMENT = REPO_ROOT / "experiments" / "run_local_24gb_measurement.py"
DEPLOY = REPO_ROOT / "experiments" / "deploy_minimal_trainticket.sh"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], *, timeout: int = 60, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result


def kubectl(args: argparse.Namespace, context: str, *extra: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", context, *extra],
        timeout=timeout,
    )


def deployment_ready(args: argparse.Namespace, cluster: str, name: str, generation: int) -> bool:
    result = kubectl(args, cluster, "-n", args.namespace, "get", "deployment", name, "-o", "json")
    item = json.loads(result.stdout)
    status = item.get("status", {})
    return (
        int(item.get("metadata", {}).get("generation", 0)) >= generation
        and int(status.get("observedGeneration", 0)) >= int(item["metadata"]["generation"])
        and int(item.get("spec", {}).get("replicas", 0)) == 1
        and int(status.get("updatedReplicas", 0)) == 1
        and int(status.get("readyReplicas", 0)) == 1
        and int(status.get("availableReplicas", 0)) == 1
        and int(status.get("unavailableReplicas", 0)) == 0
    )


def restart_and_wait(args: argparse.Namespace, name: str, abort_file: Path, log: Any) -> dict[str, Any]:
    token = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    patch = json.dumps(
        {"spec": {"template": {"metadata": {"annotations": {"headlight.nephele-project.eu/restartedAt": token}}}}}
    )
    started = time.time()
    kubectl(
        args,
        "karmada-apiserver",
        "-n",
        args.namespace,
        "patch",
        "deployment",
        name,
        "--type=merge",
        "-p",
        patch,
    )
    generations: dict[str, int] = {}
    discovery_deadline = time.monotonic() + 120
    while time.monotonic() < discovery_deadline:
        for cluster in ("cluster2", "cluster3"):
            result = kubectl(args, cluster, "-n", args.namespace, "get", "deployment", name, "-o", "json")
            item = json.loads(result.stdout)
            annotations = item.get("spec", {}).get("template", {}).get("metadata", {}).get("annotations", {})
            if annotations.get("headlight.nephele-project.eu/restartedAt") == token:
                generations[cluster] = int(item["metadata"]["generation"])
        if len(generations) == 2:
            break
        if abort_file.exists():
            raise RuntimeError(abort_file.read_text(encoding="utf-8").strip())
        time.sleep(2)
    if len(generations) != 2:
        raise RuntimeError(f"{name}: restart annotation did not reach both members")

    ready_at: dict[str, float] = {}
    deadline = time.monotonic() + args.rollout_timeout_seconds
    while time.monotonic() < deadline:
        if abort_file.exists():
            raise RuntimeError(abort_file.read_text(encoding="utf-8").strip())
        for cluster in ("cluster2", "cluster3"):
            if cluster not in ready_at and deployment_ready(args, cluster, name, generations[cluster]):
                ready_at[cluster] = time.time()
        if len(ready_at) == 2:
            finished = max(ready_at.values())
            record = {
                "deployment": name,
                "restart_timestamp": token,
                "started_at": started,
                "ready_at": ready_at,
                "decision_to_all_ready_seconds": finished - started,
            }
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            return record
        time.sleep(5)
    raise RuntimeError(f"{name}: rollout did not become Ready in {args.rollout_timeout_seconds}s")


def host_rollout_summary(raw_dir: Path) -> dict[str, Any]:
    with (raw_dir / "host-series.csv").open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == "rollout"]
    available = [int(row["mem_available_bytes"]) for row in rows]
    swap = [int(row["swap_used_bytes"]) for row in rows]
    cpu = [float(row["cpu_busy_percent"]) for row in rows]
    return {
        "samples": len(rows),
        "minimum_available_bytes": min(available, default=0),
        "peak_swap_used_bytes": max(swap, default=0),
        "peak_host_cpu_percent": max(cpu, default=0.0),
        "disk_read_bytes": int(rows[-1]["disk_read_bytes"]) - int(rows[0]["disk_read_bytes"]) if len(rows) > 1 else 0,
        "disk_write_bytes": int(rows[-1]["disk_write_bytes"]) - int(rows[0]["disk_write_bytes"]) if len(rows) > 1 else 0,
    }


def seed_and_verify(args: argparse.Namespace, output_dir: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HEADLIGHT_KUBECONFIG": str(args.kubeconfig),
            "HEADLIGHT_TRAINTICKET_NAMESPACE": args.namespace,
            "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
            "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "local-24gb-tested",
            "HEADLIGHT_TRAINTICKET_DEPLOY_MODE": "karmada",
            "HEADLIGHT_TRAINTICKET_ACTIVE_CLUSTERS": "cluster2 cluster3",
            "HEADLIGHT_WORKLOAD_DEPLOY_ATTEMPTS": "1",
            "HEADLIGHT_WORKLOAD_WAIT_TIMEOUT": "600s",
            "HEADLIGHT_TRAINTICKET_SEED_ONLY": "1",
        }
    )
    result = run(["bash", str(DEPLOY)], timeout=1800, env=env)
    (output_dir / "seed-and-verify.log").write_text(result.stdout, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, default=REPO_ROOT / ".kube" / "headlight-testbed.config")
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--memory-threshold-bytes", type=int, default=3 * GIB)
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--rollout-timeout-seconds", type=int, default=600)
    parser.add_argument("--baseline-idle-seconds", type=int, default=900)
    parser.add_argument("--baseline-load-seconds", type=int, default=900)
    parser.add_argument("--target-rps", type=float, default=0.4)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or REPO_ROOT / "results" / "local-24gb-redeployment-gate" / f"run-{stamp}").resolve()
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True)
    initialize_series(raw_dir)
    phase_file = raw_dir / "phase.txt"
    stop_file = raw_dir / "watchdog.stop"
    abort_file = raw_dir / "watchdog.abort"
    phase_file.write_text("preflight\n", encoding="utf-8")
    preflight(args, raw_dir)
    snapshot(args, raw_dir, "before")

    watchdog_log = (logs_dir / "watchdog.log").open("w", encoding="utf-8")
    watchdog = subprocess.Popen(
        [
            "python3", str(WATCHDOG), "--kubeconfig", str(args.kubeconfig), "--output-dir", str(raw_dir),
            "--phase-file", str(phase_file), "--stop-file", str(stop_file), "--abort-file", str(abort_file),
            "--namespace", args.namespace, "--load-cluster", "cluster2", "--load-job", "headlight-local-fixed-load",
            "--threshold-bytes", str(args.memory_threshold_bytes),
        ],
        cwd=REPO_ROOT,
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sampler = Sampler(args, raw_dir, phase_file)
    success = False
    started_at = utc_now()
    rollout_records: list[dict[str, Any]] = []
    try:
        sampler.start()
        phase_file.write_text("rollout\n", encoding="utf-8")
        with (raw_dir / "rollout-events.jsonl").open("w", encoding="utf-8") as log:
            for name in sorted(WORKLOADS, key=lambda item: (not item.endswith("-mongo"), item)):
                rollout_records.append(restart_and_wait(args, name, abort_file, log))
        seed_and_verify(args, output_dir)
        snapshot(args, raw_dir, "after-rollout")
        sampler.stop()

        phase_file.write_text("baseline\n", encoding="utf-8")
        baseline_dir = output_dir / "baseline"
        result = run(
            [
                "python3", str(MEASUREMENT), "--kubeconfig", str(args.kubeconfig),
                "--output-dir", str(baseline_dir), "--idle-seconds", str(args.baseline_idle_seconds),
                "--load-seconds", str(args.baseline_load_seconds), "--target-rps", str(args.target_rps),
            ],
            timeout=args.baseline_idle_seconds + args.baseline_load_seconds + 900,
        )
        (logs_dir / "baseline.log").write_text(result.stdout, encoding="utf-8")
        baseline = json.loads((baseline_dir / "baseline-resource-report.json").read_text(encoding="utf-8"))
        if not baseline["load_validation"]["rate_valid"] or baseline["watchdog_triggered"]:
            raise RuntimeError("baseline validation or watchdog gate failed")
        success = True
    except Exception as exc:
        (logs_dir / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        snapshot(args, raw_dir, "failure")
        raise
    finally:
        if sampler.is_alive():
            sampler.stop()
        stop_file.touch()
        try:
            watchdog.wait(timeout=15)
        except subprocess.TimeoutExpired:
            watchdog.terminate()
            watchdog.wait(timeout=10)
        watchdog_log.close()

    summary = {
        "schema_version": "headlight.local-redeployment-gate/v1",
        "started_at": started_at,
        "finished_at": utc_now(),
        "success": success,
        "watchdog_events": [json.loads(line) for line in (raw_dir / "watchdog-events.jsonl").read_text().splitlines()],
        "rollout": host_rollout_summary(raw_dir),
        "deployment_count": len(rollout_records),
        "decision_to_ready_seconds": {row["deployment"]: row["decision_to_all_ready_seconds"] for row in rollout_records},
        "baseline_report": str((output_dir / "baseline" / "baseline-resource-report.json").resolve()),
    }
    (output_dir / "redeployment-gate-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
