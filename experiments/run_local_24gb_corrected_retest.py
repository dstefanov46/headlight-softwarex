#!/usr/bin/env python3
"""Run one corrected, bounded local-24gb-tested infrastructure retest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import run_local_24gb_conditional_retest as conditional
from experiments import run_local_24gb_infrastructure_qualification as base
from experiments.run_local_24gb_measurement import (
    GIB,
    Sampler,
    collect_prometheus,
    initialize_series,
    restart_and_event_summary,
    snapshot,
    summarize_prometheus,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "experiments" / "configs" / "qualification" / "research-v4-infrastructure-qualification.yaml"
WATCHDOG = REPO_ROOT / "experiments" / "scripts" / "local_measurement_watchdog.py"
OPEN_LOOP = REPO_ROOT / "experiments" / "testbed" / "trainticket-local-measurement" / "open_loop_load.py"
PROFILE = REPO_ROOT / "ai-module" / "configs" / "service_profiles" / "advanced-ticket-local-24gb-tested.json"
RENDERED = REPO_ROOT / "experiments" / "testbed" / "trainticket-minimal" / "rendered-advanced-ticket.yaml"
CRP_PATCH = REPO_ROOT / "experiments" / "patches" / "controlled-replica-propagation" / "0001-wait-for-resourcebinding-suspension.patch"
LOAD_JOB = "headlight-corrected-open-loop"
LOAD_CONFIGMAP = "headlight-corrected-open-loop"
TESTBED_CONTAINERS = (
    "cluster1-control-plane",
    "cluster2-control-plane",
    "cluster3-control-plane",
    "karmada-host-control-plane",
    "headlight-karmada-api-proxy",
)
SWAP_INTERVAL_SECONDS = 5.0
CGROUP_INTERVAL_SECONDS = 15.0
PROCESS_INTERVAL_SECONDS = 30.0
SWAP_LIMIT_PAGES_PER_SECOND = 1.0
MAX_CONCURRENCY = 16
QUEUE_CAPACITY = 16
REQUEST_TIMEOUT_SECONDS = 60
DRAIN_TIMEOUT_SECONDS = 125
SONAR_LABEL = "io.modelcontextprotocol.server.name=io.github.SonarSource/sonarqube-mcp-server"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True, timeout: float = 60, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        values[name] = int(raw.split()[0]) * 1024
    return values


def vmstat() -> dict[str, int]:
    return {
        name: int(value)
        for name, value in (line.split() for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines())
        if name in {"pswpin", "pswpout", "pgmajfault", "pgfault"}
    }


def pressure_totals() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/pressure/memory").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        values[fields[0]] = int(next(item.split("=", 1)[1] for item in fields if item.startswith("total=")))
    return values.get("some", 0), values.get("full", 0)


class PhaseController:
    def __init__(self, path: Path, diagnostics: "DiagnosticSampler") -> None:
        self.path = path
        self.diagnostics = diagnostics

    def set(self, name: str) -> None:
        self.diagnostics.sample_now(include_cgroups=True, include_processes=False)
        self.path.write_text(name + "\n", encoding="utf-8")
        self.diagnostics.sample_now(include_cgroups=True, include_processes=False)


class DiagnosticSampler(threading.Thread):
    """Sample global swap/cache/PSI plus available Docker cgroup evidence."""

    HOST_FIELDS = (
        "epoch", "timestamp", "phase", "mem_total_bytes", "mem_available_bytes", "mem_free_bytes",
        "cached_bytes", "sreclaimable_bytes", "shmem_bytes", "active_anon_bytes", "inactive_anon_bytes",
        "active_file_bytes", "inactive_file_bytes", "swap_used_bytes", "pswpin", "pswpout", "pgmajfault",
        "pgfault", "memory_psi_some_total_us", "memory_psi_full_total_us",
    )
    CGROUP_FIELDS = (
        "epoch", "timestamp", "phase", "container", "memory_current_bytes", "swap_current_bytes",
        "pgfault", "pgmajfault", "psi_some_total_us", "psi_full_total_us",
    )

    def __init__(self, raw_dir: Path, phase_file: Path) -> None:
        super().__init__(daemon=True)
        self.raw_dir = raw_dir
        self.phase_file = phase_file
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.error: Exception | None = None
        self.last_cgroup = 0.0
        self.last_process = 0.0
        with (raw_dir / "extended-host-series.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(self.HOST_FIELDS)
        with (raw_dir / "cgroup-memory-series.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(self.CGROUP_FIELDS)
        (raw_dir / "process-memory-series.jsonl").touch()

    def phase(self) -> str:
        return self.phase_file.read_text(encoding="utf-8").strip() if self.phase_file.exists() else "unknown"

    def sample_host(self, epoch: float, timestamp: str, phase: str) -> None:
        memory = meminfo()
        counters = vmstat()
        psi_some, psi_full = pressure_totals()
        row = {
            "epoch": epoch,
            "timestamp": timestamp,
            "phase": phase,
            "mem_total_bytes": memory["MemTotal"],
            "mem_available_bytes": memory["MemAvailable"],
            "mem_free_bytes": memory["MemFree"],
            "cached_bytes": memory.get("Cached", 0),
            "sreclaimable_bytes": memory.get("SReclaimable", 0),
            "shmem_bytes": memory.get("Shmem", 0),
            "active_anon_bytes": memory.get("Active(anon)", 0),
            "inactive_anon_bytes": memory.get("Inactive(anon)", 0),
            "active_file_bytes": memory.get("Active(file)", 0),
            "inactive_file_bytes": memory.get("Inactive(file)", 0),
            "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
            "pswpin": counters.get("pswpin", 0),
            "pswpout": counters.get("pswpout", 0),
            "pgmajfault": counters.get("pgmajfault", 0),
            "pgfault": counters.get("pgfault", 0),
            "memory_psi_some_total_us": psi_some,
            "memory_psi_full_total_us": psi_full,
        }
        with (self.raw_dir / "extended-host-series.csv").open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.HOST_FIELDS).writerow(row)

    @staticmethod
    def cgroup_values(container: str) -> dict[str, int] | None:
        command = (
            "printf '%s\\n' "
            "$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1) "
            "$(cat /sys/fs/cgroup/memory.swap.current 2>/dev/null || echo -1); "
            "grep -E '^(pgfault|pgmajfault) ' /sys/fs/cgroup/memory.stat 2>/dev/null; "
            "grep -E '^(some|full) ' /sys/fs/cgroup/memory.pressure 2>/dev/null"
        )
        result = run(["docker", "exec", container, "sh", "-c", command], check=False, timeout=10)
        if result.returncode != 0:
            return None
        lines = result.stdout.splitlines()
        if len(lines) < 6:
            return None
        values = {"memory_current_bytes": int(lines[0]), "swap_current_bytes": int(lines[1])}
        for line in lines[2:]:
            fields = line.split()
            if fields[0] in {"pgfault", "pgmajfault"}:
                values[fields[0]] = int(fields[1])
            elif fields[0] in {"some", "full"}:
                values[f"psi_{fields[0]}_total_us"] = int(next(item.split("=", 1)[1] for item in fields if item.startswith("total=")))
        return values

    def sample_cgroups(self, epoch: float, timestamp: str, phase: str) -> None:
        with (self.raw_dir / "cgroup-memory-series.csv").open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CGROUP_FIELDS)
            for container in TESTBED_CONTAINERS:
                values = self.cgroup_values(container)
                if values is None:
                    continue
                writer.writerow({"epoch": epoch, "timestamp": timestamp, "phase": phase, "container": container, **values})

    @staticmethod
    def process_values(container: str) -> list[dict[str, Any]]:
        script = r'''for f in /proc/[0-9]*/status; do
pid=${f#/proc/}; pid=${pid%/status}
name=$(awk '/^Name:/{print $2}' "$f" 2>/dev/null)
rss=$(awk '/^VmRSS:/{print $2}' "$f" 2>/dev/null)
swap=$(awk '/^VmSwap:/{print $2}' "$f" 2>/dev/null)
[ -n "$rss" ] && printf '%s\t%s\t%s\t%s\n' "$swap" "$rss" "$pid" "$name"
done | sort -nr -k1,1 -k2,2 | head -20'''
        result = run(["docker", "exec", container, "sh", "-c", script], check=False, timeout=15)
        rows: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) == 4:
                rows.append({"swap_kib": int(fields[0] or 0), "rss_kib": int(fields[1]), "pid": int(fields[2]), "name": fields[3]})
        return rows

    def sample_processes(self, epoch: float, timestamp: str, phase: str) -> None:
        with (self.raw_dir / "process-memory-series.jsonl").open("a", encoding="utf-8") as handle:
            for container in TESTBED_CONTAINERS:
                rows = self.process_values(container)
                handle.write(json.dumps({"epoch": epoch, "timestamp": timestamp, "phase": phase, "container": container, "processes": rows}, sort_keys=True) + "\n")

    def sample_now(self, *, include_cgroups: bool, include_processes: bool) -> None:
        with self.lock:
            epoch = time.time()
            timestamp = utc_now()
            phase = self.phase()
            self.sample_host(epoch, timestamp, phase)
            if include_cgroups:
                self.sample_cgroups(epoch, timestamp, phase)
                self.last_cgroup = epoch
            if include_processes:
                self.sample_processes(epoch, timestamp, phase)
                self.last_process = epoch

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                now = time.time()
                self.sample_now(
                    include_cgroups=now - self.last_cgroup >= CGROUP_INTERVAL_SECONDS,
                    include_processes=now - self.last_process >= PROCESS_INTERVAL_SECONDS,
                )
                self.stop_event.wait(SWAP_INTERVAL_SECONDS)
        except Exception as exc:
            self.error = exc

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=PROCESS_INTERVAL_SECONDS + 10)
        if self.error:
            raise self.error


class ActivityGuard(threading.Thread):
    """Abort if outer build/training containers or accessible GPU jobs appear."""

    def __init__(self, raw_dir: Path, abort_file: Path) -> None:
        super().__init__(daemon=True)
        self.raw_dir = raw_dir
        self.abort_file = abort_file
        self.stop_event = threading.Event()
        self.allowed_ids = set(run(["docker", "ps", "-q"]).stdout.split())

    def event(self, name: str, **extra: Any) -> None:
        with (self.raw_dir / "activity-guard-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": utc_now(), "event": name, **extra}, sort_keys=True) + "\n")

    def run(self) -> None:
        self.event("guard_started", allowed_container_ids=sorted(self.allowed_ids))
        while not self.stop_event.is_set():
            current = set(run(["docker", "ps", "-q"], check=False).stdout.split())
            unexpected = sorted(current - self.allowed_ids)
            gpu = run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                check=False,
                timeout=10,
            )
            if unexpected or (gpu.returncode == 0 and gpu.stdout.strip()):
                reason = {"unexpected_outer_containers": unexpected, "gpu_processes": gpu.stdout.splitlines() if gpu.returncode == 0 else []}
                self.event("guard_abort", **reason)
                self.abort_file.write_text("build/compilation/GPU activity guard triggered\n", encoding="utf-8")
                return
            self.stop_event.wait(5)
        self.event("guard_stopped")

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=10)


class SonarState:
    def __init__(self, private_dir: Path) -> None:
        self.private_dir = private_dir
        self.private_dir.mkdir(mode=0o700, parents=True)
        os.chmod(self.private_dir, 0o700)
        self.inspect: dict[str, Any] | None = None
        self.name: str | None = None
        self.was_running = False
        self.storage_path: str | None = None

    def capture_and_stop(self) -> dict[str, Any]:
        names = run(["docker", "ps", "--filter", f"label={SONAR_LABEL}", "--format", "{{.Names}}"], check=False).stdout.splitlines()
        if not names:
            return {"was_running": False}
        if len(names) != 1:
            raise RuntimeError(f"expected one Sonar MCP container, found {names}")
        self.name = names[0]
        self.inspect = json.loads(run(["docker", "inspect", self.name]).stdout)[0]
        self.was_running = True
        inspect_path = self.private_dir / "sonar-inspect.private.json"
        inspect_path.write_text(json.dumps(self.inspect, indent=2) + "\n", encoding="utf-8")
        os.chmod(inspect_path, 0o600)
        env = list(self.inspect["Config"].get("Env") or [])
        env_path = self.private_dir / "sonar-env.private.list"
        env_path.write_text("\n".join(env) + "\n", encoding="utf-8")
        os.chmod(env_path, 0o600)
        env_map = dict(item.split("=", 1) for item in env if "=" in item)
        self.storage_path = env_map.get("STORAGE_PATH")
        storage_saved = False
        if self.storage_path:
            storage_dir = self.private_dir / "storage"
            result = run(["docker", "cp", f"{self.name}:{self.storage_path}", str(storage_dir)], check=False, timeout=120)
            storage_saved = result.returncode == 0
        stats = run(["docker", "stats", "--no-stream", "--format", "{{json .}}", self.name], check=False).stdout.strip()
        run(["docker", "stop", "--time", "20", self.name], timeout=60)
        return {
            "was_running": True,
            "name": self.name,
            "image": self.inspect["Config"]["Image"],
            "container_id": self.inspect["Id"],
            "created": self.inspect["Created"],
            "storage_path": self.storage_path,
            "storage_saved": storage_saved,
            "docker_stats": json.loads(stats) if stats else None,
            "private_state_mode": "0700 directory; inspect/env files 0600",
        }

    def restore(self) -> dict[str, Any]:
        if not self.was_running or not self.inspect or not self.name:
            return {"restored": False, "reason": "Sonar was not running before measurement"}
        available = meminfo()["MemAvailable"]
        if available < 3 * GIB:
            return {"restored": False, "reason": "unsafe memory headroom", "mem_available_bytes": available}
        env_path = self.private_dir / "sonar-env.private.list"
        config = self.inspect["Config"]
        command = ["docker", "create", "--rm", "-i", "--name", self.name, "--env-file", str(env_path)]
        for key, value in (config.get("Labels") or {}).items():
            command.extend(["--label", f"{key}={value}"])
        if config.get("WorkingDir"):
            command.extend(["--workdir", str(config["WorkingDir"])])
        if config.get("User"):
            command.extend(["--user", str(config["User"])])
        command.append(str(config["Image"]))
        run(command, timeout=60)
        storage_dir = self.private_dir / "storage"
        if self.storage_path and storage_dir.exists():
            run(["docker", "cp", f"{storage_dir}/.", f"{self.name}:{self.storage_path}"], timeout=120)
        run(["docker", "start", self.name], timeout=60)
        return {
            "restored": True,
            "name": self.name,
            "image": config["Image"],
            "running": run(["docker", "inspect", "-f", "{{.State.Running}}", self.name]).stdout.strip() == "true",
            "mem_available_bytes": meminfo()["MemAvailable"],
        }


def privileged_host(command: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return run(
        ["docker", "run", "--rm", "--privileged", "--pid=host", "--entrypoint", "nsenter",
         "kindest/node:v1.32.0", "-t", "1", "-m", "--", *command],
        timeout=timeout,
    )


def wait_environment(args: argparse.Namespace, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            members = conditional.base.members_ready(args)
            ready = True
            for cluster in ("cluster2", "cluster3"):
                payload = json.loads(base.kubectl(args, cluster, "-n", args.namespace, "get", "deployments", "-o", "json").stdout)
                ready = ready and all(
                    int(item["status"].get("readyReplicas", 0)) == int(item["spec"].get("replicas", 0))
                    for item in payload["items"]
                )
            if members and ready:
                return
        except Exception:
            pass
        time.sleep(10)
    raise RuntimeError("workload members did not return Ready after safe swap preparation")


def prepare_swap(args: argparse.Namespace, preparation_dir: Path) -> dict[str, Any]:
    before = meminfo()
    record: dict[str, Any] = {
        "swappiness": int(Path("/proc/sys/vm/swappiness").read_text().strip()),
        "before": before,
        "stopped_workload_containers": [],
    }
    if record["swappiness"] != 1:
        raise RuntimeError(f"vm.swappiness must be 1, found {record['swappiness']}")
    swap_used = before["SwapTotal"] - before["SwapFree"]
    try:
        if swap_used > 128 * 1024**2:
            projected = before["MemAvailable"] - swap_used
            if projected < 3 * GIB:
                for name in ("cluster2-control-plane", "cluster3-control-plane"):
                    run(["docker", "stop", "--time", "30", name], timeout=90)
                    record["stopped_workload_containers"].append(name)
                time.sleep(5)
                safe = meminfo()
                if safe["MemAvailable"] - (safe["SwapTotal"] - safe["SwapFree"]) < 3 * GIB:
                    raise RuntimeError("swap cannot be cleared while preserving 3 GiB projected headroom")
            privileged_host(["swapoff", "/swap.img"], timeout=300)
            privileged_host(["swapon", "/swap.img"], timeout=60)
            record["swap_cleared"] = True
        else:
            record["swap_cleared"] = False
    except Exception:
        restore_stopped_workloads(record)
        raise
    record["after_clear"] = meminfo()
    (preparation_dir / "swap-preparation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def restore_stopped_workloads(record: dict[str, Any]) -> None:
    for name in record.get("stopped_workload_containers", []):
        run(["docker", "start", name], timeout=60)


def marked_block(text: str, start: str, end: str) -> str:
    return base.marked_block(text, start, end)


def load_manifest(args: argparse.Namespace, phase: str, rate: float, duration: int) -> dict[str, Any]:
    command = " ".join(
        [
            "/opt/venv/bin/python", "/app/open_loop_load.py",
            "--phase", phase,
            "--namespace", args.namespace,
            "--rate-rps", str(rate),
            "--duration-seconds", str(duration),
            "--max-concurrency", str(MAX_CONCURRENCY),
            "--queue-capacity", str(QUEUE_CAPACITY),
            "--request-timeout-seconds", str(REQUEST_TIMEOUT_SECONDS),
            "--drain-timeout-seconds", str(DRAIN_TIMEOUT_SECONDS),
            "--output", "/tmp/open-loop-events.csv",
        ]
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": LOAD_JOB, "namespace": args.load_namespace, "labels": {"headlight.nephele/measurement": "corrected-infrastructure-retest"}},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": duration + DRAIN_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS + 60,
            "template": {
                "metadata": {"labels": {"headlight.nephele/measurement": "corrected-infrastructure-retest"}, "annotations": {"sidecar.istio.io/inject": "false"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "open-loop",
                        "image": args.load_image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/sh", "-lc"],
                        "args": [command],
                        "resources": {"requests": {"cpu": "25m", "memory": "64Mi"}, "limits": {"cpu": "250m", "memory": "256Mi"}},
                        "volumeMounts": [{"name": "driver", "mountPath": "/app/open_loop_load.py", "subPath": "open_loop_load.py"}],
                    }],
                    "volumes": [{"name": "driver", "configMap": {"name": LOAD_CONFIGMAP}}],
                },
            },
        },
    }


def cleanup_load(args: argparse.Namespace) -> None:
    base.kubectl(args, "cluster2", "-n", args.load_namespace, "delete", "job", LOAD_JOB, "--ignore-not-found", "--wait=true", check=False)


def run_open_loop(
    args: argparse.Namespace,
    raw_dir: Path,
    phases: PhaseController,
    abort_file: Path,
    deadline: float,
    label: str,
    rate: float,
    duration: int = 180,
) -> dict[str, Any]:
    base.ensure_deadline(deadline, abort_file)
    cleanup_load(args)
    stage_dir = raw_dir / "loads" / label
    stage_dir.mkdir(parents=True)
    manifest = load_manifest(args, label, rate, duration)
    (stage_dir / "job.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    phases.set(label)
    started = time.time()
    run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", "cluster2", "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )
    wait_deadline = min(deadline, time.monotonic() + duration + DRAIN_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS + 120)
    while time.monotonic() < wait_deadline:
        base.ensure_deadline(deadline, abort_file)
        response = base.kubectl(args, "cluster2", "-n", args.load_namespace, "get", "job", LOAD_JOB, "-o", "json", check=False)
        if response.returncode != 0:
            raise RuntimeError(f"{label}: load Job disappeared")
        status = json.loads(response.stdout).get("status", {})
        if status.get("succeeded", 0):
            break
        if status.get("failed", 0):
            raise RuntimeError(f"{label}: load Job failed: {status}")
        time.sleep(5)
    else:
        raise RuntimeError(f"{label}: load Job exceeded its bounded deadline")
    pod = base.kubectl(
        args, "cluster2", "-n", args.load_namespace, "get", "pod", "-l", f"job-name={LOAD_JOB}",
        "-o", "jsonpath={.items[0].metadata.name}",
    ).stdout.strip()
    logs = base.kubectl(args, "cluster2", "-n", args.load_namespace, "logs", pod, timeout=60).stdout
    (stage_dir / "pod.log").write_text(logs, encoding="utf-8")
    events = marked_block(logs, "__HEADLIGHT_OPEN_LOOP_EVENTS_BEGIN__", "__HEADLIGHT_OPEN_LOOP_EVENTS_END__")
    summary_text = marked_block(logs, "__HEADLIGHT_OPEN_LOOP_SUMMARY_BEGIN__", "__HEADLIGHT_OPEN_LOOP_SUMMARY_END__").strip()
    if not events or not summary_text:
        raise RuntimeError(f"{label}: load Job did not emit complete accounting markers")
    summary = json.loads(summary_text)
    summary["job_started_epoch"] = started
    summary["job_finished_epoch"] = time.time()
    (stage_dir / "request-events.csv").write_text(events, encoding="utf-8")
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["accounting_balanced"] or summary["accounted_arrivals"] != summary["scheduled_arrivals"]:
        raise RuntimeError(f"{label}: arrival accounting is not balanced")
    cleanup_load(args)
    return summary


def summarize_swap(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: float(row["epoch"]))
    phases: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "duration_seconds": 0.0, "pages_in": 0, "pages_out": 0, "major_faults": 0,
        "psi_some_seconds": 0.0, "psi_full_seconds": 0.0, "windows": [], "samples": 0,
    })
    for row in rows:
        phases[row["phase"]]["samples"] += 1
    for left, right in zip(rows, rows[1:]):
        elapsed = float(right["epoch"]) - float(left["epoch"])
        if elapsed <= 0:
            continue
        phase = right["phase"]
        pin = max(0, int(right["pswpin"]) - int(left["pswpin"]))
        pout = max(0, int(right["pswpout"]) - int(left["pswpout"]))
        target = phases[phase]
        target["duration_seconds"] += elapsed
        target["pages_in"] += pin
        target["pages_out"] += pout
        target["major_faults"] += max(0, int(right["pgmajfault"]) - int(left["pgmajfault"]))
        target["psi_some_seconds"] += max(0, int(right["memory_psi_some_total_us"]) - int(left["memory_psi_some_total_us"])) / 1_000_000
        target["psi_full_seconds"] += max(0, int(right["memory_psi_full_total_us"]) - int(left["memory_psi_full_total_us"])) / 1_000_000
        target["windows"].append((pin / elapsed, pout / elapsed, (pin + pout) / elapsed, elapsed))
    result: dict[str, Any] = {}
    for phase, value in phases.items():
        selected = [row for row in rows if row["phase"] == phase]
        duration = value.pop("duration_seconds")
        windows = value.pop("windows")
        result[phase] = {
            **value,
            "accounted_duration_seconds": duration,
            "swap_in_pages_per_second": value["pages_in"] / duration if duration else None,
            "swap_out_pages_per_second": value["pages_out"] / duration if duration else None,
            "total_swap_pages_per_second": (value["pages_in"] + value["pages_out"]) / duration if duration else None,
            "peak_short_window_swap_in_pages_per_second": max((item[0] for item in windows), default=None),
            "peak_short_window_swap_out_pages_per_second": max((item[1] for item in windows), default=None),
            "peak_short_window_total_pages_per_second": max((item[2] for item in windows), default=None),
            "actual_window_seconds_min": min((item[3] for item in windows), default=None),
            "actual_window_seconds_max": max((item[3] for item in windows), default=None),
            "swap_occupancy_start_bytes": int(selected[0]["swap_used_bytes"]) if selected else None,
            "swap_occupancy_end_bytes": int(selected[-1]["swap_used_bytes"]) if selected else None,
            "swap_occupancy_peak_bytes": max((int(row["swap_used_bytes"]) for row in selected), default=None),
            "minimum_mem_available_bytes": min((int(row["mem_available_bytes"]) for row in selected), default=None),
            "cache_peak_bytes": max((int(row["cached_bytes"]) + int(row["sreclaimable_bytes"]) for row in selected), default=None),
        }
    return result


def summarize_cgroups(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {}
    for container in sorted({row["container"] for row in rows}):
        selected = sorted((row for row in rows if row["container"] == container), key=lambda row: float(row["epoch"]))
        phase_faults: dict[str, int] = defaultdict(int)
        for left, right in zip(selected, selected[1:]):
            phase_faults[right["phase"]] += max(0, int(right["pgmajfault"]) - int(left["pgmajfault"]))
        result[container] = {
            "swap_occupancy_peak_bytes": max((int(row["swap_current_bytes"]) for row in selected), default=0),
            "major_fault_deltas_by_phase": dict(phase_faults),
            "note": "occupancy and major-fault evidence; not direct per-cgroup swap-I/O attribution",
        }
    return result


def write_contract(output_dir: Path) -> dict[str, Any]:
    controller_image = conditional.CONTROLLER_IMAGE
    image_id = run(["docker", "image", "inspect", controller_image, "--format", "{{.Id}}"]).stdout.strip()
    deployed_image = base.kubectl(
        argparse.Namespace(kubeconfig=REPO_ROOT / ".kube" / "headlight-testbed.config"),
        "karmada-host", "-n", "crp-system", "get", "deployment", "crp-controller-manager",
        "-o", "jsonpath={.spec.template.spec.containers[0].image}",
    ).stdout.strip()
    contract = {
        "schema": "headlight.corrected-infrastructure-retest/v1",
        "attempt": 3,
        "seed": 201,
        "seed_role": "infrastructure-qualification",
        "registered_before_execution_at": utc_now(),
        "maximum_live_seconds": 3600,
        "restrictions": {"training_input": False, "final_evaluation": False, "final_matrix": False, "automatic_retry": False},
        "preserved": {
            "profile": "local-24gb-tested", "applications": 11, "databases": 7,
            "quota_requests_memory_mib": 2078, "replica_bounds_per_service_per_cluster": [1, 2],
            "maximum_extra_application_replicas_per_cluster": 1,
        },
        "load": {
            "scheduler": "fixed-arrival/open-loop", "warmup_seconds": 180, "measured_seconds": 180,
            "rates_rps": [2.4, 3.2], "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "drain_timeout_seconds": DRAIN_TIMEOUT_SECONDS, "max_concurrency": MAX_CONCURRENCY,
            "queue_capacity": QUEUE_CAPACITY, "phase_attribution": "scheduled arrival",
            "dropped_and_incomplete_arrivals": "delivery failures",
        },
        "swap": {
            "sample_interval_seconds": SWAP_INTERVAL_SECONDS,
            "boundary_assignment": "every adjacent global counter interval is assigned to the phase of its right-hand sample; explicit boundary samples prevent omission",
            "acceptance": "every execution phase with at least two samples averages <1 total swap page/s",
        },
        "sequence": ["warmup-1-1", "measured-2.4-rps", "measured-3.2-rps", "scale-route-2-2", "settle-route-2-2", "matched-3.2-rps", "recovery-1-1"],
        "controller": {
            "source_revision": conditional.CONTROLLER_SOURCE_REVISION,
            "image": controller_image,
            "image_id": image_id,
            "deployed_image": deployed_image,
            "verified_before_contract": deployed_image == controller_image,
        },
        "tooling": {
            "python": sys.version.split()[0],
            "docker": run(["docker", "version", "--format", "{{.Client.Version}}"]).stdout.strip(),
            "kubectl": run(["kubectl", "version", "--client", "-o", "json"]).stdout.strip(),
        },
        "hashes": {
            "open_loop_driver_sha256": sha256_file(OPEN_LOOP),
            "runner_sha256": sha256_file(Path(__file__)),
            "profile_sha256": sha256_file(PROFILE),
            "rendered_manifest_sha256": sha256_file(RENDERED),
            "protocol_sha256": sha256_file(PROTOCOL),
            "crp_patch_sha256": sha256_file(CRP_PATCH),
        },
    }
    if not contract["controller"]["verified_before_contract"]:
        raise RuntimeError("patched controller image is not deployed before contract registration")
    (output_dir / "attempt-contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kubeconfig", type=Path, default=REPO_ROOT / ".kube" / "headlight-testbed.config")
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--load-namespace", default="headlight-infra-qualification")
    parser.add_argument("--load-image", default="dstefanov46/locust-test:0.66.0")
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--prometheus-port", type=int, default=39330)
    parser.add_argument("--readiness-timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    args.memory_threshold_bytes = 3 * GIB
    args.admission_queue = conditional.AdmissionQueue(args)
    args.scale_action = conditional.safe_set_replicas

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    preparation_dir = output_dir / "preparation"
    private_dir = output_dir / "private-restore-state"
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(); logs_dir.mkdir(); preparation_dir.mkdir()
    contract = write_contract(output_dir)
    (output_dir / "protocol.yaml").write_text(PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")
    initialize_series(raw_dir)
    phase_file = raw_dir / "phase.txt"
    phase_file.write_text("preparation\n", encoding="utf-8")
    stop_file = raw_dir / "watchdog.stop"
    abort_file = raw_dir / "watchdog.abort"
    diagnostics = DiagnosticSampler(raw_dir, phase_file)
    phases = PhaseController(phase_file, diagnostics)
    sonar = SonarState(private_dir)
    sonar_record: dict[str, Any] = {}
    sonar_restore: dict[str, Any] = {}
    swap_record: dict[str, Any] = {}
    watchdog: subprocess.Popen[str] | None = None
    watchdog_log = None
    sampler: Sampler | None = None
    guard: ActivityGuard | None = None
    deadline = time.monotonic() + 3600
    success = False
    actions: list[dict[str, Any]] = []
    loads: list[dict[str, Any]] = []
    windows: dict[str, tuple[float, float]] = {}
    failure: str | None = None
    try:
        base.preflight(args, raw_dir)
        if not conditional.base.members_ready(args):
            raise RuntimeError("not all intended Karmada members are Ready before preparation")
        args.admission_queue.ensure()
        conditional.safe_set_replicas(args, "ts-route", 1, 1, abort_file, time.monotonic() + 600)
        sonar_record = sonar.capture_and_stop()
        (preparation_dir / "sonar-state-public.json").write_text(json.dumps(sonar_record, indent=2) + "\n", encoding="utf-8")
        swap_record = prepare_swap(args, preparation_dir)

        watchdog_log = (logs_dir / "watchdog.log").open("w", encoding="utf-8")
        watchdog = subprocess.Popen([
            "python3", str(WATCHDOG), "--kubeconfig", str(args.kubeconfig), "--output-dir", str(raw_dir),
            "--phase-file", str(phase_file), "--stop-file", str(stop_file), "--abort-file", str(abort_file),
            "--namespace", args.namespace, "--load-namespace", args.load_namespace, "--load-cluster", "cluster2",
            "--load-job", LOAD_JOB, "--threshold-bytes", str(args.memory_threshold_bytes),
        ], cwd=REPO_ROOT, stdout=watchdog_log, stderr=subprocess.STDOUT, text=True)
        restore_stopped_workloads(swap_record)
        wait_environment(args, time.monotonic() + 900)
        after_ready = meminfo()
        (preparation_dir / "ready-memory.json").write_text(json.dumps(after_ready, indent=2) + "\n", encoding="utf-8")
        if after_ready["MemAvailable"] < args.memory_threshold_bytes:
            raise RuntimeError("environment became Ready below the 3 GiB threshold")

        snapshot(args, raw_dir, "before")
        base.kubectl(args, "cluster2", "-n", args.load_namespace, "delete", "configmap", LOAD_CONFIGMAP,
                     "--ignore-not-found", "--wait=true", check=False)
        base.kubectl(args, "cluster2", "-n", args.load_namespace, "create", "configmap", LOAD_CONFIGMAP,
                     f"--from-file=open_loop_load.py={OPEN_LOOP}")
        diagnostics.start()
        sampler = Sampler(args, raw_dir, phase_file)
        sampler.start()
        guard = ActivityGuard(raw_dir, abort_file)
        guard.start()

        warmup = run_open_loop(args, raw_dir, phases, abort_file, deadline, "warmup-1-1", 2.4)
        warmup["acceptance_scope"] = "infrastructure/swap only; excluded from steady-state delivery gate"
        loads.append(warmup); windows["warmup-1-1"] = (warmup["measurement_started_epoch"], warmup["drain_finished_epoch"])
        for label, rate in (("measured-2.4-rps", 2.4), ("measured-3.2-rps", 3.2)):
            record = run_open_loop(args, raw_dir, phases, abort_file, deadline, label, rate)
            loads.append(record); windows[label] = (record["measurement_started_epoch"], record["drain_finished_epoch"])

        phases.set("scale-route-2-2")
        actions.append(conditional.safe_set_replicas(args, "ts-route", 2, 2, abort_file, deadline))
        phases.set("settle-route-2-2")
        settle_deadline = time.monotonic() + 300
        while time.monotonic() < settle_deadline:
            base.ensure_deadline(deadline, abort_file); time.sleep(1)
        matched = run_open_loop(args, raw_dir, phases, abort_file, deadline, "matched-3.2-rps", 3.2)
        loads.append(matched); windows["matched-3.2-rps"] = (matched["measurement_started_epoch"], matched["drain_finished_epoch"])
        phases.set("recovery-1-1")
        actions.append(conditional.safe_set_replicas(args, "ts-route", 1, 1, abort_file, deadline))
        recovery_deadline = time.monotonic() + 300
        while time.monotonic() < recovery_deadline:
            base.ensure_deadline(deadline, abort_file); time.sleep(1)
        phases.set("collect")
        snapshot(args, raw_dir, "after")
        success = True
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        (logs_dir / "failure.txt").write_text(failure + "\n", encoding="utf-8")
        try:
            snapshot(args, raw_dir, "failure")
        except Exception:
            pass
    finally:
        cleanup_load(args)
        try:
            conditional.safe_set_replicas(args, "ts-route", 1, 1, abort_file, time.monotonic() + 600)
        except Exception as exc:
            (logs_dir / "restore-replicas-failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        if guard:
            guard.stop()
        if sampler and sampler.is_alive():
            sampler.stop()
        if diagnostics.is_alive():
            diagnostics.stop()
        stop_file.touch()
        if watchdog:
            try:
                watchdog.wait(timeout=15)
            except subprocess.TimeoutExpired:
                watchdog.terminate(); watchdog.wait(timeout=10)
        if watchdog_log:
            watchdog_log.close()
        try:
            sonar_restore = sonar.restore()
        except Exception as exc:
            sonar_restore = {"restored": False, "reason": f"{type(exc).__name__}: {exc}"}
        (preparation_dir / "sonar-restore-public.json").write_text(json.dumps(sonar_restore, indent=2) + "\n", encoding="utf-8")

    prometheus: dict[str, Any] = {}
    telemetry_error: str | None = None
    if windows and success:
        try:
            collect_prometheus(args, raw_dir, min(start for start, _ in windows.values()), max(end for _, end in windows.values()))
            prometheus = summarize_prometheus(raw_dir, windows)
        except Exception as exc:
            telemetry_error = f"{type(exc).__name__}: {exc}"
            (logs_dir / "prometheus-collection-failure.txt").write_text(telemetry_error + "\n", encoding="utf-8")

    health: dict[str, Any] = {}
    scheduling: dict[str, Any] = {}
    if success:
        health = restart_and_event_summary(raw_dir)
        scheduling = base.scheduling_failures(raw_dir)
        for cluster, failures in scheduling.items():
            health[cluster]["scheduling_failures"] = len(failures)

    if (raw_dir / "extended-host-series.csv").exists():
        swap = summarize_swap(raw_dir / "extended-host-series.csv")
        cgroups = summarize_cgroups(raw_dir / "cgroup-memory-series.csv")
        execution = {name: row for name, row in swap.items() if row["samples"] >= 2 and name not in {"preparation", "collect"}}
        swap_gate = bool(execution) and all(float(row["total_swap_pages_per_second"] or 0) < SWAP_LIMIT_PAGES_PER_SECOND for row in execution.values())
        minimum_available = min((row["minimum_mem_available_bytes"] for row in execution.values()), default=0)
    else:
        swap, cgroups, execution, swap_gate, minimum_available = {}, {}, {}, False, 0
    watchdog_events = [json.loads(line) for line in (raw_dir / "watchdog-events.jsonl").read_text().splitlines()] if (raw_dir / "watchdog-events.jsonl").exists() else []
    watchdog_clean = not any(row["event"] in {"stop_load", "stop_testbed"} for row in watchdog_events)
    measured = [row for row in loads if row["phase"] != "warmup-1-1"]
    load_correct = bool(measured) and all(
        row["accounting_balanced"] and row["dispatch_fraction"] >= 0.99
        and row["success_fraction_of_arrivals"] >= 0.90
        and row["dropped_requests"] == 0 and row["incomplete_requests"] == 0
        for row in measured
    )
    useful_scaling = False
    if any(row["phase"] == "measured-3.2-rps" for row in loads) and any(row["phase"] == "matched-3.2-rps" for row in loads):
        baseline_record = next(row for row in loads if row["phase"] == "measured-3.2-rps")
        matched_record = next(row for row in loads if row["phase"] == "matched-3.2-rps")
        useful_scaling = float(matched_record["response_time_p95_ms"] or 0) <= float(baseline_record["response_time_p95_ms"] or 0) * 0.95
    admission_correct = bool(actions) and all(
        all(action["targets"][cluster] <= 2 for cluster in ("cluster2", "cluster3"))
        for action in actions
    ) and not any(scheduling.values())
    health_clean = bool(health) and all(
        row["restart_increase"] == 0 and row["pending_pods_after"] == 0
        and row["oom_events"] == 0 and row["eviction_events"] == 0
        and row["oomkilled_containers_after"] == 0 and row["scheduling_failures"] == 0
        for row in health.values()
    )
    telemetry_complete = telemetry_error is None and bool(prometheus)
    report = {
        "schema": "headlight.corrected-infrastructure-retest-result/v1",
        "success": success,
        "failure": failure,
        "claim_scope": "abbreviated infrastructure-only retest; not full workload qualification, training, profiling, or final evaluation",
        "contract": contract,
        "acceptance_gates": {
            "load_generation_correct": load_correct,
            "memory_headroom": minimum_available >= 3 * GIB,
            "watchdog_clean": watchdog_clean,
            "swap_per_phase_below_one_page_per_second": swap_gate,
            "workload_health": health_clean,
            "telemetry_complete": telemetry_complete,
            "admission_correct": admission_correct,
            "useful_scaling": useful_scaling,
        },
        "loads": loads,
        "swap": swap,
        "cgroups": cgroups,
        "scale_actions": actions,
        "readiness": conditional.readiness_summary(actions),
        "health": health,
        "scheduling_failures": scheduling,
        "prometheus": prometheus,
        "telemetry_error": telemetry_error,
        "watchdog_events": watchdog_events,
        "sonar_before": sonar_record,
        "sonar_restore": sonar_restore,
        "earlier_evidence": {
            "remains_usable": [
                "profile/topology/resource/quota rendering and effective admission settings",
                "CRP ResourceBinding suspension barrier regression proof",
                "prior clean 2/2 admission transition with no transient third pod",
                "prior OOM/eviction/restart/control-plane health evidence",
            ],
            "must_be_repeated_for_full_qualification": [
                "all prior closed-loop throughput and latency phases because the generator semantics changed",
                "the contaminated prior 2/1 transition",
                "rapid-ramp, repeating, and flash traces not included in this abbreviated retest",
                "readiness p95, which still lacks enough independent samples",
            ],
        },
        "limitations": [
            "cgroup swap occupancy and major faults do not directly attribute swap I/O",
            "the abbreviated retest cannot establish full workload qualification",
            "no automatic retry was attempted",
        ],
    }
    report_path = output_dir / "corrected-retest-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "status.json").write_text(json.dumps({"success": success, "finished_at": utc_now(), "failure": failure}, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
