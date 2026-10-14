#!/usr/bin/env python3
"""Run the bounded idle/load measurement for local-24gb-tested."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KUBECONFIG = REPO_ROOT / ".kube" / "headlight-testbed.config"
WATCHDOG = REPO_ROOT / "experiments" / "scripts" / "local_measurement_watchdog.py"
LOCUSTFILE = REPO_ROOT / "experiments" / "testbed" / "trainticket-local-measurement" / "locustfile.py"
WORKLOADS = {
    "ts-auth", "ts-auth-mongo", "ts-basic", "ts-price", "ts-price-mongo",
    "ts-route", "ts-route-mongo", "ts-route-plan", "ts-station",
    "ts-station-mongo", "ts-ticketinfo", "ts-train", "ts-train-mongo",
    "ts-travel", "ts-travel-mongo", "ts-travel-plan", "ts-travel2",
    "ts-travel2-mongo",
}
LOAD_JOB = "headlight-local-fixed-load"
LOAD_CONFIGMAP = "headlight-local-fixed-load"
GIB = 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
        timeout=timeout,
    )


def kubectl(args: argparse.Namespace, cluster: str, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", cluster, *extra],
        check=check,
        timeout=30,
    )


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.split()[0]) * 1024
    return values


def cpu_counters() -> tuple[int, int]:
    values = [int(value) for value in Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]]
    idle = values[3] + values[4]
    return sum(values), idle


def root_disk_device() -> str:
    result = run(["findmnt", "-no", "SOURCE", "/"])
    source = result.stdout.strip().split("[", 1)[0]
    # The sandbox can read /proc/diskstats while the matching /dev node is not
    # exposed as a block device. The mounted partition name is sufficient for
    # measuring filesystem I/O and avoids depending on that device node.
    return Path(source).name


def disk_counters(device: str) -> tuple[int, int]:
    for line in Path("/proc/diskstats").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[2] == device:
            return int(fields[5]) * 512, int(fields[9]) * 512
    raise RuntimeError(f"disk {device} is absent from /proc/diskstats")


def swap_io_counters() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        if key in {"pswpin", "pswpout"}:
            values[key] = int(value)
    return values.get("pswpin", 0), values.get("pswpout", 0)


def quantity_cpu(value: str) -> int:
    value = value.strip()
    if value.endswith("m"):
        return int(value[:-1])
    return int(float(value) * 1000)


def quantity_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9.]+)(Ki|Mi|Gi|K|M|G)?", value.strip())
    if not match:
        raise ValueError(f"unsupported memory quantity {value!r}")
    scale = {None: 1, "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "K": 1000, "M": 1000**2, "G": 1000**3}
    return int(float(match.group(1)) * scale[match.group(2)])


def docker_memory_bytes(value: str) -> int:
    return quantity_bytes(value.replace("B", "").replace("i", "i"))


class Sampler(threading.Thread):
    def __init__(self, args: argparse.Namespace, raw_dir: Path, phase_file: Path):
        super().__init__(daemon=True)
        self.args = args
        self.raw_dir = raw_dir
        self.phase_file = phase_file
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.disk = root_disk_device()
        self.previous_cpu = cpu_counters()

    def phase(self) -> str:
        return self.phase_file.read_text(encoding="utf-8").strip()

    def sample_host(self, timestamp: float, phase: str) -> None:
        memory = read_meminfo()
        total, idle = cpu_counters()
        previous_total, previous_idle = self.previous_cpu
        self.previous_cpu = total, idle
        delta_total = max(1, total - previous_total)
        busy = 100.0 * (delta_total - (idle - previous_idle)) / delta_total
        read_bytes, write_bytes = disk_counters(self.disk)
        swap_in_pages, swap_out_pages = swap_io_counters()
        usage = shutil.disk_usage(REPO_ROOT)
        with (self.raw_dir / "host-series.csv").open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [timestamp, utc_now(), phase, memory["MemTotal"], memory["MemAvailable"],
                 memory["SwapTotal"] - memory["SwapFree"], busy, read_bytes, write_bytes,
                 usage.used, usage.free, os.getloadavg()[0], swap_in_pages, swap_out_pages]
            )

    def sample_docker(self, timestamp: float, phase: str) -> None:
        result = run(["docker", "stats", "--no-stream", "--format", "{{json .}}"], check=False, timeout=20)
        with (self.raw_dir / "docker-series.jsonl").open("a", encoding="utf-8") as handle:
            for line in result.stdout.splitlines():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload.update({"timestamp": timestamp, "phase": phase})
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def sample_kubernetes(self, timestamp: float, phase: str) -> None:
        path = self.raw_dir / "kubernetes-container-series.csv"
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for cluster in ("cluster2", "cluster3"):
                result = kubectl(self.args, cluster, "top", "pods", "-A", "--containers", "--no-headers", check=False)
                if result.returncode != 0:
                    continue
                for line in result.stdout.splitlines():
                    fields = line.split()
                    if len(fields) != 5:
                        continue
                    namespace, pod, container, cpu, memory = fields
                    writer.writerow(
                        [timestamp, utc_now(), phase, cluster, namespace, pod, container,
                         quantity_cpu(cpu), quantity_bytes(memory)]
                    )

    def sample(self) -> None:
        timestamp = time.time()
        phase = self.phase()
        self.sample_host(timestamp, phase)
        self.sample_docker(timestamp, phase)
        self.sample_kubernetes(timestamp, phase)

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                self.sample()
                remaining = max(0.0, self.args.sample_interval - (time.monotonic() - started))
                self.stop_event.wait(remaining)
        except Exception as exc:
            self.error = exc

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=self.args.sample_interval + 30)
        if self.error:
            raise self.error


def initialize_series(raw_dir: Path) -> None:
    with (raw_dir / "host-series.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            ["timestamp", "timestamp_iso", "phase", "mem_total_bytes", "mem_available_bytes",
             "swap_used_bytes", "cpu_busy_percent", "disk_read_bytes", "disk_write_bytes",
             "filesystem_used_bytes", "filesystem_free_bytes", "load_average_1m",
             "swap_in_pages", "swap_out_pages"]
        )
    with (raw_dir / "kubernetes-container-series.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            ["timestamp", "timestamp_iso", "phase", "cluster", "namespace", "pod", "container",
             "cpu_millicores", "memory_bytes"]
        )
    (raw_dir / "docker-series.jsonl").write_text("", encoding="utf-8")


def current_processes() -> list[str]:
    result = run(["ps", "-eo", "pid,args"])
    patterns = ("locust", "train_adapgla", "run_capacity_qualification.py", "run_experiment.py")
    return [line.strip() for line in result.stdout.splitlines() if any(pattern in line for pattern in patterns)]


def preflight(args: argparse.Namespace, raw_dir: Path) -> dict[str, Any]:
    memory = read_meminfo()
    if memory["MemAvailable"] < args.memory_threshold_bytes:
        raise RuntimeError(
            f"available memory {memory['MemAvailable']} is below required {args.memory_threshold_bytes}"
        )
    docker_info = json.loads(run(["docker", "info", "--format", "{{json .}}"], timeout=30).stdout)
    clusters = json.loads(kubectl(args, "karmada-apiserver", "get", "clusters", "-o", "json").stdout)
    member_ready = {
        item["metadata"]["name"]: any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in item.get("status", {}).get("conditions", [])
        )
        for item in clusters["items"]
    }
    if member_ready != {"cluster1": True, "cluster2": True, "cluster3": True}:
        raise RuntimeError(f"Karmada members are not exactly Ready: {member_ready}")

    for cluster in ("cluster2", "cluster3"):
        deployments = json.loads(kubectl(args, cluster, "-n", args.namespace, "get", "deployments", "-o", "json").stdout)
        by_name = {item["metadata"]["name"]: item for item in deployments["items"]}
        if set(by_name) != WORKLOADS:
            raise RuntimeError(f"{cluster}: unexpected deployments: {sorted(set(by_name) ^ WORKLOADS)}")
        bad = [
            name for name, item in by_name.items()
            if item["spec"].get("replicas") != 1 or item.get("status", {}).get("readyReplicas", 0) != 1
        ]
        if bad:
            raise RuntimeError(f"{cluster}: deployments not at one Ready replica: {bad}")
        quota = json.loads(
            kubectl(args, cluster, "-n", args.namespace, "get", "resourcequota", "local-24gb-tested-admission", "-o", "json").stdout
        )
        if quota.get("status", {}).get("hard", {}).get("requests.memory") != "2078Mi":
            raise RuntimeError(f"{cluster}: local admission quota is not active")

    payload = {
        "timestamp": utc_now(),
        "host_mem_total_bytes": memory["MemTotal"],
        "host_mem_available_bytes": memory["MemAvailable"],
        "host_cpu_model": run(["bash", "-lc", "lscpu | awk -F: '/Model name/{sub(/^[ \\t]+/,$2); print $2; exit}'"]).stdout.strip(),
        "host_logical_cpus": os.cpu_count(),
        "docker_ncpu": docker_info.get("NCPU"),
        "docker_mem_total_bytes": docker_info.get("MemTotal"),
        "docker_driver": docker_info.get("Driver"),
        "member_ready": member_ready,
        "competing_processes": current_processes(),
        "running_containers": run(["docker", "ps", "--format", "{{json .}}"], check=False).stdout.splitlines(),
        "disk_usage": dict(zip(("total", "used", "free"), shutil.disk_usage(REPO_ROOT))),
    }
    (raw_dir / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def snapshot(args: argparse.Namespace, raw_dir: Path, label: str) -> None:
    for cluster in ("cluster2", "cluster3"):
        for resource in ("pods", "deployments", "events", "nodes"):
            result = kubectl(args, cluster, "get", resource, "-A", "-o", "json", check=False)
            (raw_dir / f"{cluster}-{resource}-{label}.json").write_text(result.stdout, encoding="utf-8")


def bounded_wait(seconds: int, abort_file: Path) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if abort_file.exists():
            raise RuntimeError(f"memory watchdog aborted the run: {abort_file.read_text(encoding='utf-8').strip()}")
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def load_job_manifest(args: argparse.Namespace) -> dict[str, Any]:
    command = "\n".join(
        [
            "/opt/venv/bin/locust --headless -f /app/locustfile.py "
            f"--users 1 --spawn-rate 1 --run-time {args.load_seconds}s "
            "--exit-code-on-error 0 --csv /tmp/headlight-local",
            'rc="$?"',
            "echo __HEADLIGHT_REQUEST_CSV_BEGIN__",
            "cat /app/headlight_requests.csv 2>/dev/null || true",
            "echo __HEADLIGHT_REQUEST_CSV_END__",
            "echo __HEADLIGHT_LOCUST_STATS_BEGIN__",
            "cat /tmp/headlight-local_stats.csv 2>/dev/null || true",
            "echo __HEADLIGHT_LOCUST_STATS_END__",
            'exit "$rc"',
        ]
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": LOAD_JOB, "namespace": args.namespace},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": args.load_seconds + 180,
            "template": {
                "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "locust",
                            "image": args.locust_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/sh", "-lc"],
                            "args": [command],
                            "env": [
                                {"name": "HEADLIGHT_REQUEST_CSV", "value": "/app/headlight_requests.csv"},
                                {"name": "HEADLIGHT_TARGET_RPS", "value": str(args.target_rps)},
                                {"name": "HEADLIGHT_TRAINTICKET_NAMESPACE", "value": args.namespace},
                            ],
                            "resources": {
                                "requests": {"cpu": "25m", "memory": "64Mi"},
                                "limits": {"cpu": "250m", "memory": "256Mi"},
                            },
                            "volumeMounts": [{"name": "locustfile", "mountPath": "/app/locustfile.py", "subPath": "locustfile.py"}],
                        }
                    ],
                    "volumes": [{"name": "locustfile", "configMap": {"name": LOAD_CONFIGMAP}}],
                },
            },
        },
    }


def start_load(args: argparse.Namespace, raw_dir: Path) -> None:
    cluster = args.load_cluster
    kubectl(args, cluster, "-n", args.namespace, "delete", "job", LOAD_JOB, "--ignore-not-found", "--wait=true", check=False)
    kubectl(args, cluster, "-n", args.namespace, "delete", "configmap", LOAD_CONFIGMAP, "--ignore-not-found", check=False)
    kubectl(
        args,
        cluster,
        "-n",
        args.namespace,
        "create",
        "configmap",
        LOAD_CONFIGMAP,
        f"--from-file=locustfile.py={LOCUSTFILE}",
    )
    manifest = load_job_manifest(args)
    (raw_dir / "locust-job.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    result = run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", cluster, "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )
    (raw_dir / "locust-apply.log").write_text(result.stdout, encoding="utf-8")


def wait_for_load(args: argparse.Namespace, abort_file: Path) -> None:
    deadline = time.monotonic() + args.load_seconds + 240
    while time.monotonic() < deadline:
        if abort_file.exists():
            raise RuntimeError(f"memory watchdog aborted the load: {abort_file.read_text(encoding='utf-8').strip()}")
        result = kubectl(args, args.load_cluster, "-n", args.namespace, "get", "job", LOAD_JOB, "-o", "json", check=False)
        if result.returncode != 0:
            raise RuntimeError(f"load job disappeared: {result.stdout.strip()}")
        job = json.loads(result.stdout)
        status = job.get("status", {})
        if status.get("succeeded", 0) >= 1:
            return
        if status.get("failed", 0) >= 1:
            raise RuntimeError(f"load job failed: {status}")
        time.sleep(5)
    raise RuntimeError("load job exceeded its bounded completion deadline")


def marked_block(text: str, start: str, end: str) -> str:
    collecting = False
    lines: list[str] = []
    for line in text.splitlines():
        if line == start:
            collecting = True
            continue
        if line == end:
            collecting = False
            continue
        if collecting:
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def collect_load(args: argparse.Namespace, raw_dir: Path) -> dict[str, Any]:
    pod = kubectl(
        args, args.load_cluster, "-n", args.namespace, "get", "pod", "-l", f"job-name={LOAD_JOB}",
        "-o", "jsonpath={.items[0].metadata.name}",
    ).stdout.strip()
    logs = kubectl(args, args.load_cluster, "-n", args.namespace, "logs", pod).stdout
    (raw_dir / "locust-pod.log").write_text(logs, encoding="utf-8")
    requests_text = marked_block(logs, "__HEADLIGHT_REQUEST_CSV_BEGIN__", "__HEADLIGHT_REQUEST_CSV_END__")
    stats_text = marked_block(logs, "__HEADLIGHT_LOCUST_STATS_BEGIN__", "__HEADLIGHT_LOCUST_STATS_END__")
    (raw_dir / "request-events.csv").write_text(requests_text, encoding="utf-8")
    (raw_dir / "locust-stats.csv").write_text(stats_text, encoding="utf-8")

    rows = list(csv.DictReader(requests_text.splitlines()))
    travel = [row for row in rows if row.get("endpoint") == "travel.left"]
    successful = [row for row in travel if row.get("status", "").startswith(("2", "3")) and not row.get("exception")]
    timestamps = [float(row["timestamp"]) for row in travel]
    active_seconds = (max(timestamps) - min(timestamps) + 1.0 / args.target_rps) if timestamps else 0.0
    achieved = len(successful) / active_seconds if active_seconds else 0.0
    coverage = active_seconds / args.load_seconds if args.load_seconds else 0.0
    lower, upper = args.target_rps * 0.90, args.target_rps * 1.10
    payload = {
        "requested_rps": args.target_rps,
        "achieved_successful_rps": achieved,
        "active_seconds": active_seconds,
        "coverage_fraction": coverage,
        "travel_requests": len(travel),
        "successful_travel_requests": len(successful),
        "failed_travel_requests": len(travel) - len(successful),
        "rate_tolerance": {"lower": lower, "upper": upper},
        "rate_valid": lower <= achieved <= upper and coverage >= 0.95,
    }
    (raw_dir / "load-validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["rate_valid"]:
        raise RuntimeError(f"fixed load did not achieve its requested rate: {payload}")
    return payload


PROMETHEUS_QUERIES = {
    "cpu_rate_by_container": 'sum by (pod, container) (rate(container_cpu_usage_seconds_total{namespace="trainticket",container!="",pod!=""}[1m]))',
    "memory_by_container": 'sum by (pod, container) (container_memory_working_set_bytes{namespace="trainticket",container!="",pod!=""})',
    "throttled_seconds_rate": 'sum by (pod, container) (rate(container_cpu_cfs_throttled_seconds_total{namespace="trainticket",container!="",pod!=""}[1m]))',
    "throttled_periods_rate": 'sum by (pod, container) (rate(container_cpu_cfs_throttled_periods_total{namespace="trainticket",container!="",pod!=""}[1m]))',
    "cpu_periods_rate": 'sum by (pod, container) (rate(container_cpu_cfs_periods_total{namespace="trainticket",container!="",pod!=""}[1m]))',
    "throttled_period_ratio": 'sum by (pod, container) (rate(container_cpu_cfs_throttled_periods_total{namespace="trainticket",container!="",pod!=""}[1m])) / clamp_min(sum by (pod, container) (rate(container_cpu_cfs_periods_total{namespace="trainticket",container!="",pod!=""}[1m])), 0.000001)',
    "request_rate": 'sum(rate(istio_requests_total{destination_workload_namespace="trainticket"}[1m]))',
    "p95_latency_ms": 'histogram_quantile(0.95, sum by (le) (rate(istio_request_duration_milliseconds_bucket{destination_workload_namespace="trainticket"}[1m])))',
    "failure_rate": 'sum(rate(istio_requests_total{destination_workload_namespace="trainticket",response_code!~"2..|3.."}[1m])) or vector(0)',
    "pending_pods": 'sum(kube_pod_status_phase{namespace="trainticket",phase="Pending"})',
}


def prometheus_range(port: int, query: str, start: float, end: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({"query": query, "start": int(start), "end": int(end), "step": "15s"})
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/query_range?{params}", timeout=30) as response:
        return json.load(response)


def collect_prometheus(args: argparse.Namespace, raw_dir: Path, start: float, end: float) -> None:
    for offset, cluster in enumerate(("cluster2", "cluster3")):
        port = args.prometheus_port + offset
        log = (raw_dir / f"{cluster}-prometheus-port-forward.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", cluster, "-n", "monitoring",
             "port-forward", "svc/prometheus-kube-prometheus-prometheus", f"{port}:9090"],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for _ in range(60):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/-/ready", timeout=1).read()
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError(f"{cluster}: Prometheus port-forward did not become ready")
            payload = {
                name: {"query": query, "response": prometheus_range(port, query, start, end)}
                for name, query in PROMETHEUS_QUERIES.items()
            }
            (raw_dir / f"{cluster}-prometheus-range.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            log.close()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def category(namespace: str, container: str) -> str:
    if namespace == "monitoring":
        return "monitoring"
    if namespace != "trainticket":
        return "infrastructure"
    if container == "istio-proxy":
        return "sidecar"
    if container.endswith("-mongo"):
        return "database"
    if container == "locust":
        return "load-generator"
    return "application"


def summarize_host(raw_dir: Path) -> dict[str, Any]:
    rows = list(csv.DictReader((raw_dir / "host-series.csv").open(encoding="utf-8")))
    result: dict[str, Any] = {}
    for phase in ("idle", "load"):
        selected = [row for row in rows if row["phase"] == phase]
        available = [int(row["mem_available_bytes"]) for row in selected]
        used = [int(row["mem_total_bytes"]) - int(row["mem_available_bytes"]) for row in selected]
        cpu = [float(row["cpu_busy_percent"]) for row in selected]
        swap = [int(row["swap_used_bytes"]) for row in selected]
        result[phase] = {
            "samples": len(selected),
            "memory_used_avg_bytes": int(statistics.fmean(used)) if used else 0,
            "memory_used_peak_bytes": max(used, default=0),
            "memory_available_min_bytes": min(available, default=0),
            "cpu_busy_avg_percent": statistics.fmean(cpu) if cpu else 0.0,
            "cpu_busy_peak_percent": max(cpu, default=0.0),
            "swap_used_peak_bytes": max(swap, default=0),
            "swap_in_pages": int(selected[-1].get("swap_in_pages", 0)) - int(selected[0].get("swap_in_pages", 0)) if len(selected) > 1 else 0,
            "swap_out_pages": int(selected[-1].get("swap_out_pages", 0)) - int(selected[0].get("swap_out_pages", 0)) if len(selected) > 1 else 0,
            "disk_read_bytes": int(selected[-1]["disk_read_bytes"]) - int(selected[0]["disk_read_bytes"]) if len(selected) > 1 else 0,
            "disk_write_bytes": int(selected[-1]["disk_write_bytes"]) - int(selected[0]["disk_write_bytes"]) if len(selected) > 1 else 0,
            "filesystem_used_delta_bytes": int(selected[-1]["filesystem_used_bytes"]) - int(selected[0]["filesystem_used_bytes"]) if len(selected) > 1 else 0,
        }
    return result


def summarize_kubernetes(raw_dir: Path) -> dict[str, Any]:
    with (raw_dir / "kubernetes-container-series.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        key = (row["phase"], row["timestamp"], category(row["namespace"], row["container"]))
        grouped[key].append((int(row["cpu_millicores"]), int(row["memory_bytes"])))
    result: dict[str, Any] = {}
    for phase in ("idle", "load"):
        phase_result: dict[str, Any] = {}
        categories = sorted({key[2] for key in grouped if key[0] == phase})
        for name in categories:
            points = [
                (sum(cpu for cpu, _ in values), sum(memory for _, memory in values))
                for key, values in grouped.items() if key[0] == phase and key[2] == name
            ]
            phase_result[name] = {
                "samples": len(points),
                "cpu_avg_millicores": statistics.fmean(cpu for cpu, _ in points),
                "cpu_peak_millicores": max(cpu for cpu, _ in points),
                "memory_avg_bytes": int(statistics.fmean(memory for _, memory in points)),
                "memory_peak_bytes": max(memory for _, memory in points),
            }
        result[phase] = phase_result
    return result


def summarize_kubernetes_containers(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    with (raw_dir / "kubernetes-container-series.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        key = (row["phase"], row["cluster"], row["namespace"], row["pod"], row["container"])
        grouped[key].append((int(row["cpu_millicores"]), int(row["memory_bytes"])))

    result: dict[str, list[dict[str, Any]]] = {"idle": [], "load": []}
    for (phase, cluster, namespace, pod, container), points in grouped.items():
        if phase not in result:
            continue
        result[phase].append(
            {
                "cluster": cluster,
                "namespace": namespace,
                "pod": pod,
                "container": container,
                "category": category(namespace, container),
                "samples": len(points),
                "cpu_avg_millicores": statistics.fmean(cpu for cpu, _ in points),
                "cpu_peak_millicores": max(cpu for cpu, _ in points),
                "memory_avg_bytes": int(statistics.fmean(memory for _, memory in points)),
                "memory_peak_bytes": max(memory for _, memory in points),
            }
        )
    for phase in result:
        result[phase].sort(key=lambda row: (row["cluster"], row["namespace"], row["pod"], row["container"]))
    return result


def summarize_docker(raw_dir: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for line in (raw_dir / "docker-series.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cpu = float(row.get("CPUPerc", "0%").rstrip("%"))
        memory = docker_memory_bytes(row.get("MemUsage", "0B / 0B").split(" / ")[0])
        grouped[(row["phase"], row["Name"])].append((cpu, memory))
    result: dict[str, Any] = {"idle": {}, "load": {}}
    for (phase, name), points in grouped.items():
        if phase not in result:
            continue
        result[phase][name] = {
            "samples": len(points),
            "cpu_avg_percent": statistics.fmean(cpu for cpu, _ in points),
            "cpu_peak_percent": max(cpu for cpu, _ in points),
            "memory_avg_bytes": int(statistics.fmean(memory for _, memory in points)),
            "memory_peak_bytes": max(memory for _, memory in points),
        }
    return result


def summarize_prometheus(raw_dir: Path, phases: dict[str, tuple[float, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cluster in ("cluster2", "cluster3"):
        payload = json.loads((raw_dir / f"{cluster}-prometheus-range.json").read_text(encoding="utf-8"))
        cluster_result: dict[str, Any] = {}
        for metric, wrapped in payload.items():
            series = wrapped["response"].get("data", {}).get("result", [])
            phase_values: dict[str, list[float]] = {phase: [] for phase in phases}
            for item in series:
                for timestamp, value in item.get("values", []):
                    for phase, (start, end) in phases.items():
                        if start <= float(timestamp) <= end:
                            try:
                                numeric = float(value)
                            except ValueError:
                                continue
                            if math.isfinite(numeric):
                                phase_values[phase].append(numeric)
            cluster_result[metric] = {
                phase: {
                    "available": bool(values),
                    "samples": len(values),
                    "avg": statistics.fmean(values) if values else None,
                    "p95": percentile(values, 0.95) if values else None,
                    "max": max(values) if values else None,
                }
                for phase, values in phase_values.items()
            }
        result[cluster] = cluster_result
    return result


def restart_and_event_summary(raw_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cluster in ("cluster2", "cluster3"):
        before = json.loads((raw_dir / f"{cluster}-pods-before.json").read_text(encoding="utf-8"))
        after = json.loads((raw_dir / f"{cluster}-pods-after.json").read_text(encoding="utf-8"))
        def restarts(payload: dict[str, Any]) -> dict[tuple[str, str], int]:
            return {
                (item["metadata"]["name"], status["name"]): int(status.get("restartCount", 0))
                for item in payload.get("items", []) if item["metadata"].get("namespace") == "trainticket"
                for status in item.get("status", {}).get("containerStatuses", [])
            }
        first, last = restarts(before), restarts(after)
        events_before = json.loads((raw_dir / f"{cluster}-events-before.json").read_text(encoding="utf-8"))
        events_after = json.loads((raw_dir / f"{cluster}-events-after.json").read_text(encoding="utf-8"))
        before_uids = {item["metadata"]["uid"] for item in events_before.get("items", [])}
        new_events = [item for item in events_after.get("items", []) if item["metadata"]["uid"] not in before_uids]
        reasons = [item.get("reason", "") for item in new_events]
        result[cluster] = {
            "restart_increase": sum(max(0, value - first.get(key, 0)) for key, value in last.items()),
            "pending_pods_after": sum(item.get("status", {}).get("phase") == "Pending" for item in after.get("items", [])),
            "oom_events": sum(reason == "OOMKilling" for reason in reasons),
            "eviction_events": sum(reason == "Evicted" for reason in reasons),
            "new_events": len(new_events),
            "oomkilled_containers_after": sum(
                status.get("lastState", {}).get("terminated", {}).get("reason") == "OOMKilled"
                for item in after.get("items", []) for status in item.get("status", {}).get("containerStatuses", [])
            ),
        }
    return result


def write_report(
    output_dir: Path,
    raw_dir: Path,
    phases: dict[str, tuple[float, float]],
    load_validation: dict[str, Any],
) -> dict[str, Any]:
    watchdog_events = [json.loads(line) for line in (raw_dir / "watchdog-events.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {
        "schema_version": "headlight.local-baseline/v1",
        "generated_at": utc_now(),
        "claim_scope": "bounded local startup/idle/fixed-load baseline; not experimental qualification",
        "phase_windows": {name: {"start": start, "end": end, "duration_seconds": end - start} for name, (start, end) in phases.items()},
        "load_validation": load_validation,
        "host": summarize_host(raw_dir),
        "docker": summarize_docker(raw_dir),
        "kubernetes_categories": summarize_kubernetes(raw_dir),
        "kubernetes_containers": summarize_kubernetes_containers(raw_dir),
        "prometheus": summarize_prometheus(raw_dir, phases),
        "health": restart_and_event_summary(raw_dir),
        "watchdog_events": watchdog_events,
        "watchdog_triggered": any(event["event"] in {"stop_load", "stop_testbed"} for event in watchdog_events),
    }
    (output_dir / "baseline-resource-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# local-24gb-tested bounded baseline",
        "",
        "This is a local startup/idle/fixed-load feasibility baseline, not full experimental qualification.",
        "",
        "## Load validation",
        "",
        f"- Requested rate: {load_validation['requested_rps']:.3f} RPS",
        f"- Achieved successful rate: {load_validation['achieved_successful_rps']:.3f} RPS",
        f"- Active coverage: {load_validation['coverage_fraction']:.3%}",
        f"- Successful/failed travel requests: {load_validation['successful_travel_requests']}/{load_validation['failed_travel_requests']}",
        "",
        "## Host baseline",
        "",
        "| Phase | Avg used GiB | Peak used GiB | Min available GiB | Avg CPU | Peak CPU | Peak swap GiB | Disk read MiB | Disk write MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in ("idle", "load"):
        row = report["host"][phase]
        lines.append(
            f"| {phase} | {row['memory_used_avg_bytes']/GIB:.2f} | {row['memory_used_peak_bytes']/GIB:.2f} | "
            f"{row['memory_available_min_bytes']/GIB:.2f} | {row['cpu_busy_avg_percent']:.1f}% | "
            f"{row['cpu_busy_peak_percent']:.1f}% | {row['swap_used_peak_bytes']/GIB:.2f} | "
            f"{row['disk_read_bytes']/1024**2:.1f} | {row['disk_write_bytes']/1024**2:.1f} |"
        )
    lines += ["", "## Kubernetes container categories", ""]
    for phase in ("idle", "load"):
        lines += [f"### {phase}", "", "| Category | Avg CPU m | Peak CPU m | Avg memory MiB | Peak memory MiB |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, row in sorted(report["kubernetes_categories"][phase].items()):
            lines.append(
                f"| {name} | {row['cpu_avg_millicores']:.1f} | {row['cpu_peak_millicores']} | "
                f"{row['memory_avg_bytes']/1024**2:.1f} | {row['memory_peak_bytes']/1024**2:.1f} |"
            )
        lines.append("")
    lines += [
        "## Kubernetes per-container peaks",
        "",
        "The JSON report contains every sampled container. The tables below show the ten highest CPU and memory peaks in each phase.",
        "",
    ]
    for phase in ("idle", "load"):
        containers = report["kubernetes_containers"][phase]
        for dimension, key, unit in (
            ("CPU", "cpu_peak_millicores", "m"),
            ("memory", "memory_peak_bytes", "MiB"),
        ):
            lines += [
                f"### {phase}: top {dimension} peaks",
                "",
                f"| Cluster | Namespace | Pod | Container | Category | Peak {dimension} |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
            for row in sorted(containers, key=lambda item: item[key], reverse=True)[:10]:
                value = row[key] if unit == "m" else row[key] / 1024**2
                rendered = f"{value:.0f} m" if unit == "m" else f"{value:.1f} MiB"
                lines.append(
                    f"| {row['cluster']} | {row['namespace']} | {row['pod']} | {row['container']} | "
                    f"{row['category']} | {rendered} |"
                )
            lines.append("")
    lines += [
        "## CPU throttling",
        "",
        "| Cluster | Phase | Max throttled periods/s | Max total periods/s | Max throttled-period ratio | Max throttled seconds/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cluster in ("cluster2", "cluster3"):
        for phase in ("idle", "load"):
            throttled_periods = report["prometheus"][cluster]["throttled_periods_rate"][phase]["max"]
            total_periods = report["prometheus"][cluster]["cpu_periods_rate"][phase]["max"]
            ratio = report["prometheus"][cluster]["throttled_period_ratio"][phase]["max"]
            seconds = report["prometheus"][cluster]["throttled_seconds_rate"][phase]["max"]
            throttled_periods_text = f"{throttled_periods:.6f}" if throttled_periods is not None else "unavailable"
            total_periods_text = f"{total_periods:.6f}" if total_periods is not None else "unavailable"
            ratio_text = f"{ratio:.6f}" if ratio is not None else "unavailable"
            seconds_text = f"{seconds:.6f}" if seconds is not None else "unavailable"
            lines.append(
                f"| {cluster} | {phase} | {throttled_periods_text} | {total_periods_text} | "
                f"{ratio_text} | {seconds_text} |"
            )
    lines += [
        "",
        "## Safety and health",
        "",
        f"- Watchdog triggered: {report['watchdog_triggered']}",
        f"- Cluster2: {report['health']['cluster2']}",
        f"- Cluster3: {report['health']['cluster3']}",
        "- Full CPU throttling time series are retained in the Prometheus exports.",
        "",
    ]
    (output_dir / "baseline-resource-report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def cleanup_load(args: argparse.Namespace) -> None:
    kubectl(args, args.load_cluster, "-n", args.namespace, "delete", "job", LOAD_JOB, "--ignore-not-found", "--wait=false", check=False)
    kubectl(args, args.load_cluster, "-n", args.namespace, "delete", "configmap", LOAD_CONFIGMAP, "--ignore-not-found", check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, default=DEFAULT_KUBECONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--load-cluster", default="cluster2")
    parser.add_argument("--idle-seconds", type=int, default=900)
    parser.add_argument("--load-seconds", type=int, default=900)
    parser.add_argument("--target-rps", type=float, default=0.4)
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--memory-threshold-bytes", type=int, default=3 * GIB)
    parser.add_argument("--prometheus-port", type=int, default=39120)
    parser.add_argument("--locust-image", default="dstefanov46/locust-test:0.66.0")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "results" / "local-24gb-measurement" / f"run-{stamp}"
    output_dir = output_dir.resolve()
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True)
    initialize_series(raw_dir)

    phase_file = raw_dir / "phase.txt"
    stop_file = raw_dir / "watchdog.stop"
    abort_file = raw_dir / "watchdog.abort"
    phase_file.write_text("preflight\n", encoding="utf-8")
    preflight_payload = preflight(args, raw_dir)
    contract = {
        "schema_version": "headlight.local-measurement/v1",
        "profile": "local-24gb-tested",
        "idle_seconds": args.idle_seconds,
        "load_seconds": args.load_seconds,
        "target_rps": args.target_rps,
        "memory_threshold_bytes": args.memory_threshold_bytes,
        "watchdog_sustained_seconds": 6,
        "watchdog_stop_testbed_after_additional_seconds": 20,
        "preflight": preflight_payload,
        "claim_scope": "measurement-only; no profiling, training, Stage 8, or final evaluation",
    }
    (output_dir / "measurement-contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    watchdog_log = (logs_dir / "watchdog.log").open("w", encoding="utf-8")
    watchdog = subprocess.Popen(
        [
            "python3", str(WATCHDOG), "--kubeconfig", str(args.kubeconfig), "--output-dir", str(raw_dir),
            "--phase-file", str(phase_file), "--stop-file", str(stop_file), "--abort-file", str(abort_file),
            "--namespace", args.namespace, "--load-cluster", args.load_cluster, "--load-job", LOAD_JOB,
            "--threshold-bytes", str(args.memory_threshold_bytes),
        ],
        cwd=REPO_ROOT,
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sampler = Sampler(args, raw_dir, phase_file)
    phases: dict[str, tuple[float, float]] = {}
    load_validation: dict[str, Any] | None = None
    success = False
    try:
        snapshot(args, raw_dir, "before")
        sampler.start()

        phase_file.write_text("idle\n", encoding="utf-8")
        idle_start = time.time()
        bounded_wait(args.idle_seconds, abort_file)
        idle_end = time.time()
        phases["idle"] = (idle_start, idle_end)

        start_load(args, raw_dir)
        phase_file.write_text("load\n", encoding="utf-8")
        load_start = time.time()
        wait_for_load(args, abort_file)
        load_end = time.time()
        phases["load"] = (load_start, load_end)
        load_validation = collect_load(args, raw_dir)

        phase_file.write_text("collect\n", encoding="utf-8")
        snapshot(args, raw_dir, "after")
        sampler.stop()
        collect_prometheus(args, raw_dir, idle_start, load_end)
        stop_file.touch()
        watchdog.wait(timeout=15)
        write_report(output_dir, raw_dir, phases, load_validation)
        success = True
    except Exception as exc:
        (logs_dir / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        snapshot(args, raw_dir, "failure")
        raise
    finally:
        if sampler.is_alive():
            try:
                sampler.stop()
            except Exception as exc:
                (logs_dir / "sampler-failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        stop_file.touch()
        try:
            watchdog.wait(timeout=15)
        except subprocess.TimeoutExpired:
            watchdog.terminate()
            watchdog.wait(timeout=10)
        watchdog_log.close()
        if success:
            cleanup_load(args)

    print(f"Bounded local measurement complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
