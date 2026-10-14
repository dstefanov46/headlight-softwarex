#!/usr/bin/env python3
"""Run the bounded seed-201 local infrastructure qualification campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from experiments.run_local_24gb_measurement import (
    GIB,
    LOCUSTFILE,
    REPO_ROOT,
    Sampler,
    collect_prometheus,
    initialize_series,
    preflight,
    restart_and_event_summary,
    snapshot,
    summarize_prometheus,
)


PROTOCOL_PATH = REPO_ROOT / "experiments" / "configs" / "qualification" / "research-v4-infrastructure-qualification.yaml"
WATCHDOG = REPO_ROOT / "experiments" / "scripts" / "local_measurement_watchdog.py"
LOAD_JOB = "headlight-infra-load"
LOAD_CONFIGMAP = "headlight-infra-load"
APPLICATION_MEMORY_REQUEST_MIB = 144


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    *,
    timeout: float = 60,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result


def kubectl(args: argparse.Namespace, context: str, *extra: str, timeout: float = 60, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", context, *extra],
        timeout=timeout,
        check=check,
    )


def validate_protocol(protocol: dict[str, Any], seed: int) -> dict[str, Any]:
    spec = protocol["spec"]
    if spec["seed_role"] != "infrastructure-qualification" or seed not in spec["seeds"]:
        raise RuntimeError(f"seed {seed} is not registered for infrastructure qualification")
    if any(spec["restrictions"].get(name) for name in ("training_input", "final_evaluation", "final_matrix")):
        raise RuntimeError("infrastructure protocol has unsafe result-use permissions")
    return spec


def ensure_deadline(deadline: float, abort_file: Path) -> None:
    if abort_file.exists():
        raise RuntimeError(abort_file.read_text(encoding="utf-8").strip())
    if time.monotonic() >= deadline:
        raise RuntimeError("infrastructure qualification exceeded its registered three-hour bound")


def cleanup_load(args: argparse.Namespace) -> None:
    for kind, name in (("job", LOAD_JOB), ("configmap", LOAD_CONFIGMAP)):
        kubectl(args, "cluster2", "-n", args.load_namespace, "delete", kind, name, "--ignore-not-found", "--wait=true", check=False)


def load_manifest(args: argparse.Namespace, segments: list[dict[str, int]]) -> dict[str, Any]:
    duration = sum(segment["duration_seconds"] for segment in segments)
    users = max(segment["users"] for segment in segments)
    command = "\n".join(
        [
            "/opt/venv/bin/locust --headless -f /app/locustfile.py "
            f"--users {users} --spawn-rate {users} --run-time {duration + 5}s "
            "--exit-code-on-error 0 --csv /tmp/headlight-infra",
            'rc="$?"',
            "echo __HEADLIGHT_REQUEST_CSV_BEGIN__",
            "cat /app/headlight_requests.csv 2>/dev/null || true",
            "echo __HEADLIGHT_REQUEST_CSV_END__",
            "echo __HEADLIGHT_LOCUST_STATS_BEGIN__",
            "cat /tmp/headlight-infra_stats.csv 2>/dev/null || true",
            "echo __HEADLIGHT_LOCUST_STATS_END__",
            'exit "$rc"',
        ]
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": LOAD_JOB,
            "namespace": args.load_namespace,
            "labels": {"headlight.nephele/measurement": "infrastructure-qualification"},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": duration + 180,
            "template": {
                "metadata": {
                    "labels": {"headlight.nephele/measurement": "infrastructure-qualification"},
                    "annotations": {"sidecar.istio.io/inject": "false"},
                },
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
                                {"name": "HEADLIGHT_TARGET_RPS", "value": str(args.per_user_rps)},
                                {"name": "HEADLIGHT_TRAINTICKET_NAMESPACE", "value": args.namespace},
                                {"name": "HEADLIGHT_TRACE_SEGMENTS", "value": json.dumps(segments, separators=(",", ":"))},
                            ],
                            "resources": {
                                "requests": {"cpu": "25m", "memory": "64Mi"},
                                "limits": {"cpu": "250m", "memory": "256Mi"},
                            },
                            "volumeMounts": [
                                {"name": "locustfile", "mountPath": "/app/locustfile.py", "subPath": "locustfile.py"}
                            ],
                        }
                    ],
                    "volumes": [{"name": "locustfile", "configMap": {"name": LOAD_CONFIGMAP}}],
                },
            },
        },
    }


def marked_block(text: str, start: str, end: str) -> str:
    active = False
    result: list[str] = []
    for line in text.splitlines():
        if line == start:
            active = True
            continue
        if line == end:
            active = False
            continue
        if active:
            result.append(line)
    return "\n".join(result) + ("\n" if result else "")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def choose_operating_record(records: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """Choose the highest offered search point whose demand was actually delivered."""
    reliable = [
        record
        for record in records
        if record["success_fraction"] >= 0.99 and record["delivery_fraction"] >= 0.90
    ]
    if reliable:
        return max(reliable, key=lambda record: record["requested_average_rps"]), True
    return max(records, key=lambda record: record["achieved_successful_rps"]), False


def run_load(
    args: argparse.Namespace,
    raw_dir: Path,
    phase_file: Path,
    abort_file: Path,
    deadline: float,
    label: str,
    segments: list[dict[str, int]],
    action_service: str | None = None,
    scheduled_actions: list[dict[str, int]] | None = None,
    action_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ensure_deadline(deadline, abort_file)
    cleanup_load(args)
    kubectl(
        args,
        "cluster2",
        "-n",
        args.load_namespace,
        "create",
        "configmap",
        LOAD_CONFIGMAP,
        f"--from-file=locustfile.py={LOCUSTFILE}",
    )
    manifest = load_manifest(args, segments)
    stage_dir = raw_dir / "loads" / label
    stage_dir.mkdir(parents=True)
    (stage_dir / "job.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", "cluster2", "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )
    phase_file.write_text(label + "\n", encoding="utf-8")
    started = time.time()
    duration = sum(segment["duration_seconds"] for segment in segments)
    action_errors: list[str] = []

    def exercise_actions() -> None:
        for action in scheduled_actions or []:
            while time.time() - started < action["offset_seconds"]:
                try:
                    ensure_deadline(deadline, abort_file)
                except Exception as exc:
                    action_errors.append(str(exc))
                    return
                time.sleep(1)
            try:
                action_runner = getattr(args, "scale_action", set_replicas)
                record = action_runner(
                    args,
                    str(action_service),
                    action["cluster2"],
                    action["cluster3"],
                    abort_file,
                    deadline,
                )
                record.update({"trace": label, "scheduled_offset_seconds": action["offset_seconds"]})
                if action_records is not None:
                    action_records.append(record)
            except Exception as exc:
                action_errors.append(str(exc))
                return

    action_thread: threading.Thread | None = None
    if scheduled_actions:
        if not action_service:
            raise RuntimeError(f"{label}: scheduled scale actions require a service")
        action_thread = threading.Thread(target=exercise_actions, daemon=True)
        action_thread.start()
    wait_deadline = time.monotonic() + duration + 240
    mongostat_sampled = False
    while time.monotonic() < wait_deadline:
        ensure_deadline(deadline, abort_file)
        if not mongostat_sampled and time.time() - started >= duration / 2:
            sample_mongostat(args, raw_dir, label)
            mongostat_sampled = True
        response = kubectl(args, "cluster2", "-n", args.load_namespace, "get", "job", LOAD_JOB, "-o", "json", check=False)
        if response.returncode != 0:
            raise RuntimeError(f"{label}: load Job disappeared")
        status = json.loads(response.stdout).get("status", {})
        if status.get("succeeded", 0):
            break
        if status.get("failed", 0):
            raise RuntimeError(f"{label}: load Job failed: {status}")
        time.sleep(5)
    else:
        raise RuntimeError(f"{label}: load Job exceeded its bounded completion deadline")
    if action_thread is not None:
        action_thread.join(timeout=args.readiness_timeout_seconds + 30)
        if action_thread.is_alive():
            raise RuntimeError(f"{label}: scheduled scale actions exceeded their bound")
        if action_errors:
            raise RuntimeError(f"{label}: scheduled scale action failed: {action_errors[0]}")
    finished = time.time()
    pod = kubectl(
        args,
        "cluster2",
        "-n",
        args.load_namespace,
        "get",
        "pod",
        "-l",
        f"job-name={LOAD_JOB}",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).stdout.strip()
    logs = kubectl(args, "cluster2", "-n", args.load_namespace, "logs", pod).stdout
    (stage_dir / "pod.log").write_text(logs, encoding="utf-8")
    requests_text = marked_block(logs, "__HEADLIGHT_REQUEST_CSV_BEGIN__", "__HEADLIGHT_REQUEST_CSV_END__")
    (stage_dir / "request-events.csv").write_text(requests_text, encoding="utf-8")
    (stage_dir / "locust-stats.csv").write_text(
        marked_block(logs, "__HEADLIGHT_LOCUST_STATS_BEGIN__", "__HEADLIGHT_LOCUST_STATS_END__"),
        encoding="utf-8",
    )
    rows = [row for row in csv.DictReader(requests_text.splitlines()) if row.get("endpoint") == "travel.left"]
    successful = [row for row in rows if row.get("status", "").startswith(("2", "3")) and not row.get("exception")]
    latencies = [float(row["response_time_ms"]) for row in successful]
    requested = sum(segment["duration_seconds"] * segment["users"] * args.per_user_rps for segment in segments)
    record = {
        "label": label,
        "segments": segments,
        "started_at": started,
        "finished_at": finished,
        "nominal_duration_seconds": duration,
        "wall_duration_seconds": finished - started,
        "requested_requests": requested,
        "requested_average_rps": requested / duration,
        "delivered_requests": len(rows),
        "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "achieved_successful_rps": len(successful) / duration,
        "delivery_fraction": len(successful) / requested if requested else 0.0,
        "success_fraction": len(successful) / len(rows) if rows else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
    }
    (stage_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cleanup_load(args)
    return record


def set_replicas(
    args: argparse.Namespace,
    service: str,
    cluster2: int,
    cluster3: int,
    abort_file: Path,
    deadline: float,
) -> dict[str, Any]:
    started = time.time()
    placements = [
        {"clusterName": "cluster2", "replicas": cluster2},
        {"clusterName": "cluster3", "replicas": cluster3},
    ]
    kubectl(
        args,
        "karmada-apiserver",
        "-n",
        args.namespace,
        "patch",
        "controlledreplicapropagation",
        service,
        "--type=merge",
        "-p",
        json.dumps({"spec": {"placements": placements}}),
    )
    expected = {"cluster2": cluster2, "cluster3": cluster3}
    ready_at: dict[str, float] = {}
    wait_deadline = time.monotonic() + args.readiness_timeout_seconds
    while time.monotonic() < wait_deadline:
        ensure_deadline(deadline, abort_file)
        for cluster, replicas in expected.items():
            if cluster in ready_at:
                continue
            item = json.loads(
                kubectl(args, cluster, "-n", args.namespace, "get", "deployment", service, "-o", "json").stdout
            )
            status = item.get("status", {})
            if (
                int(item.get("spec", {}).get("replicas", -1)) == replicas
                and int(status.get("readyReplicas", 0)) == replicas
                and int(status.get("availableReplicas", 0)) == replicas
                and int(status.get("updatedReplicas", 0)) == replicas
                and int(status.get("unavailableReplicas", 0)) == 0
            ):
                ready_at[cluster] = time.time()
        if len(ready_at) == 2:
            break
        time.sleep(5)
    if len(ready_at) != 2:
        raise RuntimeError(f"{service}: placement {cluster2}/{cluster3} did not become Ready")
    return {
        "service": service,
        "cluster2": cluster2,
        "cluster3": cluster3,
        "decision_at": started,
        "ready_at": ready_at,
        "decision_to_all_ready_seconds": max(ready_at.values()) - started,
    }


def select_bottleneck(raw_dir: Path, phases: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    with (raw_dir / "kubernetes-container-series.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["phase"] in phases
                and row["cluster"] == "cluster2"
                and row["namespace"] == "trainticket"
                and row["container"].startswith("ts-")
                and not row["container"].endswith("-mongo")
                and row["container"] != "istio-proxy"
            ):
                grouped[row["container"]].append(int(row["cpu_millicores"]))
    if not grouped:
        raise RuntimeError("no application CPU samples available to identify a bottleneck")
    ranked = sorted(
        (
            {
                "service": service,
                "samples": len(values),
                "mean_cpu_millicores": statistics.fmean(values),
                "peak_cpu_millicores": max(values),
                "peak_fraction_of_200m_limit": max(values) / 200.0,
            }
            for service, values in grouped.items()
        ),
        key=lambda row: (row["mean_cpu_millicores"], row["peak_cpu_millicores"]),
        reverse=True,
    )
    return {"selected": ranked[0], "ranking": ranked}


def sample_mongostat(args: argparse.Namespace, raw_dir: Path, label: str) -> None:
    path = raw_dir / "mongostat.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for cluster in ("cluster2", "cluster3"):
            pods = json.loads(
                kubectl(args, cluster, "-n", args.namespace, "get", "pods", "-o", "json").stdout
            )
            for item in pods.get("items", []):
                containers = [container["name"] for container in item.get("spec", {}).get("containers", [])]
                mongo = next((name for name in containers if name.endswith("-mongo")), None)
                if not mongo:
                    continue
                result = kubectl(
                    args,
                    cluster,
                    "-n",
                    args.namespace,
                    "exec",
                    item["metadata"]["name"],
                    "-c",
                    mongo,
                    "--",
                    "mongostat",
                    "--rowcount",
                    "1",
                    "--json",
                    timeout=30,
                    check=False,
                )
                handle.write(
                    json.dumps(
                        {
                            "timestamp": utc_now(),
                            "phase": label,
                            "cluster": cluster,
                            "pod": item["metadata"]["name"],
                            "container": mongo,
                            "returncode": result.returncode,
                            "output": result.stdout.strip(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def host_phases(raw_dir: Path, phase_names: list[str]) -> dict[str, Any]:
    with (raw_dir / "host-series.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {}
    for phase in phase_names:
        selected = [row for row in rows if row["phase"] == phase]
        available = [int(row["mem_available_bytes"]) for row in selected]
        cpu = [float(row["cpu_busy_percent"]) for row in selected]
        result[phase] = {
            "samples": len(selected),
            "minimum_available_bytes": min(available, default=0),
            "peak_cpu_percent": max(cpu, default=0.0),
            "average_cpu_percent": statistics.fmean(cpu) if cpu else 0.0,
            "peak_swap_used_bytes": max((int(row["swap_used_bytes"]) for row in selected), default=0),
            "swap_in_pages": int(selected[-1]["swap_in_pages"]) - int(selected[0]["swap_in_pages"]) if len(selected) > 1 else 0,
            "swap_out_pages": int(selected[-1]["swap_out_pages"]) - int(selected[0]["swap_out_pages"]) if len(selected) > 1 else 0,
            "disk_read_bytes": int(selected[-1]["disk_read_bytes"]) - int(selected[0]["disk_read_bytes"]) if len(selected) > 1 else 0,
            "disk_write_bytes": int(selected[-1]["disk_write_bytes"]) - int(selected[0]["disk_write_bytes"]) if len(selected) > 1 else 0,
        }
    return result


def summarize_mongostat(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / "mongostat.jsonl"
    summaries: list[dict[str, Any]] = []
    if not path.exists():
        return {"samples": 0, "parseable_samples": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["returncode"] != 0:
            continue
        try:
            payload = json.loads(record["output"].splitlines()[-1])
            values = next(iter(payload.values()))
            queued_readers, queued_writers = (int(value) for value in values.get("qrw", "0|0").split("|"))
            active_readers, active_writers = (int(value) for value in values.get("arw", "0|0").split("|"))
            summaries.append(
                {
                    "phase": record["phase"],
                    "cluster": record["cluster"],
                    "pod": record["pod"],
                    "queued_readers": queued_readers,
                    "queued_writers": queued_writers,
                    "active_readers": active_readers,
                    "active_writers": active_writers,
                    "connections": int(values.get("conn", 0)),
                    "dirty_percent": float(str(values.get("dirty", "0%")).rstrip("%")),
                    "cache_used_percent": float(str(values.get("used", "0%")).rstrip("%")),
                }
            )
        except (ValueError, TypeError, json.JSONDecodeError, StopIteration):
            continue
    return {
        "samples": len(path.read_text(encoding="utf-8").splitlines()),
        "parseable_samples": len(summaries),
        "maximum_queued_readers": max((row["queued_readers"] for row in summaries), default=None),
        "maximum_queued_writers": max((row["queued_writers"] for row in summaries), default=None),
        "maximum_active_readers": max((row["active_readers"] for row in summaries), default=None),
        "maximum_active_writers": max((row["active_writers"] for row in summaries), default=None),
        "maximum_connections": max((row["connections"] for row in summaries), default=None),
        "maximum_dirty_percent": max((row["dirty_percent"] for row in summaries), default=None),
        "maximum_cache_used_percent": max((row["cache_used_percent"] for row in summaries), default=None),
        "records": summaries,
    }


def members_ready(args: argparse.Namespace) -> bool:
    payload = json.loads(kubectl(args, "karmada-apiserver", "get", "clusters", "-o", "json").stdout)
    return all(
        any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in item.get("status", {}).get("conditions", []))
        for item in payload.get("items", [])
    )


def scheduling_failures(raw_dir: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for cluster in ("cluster2", "cluster3"):
        before = json.loads((raw_dir / f"{cluster}-events-before.json").read_text(encoding="utf-8"))
        after = json.loads((raw_dir / f"{cluster}-events-after.json").read_text(encoding="utf-8"))
        known = {item.get("metadata", {}).get("uid") for item in before.get("items", [])}
        result[cluster] = [
            {
                "timestamp": str(item.get("eventTime") or item.get("lastTimestamp") or item.get("firstTimestamp") or ""),
                "reason": str(item.get("reason", "")),
                "object": str(item.get("involvedObject", {}).get("name", "")),
                "message": str(item.get("message", "")),
            }
            for item in after.get("items", [])
            if item.get("metadata", {}).get("uid") not in known
            and item.get("reason") in {"FailedCreate", "FailedScheduling"}
        ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--kubeconfig", type=Path, default=REPO_ROOT / ".kube" / "headlight-testbed.config")
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--readiness-timeout-seconds", type=int, default=600)
    parser.add_argument("--prometheus-port", type=int, default=39130)
    parser.add_argument("--locust-image", default="dstefanov46/locust-test:0.66.0")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    spec = validate_protocol(protocol, args.seed)
    args.memory_threshold_bytes = int(spec["memory_guard"]["minimum_available_bytes"])
    args.per_user_rps = float(spec["load"]["per_user_target_rps"])
    args.load_namespace = str(spec["load_generator_namespace"])
    maximum_duration = int(spec["maximum_duration_seconds"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = REPO_ROOT / spec["artifact_namespace"] / f"seed-{args.seed}"
    output_dir = (args.output_dir or root / f"run-{stamp}").resolve()
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True)
    (output_dir / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    initialize_series(raw_dir)
    phase_file = raw_dir / "phase.txt"
    stop_file = raw_dir / "watchdog.stop"
    abort_file = raw_dir / "watchdog.abort"
    phase_file.write_text("preflight\n", encoding="utf-8")
    preflight(args, raw_dir)
    namespace = kubectl(args, "cluster2", "get", "namespace", args.load_namespace, check=False)
    if namespace.returncode != 0:
        kubectl(args, "cluster2", "create", "namespace", args.load_namespace)
    snapshot(args, raw_dir, "before")

    watchdog_log = (logs_dir / "watchdog.log").open("w", encoding="utf-8")
    watchdog = subprocess.Popen(
        [
            "python3", str(WATCHDOG), "--kubeconfig", str(args.kubeconfig), "--output-dir", str(raw_dir),
            "--phase-file", str(phase_file), "--stop-file", str(stop_file), "--abort-file", str(abort_file),
            "--namespace", args.namespace, "--load-namespace", args.load_namespace,
            "--load-cluster", "cluster2", "--load-job", LOAD_JOB,
            "--threshold-bytes", str(args.memory_threshold_bytes),
        ],
        cwd=REPO_ROOT,
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sampler = Sampler(args, raw_dir, phase_file)
    sampler.start()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + maximum_duration
    load_records: list[dict[str, Any]] = []
    scale_records: list[dict[str, Any]] = []
    phase_windows: dict[str, tuple[float, float]] = {}
    bottleneck: dict[str, Any] | None = None
    restored_service: str | None = None
    success = False
    try:
        search_phases: list[str] = []
        search_records: list[dict[str, Any]] = []
        for users in spec["load"]["search_users"]:
            label = f"search-{users}-users"
            record = run_load(
                args,
                raw_dir,
                phase_file,
                abort_file,
                deadline,
                label,
                [{"duration_seconds": int(spec["load"]["fixed_stage_seconds"]), "users": int(users)}],
            )
            load_records.append(record)
            search_records.append(record)
            phase_windows[label] = (record["started_at"], record["finished_at"])
            search_phases.append(label)
            sample_mongostat(args, raw_dir, label)

        bottleneck = select_bottleneck(raw_dir, search_phases)
        (output_dir / "bottleneck.json").write_text(json.dumps(bottleneck, indent=2) + "\n", encoding="utf-8")
        service = bottleneck["selected"]["service"]
        restored_service = service
        operating_record, operating_record_reliable = choose_operating_record(search_records)
        comparable_users = int(operating_record["segments"][0]["users"])

        phase_file.write_text("scale-out-cluster2\n", encoding="utf-8")
        scale_records.append(set_replicas(args, service, 2, 1, abort_file, deadline))
        record = run_load(
            args,
            raw_dir,
            phase_file,
            abort_file,
            deadline,
            "scaled-cluster2",
            [{"duration_seconds": int(spec["load"]["fixed_stage_seconds"]), "users": comparable_users}],
        )
        load_records.append(record)
        phase_windows[record["label"]] = (record["started_at"], record["finished_at"])
        sample_mongostat(args, raw_dir, record["label"])

        phase_file.write_text("scale-out-simultaneous\n", encoding="utf-8")
        scale_records.append(set_replicas(args, service, 2, 2, abort_file, deadline))
        record = run_load(
            args,
            raw_dir,
            phase_file,
            abort_file,
            deadline,
            "scaled-simultaneous",
            [{"duration_seconds": int(spec["load"]["fixed_stage_seconds"]), "users": comparable_users}],
        )
        load_records.append(record)
        phase_windows[record["label"]] = (record["started_at"], record["finished_at"])
        sample_mongostat(args, raw_dir, record["label"])

        phase_file.write_text("scale-in-recovery\n", encoding="utf-8")
        scale_records.append(set_replicas(args, service, 1, 1, abort_file, deadline))
        recovery_start = time.time()
        while time.time() - recovery_start < 120:
            ensure_deadline(deadline, abort_file)
            time.sleep(2)

        high_users = comparable_users
        traces = {
            "rapid-ramp": {
                "segments": [
                    {"duration_seconds": 60, "users": 1},
                    {"duration_seconds": 300, "users": high_users},
                    {"duration_seconds": 180, "users": 2},
                ],
                "actions": [
                    {"offset_seconds": 60, "cluster2": 2, "cluster3": 1},
                    {"offset_seconds": 360, "cluster2": 1, "cluster3": 1},
                ],
            },
            "repeating": {
                "segments": [
                    {"duration_seconds": 60, "users": 2},
                    {"duration_seconds": 240, "users": high_users},
                    {"duration_seconds": 180, "users": 2},
                    {"duration_seconds": 240, "users": high_users},
                    {"duration_seconds": 180, "users": 2},
                ],
                "actions": [
                    {"offset_seconds": 60, "cluster2": 2, "cluster3": 1},
                    {"offset_seconds": 300, "cluster2": 1, "cluster3": 1},
                    {"offset_seconds": 480, "cluster2": 2, "cluster3": 1},
                    {"offset_seconds": 720, "cluster2": 1, "cluster3": 1},
                ],
            },
            "flash-crowd": {
                "segments": [
                    {"duration_seconds": 120, "users": 1},
                    {"duration_seconds": 300, "users": high_users},
                    {"duration_seconds": 240, "users": 1},
                ],
                "actions": [
                    {"offset_seconds": 120, "cluster2": 2, "cluster3": 2},
                    {"offset_seconds": 420, "cluster2": 1, "cluster3": 1},
                ],
            },
        }
        for label, trace in traces.items():
            record = run_load(
                args,
                raw_dir,
                phase_file,
                abort_file,
                deadline,
                label,
                trace["segments"],
                action_service=service,
                scheduled_actions=trace["actions"],
                action_records=scale_records,
            )
            load_records.append(record)
            phase_windows[label] = (record["started_at"], record["finished_at"])
            sample_mongostat(args, raw_dir, label)

        phase_file.write_text("collect\n", encoding="utf-8")
        snapshot(args, raw_dir, "after")
        sampler.stop()
        collect_prometheus(args, raw_dir, min(start for start, _ in phase_windows.values()), max(end for _, end in phase_windows.values()))
        prometheus = summarize_prometheus(raw_dir, phase_windows)
        health = restart_and_event_summary(raw_dir)
        scheduling = scheduling_failures(raw_dir)
        for cluster, failures in scheduling.items():
            health[cluster]["scheduling_failures"] = len(failures)
        watchdog_events = [json.loads(line) for line in (raw_dir / "watchdog-events.jsonl").read_text().splitlines()]
        host = host_phases(raw_dir, list(phase_windows))
        minimum_available = min((row["minimum_available_bytes"] for row in host.values()), default=0)
        watchdog_triggered = any(event["event"] in {"stop_load", "stop_testbed"} for event in watchdog_events)
        health_clean = all(
            row["restart_increase"] == 0
            and row["pending_pods_after"] == 0
            and row["oom_events"] == 0
            and row["eviction_events"] == 0
            and row["oomkilled_containers_after"] == 0
            and row["scheduling_failures"] == 0
            for row in health.values()
        )
        telemetry_complete = all(
            prometheus[cluster][metric][phase]["available"]
            for cluster in ("cluster2", "cluster3")
            for metric in ("cpu_rate_by_container", "memory_by_container", "throttled_period_ratio", "pending_pods")
            for phase in phase_windows
        ) and all(
            prometheus["cluster2"][metric][phase]["available"]
            for metric in ("request_rate", "failure_rate")
            for phase in phase_windows
        )
        comparison_and_trace_records = [
            row for row in load_records if not row["label"].startswith("search-")
        ]
        delivered_reliably = operating_record_reliable and all(
            row["success_fraction"] >= 0.99 and row["delivery_fraction"] >= 0.90
            for row in comparison_and_trace_records
        )
        baseline_record = operating_record
        scaled_record = next(row for row in load_records if row["label"] == "scaled-cluster2")
        throughput_gain = scaled_record["achieved_successful_rps"] / max(baseline_record["achieved_successful_rps"], 1e-9) - 1
        latency_gain = 1 - scaled_record["p95_latency_ms"] / max(baseline_record["p95_latency_ms"], 1e-9)
        useful_scale_out = throughput_gain >= 0.05 or latency_gain >= 0.05
        control_plane_stable = members_ready(args)
        database_contention = summarize_mongostat(raw_dir)
        measured_seconds = sum(end - start for start, end in phase_windows.values())
        swap_in_pages = sum(row["swap_in_pages"] for row in host.values())
        swap_out_pages = sum(row["swap_out_pages"] for row in host.values())
        swap_io_pages_per_second = (swap_in_pages + swap_out_pages) / max(measured_seconds, 1.0)
        pass_gates = {
            "memory_headroom": minimum_available >= args.memory_threshold_bytes,
            "watchdog": not watchdog_triggered,
            "health": health_clean,
            "telemetry": telemetry_complete,
            "delivered_load": delivered_reliably,
            "useful_scale_out": useful_scale_out,
            "control_plane": control_plane_stable,
            "no_sustained_swap_io": swap_io_pages_per_second < 1.0,
        }
        verdict = "PASS" if all(pass_gates.values()) else "CONDITIONAL"
        report = {
            "schema_version": "headlight.infrastructure-qualification/v1",
            "claim_scope": "infrastructure qualification only; never training or final evaluation data",
            "protocol": protocol,
            "seed": args.seed,
            "duration_seconds": time.monotonic() - started_monotonic,
            "verdict": verdict,
            "pass_gates": pass_gates,
            "minimum_available_bytes": minimum_available,
            "swap_io": {
                "swap_in_pages": swap_in_pages,
                "swap_out_pages": swap_out_pages,
                "pages_per_second": swap_io_pages_per_second,
                "sustained_threshold_pages_per_second": 1.0,
            },
            "load_stages": load_records,
            "operating_point": {
                "record": operating_record,
                "reliably_delivered": operating_record_reliable,
                "selection": "highest offered search point with >=99% success and >=90% delivered demand",
            },
            "bottleneck": bottleneck,
            "scale_actions": scale_records,
            "scale_comparison": {
                "baseline": baseline_record["label"],
                "scaled": scaled_record["label"],
                "throughput_gain_fraction": throughput_gain,
                "p95_latency_gain_fraction": latency_gain,
            },
            "host": host,
            "database_contention": database_contention,
            "prometheus": prometheus,
            "health": health,
            "scheduling_failures": scheduling,
            "watchdog_events": watchdog_events,
            "qualified_configuration": {
                "profile": "local-24gb-tested",
                "workload_clusters": ["cluster2", "cluster3"],
                "applications": 11,
                "databases": 7,
                "initial_replicas_per_service_per_cluster": 1,
                "tested_extra_application_budget_per_cluster": 1,
                "maximum_replicas_per_service_per_cluster": 2,
                "namespace_requests_memory_quota_mib": 2078,
                "load_generator_namespace": args.load_namespace,
                "fair_concurrency_rule": (
                    "Serialize scale actions through one admission queue; choose contenders by oldest decision timestamp, "
                    "then stable service name. ResourceQuota is the hard backstop, not the fairness mechanism."
                ),
                "controller_constraint": (
                    "Every controller must use this service profile, the same per-cluster 1..2 bounds, and the same "
                    "serialized global one-extra-application admission queue."
                ),
            },
        }
        (output_dir / "qualification-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        success = True
    except Exception as exc:
        (logs_dir / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        snapshot(args, raw_dir, "failure")
        raise
    finally:
        cleanup_load(args)
        if restored_service:
            try:
                set_replicas(args, restored_service, 1, 1, abort_file, time.monotonic() + 600)
            except Exception as exc:
                (logs_dir / "restore-failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        if sampler.is_alive():
            sampler.stop()
        stop_file.touch()
        try:
            watchdog.wait(timeout=15)
        except subprocess.TimeoutExpired:
            watchdog.terminate()
            watchdog.wait(timeout=10)
        watchdog_log.close()
        (output_dir / "status.json").write_text(
            json.dumps({"success": success, "finished_at": utc_now()}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
