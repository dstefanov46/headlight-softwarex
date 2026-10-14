#!/usr/bin/env python3
"""Run the single bounded seed-201 CONDITIONAL qualification retest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import run_local_24gb_infrastructure_qualification as base
from experiments.run_local_24gb_measurement import (
    Sampler,
    collect_prometheus,
    initialize_series,
    preflight,
    restart_and_event_summary,
    snapshot,
    summarize_prometheus,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = REPO_ROOT / "experiments" / "scripts" / "local_measurement_watchdog.py"
PROTOCOL = REPO_ROOT / "experiments" / "configs" / "qualification" / "research-v4-infrastructure-qualification.yaml"
PROFILE = REPO_ROOT / "ai-module" / "configs" / "service_profiles" / "advanced-ticket-local-24gb-tested.json"
RENDERED = REPO_ROOT / "experiments" / "testbed" / "trainticket-minimal" / "rendered-advanced-ticket.yaml"
CRP_PATCH = REPO_ROOT / "experiments" / "patches" / "controlled-replica-propagation" / "0001-wait-for-resourcebinding-suspension.patch"
ADMISSION_NAMESPACE = "headlight-infra-qualification"
ADMISSION_CONFIGMAP = "local-24gb-tested-scale-admission"
QUOTA_NAME = "local-24gb-tested-admission"
APP_REQUEST_MIB = 144
BASELINE_REPLICAS = 1
MAX_EXTRA_APPS_PER_CLUSTER = 1
SWAP_SAMPLE_SECONDS = 5.0
SWAP_PHASE_AVERAGE_LIMIT = 1.0
CONTROLLER_SOURCE_REVISION = "8d11e02ab7b3005e7c0f9c74a32b3aaa67f35c25"
CONTROLLER_IMAGE = "headlight/controlled-replica-propagation:local-24gb-tested-rb-suspension"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_queue(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: (float(item["decision_epoch"]), str(item["service"]), str(item["id"])))


def projected_extra_replicas(current: dict[str, int], service: str, target: int) -> int:
    projected = dict(current)
    projected[service] = target
    return sum(max(0, replicas - BASELINE_REPLICAS) for replicas in projected.values())


class AdmissionQueue:
    """Optimistic-concurrency queue shared through the Karmada API."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def ensure(self) -> None:
        ns = base.kubectl(self.args, "karmada-apiserver", "get", "namespace", ADMISSION_NAMESPACE, check=False)
        if ns.returncode != 0:
            base.kubectl(self.args, "karmada-apiserver", "create", "namespace", ADMISSION_NAMESPACE)
        existing = base.kubectl(
            self.args, "karmada-apiserver", "-n", ADMISSION_NAMESPACE,
            "get", "configmap", ADMISSION_CONFIGMAP, check=False,
        )
        if existing.returncode == 0:
            return
        state = {"schema": "headlight.scale-admission/v1", "active": None, "queue": []}
        manifest = self._manifest(state)
        base.run(
            ["kubectl", "--kubeconfig", str(self.args.kubeconfig), "--context", "karmada-apiserver", "create", "-f", "-"],
            input_text=yaml.safe_dump(manifest, sort_keys=False), check=False,
        )

    def _manifest(self, state: dict[str, Any], resource_version: str | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {"name": ADMISSION_CONFIGMAP, "namespace": ADMISSION_NAMESPACE}
        if resource_version:
            metadata["resourceVersion"] = resource_version
        return {
            "apiVersion": "v1", "kind": "ConfigMap", "metadata": metadata,
            "data": {"state.json": json.dumps(state, sort_keys=True, separators=(",", ":"))},
        }

    def read(self) -> tuple[dict[str, Any], str]:
        payload = json.loads(base.kubectl(
            self.args, "karmada-apiserver", "-n", ADMISSION_NAMESPACE,
            "get", "configmap", ADMISSION_CONFIGMAP, "-o", "json",
        ).stdout)
        return json.loads(payload["data"]["state.json"]), payload["metadata"]["resourceVersion"]

    def replace(self, state: dict[str, Any], resource_version: str) -> bool:
        result = base.run(
            ["kubectl", "--kubeconfig", str(self.args.kubeconfig), "--context", "karmada-apiserver", "replace", "-f", "-"],
            input_text=yaml.safe_dump(self._manifest(state, resource_version), sort_keys=False), check=False,
        )
        return result.returncode == 0

    def acquire(self, service: str, targets: dict[str, int], deadline: float) -> dict[str, Any]:
        request = {
            "id": str(uuid.uuid4()), "service": service, "targets": targets,
            "decision_epoch": time.time(), "decision_at": utc_now(), "pid": os.getpid(),
        }
        registered = False
        while time.monotonic() < deadline:
            state, resource_version = self.read()
            queue = [item for item in state.get("queue", []) if item.get("id") != request["id"]]
            if not registered:
                queue.append(request)
            queue = ordered_queue(queue)
            active = state.get("active")
            if active is None and queue and queue[0]["id"] == request["id"]:
                active = request["id"]
            replacement = {"schema": "headlight.scale-admission/v1", "active": active, "queue": queue}
            if self.replace(replacement, resource_version):
                registered = True
                if active == request["id"]:
                    request["admitted_at"] = utc_now()
                    request["admitted_epoch"] = time.time()
                    return request
            time.sleep(0.5)
        raise RuntimeError(f"admission timeout for {service}; queue was not bypassed or stolen")

    def release(self, request_id: str) -> None:
        for _ in range(60):
            state, resource_version = self.read()
            queue = [item for item in state.get("queue", []) if item.get("id") != request_id]
            active = None if state.get("active") == request_id else state.get("active")
            replacement = {"schema": "headlight.scale-admission/v1", "active": active, "queue": ordered_queue(queue)}
            if self.replace(replacement, resource_version):
                return
            time.sleep(0.5)
        raise RuntimeError(f"failed to release admission request {request_id}")


def stable_cluster_state(args: argparse.Namespace, cluster: str, service: str, target: int) -> dict[str, Any]:
    deployments = json.loads(base.kubectl(
        args, cluster, "-n", args.namespace, "get", "deployments", "-o", "json",
    ).stdout)["items"]
    active = [item for item in deployments if item["metadata"]["name"].startswith("ts-")]
    unstable: list[str] = []
    app_replicas: dict[str, int] = {}
    for item in active:
        name = item["metadata"]["name"]
        desired = int(item.get("spec", {}).get("replicas", 0))
        status = item.get("status", {})
        if not name.endswith("-mongo"):
            app_replicas[name] = desired
        if not (
            int(status.get("observedGeneration", 0)) >= int(item["metadata"].get("generation", 0))
            and int(status.get("replicas", 0)) == desired
            and int(status.get("updatedReplicas", 0)) == desired
            and int(status.get("readyReplicas", 0)) == desired
            and int(status.get("availableReplicas", 0)) == desired
            and int(status.get("unavailableReplicas", 0)) == 0
        ):
            unstable.append(name)
    pods = json.loads(base.kubectl(args, cluster, "-n", args.namespace, "get", "pods", "-o", "json").stdout)["items"]
    in_flight = [
        item["metadata"]["name"] for item in pods
        if item["metadata"].get("labels", {}).get("app", "").startswith("ts-")
        and (item["metadata"].get("deletionTimestamp") or item.get("status", {}).get("phase") not in {"Running", "Succeeded"})
    ]
    if unstable or in_flight:
        raise RuntimeError(f"{cluster}: in-flight workload blocks admission: deployments={unstable}, pods={in_flight}")
    extras = projected_extra_replicas(app_replicas, service, target)
    if extras > MAX_EXTRA_APPS_PER_CLUSTER:
        raise RuntimeError(f"{cluster}: projected extra app replicas {extras} exceeds measured budget 1")
    quota = json.loads(base.kubectl(
        args, cluster, "-n", args.namespace, "get", "resourcequota", QUOTA_NAME, "-o", "json",
    ).stdout)
    used_mib = int(quota["status"]["used"]["requests.memory"].removesuffix("Mi"))
    hard_mib = int(quota["status"]["hard"]["requests.memory"].removesuffix("Mi"))
    current_target = app_replicas[service]
    projected_used = used_mib + max(0, target - current_target) * APP_REQUEST_MIB
    if projected_used > hard_mib:
        raise RuntimeError(f"{cluster}: projected requests.memory {projected_used}Mi exceeds quota {hard_mib}Mi")
    return {
        "cluster": cluster, "applications": len(app_replicas), "deployments": len(active),
        "current_target_replicas": current_target, "projected_extra_app_replicas": extras,
        "quota_used_mib": used_mib, "quota_hard_mib": hard_mib, "projected_quota_used_mib": projected_used,
    }


def epoch(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def pod_timings(item: dict[str, Any], service: str) -> dict[str, Any]:
    status = item.get("status", {})
    conditions = {condition["type"]: condition for condition in status.get("conditions", [])}
    app = next((row for row in status.get("containerStatuses", []) if row["name"] == service), None)
    started = app.get("state", {}).get("running", {}).get("startedAt") if app else None
    scheduled = conditions.get("PodScheduled", {}).get("lastTransitionTime")
    ready = conditions.get("Ready", {}).get("lastTransitionTime")
    return {
        "pod": item["metadata"]["name"], "uid": item["metadata"]["uid"],
        "created_at": item["metadata"].get("creationTimestamp"), "scheduled_at": scheduled,
        "container_started_at": started, "ready_at": ready,
        "creation_to_scheduled_seconds": (epoch(scheduled) - epoch(item["metadata"].get("creationTimestamp"))) if scheduled else None,
        "scheduled_to_container_start_seconds": (epoch(started) - epoch(scheduled)) if started and scheduled else None,
        "container_start_to_ready_seconds": (epoch(ready) - epoch(started)) if ready and started else None,
    }


def safe_set_replicas(
    args: argparse.Namespace, service: str, cluster2: int, cluster3: int,
    abort_file: Path, deadline: float,
) -> dict[str, Any]:
    base.ensure_deadline(deadline, abort_file)
    decision_epoch = time.time()
    targets = {"cluster2": cluster2, "cluster3": cluster3}
    queue: AdmissionQueue = args.admission_queue
    request = queue.acquire(service, targets, min(deadline, time.monotonic() + 180))
    record: dict[str, Any] = {
        "service": service, "targets": targets, "decision_at": utc_now(), "decision_epoch": decision_epoch,
        "admission": request, "preconditions": {}, "member_spec_observed_at": {}, "ready_at": {},
        "policy_suspended_at": None, "binding_suspended_at": None, "placement_at": None, "new_pods": [],
    }
    try:
        before_uids: dict[str, set[str]] = {}
        for cluster, target in targets.items():
            record["preconditions"][cluster] = stable_cluster_state(args, cluster, service, target)
            items = json.loads(base.kubectl(
                args, cluster, "-n", args.namespace, "get", "pods", "-l", f"app={service}", "-o", "json",
            ).stdout)["items"]
            before_uids[cluster] = {item["metadata"]["uid"] for item in items}
        placements = [{"clusterName": cluster, "replicas": replicas} for cluster, replicas in targets.items()]
        base.kubectl(
            args, "karmada-apiserver", "-n", args.namespace, "patch", "controlledreplicapropagation", service,
            "--type=merge", "-p", json.dumps({"spec": {"placements": placements}}),
        )
        record["patch_accepted_at"] = utc_now()
        record["patch_accepted_epoch"] = time.time()
        wait_deadline = min(deadline, time.monotonic() + args.readiness_timeout_seconds)
        while time.monotonic() < wait_deadline:
            base.ensure_deadline(deadline, abort_file)
            now = time.time()
            policy = json.loads(base.kubectl(
                args, "karmada-apiserver", "-n", args.namespace,
                "get", "propagationpolicy", f"crp-owned-{service}", "-o", "json",
            ).stdout)
            if policy.get("spec", {}).get("suspension", {}).get("dispatching") is True and record["policy_suspended_at"] is None:
                record["policy_suspended_at"] = now
            bindings = json.loads(base.kubectl(
                args, "karmada-apiserver", "-n", args.namespace, "get", "resourcebindings", "-o", "json",
            ).stdout)["items"]
            binding = next(item for item in bindings if item.get("spec", {}).get("resource", {}).get("name") == service)
            observed = int(binding.get("status", {}).get("schedulerObservedGeneration", 0)) >= int(binding["metadata"].get("generation", 0))
            if binding.get("spec", {}).get("suspension", {}).get("dispatching") is True and observed and record["binding_suspended_at"] is None:
                record["binding_suspended_at"] = now
            binding_targets = {item["name"]: int(item.get("replicas", 0)) for item in binding.get("spec", {}).get("clusters", [])}
            if observed and int(binding.get("spec", {}).get("replicas", -1)) == sum(targets.values()) and binding_targets == targets:
                record["placement_at"] = record["placement_at"] or now
            for cluster, target in targets.items():
                deployment = json.loads(base.kubectl(
                    args, cluster, "-n", args.namespace, "get", "deployment", service, "-o", "json",
                ).stdout)
                status = deployment.get("status", {})
                if int(deployment.get("spec", {}).get("replicas", -1)) == target:
                    record["member_spec_observed_at"].setdefault(cluster, now)
                if (
                    int(deployment.get("spec", {}).get("replicas", -1)) == target
                    and int(status.get("observedGeneration", 0)) >= int(deployment["metadata"].get("generation", 0))
                    and int(status.get("replicas", 0)) == target
                    and int(status.get("updatedReplicas", 0)) == target
                    and int(status.get("readyReplicas", 0)) == target
                    and int(status.get("availableReplicas", 0)) == target
                    and int(status.get("unavailableReplicas", 0)) == 0
                ):
                    record["ready_at"].setdefault(cluster, now)
            if len(record["ready_at"]) == 2 and record["placement_at"]:
                break
            time.sleep(1)
        if len(record["ready_at"]) != 2 or not record["placement_at"]:
            raise RuntimeError(f"{service}: placement {cluster2}/{cluster3} did not converge within the bound")
        for cluster in targets:
            items = json.loads(base.kubectl(
                args, cluster, "-n", args.namespace, "get", "pods", "-l", f"app={service}", "-o", "json",
            ).stdout)["items"]
            for item in items:
                if item["metadata"]["uid"] not in before_uids[cluster]:
                    timing = pod_timings(item, service)
                    timing["cluster"] = cluster
                    record["new_pods"].append(timing)
        record["decision_to_placement_seconds"] = record["placement_at"] - decision_epoch
        record["decision_to_all_ready_seconds"] = max(record["ready_at"].values()) - decision_epoch
        record["placement_to_all_ready_seconds"] = max(record["ready_at"].values()) - record["placement_at"]
        return record
    finally:
        queue.release(request["id"])


class SwapSampler(threading.Thread):
    def __init__(self, path: Path, phase_file: Path, interval: float = SWAP_SAMPLE_SECONDS) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.phase_file = phase_file
        self.interval = interval
        self._stop_event = threading.Event()

    @staticmethod
    def counters() -> tuple[int, int]:
        values: dict[str, int] = {}
        for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
            name, value = line.split()
            if name in {"pswpin", "pswpout"}:
                values[name] = int(value)
        return values["pswpin"], values["pswpout"]

    def run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "timestamp", "phase", "pswpin", "pswpout"])
            while not self._stop_event.is_set():
                swap_in, swap_out = self.counters()
                phase = self.phase_file.read_text(encoding="utf-8").strip() if self.phase_file.exists() else "unknown"
                writer.writerow([time.time(), utc_now(), phase, swap_in, swap_out])
                handle.flush()
                self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=self.interval + 2)


def summarize_swap(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    phases: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        phases.setdefault(row["phase"], []).append(row)
    result: dict[str, Any] = {}
    for phase, selected in phases.items():
        selected.sort(key=lambda row: float(row["epoch"]))
        windows: list[float] = []
        for left, right in zip(selected, selected[1:]):
            elapsed = float(right["epoch"]) - float(left["epoch"])
            pages = (int(right["pswpin"]) - int(left["pswpin"])) + (int(right["pswpout"]) - int(left["pswpout"]))
            if elapsed > 0:
                windows.append(max(0.0, pages / elapsed))
        duration = float(selected[-1]["epoch"]) - float(selected[0]["epoch"]) if len(selected) > 1 else 0.0
        pages_in = int(selected[-1]["pswpin"]) - int(selected[0]["pswpin"]) if len(selected) > 1 else 0
        pages_out = int(selected[-1]["pswpout"]) - int(selected[0]["pswpout"]) if len(selected) > 1 else 0
        result[phase] = {
            "samples": len(selected), "duration_seconds": duration,
            "pages_in": pages_in, "pages_out": pages_out,
            "average_total_pages_per_second": (pages_in + pages_out) / duration if duration else None,
            "peak_short_window_total_pages_per_second": max(windows, default=None),
            "short_window_seconds": SWAP_SAMPLE_SECONDS,
        }
    return result


def readiness_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scale_out = [
        row for row in records
        if any(row["targets"][cluster] > row["preconditions"][cluster]["current_target_replicas"] for cluster in row["targets"])
    ]
    values = [float(row["decision_to_all_ready_seconds"]) for row in scale_out]
    pods = [pod for row in scale_out for pod in row.get("new_pods", [])]
    def range_for(name: str) -> dict[str, Any]:
        available = [float(pod[name]) for pod in pods if pod.get(name) is not None]
        return {"samples": len(available), "minimum_seconds": min(available, default=None), "maximum_seconds": max(available, default=None)}
    return {
        "scale_out_transition_samples": len(values),
        "decision_to_all_ready_seconds": {
            "minimum": min(values, default=None), "maximum": max(values, default=None),
            "median": statistics.median(values) if values else None,
        },
        "new_pod_samples": len(pods),
        "creation_to_scheduling": range_for("creation_to_scheduled_seconds"),
        "scheduling_to_container_start": range_for("scheduled_to_container_start_seconds"),
        "container_start_to_application_ready": range_for("container_start_to_ready_seconds"),
        "uncertainty": "Small transition sample; ranges and medians only. No readiness p95 is inferred.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, default=REPO_ROOT / ".kube" / "headlight-testbed.config")
    parser.add_argument("--namespace", default="trainticket")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--readiness-timeout-seconds", type=int, default=600)
    parser.add_argument("--prometheus-port", type=int, default=39230)
    parser.add_argument("--locust-image", default="dstefanov46/locust-test:0.66.0")
    args = parser.parse_args()
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    spec = base.validate_protocol(protocol, 201)
    args.memory_threshold_bytes = int(spec["memory_guard"]["minimum_available_bytes"])
    args.per_user_rps = float(spec["load"]["per_user_target_rps"])
    args.load_namespace = str(spec["load_generator_namespace"])
    args.admission_queue = AdmissionQueue(args)
    args.scale_action = safe_set_replicas
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True)
    initialize_series(raw_dir)
    phase_file = raw_dir / "phase.txt"
    stop_file = raw_dir / "watchdog.stop"
    abort_file = raw_dir / "watchdog.abort"
    phase_file.write_text("preflight\n", encoding="utf-8")
    contract = {
        "schema": "headlight.conditional-qualification-retest/v1", "attempt": 2,
        "seed": 201, "seed_role": "infrastructure-qualification", "maximum_live_seconds": 10800,
        "training_input": False, "final_evaluation": False, "final_matrix": False,
        "swap_aggregation": {
            "sample_interval_seconds": SWAP_SAMPLE_SECONDS,
            "per_phase_average": "(delta pswpin + delta pswpout) / phase sample span",
            "short_window_peak": "maximum consecutive-sample total page delta / elapsed seconds",
            "acceptance": "every execution phase with at least two samples averages <1 total page/s; short-window peaks are reported separately",
        },
        "load_points_rps": [2.4, 3.2], "matched_route_placements": ["2/1", "2/2"],
        "flash": {"segments": [{"seconds": 120, "users": 1}, {"seconds": 480, "users": 8}, {"seconds": 300, "users": 1}]},
    }
    (output_dir / "retest-contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (output_dir / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    preflight(args, raw_dir)
    args.admission_queue.ensure()
    if base.kubectl(args, "cluster2", "get", "namespace", args.load_namespace, check=False).returncode != 0:
        base.kubectl(args, "cluster2", "create", "namespace", args.load_namespace)
    snapshot(args, raw_dir, "before")
    controller_before = base.kubectl(args, "karmada-host", "-n", "crp-system", "logs", "deployment/crp-controller-manager").stdout
    (logs_dir / "crp-controller-before.log").write_text(controller_before, encoding="utf-8")
    watchdog_log = (logs_dir / "watchdog.log").open("w", encoding="utf-8")
    watchdog = subprocess.Popen([
        "python3", str(WATCHDOG), "--kubeconfig", str(args.kubeconfig), "--output-dir", str(raw_dir),
        "--phase-file", str(phase_file), "--stop-file", str(stop_file), "--abort-file", str(abort_file),
        "--namespace", args.namespace, "--load-namespace", args.load_namespace,
        "--load-cluster", "cluster2", "--load-job", base.LOAD_JOB,
        "--threshold-bytes", str(args.memory_threshold_bytes),
    ], cwd=REPO_ROOT, stdout=watchdog_log, stderr=subprocess.STDOUT, text=True)
    sampler = Sampler(args, raw_dir, phase_file)
    swap_sampler = SwapSampler(raw_dir / "swap-series.csv", phase_file)
    sampler.start()
    swap_sampler.start()
    deadline = time.monotonic() + 10800
    loads: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    windows: dict[str, tuple[float, float]] = {}
    success = False
    try:
        for users in (6, 8):
            label = f"retest-{users}-users"
            record = base.run_load(args, raw_dir, phase_file, abort_file, deadline, label, [{"duration_seconds": 180, "users": users}])
            loads.append(record); windows[label] = (record["started_at"], record["finished_at"])
            base.sample_mongostat(args, raw_dir, label)
        for cluster2, cluster3, label in ((2, 1, "matched-route-2-1"), (2, 2, "matched-route-2-2")):
            phase_file.write_text(f"scale-route-{cluster2}-{cluster3}\n", encoding="utf-8")
            actions.append(safe_set_replicas(args, "ts-route", cluster2, cluster3, abort_file, deadline))
            phase_file.write_text(f"settle-route-{cluster2}-{cluster3}\n", encoding="utf-8")
            settle_until = time.monotonic() + 60
            while time.monotonic() < settle_until:
                base.ensure_deadline(deadline, abort_file); time.sleep(1)
            record = base.run_load(args, raw_dir, phase_file, abort_file, deadline, label, [{"duration_seconds": 180, "users": 8}])
            loads.append(record); windows[label] = (record["started_at"], record["finished_at"])
            base.sample_mongostat(args, raw_dir, label)
        phase_file.write_text("recovery-route-1-1\n", encoding="utf-8")
        actions.append(safe_set_replicas(args, "ts-route", 1, 1, abort_file, deadline))
        recovery_until = time.monotonic() + 120
        while time.monotonic() < recovery_until:
            base.ensure_deadline(deadline, abort_file); time.sleep(1)
        flash = base.run_load(
            args, raw_dir, phase_file, abort_file, deadline, "retest-flash",
            [{"duration_seconds": 120, "users": 1}, {"duration_seconds": 480, "users": 8}, {"duration_seconds": 300, "users": 1}],
            action_service="ts-route",
            scheduled_actions=[{"offset_seconds": 120, "cluster2": 2, "cluster3": 2}, {"offset_seconds": 600, "cluster2": 1, "cluster3": 1}],
            action_records=actions,
        )
        loads.append(flash); windows[flash["label"]] = (flash["started_at"], flash["finished_at"])
        phase_file.write_text("collect\n", encoding="utf-8")
        snapshot(args, raw_dir, "after")
        controller_after = base.kubectl(args, "karmada-host", "-n", "crp-system", "logs", "deployment/crp-controller-manager").stdout
        (logs_dir / "crp-controller-after.log").write_text(controller_after, encoding="utf-8")
        sampler.stop(); swap_sampler.stop()
        collect_prometheus(args, raw_dir, min(start for start, _ in windows.values()), max(end for _, end in windows.values()))
        prometheus = summarize_prometheus(raw_dir, windows)
        health = restart_and_event_summary(raw_dir)
        scheduling = base.scheduling_failures(raw_dir)
        for cluster, failures in scheduling.items():
            health[cluster]["scheduling_failures"] = len(failures)
        swap = summarize_swap(raw_dir / "swap-series.csv")
        execution_phases = {
            name: row for name, row in swap.items()
            if row["samples"] >= 2 and name not in {"preflight", "collect"}
        }
        swap_clean = all(float(row["average_total_pages_per_second"]) < SWAP_PHASE_AVERAGE_LIMIT for row in execution_phases.values())
        host_rows = list(csv.DictReader((raw_dir / "host-series.csv").open(encoding="utf-8")))
        minimum_available = min(int(row["mem_available_bytes"]) for row in host_rows)
        watchdog_events = [json.loads(line) for line in (raw_dir / "watchdog-events.jsonl").read_text(encoding="utf-8").splitlines()]
        watchdog_triggered = any(item["event"] in {"stop_load", "stop_testbed"} for item in watchdog_events)
        load_reliable = all(row["success_fraction"] >= 0.99 and row["delivery_fraction"] >= 0.90 for row in loads)
        health_clean = all(
            row["restart_increase"] == 0 and row["pending_pods_after"] == 0 and row["oom_events"] == 0
            and row["eviction_events"] == 0 and row["oomkilled_containers_after"] == 0 and row["scheduling_failures"] == 0
            for row in health.values()
        )
        telemetry_complete = all(
            prometheus[cluster][metric][phase]["available"]
            for cluster in ("cluster2", "cluster3")
            for metric in ("cpu_rate_by_container", "memory_by_container", "throttled_period_ratio", "pending_pods")
            for phase in windows
        ) and all(
            prometheus["cluster2"][metric][phase]["available"]
            for metric in ("request_rate", "failure_rate") for phase in windows
        )
        baseline = next(row for row in loads if row["label"] == "retest-8-users")
        matched21 = next(row for row in loads if row["label"] == "matched-route-2-1")
        matched22 = next(row for row in loads if row["label"] == "matched-route-2-2")
        useful_scaleout = min(matched21["p95_latency_ms"], matched22["p95_latency_ms"]) < baseline["p95_latency_ms"] * 0.95
        control_plane = base.members_ready(args)
        gates = {
            "memory_headroom": minimum_available >= args.memory_threshold_bytes,
            "watchdog": not watchdog_triggered, "health": health_clean,
            "telemetry": telemetry_complete, "delivered_load": load_reliable,
            "useful_scale_out": useful_scaleout, "control_plane": control_plane,
            "no_sustained_swap_io": swap_clean,
            "admission_race_eliminated": not any(scheduling.values()),
        }
        verdict = "PASS" if all(gates.values()) else "CONDITIONAL"
        if not gates["memory_headroom"] or not gates["control_plane"] or not gates["health"]:
            verdict = "FAIL"
        hashes = {
            "controller_source_revision": CONTROLLER_SOURCE_REVISION,
            "controller_patch_sha256": sha256_file(CRP_PATCH),
            "controller_image": CONTROLLER_IMAGE,
            "profile_sha256": sha256_file(PROFILE), "rendered_manifest_sha256": sha256_file(RENDERED),
            "protocol_sha256": sha256_file(PROTOCOL), "retest_runner_sha256": sha256_file(Path(__file__)),
        }
        report = {
            "schema": "headlight.conditional-qualification-retest-result/v1",
            "verdict": verdict, "claim_scope": "infrastructure qualification only; not training or final evaluation data",
            "gates": gates, "hashes": hashes, "minimum_available_bytes": minimum_available,
            "swap": {"aggregation_contract": contract["swap_aggregation"], "phases": swap},
            "loads": loads, "scale_actions": actions, "readiness": readiness_summary(actions),
            "latency_comparison": {
                "baseline_3_2_rps_p95_ms": baseline["p95_latency_ms"], "route_2_1_p95_ms": matched21["p95_latency_ms"],
                "route_2_2_p95_ms": matched22["p95_latency_ms"],
            },
            "health": health, "scheduling_failures": scheduling, "watchdog_events": watchdog_events,
            "prometheus": prometheus, "database_contention": base.summarize_mongostat(raw_dir),
            "qualified_envelope": {
                "profile": "local-24gb-tested", "workload_clusters": ["cluster2", "cluster3"],
                "applications": 11, "databases": 7, "quota_requests_memory_mib": 2078,
                "baseline": "one replica of every selected service per workload cluster",
                "tested_route_placements": ["1/1", "2/1", "2/2"], "per_cluster_extra_application_budget": 1,
                "per_service_per_cluster_bounds": [1, 2], "offered_load_rps": [2.4, 3.2],
                "admission": "Karmada ConfigMap CAS queue; oldest decision timestamp then service/name; one active transition; quota is only the hard backstop",
            },
            "limitations": [
                "The shared admission queue is enforced by this qualification actuation path; every experiment controller must use the same queue client.",
                "Readiness sample size is intentionally small and supports ranges, not a p95 estimate.",
                "Only ts-route 1/1, 2/1, and 2/2 were retested; no final experiment matrix was launched.",
            ],
        }
        report_path = output_dir / "qualification-retest-report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["hashes"]["report_sha256_before_embedded_hash"] = sha256_file(report_path)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        success = True
    except Exception as exc:
        (logs_dir / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        snapshot(args, raw_dir, "failure")
        raise
    finally:
        base.cleanup_load(args)
        try:
            safe_set_replicas(args, "ts-route", 1, 1, abort_file, time.monotonic() + 600)
        except Exception as exc:
            (logs_dir / "restore-failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        if sampler.is_alive(): sampler.stop()
        if swap_sampler.is_alive(): swap_sampler.stop()
        stop_file.touch()
        try:
            watchdog.wait(timeout=15)
        except subprocess.TimeoutExpired:
            watchdog.terminate(); watchdog.wait(timeout=10)
        watchdog_log.close()
        (output_dir / "status.json").write_text(json.dumps({"success": success, "finished_at": utc_now()}, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
