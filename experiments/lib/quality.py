"""Shared quality gates for measured experiment bundles.

The functions in this module deliberately contain no Kubernetes or process
control.  They are used by live runners, campaign pilots, matrix resume checks,
and artifact generation so those paths cannot disagree about valid evidence.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.lib.metrics import percentile


PREDICTIVE_CONTROLLERS = {"headlight", "lstm"}
REQUIRED_PROMETHEUS_METRICS = {
    "cpu_seconds",
    "memory_working_set_bytes",
    "pod_count",
}


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    capacity_knee: bool = False


class BundleQualityError(RuntimeError):
    def __init__(self, bundle: Path, issues: Iterable[QualityIssue]):
        self.bundle = bundle
        self.issues = list(issues)
        super().__init__(f"{bundle}: " + "; ".join(issue.message for issue in self.issues))


def is_success_status(status: str | int | None) -> bool:
    try:
        value = int(str(status))
    except (TypeError, ValueError):
        return False
    return 200 <= value < 400


def request_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if not row.get("exception") and is_success_status(row.get("status"))]
    latencies: list[float] = []
    for row in successful:
        try:
            latencies.append(float(row["response_time_ms"]))
        except (KeyError, TypeError, ValueError):
            pass
    timestamps: list[float] = []
    for row in rows:
        try:
            timestamps.append(float(row["timestamp"]))
        except (KeyError, TypeError, ValueError):
            pass
    exceptions = sum(bool(row.get("exception")) for row in rows)
    return {
        "requests": len(rows),
        "successful": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "exceptions": exceptions,
        "p95_latency_ms": percentile(latencies, 0.95),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


def duration_is_covered(first: float | None, last: float | None, duration_seconds: int, ratio: float = 0.90) -> bool:
    return first is not None and last is not None and last - first >= duration_seconds * ratio


def _series_samples(metric: dict[str, Any]) -> list[list[float]]:
    response = metric.get("response", metric)
    result = response.get("data", {}).get("result", [])
    samples: list[list[float]] = []
    for series in result:
        parsed: list[float] = []
        for value in series.get("values", []):
            try:
                parsed.append(float(value[0]))
            except (IndexError, TypeError, ValueError):
                continue
        if parsed:
            samples.append(parsed)
    return samples


def prometheus_coverage_issues(
    payload: dict[str, Any], started_at: float, finished_at: float, interval_seconds: int
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    expected = max(2, math.floor((finished_at - started_at) / interval_seconds) + 1)
    minimum_samples = max(2, math.floor(expected * 0.75))
    edge_slack = interval_seconds * 2
    for name in sorted(REQUIRED_PROMETHEUS_METRICS):
        series = _series_samples(payload.get(name, {}))
        if not series:
            issues.append(QualityIssue("prometheus_empty", f"Prometheus metric {name} has no samples"))
            continue
        covered = any(
            len(values) >= minimum_samples
            and min(values) <= started_at + edge_slack
            and max(values) >= finished_at - edge_slack
            for values in series
        )
        if not covered:
            issues.append(
                QualityIssue(
                    "prometheus_coverage",
                    f"Prometheus metric {name} does not span the measurement window with {minimum_samples} samples",
                )
            )
    return issues


def pod_restart_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cluster, cluster_payload in payload.items():
        for pod in cluster_payload.get("items", []):
            pod_name = pod.get("metadata", {}).get("name", "unknown")
            for container in pod.get("status", {}).get("containerStatuses", []):
                key = f"{cluster}/{pod_name}/{container.get('name', 'unknown')}"
                counts[key] = int(container.get("restartCount", 0))
    return counts


def restart_increases(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    old = pod_restart_counts(before)
    new = pod_restart_counts(after)
    return {key: value - old.get(key, 0) for key, value in new.items() if value > old.get(key, 0)}


def unready_deployments(payload: dict[str, Any], allowed: set[str] | None = None) -> list[str]:
    allowed = allowed or set()
    unready: list[str] = []
    for cluster, cluster_payload in payload.items():
        for deployment in cluster_payload.get("items", []):
            name = deployment.get("metadata", {}).get("name", "unknown")
            if name in allowed:
                continue
            desired = int(deployment.get("spec", {}).get("replicas", 1))
            ready = int(deployment.get("status", {}).get("readyReplicas", 0))
            if ready < desired:
                unready.append(f"{cluster}/{name} {ready}/{desired}")
    return unready


def inference_timing_issues(path: Path, controller: str) -> list[QualityIssue]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    applicable = controller in PREDICTIVE_CONTROLLERS
    measured = [row for row in rows if row.get("duration_ms")]
    legacy_uninstrumented = any(row.get("applicable", "").lower() == "legacy-uninstrumented" for row in rows)
    if applicable and legacy_uninstrumented:
        return []
    if applicable and not measured:
        return [QualityIssue("inference_timing", f"{controller} has no model-forward timing rows")]
    if not applicable and not any(row.get("applicable", "").lower() == "false" for row in rows):
        return [QualityIssue("inference_not_applicable", f"{controller} timing is not explicitly marked not applicable")]
    return []


def _object_generations(payload: dict[str, Any]) -> dict[str, int]:
    return {
        f"{item.get('metadata', {}).get('namespace', '')}/{item.get('metadata', {}).get('name', '')}": int(
            item.get("metadata", {}).get("generation", 0) or 0
        )
        for item in payload.get("items", [])
    }


def _ready_deployments(payload: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for cluster, cluster_payload in payload.items():
        for item in cluster_payload.get("items", []):
            name = str(item.get("metadata", {}).get("name", ""))
            result[f"{cluster}/{name}"] = int(item.get("status", {}).get("readyReplicas", 0) or 0)
    return result


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def headlight_acceptance_issues(seed_dir: Path, summary: dict[str, Any], sla_ms: float) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    before_crps = json.loads((seed_dir / "raw" / "crps_before.json").read_text(encoding="utf-8"))
    after_crps = json.loads((seed_dir / "raw" / "crps_after.json").read_text(encoding="utf-8"))
    old_generations = _object_generations(before_crps)
    new_generations = _object_generations(after_crps)
    if not any(
        generation > old_generations.get(name, 0)
        for name, generation in new_generations.items()
    ):
        issues.append(QualityIssue("crp_generation", "no CRP generation changed during qualification"))

    before_deployments = json.loads(
        (seed_dir / "raw" / "deployments_before.json").read_text(encoding="utf-8")
    )
    after_deployments = json.loads(
        (seed_dir / "raw" / "deployments_after.json").read_text(encoding="utf-8")
    )
    old_ready = _ready_deployments(before_deployments)
    new_ready = _ready_deployments(after_deployments)
    if not any(value > old_ready.get(name, 0) for name, value in new_ready.items()):
        issues.append(QualityIssue("ready_scale_up", "no added replica became Ready"))

    action_rows = _json_lines(seed_dir / "raw" / "controller_actions.jsonl")
    changed_actions = [
        row
        for row in action_rows
        if row.get("desired_replicas") != row.get("previous_replicas")
        and row.get("batch_successful")
    ]
    if not changed_actions:
        issues.append(QualityIssue("controller_actions", "Headlight emitted no successful replica change"))
    invalid_actions = [
        row
        for row in changed_actions
        if row.get("policy_reason") in {"absent-service", "fixed-database"}
        or str(row.get("service", "")).endswith(("-mongo", "-mysql"))
    ]
    if invalid_actions:
        issues.append(QualityIssue("invalid_service_action", "Headlight acted on an absent or database service"))

    action_handler = json.loads(
        (seed_dir / "raw" / "actions_handler_batches.json").read_text(encoding="utf-8")
    )
    log_text = "\n".join(action_handler.get("logs", []))
    if "Failed to find CRP targeting" in log_text:
        issues.append(QualityIssue("missing_crp_action", "Actions Handler recorded a missing-CRP action"))
    if float(summary["p95_latency_ms"]) > sla_ms:
        issues.append(
            QualityIssue(
                "p95_sla",
                f"qualification P95 {summary['p95_latency_ms']:.2f}ms exceeds {sla_ms:.2f}ms",
            )
        )
    try:
        bundle_metadata = json.loads(
            (seed_dir / "raw" / "model_bundle_metadata.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        bundle_metadata = {}
    if not bundle_metadata.get("bundle_id") or bundle_metadata.get("frozen") is not True:
        issues.append(QualityIssue("model_bundle", "frozen model-bundle metadata is missing"))
    return issues


def validate_bundle(seed_dir: Path, manifest: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    required = [
        "raw/load_profile.csv",
        "raw/prometheus_queries.json",
        "raw/request_events.csv",
        "raw/kubernetes_events.json",
        "raw/crp_transitions.json",
        "raw/actions_handler_batches.json",
        "raw/ai_inference_times.csv",
        "raw/prometheus_range_data.json",
        "logs/runner.log",
        "summaries/summary.csv",
    ]
    if not dry_run:
        required += [
            "raw/measurement_window.json",
            "raw/pods_before.json",
            "raw/pods_after.json",
            "raw/deployments_after.json",
            "raw/crps_before.json",
            "raw/crps_after.json",
        ]
    missing = [relative for relative in required if not (seed_dir / relative).exists()]
    if missing:
        raise BundleQualityError(seed_dir, [QualityIssue("missing_files", f"missing {', '.join(missing)}")])
    if dry_run:
        return {"valid": True, "dry_run": True}

    run = manifest["spec"]["run"]
    qualification_kind = run.get("qualification_overrides", {}).get("kind")
    capacity_pilot = qualification_kind == "capacity-pilot"
    interval = int(run.get("sampling_interval_seconds", 15))
    duration = int(run["duration_seconds"])
    summary, issues, _, _ = _validate_bundle_telemetry(seed_dir, run, duration, interval)
    issues.extend(_validate_bundle_runtime(seed_dir, manifest, run, qualification_kind, capacity_pilot, summary))
    if issues:
        raise BundleQualityError(seed_dir, issues)
    return {"valid": True, **summary}


def _validate_bundle_telemetry(
    seed_dir: Path,
    run: dict[str, Any],
    duration: int,
    interval: int,
) -> tuple[dict[str, Any], list[QualityIssue], float, float]:
    summary = request_summary(seed_dir / "raw" / "request_events.csv")
    issues: list[QualityIssue] = []
    if not summary["requests"]:
        issues.append(QualityIssue("requests_empty", "request telemetry is empty"))
    if summary["success_rate"] < float(run["sla"].get("minimum_success_rate", 0.99)):
        issues.append(QualityIssue("request_success", f"request success rate is {summary['success_rate']:.4f}", True))
    if summary["exceptions"]:
        issues.append(QualityIssue("locust_exceptions", f"Locust recorded {summary['exceptions']} exceptions", True))
    if not duration_is_covered(summary["first_timestamp"], summary["last_timestamp"], duration):
        issues.append(QualityIssue("request_coverage", "request telemetry does not span the expected duration"))

    window = json.loads((seed_dir / "raw" / "measurement_window.json").read_text(encoding="utf-8"))
    started_at = float(window["started_at_unix"])
    finished_at = float(window["finished_at_unix"])
    if finished_at - started_at < duration * 0.90:
        issues.append(QualityIssue("execution_duration", "runner measurement window is shorter than configured duration"))
    prometheus = json.loads((seed_dir / "raw" / "prometheus_range_data.json").read_text(encoding="utf-8"))
    if not run.get("smoke_validation", False):
        issues.extend(prometheus_coverage_issues(prometheus, started_at, finished_at, interval))

    return summary, issues, started_at, finished_at


def _validate_bundle_runtime(
    seed_dir: Path,
    manifest: dict[str, Any],
    run: dict[str, Any],
    qualification_kind: str | None,
    capacity_pilot: bool,
    summary: dict[str, Any],
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    before = json.loads((seed_dir / "raw" / "pods_before.json").read_text(encoding="utf-8"))
    after = json.loads((seed_dir / "raw" / "pods_after.json").read_text(encoding="utf-8"))
    increases = restart_increases(before, after)
    if increases:
        issues.append(QualityIssue("pod_restarts", f"containers restarted: {increases}"))
    deployments = json.loads((seed_dir / "raw" / "deployments_after.json").read_text(encoding="utf-8"))
    allowed = set(manifest["spec"]["run"].get("allowed_unready_deployments", []))
    unready = unready_deployments(deployments, allowed)
    if unready:
        issues.append(QualityIssue("deployments_unready", f"deployments not Ready: {', '.join(unready)}"))
    controller = manifest["spec"]["controller"]["controller"]
    if controller in PREDICTIVE_CONTROLLERS:
        for relative in ["raw/controller_actions.jsonl", "raw/model_bundle_metadata.json"]:
            if not (seed_dir / relative).exists():
                issues.append(QualityIssue("missing_files", f"missing {relative}"))
    if not run.get("smoke_validation", False) and not capacity_pilot:
        issues.extend(inference_timing_issues(seed_dir / "raw" / "ai_inference_times.csv", controller))
    if (
        controller == "headlight"
        and qualification_kind == "capacity-qualification"
        and not run.get("smoke_validation", False)
    ):
        issues.extend(
            headlight_acceptance_issues(
                seed_dir,
                summary,
                float(run["sla"].get("p95_latency_ms", 1000)),
            )
        )
    return issues


def capacity_is_healthy(summary: dict[str, Any], baseline_p95_ms: float | None) -> tuple[bool, str]:
    p95 = float(summary["p95_latency_ms"])
    if p95 > 1000:
        return False, f"P95 {p95:.2f}ms exceeds the 1000ms SLA"
    if baseline_p95_ms is not None and p95 > baseline_p95_ms * 2:
        return False, f"P95 {p95:.2f}ms exceeds twice the baseline {baseline_p95_ms:.2f}ms"
    return True, "healthy"


def select_safe_capacity(levels: Iterable[tuple[int, bool]]) -> int | None:
    selected: int | None = None
    for users, healthy in levels:
        if not healthy:
            break
        selected = users
    return selected
