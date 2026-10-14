"""Shared quality gates for measured experiment bundles.

The functions in this module deliberately contain no Kubernetes or process
control.  They are used by live runners, campaign pilots, matrix resume checks,
and artifact generation so those paths cannot disagree about valid evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.lib.metrics import percentile
from experiments.lib.protocol import load_protocol


PREDICTIVE_CONTROLLERS = {"headlight", "headlight-v2", "lstm", "lstm-matched"}
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


def technical_validity_payload(
    issues: Iterable[QualityIssue],
    *,
    retry_rule: str = "research-v3 technical invalidity retry",
) -> dict[str, Any]:
    issue_list = list(issues)
    if not issue_list:
        return {
            "technical_validity": "valid",
            "invalidity_reason": "",
            "retry_allowed": False,
            "retry_rule": "",
            "failure_category": "",
        }
    return {
        "technical_validity": "invalid",
        "invalidity_reason": "; ".join(f"{issue.code}: {issue.message}" for issue in issue_list),
        "retry_allowed": True,
        "retry_rule": retry_rule,
        "failure_category": issue_list[0].code,
    }


def performance_outcome(
    summary: dict[str, Any],
    run: dict[str, Any],
    ceiling: dict[str, Any] | None = None,
) -> str:
    sla = run.get("sla", {})
    p95_limit = float(sla.get("p95_latency_ms", 0) or 0)
    success_limit = float(sla.get("minimum_success_rate", 0) or 0)
    p95 = float(summary.get("p95_latency_ms", 0) or 0)
    success_rate = float(summary.get("success_rate", 0) or 0)
    ceiling_hit = bool((ceiling or {}).get("services_at_ceiling"))
    if (p95_limit and p95 > p95_limit) or (success_limit and success_rate < success_limit) or ceiling_hit:
        return "poor_but_valid"
    return "measured"


def is_success_status(status: str | int | None) -> bool:
    try:
        value = int(str(status))
    except (TypeError, ValueError):
        return False
    return 200 <= value < 400


def request_summary(
    path: Path,
    *,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if started_at is not None or finished_at is not None:
                try:
                    timestamp = float(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                if started_at is not None and timestamp < started_at:
                    continue
                if finished_at is not None and timestamp > finished_at:
                    continue
            rows.append(row)
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
        "raw/protocol.json",
        "raw/topology.json",
        "raw/workload_manifest_hashes.json",
        "raw/phase_timestamps.json",
        "raw/measurement_window.json",
        "raw/resource_calculation.json",
        "raw/model_bundle_provenance.json",
        "raw/policy_config.json",
        "raw/static_plan.json",
        "raw/prometheus_queries.json",
        "raw/request_events.csv",
        "raw/request_events_measurement.csv",
        "raw/kubernetes_events.json",
        "raw/crp_transitions.json",
        "raw/actions_handler_batches.json",
        "raw/ai_inference_times.csv",
        "raw/prometheus_range_data.json",
        "raw/ceiling_binding.json",
        "logs/runner.log",
        "summaries/summary.csv",
    ]
    if not dry_run:
        required += [
            "raw/pods_before.json",
            "raw/pods_after.json",
            "raw/deployments_after.json",
            "raw/crps_before.json",
            "raw/crps_after.json",
        ]
    gateway_reference = manifest["spec"]["run"].get("protocol", {}).get("gateway_protocol")
    matched_source = manifest["spec"]["run"].get("matched_baseline_manifest")
    evaluation = manifest["spec"]["run"].get("evaluation_trial")
    if evaluation is not None:
        required.append("raw/evaluation_trial.json")
        if not dry_run:
            required.append("raw/controller_runtime.json")
    if matched_source is not None:
        required.append("raw/matched_baseline_comparison.json")
        if not dry_run:
            required.append("raw/matched_controller_runtime.json")
    if gateway_reference is not None:
        required.append("raw/gateway_protocol.json")
        if not dry_run:
            required.extend(["raw/gateway_requests.log", "logs/gateway-weights.jsonl"])
    missing = [relative for relative in required if not (seed_dir / relative).exists()]
    if missing:
        raise BundleQualityError(seed_dir, [QualityIssue("missing_files", f"missing {', '.join(missing)}")])
    with (seed_dir / "summaries/summary.csv").open(newline="") as handle:
        scope_summary = next(csv.DictReader(handle), {})
    evaluation_indicators = ((seed_dir / "raw/evaluation_trial.json").exists()
                              or manifest["spec"]["run"].get("protocol", {}).get("experiment_scope") == "evaluation_trial"
                              or scope_summary.get("experiment_scope") == "evaluation_trial")
    if evaluation is None and evaluation_indicators:
        raise BundleQualityError(seed_dir, [QualityIssue("evaluation_trial", "evaluation evidence lacks its manifest scope marker")])
    if evaluation is not None:
        try:
            recorded_trial = json.loads((seed_dir / "raw/evaluation_trial.json").read_text())
            with (seed_dir / "summaries/summary.csv").open(newline="") as handle:
                trial_summary = next(csv.DictReader(handle))
            consistent = (recorded_trial == evaluation
                          and recorded_trial["capacity_qualified"] is False
                          and recorded_trial["matched_comparison_eligible"] is False
                          and recorded_trial["comparison_status"] == "evaluation_not_matched"
                          and f"seed-{recorded_trial['seed']}" == seed_dir.name
                          and recorded_trial.get("workload", manifest["spec"]["workload"]["pattern"])
                              == manifest["spec"]["workload"]["pattern"]
                          and matched_source is None
                          and trial_summary["experiment_scope"] == "evaluation_trial"
                          and trial_summary["matched_comparison_eligible"] == "False"
                          and trial_summary["capacity_qualified"] == "False"
                          and recorded_trial["load_trace_sha256"] == hashlib.sha256((seed_dir / "raw/load_profile.csv").read_bytes()).hexdigest()
                          and not (seed_dir / "raw/capacity_contract.json").exists())
            if not dry_run:
                runtime = json.loads((seed_dir / "raw/controller_runtime.json").read_text())
                consistent = (consistent and runtime["active_actuation"] is True and runtime["shadow_mode"] == "0"
                              and runtime["controller"] == "threshold"
                              and runtime["image"] == recorded_trial["controller_image"]
                              and bool(recorded_trial["controller_image_digest"])
                              and runtime["expected_image_digest"] == recorded_trial["controller_image_digest"]
                              and bool(runtime["pod_images"])
                              and all(p["image_id"].split("@")[-1] == recorded_trial["controller_image_digest"] for p in runtime["pod_images"]))
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            consistent = False
        if not consistent:
            raise BundleQualityError(seed_dir, [QualityIssue("evaluation_trial", "evaluation scope or active-controller evidence is inconsistent")])
    if gateway_reference is not None:
        recorded = json.loads((seed_dir / "raw" / "gateway_protocol.json").read_text(encoding="utf-8"))
        if recorded.get("reference") != gateway_reference:
            raise BundleQualityError(seed_dir, [QualityIssue("gateway_protocol", "gateway protocol reference differs from the run manifest")])
    if matched_source is not None:
        try:
            comparison = json.loads((seed_dir / "raw" / "matched_baseline_comparison.json").read_text(encoding="utf-8"))
            consistent = (
                comparison["schema_version"] == "headlight.matched-v1-comparison/v1"
                and comparison["comparison_status"] == "matched_baseline_evidence_validated"
                and comparison["baseline_source_manifest"] == matched_source
                and comparison["gateway_protocol"] == gateway_reference
                and comparison["capacity_users"] == 4
                and comparison["workload"] == manifest["spec"]["workload"]["pattern"]
                and f"seed-{comparison['seed']}" == seed_dir.name
                and manifest["spec"]["controller"]["controller"] in {"headlight", "threshold"}
                and comparison.get("comparison_controller", "headlight") == manifest["spec"]["controller"]["controller"]
                and set(comparison["baselines"]) == {"hpa", "keda"}
                and comparison["load_trace_sha256"] == hashlib.sha256((seed_dir / "raw" / "load_profile.csv").read_bytes()).hexdigest()
            )
        except (OSError, ValueError, KeyError, TypeError):
            consistent = False
        if not consistent:
            raise BundleQualityError(seed_dir, [QualityIssue("matched_baseline_comparison", "matched comparison evidence differs from the run bundle")])
        if not dry_run:
            try:
                runtime = json.loads((seed_dir / "raw" / "matched_controller_runtime.json").read_text(encoding="utf-8"))
                active = runtime["active_actuation"] is True and runtime["shadow_mode"] == "0"
            except (OSError, ValueError, KeyError, TypeError):
                active = False
            if not active:
                raise BundleQualityError(seed_dir, [QualityIssue("matched_controller_runtime", "matched controller lacks active-actuation evidence")])
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
    issues: list[QualityIssue] = []
    window = json.loads((seed_dir / "raw" / "measurement_window.json").read_text(encoding="utf-8"))
    started_at = float(window["started_at_unix"])
    finished_at = float(window["finished_at_unix"])
    summary = request_summary(
        seed_dir / "raw" / "request_events.csv",
        started_at=started_at,
        finished_at=finished_at,
    )
    if not summary["requests"]:
        issues.append(QualityIssue("requests_empty", "request telemetry is empty"))
    if not duration_is_covered(summary["first_timestamp"], summary["last_timestamp"], duration):
        issues.append(QualityIssue("request_coverage", "request telemetry does not span the expected duration"))

    if finished_at - started_at < duration * 0.90:
        issues.append(QualityIssue("execution_duration", "runner measurement window is shorter than configured duration"))
    if run.get("matched_baseline_manifest") is not None:
        try:
            locust_started = float(window["locust_started_at_unix"])
            matching_timing = (
                all(math.isfinite(value) for value in (locust_started, started_at, finished_at))
                and started_at - locust_started == int(run["warmup_seconds"])
                and finished_at - started_at == duration
            )
        except (KeyError, TypeError, ValueError):
            matching_timing = False
        if not matching_timing:
            issues.append(QualityIssue("matched_timing", "matched comparison requires the full measurement duration and actual configured warmup"))
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
    controller = manifest["spec"]["controller"]["controller"]
    if controller in PREDICTIVE_CONTROLLERS:
        for relative in [
            "raw/controller_actions.jsonl",
            "raw/model_bundle_metadata.json",
            "raw/model_bundle_provenance.json",
            "raw/policy_config.json",
        ]:
            if not (seed_dir / relative).exists():
                issues.append(QualityIssue("missing_files", f"missing {relative}"))
        try:
            bundle_provenance = json.loads((seed_dir / "raw" / "model_bundle_provenance.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bundle_provenance = {}
        if not bundle_provenance.get("bundle_id") or not bundle_provenance.get("bundle_sha256"):
            issues.append(QualityIssue("model_bundle_provenance", "bundle ID and hash are not recorded"))
        if bundle_provenance.get("bundle_unchanged_during_evaluation") is False:
            issues.append(QualityIssue("model_bundle_mutated", "bundle directory changed during evaluation"))
        if bundle_provenance.get("no_online_retraining") is not True:
            issues.append(QualityIssue("online_retraining", "bundle metadata does not disable runtime retraining"))
        training = bundle_provenance.get("bundle_metadata", {}).get("training", {})
        required_training = (
            {
                "dataset_hash",
                "feature_schema_hash",
                "adjacency_hash",
                "fitting_seeds",
                "calibration_seeds",
                "holdout_seeds",
                "selection_procedure",
                "created_before_evaluation",
            }
            if controller in {"headlight-v2", "lstm-matched"}
            else {
                "dataset_sha256",
                "adjacency_sha256",
                "profile_hash",
                "sources",
                "selection_procedure",
                "created_before_evaluation",
            }
        )
        if controller=='lstm-matched':
            required_training.discard('adjacency_hash')
            required_training.add('profile_hash')
        missing_training = sorted(required_training - set(training if isinstance(training, dict) else {}))
        if missing_training:
            issues.append(
                QualityIssue(
                    "model_bundle_training_metadata",
                    f"bundle training metadata missing: {', '.join(missing_training)}",
                )
            )
        try:
            policy = json.loads((seed_dir / "raw" / "policy_config.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            policy = {}
        if controller == "headlight-v2":
            if policy.get("status") != "frozen-before-evaluation" or not policy.get("artifacts"):
                issues.append(QualityIssue("policy_provenance", "frozen v2 policy artifacts are missing"))
            policy_seed_source = policy.get("selection", {})
        else:
            if not policy.get("policy_configuration_hash") or not policy.get("provenance"):
                issues.append(QualityIssue("policy_provenance", "policy configuration hash/provenance is missing"))
            policy_seed_source = policy.get("provenance", {})
        policy_seeds = _metadata_seed_set(policy_seed_source)
        leaked = sorted(set(load_protocol()["spec"]["seeds"]["evaluation"]) & policy_seeds)
        if leaked:
            issues.append(QualityIssue("policy_seed_leakage", f"policy metadata references evaluation seeds: {leaked}"))
    if controller == "threshold" and not (seed_dir / "raw" / "controller_actions.jsonl").exists():
        issues.append(QualityIssue("controller_actions", "threshold controller action log is missing"))
    if controller in {"static", "static-matched"} and not capacity_pilot:
        try:
            static_plan = json.loads((seed_dir / "raw" / "static_plan.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            static_plan = {}
        expected_schema='headlight.static-plan/matched-v2' if controller=='static-matched' else 'headlight.static-plan/v1'
        hash_field='selection_hash' if controller=='static-matched' else 'plan_hash'
        if static_plan.get("schema_version") != expected_schema or not static_plan.get(hash_field):
            issues.append(QualityIssue("static_plan", "frozen static plan metadata is missing"))
        leaked = sorted(set(load_protocol()["spec"]["seeds"]["evaluation"]) & _metadata_seed_set(static_plan))
        if leaked:
            issues.append(QualityIssue("static_seed_leakage", f"static plan references evaluation seeds: {leaked}"))
    if controller == "hpa":
        rendered = sorted((seed_dir / "raw" / "hpa").glob("hpa-*.yaml"))
        if not rendered:
            issues.append(QualityIssue("hpa_rendered_yaml", "rendered HPA YAML was not captured"))
        try:
            hpa_metadata = json.loads((seed_dir / "raw" / "hpa_controller_metadata.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hpa_metadata = {}
        if hpa_metadata.get("scale_up_behavior") != "kubernetes-default":
            issues.append(QualityIssue("hpa_scale_up_behavior", "HPA scale-up behavior is not recorded as Kubernetes default"))
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


def _metadata_seed_set(value: Any) -> set[int]:
    seeds: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"seed", "seeds", "capacity_seed", "calibration_seeds", "tuning_seeds"}:
                if isinstance(item, list):
                    for candidate in item:
                        try:
                            seeds.add(int(candidate))
                        except (TypeError, ValueError):
                            pass
                else:
                    try:
                        seeds.add(int(item))
                    except (TypeError, ValueError):
                        pass
            seeds.update(_metadata_seed_set(item))
    elif isinstance(value, list):
        for item in value:
            seeds.update(_metadata_seed_set(item))
    return seeds


def capacity_is_healthy(summary: dict[str, Any], baseline_p95_ms: float | None) -> tuple[bool, str]:
    rule = load_protocol()["spec"]["capacity_normalization"]["health_rule"]
    p95_ceiling = float(rule["p95_latency_ceiling_ms"])
    baseline_multiplier = float(rule["max_p95_multiplier_over_baseline"])
    p95 = float(summary["p95_latency_ms"])
    if p95 > p95_ceiling:
        return False, f"P95 {p95:.2f}ms exceeds the {p95_ceiling:.0f}ms SLA"
    if baseline_p95_ms is not None and p95 > baseline_p95_ms * baseline_multiplier:
        return False, f"P95 {p95:.2f}ms exceeds {baseline_multiplier:g}x the baseline {baseline_p95_ms:.2f}ms"
    return True, "healthy"


def select_safe_capacity(levels: Iterable[tuple[int, bool]]) -> int | None:
    selected: int | None = None
    for users, healthy in levels:
        if not healthy:
            break
        selected = users
    return selected
