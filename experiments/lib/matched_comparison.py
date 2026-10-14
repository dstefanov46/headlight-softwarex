"""Validate existing HPA/KEDA runs as matched v1 comparison baselines."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml

from experiments.lib.config import REPO_ROOT
from experiments.lib.workloads import generate_shape


class MatchedComparisonError(RuntimeError):
    """Raised when an existing baseline cannot support a matched comparison."""


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_trace_sha256(
    workload: dict[str, Any],
    seed: int,
    capacity: int,
    warmup_seconds: int,
    measurement_seconds: int,
    sampling_interval_seconds: int,
) -> str:
    capacity_env = str(workload.get("capacity_users_env", "HEADLIGHT_EVALUATION_CAPACITY"))
    configured_capacity = os.environ.get(capacity_env)
    if configured_capacity not in (None, "") and int(configured_capacity) != capacity:
        raise MatchedComparisonError(
            f"configured {capacity_env}={configured_capacity} differs from matched capacity C={capacity}"
        )

    points = generate_shape(
        workload,
        seed,
        sampling_interval_seconds,
        warmup_seconds=warmup_seconds,
        duration_seconds=measurement_seconds,
        capacity_users=capacity,
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["elapsed_seconds", "users", "spawn_rate"])
    writer.writeheader()
    for point in points:
        writer.writerow(point.__dict__)
    return hashlib.sha256(output.getvalue().encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedComparisonError(f"cannot read comparison evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedComparisonError(f"comparison evidence must be a JSON object: {path}")
    return value


def _validate_one_baseline(
    row: dict[str, Any],
    *,
    controller: str,
    workload_pattern: str,
    seed: int,
    workload: dict[str, Any],
    measurement_seconds: int,
    warmup_seconds: int,
    sampling_interval_seconds: int,
    sla: dict[str, Any],
    gateway_reference: dict[str, Any],
    workload_manifest_hashes: dict[str, Any],
    expected_trace_sha256: str,
) -> dict[str, Any]:
    if row.get("technical_validity") != "valid" or row.get("performance_outcome") not in {"measured", "poor_but_valid"}:
        raise MatchedComparisonError(f"{controller}/{workload_pattern}/seed-{seed} is not a valid measured baseline")

    output_dir = _path(str(row.get("output_dir", "")))
    manifest_path = output_dir / "manifest.yaml"
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MatchedComparisonError(f"cannot read {controller} run manifest {manifest_path}: {exc}") from exc
    spec = manifest.get("spec", {}) if isinstance(manifest, dict) else {}
    controller_spec = spec.get("controller", {})
    workload_spec = spec.get("workload", {})
    run_spec = spec.get("run", {})
    if controller_spec.get("controller") != controller:
        raise MatchedComparisonError(f"run manifest controller does not match {controller}: {manifest_path}")
    if workload_spec.get("pattern") != workload_pattern:
        raise MatchedComparisonError(f"run manifest workload does not match {workload_pattern}: {manifest_path}")
    if workload_spec.get("locust") != workload.get("locust"):
        raise MatchedComparisonError(f"{controller} workload shape differs from the v1 run")
    actual_timing = (
        int(run_spec.get("duration_seconds", -1)),
        int(run_spec.get("warmup_seconds", -1)),
        int(run_spec.get("sampling_interval_seconds", -1)),
    )
    expected_timing = (measurement_seconds, warmup_seconds, sampling_interval_seconds)
    if actual_timing != expected_timing:
        raise MatchedComparisonError(f"{controller} run timing differs from the v1 run")
    if run_spec.get("sla") != sla:
        raise MatchedComparisonError(f"{controller} SLA settings differ from the v1 run")

    seed_dir = output_dir / f"seed-{seed}"
    expected_paths = {
        "measurement_window_json": seed_dir / "raw" / "measurement_window.json",
        "request_events_csv": seed_dir / "raw" / "request_events.csv",
        "summary_csv": seed_dir / "summaries" / "summary.csv",
    }
    for field, expected_path in expected_paths.items():
        if _path(str(row.get(field, ""))).resolve() != expected_path.resolve():
            raise MatchedComparisonError(f"{controller} {field} must come from the same seed bundle")
    raw_dir = seed_dir / "raw"
    window = _read_json(raw_dir / "measurement_window.json")
    if int(window.get("warmup_seconds", -1)) != warmup_seconds or int(
        window.get("measurement_seconds", -1)
    ) != measurement_seconds:
        raise MatchedComparisonError(f"{controller} measured window differs from the v1 run")
    try:
        started = float(window["started_at_unix"])
        finished = float(window["finished_at_unix"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MatchedComparisonError(f"{controller} measured window has no actual timestamps") from exc
    if not math.isfinite(started) or not math.isfinite(finished) or finished - started != measurement_seconds:
        raise MatchedComparisonError(f"{controller} actual measured window differs from the v1 run")
    try:
        locust_started = float(window["locust_started_at_unix"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MatchedComparisonError(f"{controller} actual warmup has no Locust start timestamp") from exc
    if not math.isfinite(locust_started) or started - locust_started != warmup_seconds:
        raise MatchedComparisonError(f"{controller} actual warmup differs from the v1 run")

    gateway = _read_json(raw_dir / "gateway_protocol.json")
    if gateway.get("reference") != gateway_reference:
        raise MatchedComparisonError(f"{controller} gateway protocol differs from the v1 run")

    recorded_workload_hashes = _read_json(raw_dir / "workload_manifest_hashes.json")
    if recorded_workload_hashes != workload_manifest_hashes:
        raise MatchedComparisonError(f"{controller} deployed workload manifests differ from the v1 run")

    trace_path = raw_dir / "load_profile.csv"
    actual_trace_hash = _sha256(trace_path)
    if actual_trace_hash != row.get("load_trace_sha256") or actual_trace_hash != expected_trace_sha256:
        raise MatchedComparisonError(f"{controller} load trace differs from the matched v1 workload")

    request_events = _path(str(row.get("request_events_csv", "")))
    summary_path = _path(str(row.get("summary_csv", "")))
    if not request_events.is_file() or not summary_path.is_file():
        raise MatchedComparisonError(f"{controller} request or summary evidence is missing")
    try:
        with request_events.open(encoding="utf-8", newline="") as handle:
            measured_request = any(
                started <= float(event["timestamp"]) < finished
                for event in csv.DictReader(handle)
            )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise MatchedComparisonError(f"cannot read {controller} request evidence: {exc}") from exc
    if not measured_request:
        raise MatchedComparisonError(f"{controller} request evidence contains no measured requests")
    try:
        with summary_path.open(encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise MatchedComparisonError(f"cannot read {controller} summary {summary_path}: {exc}") from exc
    if len(summary_rows) != 1:
        raise MatchedComparisonError(f"{controller} summary must contain exactly one seed-level row")
    summary = summary_rows[0]
    if (
        int(summary.get("seed", -1)) != seed
        or summary.get("controller") != controller
        or summary.get("workload") != workload_pattern
        or summary.get("technical_validity") != "valid"
        or summary.get("performance_outcome") not in {"measured", "poor_but_valid"}
    ):
        raise MatchedComparisonError(f"{controller} summary does not confirm a valid measured cell")

    return {
        "controller": controller,
        "output_dir": str(output_dir),
        "load_trace_sha256": actual_trace_hash,
        "summary_csv": str(summary_path),
        "request_events_csv": str(request_events),
        "git_commit": _read_json(output_dir / "environment.json").get("git_commit"),
        "artifact_sha256": {
            name: _sha256(path) for name, path in {
                "manifest.yaml": manifest_path,
                "environment.json": output_dir / "environment.json",
                "measurement_window.json": raw_dir / "measurement_window.json",
                "gateway_protocol.json": raw_dir / "gateway_protocol.json",
                "workload_manifest_hashes.json": raw_dir / "workload_manifest_hashes.json",
                "load_profile.csv": trace_path,
                "request_events.csv": request_events,
                "summary.csv": summary_path,
            }.items()
        },
    }


def validate_matched_baseline_pair(
    source_manifest_path: Path,
    *,
    controller: str,
    workload_pattern: str,
    seed: int,
    capacity: int,
    workload: dict[str, Any],
    measurement_seconds: int,
    warmup_seconds: int,
    sampling_interval_seconds: int,
    sla: dict[str, Any],
    gateway_reference: dict[str, Any],
    workload_manifest_hashes: dict[str, Any],
) -> dict[str, Any]:
    """Require matching, measured HPA and KEDA cells for one v1 run."""
    if controller not in {"headlight", "threshold"}:
        raise MatchedComparisonError("matched v1 comparison mode only supports Headlight and Threshold")
    supported_seeds = {101, 102, 103} if controller == "headlight" else {101, 102, 103, 104, 105}
    if seed not in supported_seeds:
        permitted = "101, 102, and 103" if controller == "headlight" else "101 through 105"
        raise MatchedComparisonError(f"matched v1 comparison mode only supports seeds {permitted}")
    if workload_pattern not in {"gradual-ramp", "flash-crowd", "diurnal"}:
        raise MatchedComparisonError(f"unsupported matched workload: {workload_pattern}")
    if capacity != 4:
        raise MatchedComparisonError("matched baseline runs use capacity C=4")

    source_manifest_path = source_manifest_path.resolve()
    source_manifest = _read_json(source_manifest_path)
    rows = [
        row for row in source_manifest.get("sources", [])
        if row.get("controller") in {"hpa", "keda"}
        and row.get("workload") == workload_pattern
        and int(row.get("seed", -1)) == seed
    ]
    selected = {str(row.get("controller")): row for row in rows}
    if len(rows) != 2 or set(selected) != {"hpa", "keda"}:
        raise MatchedComparisonError(
            f"baseline manifest must contain exactly one HPA and one KEDA cell for {workload_pattern}/seed-{seed}"
        )

    expected_trace = _expected_trace_sha256(
        workload,
        seed,
        capacity,
        warmup_seconds,
        measurement_seconds,
        sampling_interval_seconds,
    )
    baselines = {
        name: _validate_one_baseline(
            selected[name],
            controller=name,
            workload_pattern=workload_pattern,
            seed=seed,
            workload=workload,
            measurement_seconds=measurement_seconds,
            warmup_seconds=warmup_seconds,
            sampling_interval_seconds=sampling_interval_seconds,
            sla=sla,
            gateway_reference=gateway_reference,
            workload_manifest_hashes=workload_manifest_hashes,
            expected_trace_sha256=expected_trace,
        )
        for name in ("hpa", "keda")
    }
    return {
        "schema_version": "headlight.matched-v1-comparison/v1",
        "comparison_status": "matched_baseline_evidence_validated",
        "comparison_controller": controller,
        "environment_equivalence_established": False,
        "baseline_source_manifest": str(source_manifest_path),
        "baseline_source_manifest_sha256": _sha256(source_manifest_path),
        "capacity_users": capacity,
        "gateway_protocol": gateway_reference,
        "workload": workload_pattern,
        "seed": seed,
        "load_trace_sha256": expected_trace,
        "baselines": baselines,
    }
