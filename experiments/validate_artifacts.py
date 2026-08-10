#!/usr/bin/env python3
"""Validate measured experiment bundles before generating measured artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lib.config import REPO_ROOT
from experiments.lib.quality import BundleQualityError, validate_bundle
from experiments.validate_configs import validate_all


VALIDATION_SCHEMA = "headlight.artifact-validation/v1"
ARTIFACT_MANIFEST = "artifact_validation_manifest.json"
REQUIRED_SUMMARY_FIELDS = {
    "seed",
    "controller",
    "workload",
    "sla_violation_rate",
    "excess_response_time_seconds",
    "cpu_seconds",
    "mean_pod_count",
    "p95_latency_ms",
}
ZERO_GUARD_FIELDS = (
    "sla_violation_rate",
    "excess_response_time_seconds",
    "cpu_seconds",
    "mean_pod_count",
    "p95_latency_ms",
)


class ArtifactValidationError(RuntimeError):
    def __init__(self, issues: list[dict[str, str]]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{issue['code']}: {issue['message']}" for issue in issues))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _issue(code: str, message: str, path: Path | None = None) -> dict[str, str]:
    payload = {"code": code, "message": message}
    if path is not None:
        payload["path"] = str(path)
    return payload


def _parse_list(value: str | None) -> set[str] | None:
    if value is None or value == "":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_seeds(value: str | None) -> set[int] | None:
    if value is None or value == "":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _expected_matrix(
    controllers: set[str] | None,
    workloads: set[str] | None,
    seeds: set[int] | None,
) -> set[tuple[str, str, int]]:
    if controllers is not None and workloads is not None and seeds is not None:
        return {(controller, workload, seed) for controller in controllers for workload in workloads for seed in seeds}
    manifests, _ = validate_all()
    expected: set[tuple[str, str, int]] = set()
    for manifest in manifests:
        controller = str(manifest["spec"]["controller"]["controller"])
        workload = str(manifest["spec"]["workload"]["pattern"])
        if controllers is not None and controller not in controllers:
            continue
        if workloads is not None and workload not in workloads:
            continue
        configured_seeds = {int(seed) for seed in manifest["spec"]["run"]["seeds"]}
        selected_seeds = seeds if seeds is not None else configured_seeds
        for seed in sorted(configured_seeds & selected_seeds):
            expected.add((controller, workload, seed))
    return expected


def _read_summary(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    missing = sorted(REQUIRED_SUMMARY_FIELDS - set(reader.fieldnames or []))
    if missing:
        issues.append(_issue("summary_fields", f"missing summary fields: {', '.join(missing)}", path))
    if len(rows) != 1:
        issues.append(_issue("summary_row_count", f"expected exactly one summary row, found {len(rows)}", path))
    return rows, issues


def _summary_has_invalid_zeroes(row: dict[str, str]) -> bool:
    values: list[float] = []
    for field in ZERO_GUARD_FIELDS:
        try:
            parsed = float(row.get(field, ""))
        except (TypeError, ValueError):
            return True
        if not math.isfinite(parsed):
            return True
        values.append(parsed)
    return all(math.isclose(value, 0.0, abs_tol=1e-12) for value in values)


def _bundle_environment(bundle_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    path = bundle_dir / "environment.json"
    if not path.exists():
        return None, [_issue("missing_environment", "environment.json is required for measured artifacts", path)]
    try:
        environment = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, [_issue("invalid_environment", "environment.json is not valid JSON", path)]
    return environment, []


def _validate_seed_summary(
    bundle_dir: Path,
    manifest: dict[str, Any],
    root: Path,
    controller: str,
    workload: str,
    source_run: str,
    config_hash: str,
    commit: str,
    seed: int,
    seen: dict[tuple[str, str, int], Path],
) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    seed_dir = bundle_dir / f"seed-{seed}"
    summary_path = seed_dir / "summaries" / "summary.csv"
    if not summary_path.exists():
        return None, [_issue("missing_run", f"missing seed bundle for {controller}/{workload}/seed-{seed}", seed_dir)]
    key = (controller, workload, seed)
    if key in seen:
        return None, [_issue("duplicate_run", f"duplicate measured run for controller={controller} workload={workload} seed={seed}; first={seen[key]}", seed_dir)]
    seen[key] = seed_dir
    try:
        validate_bundle(seed_dir, manifest, dry_run=False)
    except BundleQualityError as exc:
        return None, [_issue(f"quality_{item.code}", item.message, seed_dir) for item in exc.issues]
    except (OSError, KeyError, ValueError) as exc:
        return None, [_issue("quality_gate_error", str(exc), seed_dir)]
    summary_rows, summary_issues = _read_summary(summary_path)
    issues.extend(summary_issues)
    if len(summary_rows) != 1:
        return None, issues
    row = dict(summary_rows[0])
    if str(row.get("controller", "")) != controller:
        issues.append(_issue("summary_controller", f"summary controller {row.get('controller')} != {controller}", summary_path))
    if str(row.get("workload", "")) != workload:
        issues.append(_issue("summary_workload", f"summary workload {row.get('workload')} != {workload}", summary_path))
    if int(row.get("seed", "-1")) != seed:
        issues.append(_issue("summary_seed", f"summary seed {row.get('seed')} != {seed}", summary_path))
    if _summary_has_invalid_zeroes(row):
        issues.append(_issue("invalid_zero_run", "measured summary is all-zero or non-finite", summary_path))
        return None, issues
    row.update({
        "source_run": source_run,
        "source_bundle": str(bundle_dir.relative_to(root)),
        "source_summary": str(summary_path.relative_to(root)),
        "source_summary_sha256": _sha256(summary_path),
        "configuration_sha256": config_hash,
        "commit_hash": commit,
        "validation_status": "validated",
    })
    return row, issues


def _load_selected_manifest(
    manifest_path: Path,
    controllers: set[str] | None,
    workloads: set[str] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [_issue("invalid_manifest", f"manifest is not valid YAML: {exc}", manifest_path)]
    if not isinstance(manifest, dict) or "spec" not in manifest:
        return None, [_issue("invalid_manifest", "manifest is not a resolved experiment manifest", manifest_path)]
    controller = str(manifest["spec"]["controller"]["controller"])
    workload = str(manifest["spec"]["workload"]["pattern"])
    if (controllers is not None and controller not in controllers) or (workloads is not None and workload not in workloads):
        return None, []
    return manifest, []


def _validate_manifest(
    manifest_path: Path,
    root: Path,
    controllers: set[str] | None,
    workloads: set[str] | None,
    seeds: set[int] | None,
    seen: dict[tuple[str, str, int], Path],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    bundle_dir = manifest_path.parent
    manifest, load_issues = _load_selected_manifest(manifest_path, controllers, workloads)
    if manifest is None:
        return [], load_issues
    controller = str(manifest["spec"]["controller"]["controller"])
    workload = str(manifest["spec"]["workload"]["pattern"])
    environment, issues = _bundle_environment(bundle_dir)
    if environment and environment.get("dry_run", False):
        return [], issues
    commit = str((environment or {}).get("git_commit") or "")
    if not commit:
        issues.append(_issue("missing_commit", "measured bundle must record git_commit", bundle_dir / "environment.json"))
    source_run = str(manifest.get("metadata", {}).get("name", bundle_dir.name))
    config_hash = _json_sha256(manifest)
    rows: list[dict[str, str]] = []
    selected_seeds = (int(value) for value in manifest["spec"]["run"]["seeds"] if seeds is None or int(value) in seeds)
    for seed in selected_seeds:
        row, seed_issues = _validate_seed_summary(bundle_dir, manifest, root, controller, workload, source_run, config_hash, commit, seed, seen)
        issues.extend(seed_issues)
        if row is not None:
            rows.append(row)
    return rows, issues


def validated_summary_rows(
    results_root: Path,
    *,
    controllers: set[str] | None = None,
    workloads: set[str] | None = None,
    seeds: set[int] | None = None,
    require_complete_matrix: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = results_root if results_root.is_absolute() else REPO_ROOT / results_root
    issues: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    seen: dict[tuple[str, str, int], Path] = {}

    for manifest_path in sorted(root.glob("**/manifest.yaml")):
        if ARTIFACT_MANIFEST in {part.name for part in manifest_path.parent.parents}:
            continue
        manifest_rows, manifest_issues = _validate_manifest(manifest_path, root, controllers, workloads, seeds, seen)
        rows.extend(manifest_rows)
        issues.extend(manifest_issues)

    expected = _expected_matrix(controllers, workloads, seeds)
    observed = {(row["controller"], row["workload"], int(row["seed"])) for row in rows}
    if require_complete_matrix:
        for controller, workload, seed in sorted(expected - observed):
            issues.append(_issue("incomplete_matrix", f"missing validated run for {controller}/{workload}/seed-{seed}"))
    if not rows:
        issues.append(_issue("missing_runs", f"no validated measured runs found under {root}"))
    if issues:
        raise ArtifactValidationError(issues)

    source_digest = _json_sha256(
        [
            {
                "controller": row["controller"],
                "workload": row["workload"],
                "seed": row["seed"],
                "source_summary_sha256": row["source_summary_sha256"],
                "configuration_sha256": row["configuration_sha256"],
                "commit_hash": row["commit_hash"],
            }
            for row in sorted(rows, key=lambda value: (value["controller"], value["workload"], int(value["seed"])))
        ]
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "validation_status": "validated",
        "results_root": str(root.relative_to(REPO_ROOT) if root.is_relative_to(REPO_ROOT) else root),
        "source_digest": source_digest,
        "validated_rows": len(rows),
        "matrix": sorted([list(item) for item in observed]),
    }
    return rows, validation


def assert_artifacts_not_stale(output_dir: Path, source_digest: str, *, replace_stale: bool = False) -> None:
    manifest_path = output_dir / ARTIFACT_MANIFEST
    if not manifest_path.exists():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError([_issue("stale_artifact_manifest", f"cannot parse existing artifact manifest: {exc}", manifest_path)])
    if existing.get("source_digest") != source_digest and not replace_stale:
        raise ArtifactValidationError(
            [
                _issue(
                    "stale_generated_artifacts",
                    "existing generated artifacts were derived from a different source set; rerun with --replace-stale-artifacts after reviewing the changed inputs",
                    manifest_path,
                )
            ]
        )


def write_validation_manifest(output_dir: Path, validation: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ARTIFACT_MANIFEST).write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/softwarex")
    parser.add_argument("--controllers")
    parser.add_argument("--workloads")
    parser.add_argument("--seeds")
    parser.add_argument("--allow-incomplete-matrix", action="store_true")
    args = parser.parse_args()
    try:
        rows, validation = validated_summary_rows(
            Path(args.results_root),
            controllers=_parse_list(args.controllers),
            workloads=_parse_list(args.workloads),
            seeds=_parse_seeds(args.seeds),
            require_complete_matrix=not args.allow_incomplete_matrix,
        )
    except ArtifactValidationError as exc:
        for issue in exc.issues:
            location = f" ({issue['path']})" if "path" in issue else ""
            print(f"{issue['code']}: {issue['message']}{location}", file=sys.stderr)
        return 1
    print(json.dumps({**validation, "validated_rows": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
