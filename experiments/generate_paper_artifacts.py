#!/usr/bin/env python3
"""Generate measured CSV artifacts from result bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.validate_configs import REPO_ROOT
from experiments.lib.quality import validate_bundle
from experiments.validate_artifacts import (
    ArtifactValidationError,
    assert_artifacts_not_stale,
    _parse_list,
    _parse_seeds,
    validated_summary_rows,
    write_validation_manifest,
)


ARTIFACTS = [
    "main_comparison.csv",
    "load_profiles.csv",
    "ablation.csv",
    "migration_offload.csv",
    "overhead.csv",
]


def measured_bundle(bundle_dir: Path) -> bool:
    environment_path = bundle_dir / "environment.json"
    if not environment_path.exists():
        return True
    try:
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return not environment.get("dry_run", False)


def validate_measured_seed(seed_dir: Path) -> dict:
    bundle_dir = seed_dir.parent
    manifest_path = bundle_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"{seed_dir}: bundle manifest is missing")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    validate_bundle(seed_dir, manifest, dry_run=False)
    return manifest


def summary_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_summaries(results_root: Path, validated_rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    if validated_rows is not None:
        return list(validated_rows)
    rows: list[dict[str, str]] = []
    for path in sorted(results_root.glob("**/seed-*/summaries/summary.csv")):
        if not measured_bundle(path.parents[2]):
            continue
        validate_measured_seed(path.parents[1])
        digest = summary_sha256(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["bundle"] = str(path.parents[2].relative_to(results_root))
                row["source_summary_sha256"] = digest
                rows.append(row)
    return rows


PROVENANCE_FIELDS = [
    "source_run",
    "source_bundle",
    "seed",
    "controller",
    "workload",
    "configuration_sha256",
    "commit_hash",
    "validation_status",
]


def provenance(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in PROVENANCE_FIELDS}


def seed_dir_from_row(results_root: Path, row: dict[str, str]) -> Path:
    return results_root / row["source_bundle"] / f"seed-{row['seed']}"


def _read_profile_rows(path: Path, metadata: dict[str, str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(metadata)
    return rows


def _collect_validated_load_profiles(results_root: Path, validated_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for validated in validated_rows:
        path = seed_dir_from_row(results_root, validated) / "raw" / "load_profile.csv"
        rows.extend(_read_profile_rows(path, provenance(validated)))
    return rows


def _summary_metadata(path: Path) -> dict[str, str]:
    summary = path / "summaries" / "summary.csv"
    if not summary.exists():
        return {}
    with summary.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {"controller": rows[0].get("controller", ""), "workload": rows[0].get("workload", "")}


def _collect_discovered_load_profiles(results_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(results_root.glob("**/seed-*/raw/load_profile.csv")):
        if not measured_bundle(path.parents[2]):
            continue
        validate_measured_seed(path.parents[1])
        bundle = str(path.parents[2].relative_to(results_root))
        seed = path.parents[1].name.removeprefix("seed-")
        metadata = {"bundle": bundle, "seed": seed, **_summary_metadata(path.parents[1])}
        rows.extend(_read_profile_rows(path, metadata))
    return rows


def collect_load_profiles(results_root: Path, validated_rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    return (
        _collect_validated_load_profiles(results_root, validated_rows)
        if validated_rows is not None
        else _collect_discovered_load_profiles(results_root)
    )


def collect_migration_offload(results_root: Path, validated_rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    if validated_rows is None:
        paths = sorted(results_root.glob("**/seed-*/raw/crp_transitions.json"))
        metadata_by_path = {path: {} for path in paths}
    else:
        paths = [seed_dir_from_row(results_root, row) / "raw" / "crp_transitions.json" for row in validated_rows]
        metadata_by_path = {path: provenance(row) for path, row in zip(paths, validated_rows)}
    return _collect_migration_paths(results_root, paths, metadata_by_path, validated_rows is not None)


def _collect_migration_paths(
    results_root: Path,
    paths: list[Path],
    metadata_by_path: dict[Path, dict[str, str]],
    validated: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_migration_rows_for_path(results_root, path, metadata_by_path[path], validated))
    return rows


def _migration_rows_for_path(
    results_root: Path,
    path: Path,
    metadata: dict[str, str],
    validated: bool,
) -> list[dict[str, str]]:
    if not validated and not measured_bundle(path.parents[2]):
        return []
    if not validated:
        validate_measured_seed(path.parents[1])
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    items = data.get("items", []) if isinstance(data, dict) else []
    bundle = str(path.parents[2].relative_to(results_root))
    seed = path.parents[1].name.removeprefix("seed-")
    rows: list[dict[str, str]] = []
    for item in items:
        placements = item.get("spec", {}).get("placements", [])
        service = item.get("metadata", {}).get("name", "")
        rows.append({"bundle": bundle, "seed": seed, "service": service, "metric": "placement_count", "value": str(len(placements)), **metadata})
        rows.extend(
            {"bundle": bundle, "seed": seed, "service": service, "cluster": placement.get("clusterName", ""), "metric": "replicas", "value": str(placement.get("replicas", "")), **metadata}
            for placement in placements
        )
    return rows


def collect_overhead(results_root: Path, validated_rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if validated_rows is None:
        paths = sorted(results_root.glob("**/seed-*/raw/ai_inference_times.csv"))
        metadata_by_path = {path: {} for path in paths}
    else:
        paths = [seed_dir_from_row(results_root, row) / "raw" / "ai_inference_times.csv" for row in validated_rows]
        metadata_by_path = {
            seed_dir_from_row(results_root, row) / "raw" / "ai_inference_times.csv": provenance(row)
            for row in validated_rows
        }
    for path in paths:
        if validated_rows is None:
            if not measured_bundle(path.parents[2]):
                continue
            validate_measured_seed(path.parents[1])
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            values = [float(row["duration_ms"]) for row in reader if row.get("duration_ms")]
        if not values:
            continue
        rows.append(
            {
                "bundle": str(path.parents[2].relative_to(results_root)),
                "seed": path.parents[1].name.removeprefix("seed-"),
                "metric": "ai_inference_duration_ms_mean",
                "value": f"{sum(values) / len(values):.6f}",
                **metadata_by_path[path],
            }
        )
    return rows


def collect_ablation(results_root: Path, validated_rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    return [
        row
        for row in collect_summaries(results_root, validated_rows)
        if any(token in row.get("bundle", "") for token in ["no-attention", "no-graph", "static-adjacency"])
        or any(token in row.get("source_bundle", "") for token in ["no-attention", "no-graph", "static-adjacency"])
    ]


def write_csv(path: Path, rows: list[dict[str, str]], default_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(default_fields + [field for row in rows for field in row.keys()]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate measured CSV artifacts")
    parser.add_argument("--results-root", default="results/softwarex")
    parser.add_argument("--output-dir", default="results/softwarex/artifacts")
    parser.add_argument("--replace-stale-artifacts", action="store_true")
    parser.add_argument("--allow-incomplete-matrix", action="store_true")
    parser.add_argument("--controllers", help="Comma-separated controller filter")
    parser.add_argument("--workloads", help="Comma-separated workload filter")
    parser.add_argument("--seeds", help="Comma-separated seed filter")
    args = parser.parse_args()
    results_root = REPO_ROOT / args.results_root
    output_dir = REPO_ROOT / args.output_dir
    try:
        rows, validation = validated_summary_rows(
            results_root,
            controllers=_parse_list(args.controllers),
            workloads=_parse_list(args.workloads),
            seeds=_parse_seeds(args.seeds),
            require_complete_matrix=not args.allow_incomplete_matrix,
        )
        assert_artifacts_not_stale(
            output_dir,
            validation["source_digest"],
            replace_stale=args.replace_stale_artifacts,
        )
    except ArtifactValidationError as exc:
        for issue in exc.issues:
            location = f" ({issue['path']})" if "path" in issue else ""
            print(f"{issue['code']}: {issue['message']}{location}", file=sys.stderr)
        return 1
    write_csv(
        output_dir / "main_comparison.csv",
        rows,
        [
            "source_run",
            "source_bundle",
            "seed",
            "controller",
            "workload",
            "configuration_sha256",
            "commit_hash",
            "validation_status",
            "sla_violation_rate",
            "excess_response_time_seconds",
            "cpu_seconds",
            "mean_pod_count",
            "p95_latency_ms",
        ],
    )
    write_csv(output_dir / "load_profiles.csv", collect_load_profiles(results_root, rows), [*PROVENANCE_FIELDS, "elapsed_seconds", "users", "spawn_rate"])
    write_csv(output_dir / "ablation.csv", collect_ablation(results_root, rows), [*PROVENANCE_FIELDS, "metric", "value"])
    write_csv(output_dir / "migration_offload.csv", collect_migration_offload(results_root, rows), [*PROVENANCE_FIELDS, "service", "cluster", "metric", "value"])
    write_csv(output_dir / "overhead.csv", collect_overhead(results_root, rows), [*PROVENANCE_FIELDS, "metric", "value"])
    write_validation_manifest(output_dir, validation)
    print(f"wrote artifacts: {output_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
