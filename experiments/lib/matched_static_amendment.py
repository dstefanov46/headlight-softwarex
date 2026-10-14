"""Validate and expand the configured, static-only three-case amendment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.lib.matched_baselines import ROOT, identity, read_json, sha256
from experiments.lib.static_plan_matched import validate_matched_static_plan

SCHEMA = "headlight.static-selection-amendment-artifact/v1"
EXPECTED_CASES = ["gradual-ramp", "flash-crowd", "diurnal"]
EXPECTED_SEEDS = [101, 102, 103, 104, 105]


def validate_static_amendment(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = read_json(root / "static-amendment.json")
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported static amendment artifact")
    if manifest.get("status") != "configuredly-selected":
        raise ValueError("static amendment is not configuredly selected")
    if manifest.get("selected_candidate_id") != "static-14":
        raise ValueError("amendment does not select static-14")
    if manifest.get("primary_cases") != EXPECTED_CASES or manifest.get("evaluation_workloads") != EXPECTED_CASES:
        raise ValueError("static amendment workload scope mismatch")
    if manifest.get("evaluation_seeds") != EXPECTED_SEEDS:
        raise ValueError("static amendment evaluation seed list mismatch")
    if "capacity-readiness-failure" in manifest["primary_cases"]:
        raise ValueError("pending-validation capacity stress case cannot be primary in this amendment")
    source_episodes = Path(manifest["source_static_episodes_path"]).resolve()
    if not source_episodes.is_file() or sha256(source_episodes) != manifest.get("source_static_episodes_sha256"):
        raise ValueError("source static episode identity mismatch")
    parent_protocol = Path(manifest["parent_protocol_path"]).resolve()
    amendment_config = Path(manifest["amendment_config_path"]).resolve()
    if not parent_protocol.is_file() or sha256(parent_protocol) != manifest.get("parent_protocol_sha256"):
        raise ValueError("parent protocol identity mismatch")
    if not amendment_config.is_file() or sha256(amendment_config) != manifest.get("amendment_config_sha256"):
        raise ValueError("amendment configuration identity mismatch")
    amendment = read_json(amendment_config)
    configured_parent = (ROOT / amendment["parent_protocol"]).resolve()
    if (
        amendment.get("amendment_id") != manifest.get("amendment_id")
        or amendment.get("primary_cases") != EXPECTED_CASES
        or amendment.get("evaluation_workloads") != EXPECTED_CASES
        or amendment.get("evaluation_seeds") != EXPECTED_SEEDS
        or configured_parent != parent_protocol
    ):
        raise ValueError("amendment configuration scope or parent mismatch")
    for name, digest in manifest.get("artifacts", {}).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"static amendment artifact hash mismatch: {name}")
    selection = read_json(root / "static-selection.json")
    if (
        selection.get("status") != "configured"
        or selection.get("protocol_amendment_id") != manifest.get("amendment_id")
        or selection.get("selected", {}).get("candidate_id") != "static-14"
    ):
        raise ValueError("static selection record does not select static-14")
    profile_path = Path(manifest["service_profile_path"]).resolve()
    if not profile_path.is_file() or sha256(profile_path) != manifest.get("service_profile_sha256"):
        raise ValueError("service profile identity mismatch")
    plan = validate_matched_static_plan(root / "static-plan.json", profile_path)
    if plan.get("replicas_per_application_service") != 14:
        raise ValueError("static plan replica total is not 14")
    if plan.get("selection_hash") != identity(selection["selected"]):
        raise ValueError("static plan selection binding mismatch")
    return manifest


def static_amendment_cells(root: Path) -> list[dict[str, Any]]:
    manifest = validate_static_amendment(root)
    return [
        {
            "controller": "static-matched",
            "workload": workload,
            "seed": seed,
            "artifact": str((root.resolve() / "static-plan.json")),
            "amendment": str((root.resolve() / "static-amendment.json")),
        }
        for workload in manifest["evaluation_workloads"]
        for seed in manifest["evaluation_seeds"]
    ]


def write_static_amendment(
    *,
    output_dir: Path,
    amendment_config: Path,
    parent_protocol: Path,
    source_episodes: Path,
    service_profile: Path,
    selection: dict[str, Any],
    candidates: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Write append-only selection, plan, and amendment provenance artifacts."""
    from experiments.lib.static_plan_matched import build_matched_static_plan

    output_dir.mkdir(parents=True, exist_ok=False)
    selection_record = {
        "schema_version": "headlight.static-selection/matched-v2-amended",
        "status": "configured",
        "protocol_amendment_id": "static-primary-three-case-v1",
        "selected": selection,
        "candidates": candidates,
        "episodes_per_candidate": 9,
        "episode_count": 72,
        "excluded_case_retained_as_secondary": "capacity-readiness-failure",
    }
    (output_dir / "static-selection.json").write_text(
        json.dumps(selection_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan_comparison = {
        **comparison,
        "cases": EXPECTED_CASES,
        "protocol_amendment_id": "static-primary-three-case-v1",
    }
    plan = build_matched_static_plan(selection, plan_comparison)
    (output_dir / "static-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = {
        "schema_version": SCHEMA,
        "status": "configuredly-selected",
        "amendment_id": "static-primary-three-case-v1",
        "amendment_status": "post-hoc-development-amendment",
        "parent_protocol_path": str(parent_protocol.resolve()),
        "parent_protocol_sha256": sha256(parent_protocol),
        "amendment_config_path": str(amendment_config.resolve()),
        "amendment_config_sha256": sha256(amendment_config),
        "source_static_episodes_path": str(source_episodes.resolve()),
        "source_static_episodes_sha256": sha256(source_episodes),
        "selected_candidate_id": "static-14",
        "primary_cases": EXPECTED_CASES,
        "secondary_stress_cases": ["capacity-readiness-failure"],
        "development_seeds": [711, 712, 713],
        "evaluation_workloads": EXPECTED_CASES,
        "evaluation_seeds": EXPECTED_SEEDS,
        "service_profile_path": str(service_profile.resolve()),
        "service_profile_sha256": sha256(service_profile),
        "claim_scope": "Configured offline development selection; independent variation-seed evaluation and live readiness are pending.",
        "artifacts": {
            "static-selection.json": sha256(output_dir / "static-selection.json"),
            "static-plan.json": sha256(output_dir / "static-plan.json"),
        },
    }
    (output_dir / "static-amendment.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
