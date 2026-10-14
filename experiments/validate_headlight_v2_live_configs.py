#!/usr/bin/env python3
"""Validate hashes and one-seed invariants for Headlight live configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "experiments/configs/live-v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_extension_config(path: Path) -> dict:
    """Validate the selected follow-up cell without altering the original lock."""
    from experiments.lib.baseline_extension import TRACE_HASHES

    live = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = live["spec"]
    lock = json.loads((CONFIG_ROOT / "runtime-lock.json").read_text())
    workload = REPO_ROOT / spec["workload_profile"]
    pattern = yaml.safe_load(workload.read_text())["spec"]["pattern"]
    if spec.get("seed") not in (104, 105) or pattern not in TRACE_HASHES:
        raise ValueError(f"{path}: unsupported v2 extension seed/workload")
    expected = {
        "workload_profile": f"experiments/configs/workloads/{pattern}.yaml",
        "workload_profile_sha256": sha256(workload),
        "expected_comparator_trace_sha256": TRACE_HASHES[pattern],
        "bundle_id": lock["bundle"]["bundle_id"],
        "bundle_archive_sha256": lock["bundle"]["archive_sha256"],
        "policy_sha256": lock["policy"]["sha256"],
        "shadow_smoke_required": True,
        "warmup_seconds": 480,
        "measurement_seconds": 2400 if pattern == "diurnal" else 1200,
        "sampling_interval_seconds": 15,
        "gateway_protocol": "experiments/configs/gateway-protocols/research-v4-gateway.yaml",
        "gateway_protocol_sha256": sha256(REPO_ROOT / "experiments/configs/gateway-protocols/research-v4-gateway.yaml"),
        "capacity_users": 4,
        "mode": "reactive-emergency-disabled",
        "diagnostics_required": True,
        "diagnostic_dir": "/artifacts/diagnostics",
        "model_changed": False,
        "parameters_tuned": False,
        "claim_scope": "seed104-105-headlight-v2-followup",
    }
    for name, value in expected.items():
        if spec.get(name) != value or type(spec.get(name)) is not type(value):
            raise ValueError(f"{path}: {name} differs from the retained v2 protocol")
    if sha256(REPO_ROOT / lock["policy"]["path"]) != spec["policy_sha256"]:
        raise ValueError(f"{path}: current v2 policy hash mismatch")
    return live


def main() -> int:
    lock = json.loads((CONFIG_ROOT / "runtime-lock.json").read_text(encoding="utf-8"))
    policy = REPO_ROOT / lock["policy"]["path"]
    if sha256(policy) != lock["policy"]["sha256"]:
        raise ValueError("runtime-lock policy hash mismatch")
    bundle = REPO_ROOT / "results/headlight-v2-live/bundle-full-seed-601"
    if bundle.exists():
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        if metadata["bundle_id"] != lock["bundle"]["bundle_id"]:
            raise ValueError("runtime-lock bundle ID mismatch")
        if metadata["component_hashes"] != lock["bundle"]["component_hashes"]:
            raise ValueError("runtime-lock component hashes mismatch")
    archive = REPO_ROOT / "results/headlight-v2-live" / lock["bundle"]["archive_name"]
    if archive.exists() and sha256(archive) != lock["bundle"]["archive_sha256"]:
        raise ValueError("runtime-lock bundle archive hash mismatch")

    configs = sorted(CONFIG_ROOT.glob("headlight-v2-*-seed-101.yaml"))
    if len(configs) != 3:
        raise ValueError("exactly three Headlight live run configs are required")
    for path in configs:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))["spec"]
        workload_name = path.name.removeprefix("headlight-v2-").removesuffix("-seed-101.yaml")
        evidence = lock["comparator_evidence"][workload_name]
        if spec["seed"] != 101 or not spec["shadow_smoke_required"]:
            raise ValueError(f"{path}: seed 101 and shadow smoke are mandatory")
        if spec["bundle_id"] != lock["bundle"]["bundle_id"]:
            raise ValueError(f"{path}: bundle ID differs from runtime lock")
        if spec["bundle_archive_sha256"] != lock["bundle"]["archive_sha256"]:
            raise ValueError(f"{path}: bundle archive hash differs from runtime lock")
        if spec["policy_sha256"] != lock["policy"]["sha256"]:
            raise ValueError(f"{path}: policy hash differs from runtime lock")
        if spec["comparator_trace_sha256"] != evidence["trace_sha256"]:
            raise ValueError(f"{path}: comparator trace hash differs from runtime lock")
        workload = REPO_ROOT / spec["workload_profile"]
        if sha256(workload) != spec["workload_profile_sha256"]:
            raise ValueError(f"{path}: workload profile hash mismatch")
        if sorted(spec["comparator_trace_verified_equal_for"]) != ["headlight", "lstm", "threshold"]:
            raise ValueError(f"{path}: comparator trace set is incomplete")
        manifests = evidence["attempt_manifests"]
        manifest_paths = [REPO_ROOT / item["path"] for item in manifests.values()]
        if any(item.exists() for item in manifest_paths):
            if not all(item.exists() for item in manifest_paths):
                raise ValueError(f"{path}: comparator evidence is only partially available")
            for controller, item in manifests.items():
                manifest_path = REPO_ROOT / item["path"]
                if sha256(manifest_path) != item["sha256"]:
                    raise ValueError(f"{path}: {controller} attempt manifest hash mismatch")
                attempt = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["spec"]
                if attempt["controller"]["controller"] != controller:
                    raise ValueError(f"{path}: {controller} attempt declares another controller")
                if attempt["workload"]["pattern"] != workload_name or 101 not in attempt["run"]["seeds"]:
                    raise ValueError(f"{path}: {controller} attempt is not the expected seed/workload cell")
                trace = manifest_path.parent / "seed-101/raw/load_profile.csv"
                if sha256(trace) != evidence["trace_sha256"]:
                    raise ValueError(f"{path}: {controller} trace hash mismatch")
    print(f"HEADLIGHT_V2_LIVE_CONFIGS_OK runs={len(configs)} seed=101 bundle_id={lock['bundle']['bundle_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
