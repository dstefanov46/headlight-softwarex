"""Provenance and environment for the HPA/KEDA seed-104/105 extension."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.lib.config import REPO_ROOT, resolve_run
from experiments.scripts.render_minimal_trainticket import render
import yaml


SOURCE_MANIFEST = REPO_ROOT / "results/headlight-v2-live/seed101-103-aggregate-20261005/source-manifest.json"
GATEWAY = REPO_ROOT / "experiments/configs/gateway-protocols/research-v4-gateway.yaml"
TRACE_HASHES = {
    "gradual-ramp": "b1a3d85fce1cd8a4ce44bef382794c8c12fcb2647f389ae789a284cc19cb00ca",
    "diurnal": "ced8b35581298a0e88cdadc9939033d41372591ed75913aaea118fa99c057ce2",
    "flash-crowd": "015943997cc853524e85ae0aeb965bcdf92da58e4d1be2003a0ae6a485f73116",
}
MANIFEST_HASHES = {
    "data_plane_no_policies_sha256": "0c752bcc700389e1f84b80795a7f2a3e8323a147f6dda0a037938ee09fbc6899",
    "karmada_with_policies_sha256": "134a0ae9078b6833961ef9cd6d83ca0659789ccc2e66b7348c2e8eb2fbe27985",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_environment(root: Path) -> dict[str, str]:
    """Use explicit baseline settings, dropping inherited development overrides."""
    if os.environ.get("HEADLIGHT_TRAINTICKET_MONGO_STORAGE", "image-defined") != "image-defined":
        raise ValueError("historical extension requires Mongo storage image-defined")
    config = os.environ.get("HEADLIGHT_KUBECONFIG", str(REPO_ROOT / ".kube/headlight-testbed.config"))
    internal = os.environ.get("HEADLIGHT_INTERNAL_KUBECONFIG", str(REPO_ROOT / ".kube/headlight-members-internal.config"))
    endpoints = os.environ.get("HEADLIGHT_PREFLIGHT_PROMETHEUS_URL", "")
    return {
        "KUBECONFIG": config,
        "HEADLIGHT_KUBECONFIG": config,
        "HEADLIGHT_INTERNAL_KUBECONFIG": internal,
        "HEADLIGHT_POD_KUBECONFIG": internal,
        "HEADLIGHT_KARMADA_DATA": os.environ.get("HEADLIGHT_KARMADA_DATA", str(REPO_ROOT / ".karmada")),
        "HEADLIGHT_PREFLIGHT_PROMETHEUS_URL": endpoints,
        "HEADLIGHT_CONTROLLER_PROMETHEUS_URL": endpoints,
        "HEADLIGHT_GATEWAY_PROTOCOL_PATH": str(GATEWAY),
        "HEADLIGHT_EVALUATION_CAPACITY": "4",
        "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
        "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
        "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
        "HEADLIGHT_MINIMAL_TRAINTICKET_MANIFEST": str(root / "runtime/rendered-advanced-ticket.yaml"),
        "HEADLIGHT_PREFLIGHT_ARTIFACT_DIR": str(root / "preflight"),
    }


def audit_baselines() -> dict[str, Any]:
    """Reject changed data plane, gateway, timing, or missing historical cells."""
    if os.environ.get("HEADLIGHT_TRAINTICKET_MONGO_STORAGE") != "image-defined":
        raise ValueError("historical extension requires Mongo storage image-defined")
    for key, policies in (("data_plane_no_policies_sha256", False), ("karmada_with_policies_sha256", True)):
        rendered = yaml.safe_dump_all(render("advanced-ticket", "trainticket", "capacity-vm",
                                            include_policies=policies, mongo_storage="image-defined"), sort_keys=True)
        if hashlib.sha256(rendered.encode()).hexdigest() != MANIFEST_HASHES[key]:
            raise ValueError(f"current workload manifest differs from historical baseline: {key}")
    gateway_spec = yaml.safe_load(GATEWAY.read_text())["spec"]
    gateway = {"version": gateway_spec["version"],
               "path": str(GATEWAY.relative_to(REPO_ROOT)), "sha256": sha256(GATEWAY),
               "base_protocol_sha256": sha256(REPO_ROOT / gateway_spec["base_protocol_path"])}
    sources = json.loads(SOURCE_MANIFEST.read_text())["sources"]
    validated = []
    identities = set()
    for source in sources:
        if source["controller"] not in {"hpa", "keda"}:
            continue
        seed, controller, workload = source["seed"], source["controller"], source["workload"]
        identity = (seed, controller, workload)
        if identity in identities:
            raise ValueError(f"duplicate baseline identity: {identity}")
        identities.add(identity)
        root = REPO_ROOT / source["output_dir"] / f"seed-{seed}"
        summary = root / "summaries/summary.csv"
        with summary.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"expected one baseline summary: {identity}")
        row = rows[0]
        if (row.get("seed"), row.get("controller"), row.get("workload"), row.get("technical_validity"), row.get("performance_outcome")) != (str(seed), controller, workload, "valid", "measured"):
            raise ValueError(f"invalid baseline summary: {identity}")
        if sha256(root / "raw/load_profile.csv") != TRACE_HASHES[workload]:
            raise ValueError(f"baseline trace mismatch: {identity}")
        recorded_hashes = json.loads((root / "raw/workload_manifest_hashes.json").read_text())
        if any(recorded_hashes.get(key) != value for key, value in MANIFEST_HASHES.items()):
            raise ValueError(f"baseline workload manifest mismatch: {identity}")
        if json.loads((root / "raw/gateway_protocol.json").read_text())["reference"] != gateway:
            raise ValueError(f"baseline gateway mismatch: {identity}")
        timing = json.loads((root / "raw/measurement_window.json").read_text())
        duration = 2400 if workload == "diurnal" else 1200
        if (timing.get("warmup_seconds"), timing.get("measurement_seconds"), timing.get("cooldown_seconds")) != (480, duration, 60):
            raise ValueError(f"baseline timing mismatch: {identity}")
        validated.append({**source, "summary_sha256": sha256(summary),
                          "workload_manifest_hashes_sha256": sha256(root / "raw/workload_manifest_hashes.json")})
    expected = {(seed, controller, workload) for seed in (101, 102, 103)
                for controller in ("hpa", "keda") for workload in TRACE_HASHES}
    if identities != expected:
        raise ValueError("historical baseline archive must contain all 18 HPA/KEDA cells")
    for controller in ("hpa", "keda"):
        for workload in TRACE_HASHES:
            manifest = resolve_run(Path(f"experiments/configs/runs/{controller}-{workload}-multi-cluster-headlight.yaml"))
            run = manifest["spec"]["run"]
            if (run["warmup_seconds"], run["duration_seconds"], run["sampling_interval_seconds"],
                run.get("diagnostic_observation_tail_seconds", 0)) != (480, 2400 if workload == "diurnal" else 1200, 15, 0):
                raise ValueError(f"extension timing differs from historical baseline: {controller}/{workload}")
    return {
        "source_manifest": str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST), "sources": validated,
        "gateway": gateway, "workload_manifest_hashes": MANIFEST_HASHES,
        "environment_equivalence_established": False,
        "comparability_note": "Docker was migrated from tmpfs to persistent disk. Historical manifest mode is preserved; new cells are separately identified follow-up evidence. Matching hashes do not establish host/storage equivalence. Do not silently pool with historical cells.",
    }
