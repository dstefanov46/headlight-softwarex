#!/usr/bin/env python3
"""Execute a frozen v2 run, or an evaluation gateway shadow-only replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_experiment import (  # noqa: E402
    EVALUATION_SEEDS,
    GATEWAY_PROTOCOL,
    apply_protocol_metadata,
    environment,
    run_one_seed,
)
from experiments.validate_headlight_v2_live_configs import main as validate_configs  # noqa: E402
from experiments.validate_headlight_v2_live_configs import validate_extension_config  # noqa: E402
from experiments.lib.config import write_yaml  # noqa: E402


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid YAML mapping: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--shadow-only",
        action="store_true",
        help="run with the opt-in gateway topology and force actuation off",
    )
    parser.add_argument("--output-dir", type=Path, help="optional result-directory override")
    args = parser.parse_args()
    validate_configs()
    run_path = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    live = load_yaml(run_path)
    spec = live["spec"]
    extension = (run_path.parent.name == "headlight-v2-extension"
                 or spec.get("claim_scope") == "seed104-105-headlight-v2-followup")
    if extension:
        validate_extension_config(run_path)
        if (GATEWAY_PROTOCOL is None
                or GATEWAY_PROTOCOL["reference"]["path"] != spec["gateway_protocol"]
                or GATEWAY_PROTOCOL["reference"]["sha256"] != spec["gateway_protocol_sha256"]):
            raise RuntimeError("v2 extension requires the shared gateway specified by its run config")
        for key, value in {
            "HEADLIGHT_EVALUATION_CAPACITY": "4",
            "HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY": "1",
            "HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket",
            "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm",
            "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined",
        }.items():
            if os.environ.get(key) != value:
                raise RuntimeError(f"v2 extension requires {key}={value}")
    seed = int(spec["seed"])
    if seed not in EVALUATION_SEEDS:
        raise ValueError(
            f"v2 live validation seed {seed} is not one of the frozen evaluation seeds "
            f"{list(EVALUATION_SEEDS)}"
        )
    if args.shadow_only and GATEWAY_PROTOCOL is None:
        raise RuntimeError("--shadow-only requires HEADLIGHT_GATEWAY_PROTOCOL_PATH")
    legacy_output = (REPO_ROOT / spec["output_dir"]).resolve()
    if GATEWAY_PROTOCOL is not None:
        requested_output = (
            args.output_dir if args.output_dir is not None and args.output_dir.is_absolute()
            else REPO_ROOT / args.output_dir if args.output_dir is not None else None
        )
        if requested_output is None or requested_output.resolve() == legacy_output:
            raise RuntimeError(
                "gateway topology requires --output-dir distinct from frozen v2 live-validation results"
            )
    os.environ.setdefault(
        "HEADLIGHT_V2_BUNDLE_DIR",
        str(REPO_ROOT / "results/headlight-v2-live/bundle-full-seed-601"),
    )
    os.environ.setdefault(
        "HEADLIGHT_V2_POLICY_CONFIG",
        str(REPO_ROOT / "experiments/configs/models/headlight-v2-closed-loop-policy.json"),
    )
    if not args.dry_run and args.shadow_only:
        os.environ["HEADLIGHT_V2_ENABLE_ACTUATION"] = "0"
    elif not args.dry_run:
        evidence = Path(os.environ.get(
            "HEADLIGHT_V2_SHADOW_EVIDENCE",
            REPO_ROOT / "results/headlight-v2-live/shadow-smoke/shadow-smoke.ok",
        ))
        lock = json.loads((REPO_ROOT / "experiments/configs/live-v2/runtime-lock.json").read_text())
        if not evidence.is_file() or f"bundle_id={lock['bundle']['bundle_id']}" not in evidence.read_text():
            raise RuntimeError("matching shadow-smoke evidence is required before live actuation")
        os.environ["HEADLIGHT_V2_ENABLE_ACTUATION"] = "1"
        os.environ["HEADLIGHT_V2_SHADOW_EVIDENCE"] = str(evidence)

    workload = load_yaml(REPO_ROOT / spec["workload_profile"])["spec"]
    topology = load_yaml(REPO_ROOT / "experiments/configs/topologies/multi-cluster-headlight.yaml")["spec"]
    manifest = {
        "apiVersion": "headlight.experiments/v1alpha1",
        "kind": "ResolvedExperimentManifest",
        "metadata": {"name": live["metadata"]["name"]},
        "spec": {
            "controller": {
                "controller": "headlight-v2",
                "controller_variant": "full-action-conditioned",
                "setup_script": "experiments/scripts/deploy_headlight_v2_shadow.sh",
                "actuation_path": "actions-handler",
            },
            "topology": topology,
            "workload": workload,
            "run": {
                "seeds": [seed],
                "duration_seconds": int(spec["measurement_seconds"]),
                "warmup_seconds": int(spec["warmup_seconds"]),
                "sampling_interval_seconds": int(spec["sampling_interval_seconds"]),
                "output_dir": spec["output_dir"],
                "sla": {"p95_latency_ms": 1000, "minimum_success_rate": 0.99},
                "reset_policy": {
                    "between_seeds": True,
                    "delete_pods": True,
                    "clear_controller_state": True,
                },
                "qualification_overrides": {
                    "kind": (
                        "headlight-v2-gateway-shadow"
                        if args.shadow_only
                        else "headlight-v2-gateway-evaluation"
                        if GATEWAY_PROTOCOL is not None
                        else "headlight-v2-live-validation"
                    ),
                    "prior_comparator_traces_directly_comparable": GATEWAY_PROTOCOL is None,
                },
                "frozen_inputs": {
                    "bundle_id": spec["bundle_id"],
                    "bundle_archive_sha256": spec["bundle_archive_sha256"],
                    "policy_sha256": spec["policy_sha256"],
                },
            },
        },
    }
    if GATEWAY_PROTOCOL is None:
        manifest["spec"]["run"]["frozen_inputs"]["comparator_trace_sha256"] = spec["comparator_trace_sha256"]
    if extension:
        manifest["spec"]["run"]["qualification_overrides"].update({
            "claim_scope": spec["claim_scope"],
            "environment_equivalence_established": False,
            "expected_comparator_trace_sha256": spec["expected_comparator_trace_sha256"],
        })
    apply_protocol_metadata(manifest)
    output = args.output_dir or (REPO_ROOT / spec["output_dir"])
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    write_yaml(output / "manifest.yaml", manifest)
    (output / "environment.json").write_text(
        json.dumps(environment(args.dry_run), indent=2) + "\n", encoding="utf-8"
    )
    run_one_seed(manifest, output, seed, args.dry_run)
    try:
        display_output = output.relative_to(REPO_ROOT)
    except ValueError:
        display_output = output
    print(f"HEADLIGHT_V2_LIVE_RUN_OK output={display_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
