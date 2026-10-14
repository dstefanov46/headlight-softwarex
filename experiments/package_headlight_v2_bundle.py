#!/usr/bin/env python3
"""Package the selected synthetic checkpoint as an immutable Headlight bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.bundle_v2 import create_v2_bundle  # noqa: E402
from headlight.observability import content_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "results/synthetic-headlight-v2/training-v2-policy-accepted/full-seed-601.pt")
    parser.add_argument("--training-report", type=Path, default=REPO_ROOT / "results/synthetic-headlight-v2/training-v2-policy-accepted/training-report.json")
    parser.add_argument("--dataset-manifest", type=Path, default=REPO_ROOT / "results/synthetic-headlight-v2/large-v2-policy/manifest.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/headlight-v2-live/bundle-full-seed-601")
    args = parser.parse_args()

    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    selected = report["selected"]["full"]
    model_config = dict(selected["model_config"])
    model_config["lookback_steps"] = int(manifest["config"]["lookback_steps"])
    feature_schema = json.loads((args.dataset_manifest.parent / "feature-schema.json").read_text(encoding="utf-8"))
    feature_schema.setdefault(
        "derived_request_features",
        ["request_rate_difference", "request_rate_short_slope", "request_rate_long_slope"],
    )
    adjacency = json.loads((args.dataset_manifest.parent / "adjacency.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("selected checkpoint is not a state dictionary")
    records = manifest["runs"]
    partitions = {
        split: [int(record["seed"]) for record in records if record["split"] == split]
        for split in ("fitting", "calibration", "holdout")
    }
    training = {
        "dataset_hash": manifest["dataset_hash"],
        "feature_schema_hash": content_sha256(feature_schema),
        "adjacency_hash": content_sha256(adjacency),
        "fitting_seeds": partitions["fitting"],
        "calibration_seeds": partitions["calibration"],
        "holdout_seeds": partitions["holdout"],
        "initialization_seed": int(selected["initialization_seed"]),
        "selection_procedure": "minimum calibration multitask loss among accepted full-model initialization seeds; selected seed 601 before live seed 101",
        "created_before_evaluation": True,
        "source_checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)),
        "source_training_report": str(args.training_report.relative_to(REPO_ROOT)),
    }
    calibration = {
        "schema_version": "headlight.calibration/identity-v1",
        "calibrated": True,
        "method": "identity",
        "evidence": {
            "calibration_sla_brier_score": selected["history"][selected["best_epoch"]]["calibration"]["sla_brier_score"],
            "calibration_reliability_brier_score": selected["history"][selected["best_epoch"]]["calibration"]["reliability_brier_score"],
        },
        "scope": "synthetic engineering checkpoint; live validation remains shadow-first",
    }
    bundle = create_v2_bundle(
        args.output,
        checkpoint_state=checkpoint,
        model_config=model_config,
        feature_schema=feature_schema,
        adjacency=adjacency,
        calibration=calibration,
        training_metadata=training,
    )
    print(json.dumps({
        "bundle": str(args.output),
        "bundle_id": bundle.bundle_id,
        "component_hashes": bundle.metadata["component_hashes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
