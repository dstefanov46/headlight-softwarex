#!/usr/bin/env python3
"""Load the frozen v2 bundle and execute one shadow decision window offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.policy_v2 import PolicyV2Config, ServiceRuntimeStateV2  # noqa: E402
from headlight.profile import load_service_profile  # noqa: E402
from headlight.runtime_v2 import HeadlightV2Runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "results/headlight-v2-live/bundle-full-seed-601")
    parser.add_argument("--profile", type=Path, default=REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json")
    parser.add_argument("--dataset-manifest", type=Path, default=REPO_ROOT / "results/synthetic-headlight-v2/large-v2-policy/manifest.json")
    args = parser.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    holdout = next(record for record in manifest["runs"] if record["split"] == "holdout")
    with np.load(args.dataset_manifest.parent / holdout["path"], allow_pickle=False) as run:
        end = int(run["sample_indices"][0]) + 1
        start = end - int(manifest["config"]["lookback_steps"])
        node_window = run["node_features"][start:end]
        edge_window = np.zeros((end - start, 18, 18, 7), dtype=np.float32)
        edge_window[:, run["edge_source"], run["edge_destination"], :] = run["edge_features"][start:end]

    profile = load_service_profile(args.profile)
    clusters = tuple(profile.cluster_limits)
    runtime = HeadlightV2Runtime.from_bundle(
        args.bundle,
        runtime_profile=profile,
        prometheus_url=",".join(f"http://unused-{cluster}.invalid" for cluster in clusters),
        prometheus_clusters=clusters,
        action_handler_url="http://unused.invalid/placements",
        policy_config=PolicyV2Config(
            policy_id="headlight-v2-live:offline-verification",
            resource_weight=0.25,
            sla_risk_weight=1.0,
            churn_weight=0.1,
            reliability_weight=0.5,
        ),
        run_id="offline-one-window",
        seed=101,
        shadow_mode=True,
        device=torch.device("cpu"),
    )
    runtime.prime(node_window, edge_window)
    placements = {cluster: int(limits["min_replicas"]) for cluster, limits in profile.cluster_limits.items()}
    states = {
        node.name: ServiceRuntimeStateV2(
            desired_replicas=placements,
            ready_replicas=placements,
            pending_replicas=dict.fromkeys(placements, 0),
            pending_age_seconds=dict.fromkeys(placements, 0.0),
            demand_rps=1.0,
            cpu_utilization=0.2,
            queue_depth=0.0,
            p95_latency_ms=100.0,
            error_rate=0.0,
            predictive_fresh=True,
            time_since_last_action_seconds=600.0,
            exists=True,
        )
        for node in profile.nodes
    }
    result = runtime.infer_states(states, timestamp=1_010.0)
    if not result["ready"] or not result["shadow_mode"] or result["actuator_called"]:
        raise RuntimeError("one-window verification did not remain ready and shadow-only")
    if len(result["decisions"]) != len(profile.nodes):
        raise RuntimeError("one-window verification did not emit one decision per service")
    print("HEADLIGHT_V2_OFFLINE_VERIFY_OK " + json.dumps({
        "bundle_id": runtime.bundle.bundle_id,
        "decision_count": len(result["decisions"]),
        "shadow_mode": result["shadow_mode"],
        "actuator_called": result["actuator_called"],
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
