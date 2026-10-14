#!/usr/bin/env python3
"""Run the immutable Headlight controller; shadow mode is the safe default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from headlight.policy_v2 import PolicyV2Config
from headlight.profile import load_service_profile
from headlight.runtime_v2 import HeadlightV2Runtime


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path: str | Path) -> tuple[PolicyV2Config, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "headlight.closed-loop-policy-freeze/v1":
        raise ValueError("unsupported Headlight frozen policy schema")
    if payload.get("status") != "frozen-before-evaluation":
        raise ValueError("Headlight live policy must be frozen before evaluation")
    weights = payload["weights"]
    return PolicyV2Config(
        policy_id=f"headlight-v2-live:{file_sha256(path)[:16]}",
        resource_weight=float(weights["resource"]),
        sla_risk_weight=float(weights["sla_risk"]),
        churn_weight=float(weights["migration"]),
        reliability_weight=float(weights["reliability"]),
        tuning_seeds=(401, 403, 407),
    ), file_sha256(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bundle", default=os.environ.get("HEADLIGHT_V2_MODEL_BUNDLE"))
    value.add_argument("--service-profile", default=os.environ.get("HEADLIGHT_SERVICE_PROFILE"))
    value.add_argument("--policy-config", default=os.environ.get("HEADLIGHT_V2_POLICY_CONFIG"))
    value.add_argument("--expected-bundle-id", default=os.environ.get("HEADLIGHT_V2_BUNDLE_ID"))
    value.add_argument("--expected-policy-sha256", default=os.environ.get("HEADLIGHT_V2_POLICY_SHA256"))
    value.add_argument("--run-id", default=os.environ.get("HEADLIGHT_RUN_ID", "headlight-v2-shadow-smoke"))
    value.add_argument("--seed", type=int, default=int(os.environ.get("HEADLIGHT_SEED", "101")))
    value.add_argument(
        "--prometheus-url",
        default=os.environ.get(
            "HEADLIGHT_PROMETHEUS_URL",
            "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
        ),
    )
    value.add_argument(
        "--action-handler-url",
        default=os.environ.get(
            "HEADLIGHT_ACTION_HANDLER_URL",
            "http://actions-handler.actions-handler-system.svc.cluster.local:8080/placements",
        ),
    )
    value.add_argument("--namespace", default=os.environ.get("HEADLIGHT_NAMESPACE"))
    value.add_argument("--prometheus-clusters", default=os.environ.get("HEADLIGHT_V2_PROMETHEUS_CLUSTERS"),
                       help="comma-separated cluster names corresponding to Prometheus URLs")
    value.add_argument(
        "--cpu-rate-window",
        default=os.environ.get("HEADLIGHT_V2_CPU_RATE_WINDOW", "90s"),
        help="Prometheus range used for v2 CPU rate features (default: 90s)",
    )
    value.add_argument(
        "--policy-revision",
        default=os.environ.get(
            "HEADLIGHT_V2_POLICY_REVISION",
            "target-pending-readiness-guard-240s-v1",
        ),
        help="experimental policy revision appended to the frozen-config identity",
    )
    value.add_argument("--action-log", default=os.environ.get("HEADLIGHT_ACTION_LOG", "/artifacts/actions-v2.jsonl"))
    value.add_argument("--diagnostic-dir", default=os.environ.get("HEADLIGHT_V2_DIAGNOSTIC_DIR"),
                       help="enable decision diagnostics and exact replay tensors in this directory")
    value.add_argument("--diagnostic-max-bytes", type=int,
                       default=int(os.environ.get("HEADLIGHT_V2_DIAGNOSTIC_MAX_BYTES", 128 * 1024 * 1024)),
                       help="combined diagnostic record/tensor storage cap")
    value.add_argument(
        "--metric-diagnostics",
        action="store_true",
        default=os.environ.get("HEADLIGHT_V2_METRIC_DIAGNOSTICS") == "1",
        help="emit per-feature Prometheus query and missingness diagnostics",
    )
    value.add_argument("--once", action="store_true")
    value.add_argument("--disable-reactive-emergency", action="store_true",
                       default=os.environ.get("HEADLIGHT_V2_DISABLE_REACTIVE_EMERGENCY") == "1",
                       help="ablation: bypass only emergency overrides and their predictive suppression")
    value.add_argument(
        "--enable-actuation",
        action="store_true",
        help="explicitly leave shadow mode; never set this for the first VM smoke test",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.bundle or not args.service_profile or not args.policy_config:
        raise SystemExit("--bundle, --service-profile, and --policy-config are required")
    policy, policy_hash = load_policy(args.policy_config)
    if args.policy_revision:
        policy = replace(policy, policy_id=f"{policy.policy_id}:{args.policy_revision}")
    if args.disable_reactive_emergency:
        policy = replace(policy, reactive_emergency_enabled=False, policy_id=f"{policy.policy_id}:emergency-disabled")
    if args.expected_policy_sha256 and policy_hash != args.expected_policy_sha256:
        raise ValueError("frozen policy SHA-256 differs from HEADLIGHT_V2_POLICY_SHA256")
    profile = load_service_profile(args.service_profile)
    runtime = HeadlightV2Runtime.from_bundle(
        args.bundle,
        runtime_profile=profile,
        prometheus_url=args.prometheus_url,
        action_handler_url=args.action_handler_url,
        policy_config=policy,
        run_id=args.run_id,
        seed=args.seed,
        namespace=args.namespace,
        shadow_mode=not args.enable_actuation,
        metric_diagnostics=args.metric_diagnostics,
        prometheus_clusters=(tuple(item.strip() for item in args.prometheus_clusters.split(","))
                             if args.prometheus_clusters else None),
        cpu_rate_window=args.cpu_rate_window,
        action_log=args.action_log,
        diagnostic_dir=args.diagnostic_dir,
        diagnostic_max_bytes=args.diagnostic_max_bytes,
    )
    if args.expected_bundle_id and runtime.bundle.bundle_id != args.expected_bundle_id:
        raise ValueError("loaded bundle ID differs from HEADLIGHT_V2_BUNDLE_ID")
    print("HEADLIGHT_V2_RUNTIME_READY " + json.dumps({
        "bundle_id": runtime.bundle.bundle_id,
        "policy_id": policy.policy_id,
        "policy_sha256": policy_hash,
        "reactive_pending_wait_seconds": policy.reactive_pending_wait_seconds,
        "profile_hash": profile.profile_hash,
        "seed": args.seed,
        "shadow_mode": runtime.shadow_mode,
        "lookback_samples": runtime.lookback,
        "cpu_rate_window": args.cpu_rate_window,
        "zero_error_rate_for_observed_traffic": True,
        "runtime_reload": False,
        "effective_policy": asdict(policy),
        "diagnostic_dir": args.diagnostic_dir,
        "diagnostic_max_bytes": args.diagnostic_max_bytes,
        "code_revision": os.environ.get("HEADLIGHT_CODE_REVISION", "unknown"),
        "image_revision": os.environ.get("HEADLIGHT_IMAGE_REVISION", "unknown"),
    }, separators=(",", ":")), flush=True)
    if args.once:
        runtime.run_once()
    else:
        runtime.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
