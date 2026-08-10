#!/usr/bin/env python3
"""Run LSTM with rolling live metrics and Headlight's shared safety policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


AI_MODULE = Path(__file__).resolve().parents[1] / "ai-module"
sys.path.insert(0, str(AI_MODULE))

from headlight.profile import load_service_profile
from headlight.runtime import ControllerRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=os.environ.get("HEADLIGHT_LSTM_BUNDLE"))
    parser.add_argument("--service-profile", default=os.environ.get("HEADLIGHT_SERVICE_PROFILE"))
    parser.add_argument(
        "--prometheus-url",
        default=os.environ.get(
            "HEADLIGHT_PROMETHEUS_URL",
            "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
        ),
    )
    parser.add_argument(
        "--action-handler-url",
        default=os.environ.get(
            "HEADLIGHT_ACTION_HANDLER_URL",
            "http://actions-handler.actions-handler-system.svc.cluster.local:8080/placements",
        ),
    )
    parser.add_argument("--namespace", default=os.environ.get("HEADLIGHT_NAMESPACE"))
    parser.add_argument(
        "--action-log",
        default=os.environ.get("HEADLIGHT_ACTION_LOG", "/app/artifacts/actions.jsonl"),
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        default=os.environ.get("HEADLIGHT_SHADOW_MODE", "0") == "1",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.bundle or not args.service_profile:
        parser.error("--bundle and --service-profile (or HEADLIGHT_* equivalents) are required")
    profile = load_service_profile(args.service_profile)
    runtime = ControllerRuntime.from_bundle(
        args.bundle,
        runtime_profile=profile,
        prometheus_url=args.prometheus_url,
        action_handler_url=args.action_handler_url,
        controller_name="lstm",
        namespace=args.namespace,
        shadow=args.shadow,
        action_log=args.action_log,
    )
    print(
        "HEADLIGHT_RUNTIME_READY "
        + json.dumps(
            {
                "controller": "lstm",
                "uses_graph": False,
                "bundle_id": runtime.bundle.bundle_id,
                "profile": profile.profile_name,
                "nodes": len(profile.nodes),
                "lookback_samples": runtime.lookback,
                "forecast_horizon_samples": runtime.bundle.model_config["horizon"],
                "runtime_reload": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    if args.once:
        runtime.run_once()
    else:
        runtime.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
