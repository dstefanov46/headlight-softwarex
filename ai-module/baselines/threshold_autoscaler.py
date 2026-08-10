#!/usr/bin/env python3
"""Threshold autoscaling baseline.

This entrypoint is intentionally small: production actuation is performed by
the Actions Handler, while the policy decision is shared with experiment tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.policies import ThresholdState, threshold_decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute one threshold scaling decision")
    parser.add_argument("--p95-latency-ms", type=float, required=True)
    parser.add_argument("--cpu-utilization", type=float, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--stable-windows", type=int, default=0)
    parser.add_argument("--cooldown-remaining-seconds", type=int, default=0)
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=8)
    args = parser.parse_args()

    next_state = threshold_decision(
        p95_latency_ms=args.p95_latency_ms,
        cpu_utilization=args.cpu_utilization,
        state=ThresholdState(args.replicas, args.stable_windows, args.cooldown_remaining_seconds),
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
    )
    print(json.dumps(next_state.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

