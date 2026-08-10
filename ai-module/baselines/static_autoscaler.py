#!/usr/bin/env python3
"""Static provisioning baseline using calibration p95 demand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.policies import static_split_replicas


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute static cluster replica split")
    parser.add_argument("--total-replicas", type=int, required=True)
    parser.add_argument("--cluster2-weight", type=int, default=3)
    parser.add_argument("--cluster3-weight", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(static_split_replicas(args.total_replicas, args.cluster2_weight, args.cluster3_weight)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

