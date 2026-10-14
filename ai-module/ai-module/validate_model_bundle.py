#!/usr/bin/env python3
"""Validate every frozen bundle component before deploying an experiment."""

from __future__ import annotations

import argparse
import json

from headlight.bundle import ModelBundle
from headlight.policy import PolicyConfig
from headlight.profile import load_service_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--service-profile", required=True)
    parser.add_argument("--controller", choices=["headlight"], required=True)
    parser.add_argument("--policy-config")
    args = parser.parse_args()
    profile = load_service_profile(args.service_profile)
    bundle = ModelBundle.load(
        args.bundle,
        runtime_profile=profile,
        expected_model_type=args.controller,
    )
    policy = PolicyConfig.load(args.policy_config)
    print(
        json.dumps(
            {
                "valid": True,
                "bundle_id": bundle.bundle_id,
                "controller": args.controller,
                "profile_hash": profile.profile_hash,
                "node_order": list(profile.node_order),
                "policy_id": policy.payload["policy_id"],
                "policy_configuration_hash": policy.configuration_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
