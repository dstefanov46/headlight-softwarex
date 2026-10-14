#!/usr/bin/env python3
"""Replay an immutable request trace through Policy-v2 in shadow mode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.policy_v2 import (  # noqa: E402
    HEADLIGHT_V2_FULL_CONTROLLER,
    CandidatePredictionV2,
    HeadlightPolicyV2,
    PolicyV2Executor,
    PolicyV2RecordContext,
    ServiceRuntimeStateV2,
    build_policy_v2_decision_record,
)
from headlight.profile import load_service_profile  # noqa: E402


DEFAULT_FIXTURE = (
    REPO_ROOT
    / "experiments"
    / "sample_data"
    / "research_v4"
    / "local-controller-smoke-seed-11-request-events.csv"
)
EXPECTED_FIXTURE_SHA256 = (
    "df7d8e517242fe4a95e4deadfe01cbb1440ad3b922f6841f74e432b274d443d2"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--intervals", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture = args.fixture.resolve()
    if args.intervals <= 0:
        raise ValueError("--intervals must be positive")
    initial_hash = sha256(fixture)
    if fixture == DEFAULT_FIXTURE.resolve() and initial_hash != EXPECTED_FIXTURE_SHA256:
        raise ValueError("the repository replay fixture hash is incompatible")
    with fixture.open(newline="", encoding="utf-8") as handle:
        rows = sorted(
            csv.DictReader(handle),
            key=lambda row: float(row["timestamp"]),
        )
    if not rows:
        raise ValueError("the replay fixture contains no request rows")

    profile = load_service_profile(
        REPO_ROOT / "ai-module" / "configs" / "service_profiles" / "advanced-ticket.json"
    )
    policy = HeadlightPolicyV2(profile)
    chunk_size = max(1, math.ceil(len(rows) / args.intervals))
    record_count = 0
    actionable_count = 0
    replayed_intervals = 0
    actuator_calls: list[object] = []

    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        timestamp = float(chunk[-1]["timestamp"])
        durations = [float(row["response_time_ms"]) for row in chunk]
        p95_index = max(0, math.ceil(0.95 * len(durations)) - 1)
        p95_latency_ms = sorted(durations)[p95_index]
        error_rate = sum(
            row["status"] not in {"200", "201", "202", "204"}
            for row in chunk
        ) / len(chunk)
        observed_demand = float(len(chunk))

        def state() -> ServiceRuntimeStateV2:
            return ServiceRuntimeStateV2(
                desired_replicas={"cluster2": 1, "cluster3": 1},
                ready_replicas={"cluster2": 1, "cluster3": 1},
                pending_replicas={"cluster2": 0, "cluster3": 0},
                pending_age_seconds={"cluster2": 0.0, "cluster3": 0.0},
                demand_rps=observed_demand,
                cpu_utilization=None,
                queue_depth=None,
                p95_latency_ms=p95_latency_ms,
                error_rate=error_rate,
                predictive_fresh=True,
                missing_fields=frozenset({"cpu_utilization", "queue_depth"}),
                time_since_last_action_seconds=600.0,
            )

        states = {node.name: state() for node in profile.nodes}

        def scorer(_service: str, candidates: tuple[object, ...]) -> dict[str, Any]:
            return {
                candidate.candidate_id: CandidatePredictionV2(
                    predicted_p95_latency_ms=p95_latency_ms,
                    sla_violation_probability=(
                        0.9 if candidate.total_replicas <= 2 else 0.001
                    ),
                )
                for candidate in candidates
            }

        decisions: list[object] = []
        execution = PolicyV2Executor(policy, shadow_mode=True).execute(
            states,
            scorer,
            recorder=decisions.append,
            actuator=actuator_calls.append,
        )
        if execution.actuator_called or actuator_calls:
            raise RuntimeError("shadow replay invoked an actuator")
        if len(decisions) != len(profile.nodes):
            raise RuntimeError("shadow replay produced an incomplete interval")
        context = PolicyV2RecordContext(
            run_id="stage5-shadow-replay-seed11",
            seed=401,
            feature_schema_hash="a" * 64,
            adjacency_hash="b" * 64,
            controller=HEADLIGHT_V2_FULL_CONTROLLER,
            bundle_id="stage5-synthetic-no-actuation",
        )
        for decision in decisions:
            record = build_policy_v2_decision_record(
                decision,
                states[decision.service],
                context,
                timestamp=timestamp,
                forecast_quantiles_by_horizon={
                    "30": {"q50": observed_demand, "q90": observed_demand * 1.1},
                    "60": {"q50": observed_demand, "q90": observed_demand * 1.2},
                },
                action_timestamps={
                    "status": "not_requested",
                    "requested_at": None,
                    "acknowledged_at": None,
                    "ready_at": None,
                    "readiness_delay_seconds": None,
                },
                shadow_mode=True,
            )
            if record["metadata"]["shadow_mode"] is not True:
                raise RuntimeError("a replay record is not marked as shadow mode")
            record_count += 1
        actionable_count += len(execution.actionable)
        replayed_intervals += 1

    final_hash = sha256(fixture)
    if final_hash != initial_hash:
        raise RuntimeError("the replay fixture changed during replay")
    print(
        json.dumps(
            {
                "fixture": str(fixture.relative_to(REPO_ROOT)),
                "fixture_sha256": initial_hash,
                "input_rows": len(rows),
                "intervals": replayed_intervals,
                "services_per_interval": len(profile.nodes),
                "decision_records": record_count,
                "actionable_decisions": actionable_count,
                "actuator_called": False,
                "all_shadow_mode": True,
                "source_unchanged": final_hash == initial_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
