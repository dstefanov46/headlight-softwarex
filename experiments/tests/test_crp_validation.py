from __future__ import annotations

import unittest
from pathlib import Path

from experiments.validate_crp_trace import CRPValidationError, load_config, load_rows, validate_rows


def row(event_type: str, available: int = 3, status: str = "true") -> dict[str, str]:
    return {
        "timestamp": "2026-07-30T00:00:00Z",
        "service": "ts-auth",
        "desired_replicas": "3",
        "available_replicas": str(available),
        "source_cluster": "cluster2",
        "destination_cluster": "cluster3",
        "placement_decision": "cluster2=1,cluster3=2",
        "event_type": event_type,
        "availability_invariant_status": status,
        "max_unavailable": "0",
    }


class CRPValidationTests(unittest.TestCase):
    def test_validates_required_events_and_max_unavailable_zero(self) -> None:
        config = load_config("experiments/config/crp_validation.yaml")
        result = validate_rows(
            [
                row("replica_propagation"),
                row("scaling"),
                row("offloading"),
            ],
            config,
        )
        self.assertEqual(result["validation_status"], "validated")
        self.assertEqual(result["max_unavailable"], 0)

    def test_fails_on_availability_violation(self) -> None:
        config = load_config("experiments/config/crp_validation.yaml")
        invalid_rows = [
            row("replica_propagation", available=2, status="false"),
            row("scaling"),
            row("offloading"),
        ]
        with self.assertRaises(CRPValidationError) as ctx:
            validate_rows(invalid_rows, config)
        self.assertIn("availability violation", str(ctx.exception))

    def test_validates_softwarex_five_five_to_two_eight_fixture(self) -> None:
        config = load_config("experiments/config/crp_validation.yaml")
        rows = load_rows(Path("experiments/sample_data/softwarex/crp_5_5_to_2_8_trace.csv"))
        result = validate_rows(rows, config)
        self.assertEqual(result["validation_status"], "validated")
        self.assertEqual(result["events"], 3)


if __name__ == "__main__":
    unittest.main()
