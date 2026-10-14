from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.run_local_24gb_infrastructure_qualification import (
    PROTOCOL_PATH,
    choose_operating_record,
    load_manifest,
    scheduling_failures,
    summarize_mongostat,
    validate_protocol,
)


class InfrastructureQualificationTests(unittest.TestCase):
    def test_protocol_registers_separate_non_evaluation_seed(self) -> None:
        protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
        spec = validate_protocol(protocol, 201)
        self.assertEqual(spec["seed_role"], "infrastructure-qualification")
        self.assertFalse(spec["restrictions"]["training_input"])
        self.assertFalse(spec["restrictions"]["final_evaluation"])
        with self.assertRaises(RuntimeError):
            validate_protocol(protocol, 501)

    def test_load_job_uses_separate_namespace(self) -> None:
        args = argparse.Namespace(
            namespace="trainticket",
            load_namespace="headlight-infra-qualification",
            locust_image="locust:test",
            per_user_rps=0.4,
        )
        manifest = load_manifest(args, [{"duration_seconds": 60, "users": 2}])
        self.assertEqual(manifest["metadata"]["namespace"], "headlight-infra-qualification")
        env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        self.assertIn(
            {"name": "HEADLIGHT_TRAINTICKET_NAMESPACE", "value": "trainticket"},
            env,
        )

    def test_mongostat_contention_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mongostat.jsonl"
            payload = {
                "returncode": 0,
                "phase": "search-2-users",
                "cluster": "cluster2",
                "pod": "ts-travel-mongo-1",
                "output": json.dumps(
                    {"localhost:27017": {"qrw": "2|1", "arw": "3|1", "conn": "7", "dirty": "1.5%", "used": "2.5%"}}
                ),
            }
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            summary = summarize_mongostat(Path(directory))
            self.assertEqual(summary["maximum_queued_readers"], 2)
            self.assertEqual(summary["maximum_queued_writers"], 1)
            self.assertEqual(summary["maximum_connections"], 7)

    def test_operating_point_uses_highest_reliably_delivered_stage(self) -> None:
        records = [
            {
                "requested_average_rps": 0.4,
                "achieved_successful_rps": 0.34,
                "success_fraction": 1.0,
                "delivery_fraction": 0.86,
            },
            {
                "requested_average_rps": 0.8,
                "achieved_successful_rps": 0.76,
                "success_fraction": 1.0,
                "delivery_fraction": 0.95,
            },
            {
                "requested_average_rps": 1.6,
                "achieved_successful_rps": 1.30,
                "success_fraction": 1.0,
                "delivery_fraction": 0.81,
            },
        ]
        selected, reliable = choose_operating_record(records)
        self.assertTrue(reliable)
        self.assertEqual(selected["requested_average_rps"], 0.8)

    def test_new_scheduling_failures_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            for cluster in ("cluster2", "cluster3"):
                (raw_dir / f"{cluster}-events-before.json").write_text(
                    json.dumps({"items": [{"metadata": {"uid": "old"}, "reason": "FailedCreate"}]}),
                    encoding="utf-8",
                )
                items = [{"metadata": {"uid": "old"}, "reason": "FailedCreate"}]
                if cluster == "cluster2":
                    items.append(
                        {
                            "metadata": {"uid": "new"},
                            "lastTimestamp": "2026-09-19T17:15:46Z",
                            "reason": "FailedCreate",
                            "involvedObject": {"name": "ts-route-rs"},
                            "message": "exceeded quota",
                        }
                    )
                (raw_dir / f"{cluster}-events-after.json").write_text(
                    json.dumps({"items": items}), encoding="utf-8"
                )
            failures = scheduling_failures(raw_dir)
            self.assertEqual(len(failures["cluster2"]), 1)
            self.assertEqual(failures["cluster2"][0]["object"], "ts-route-rs")
            self.assertEqual(failures["cluster3"], [])


if __name__ == "__main__":
    unittest.main()
