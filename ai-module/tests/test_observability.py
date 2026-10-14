from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.observability import (  # noqa: E402
    DECISION_RECORD_SCHEMA_VERSION,
    PROFILING_RECORD_SCHEMA_VERSION,
    SchemaValidationError,
    build_profiling_dataset,
    content_sha256,
    load_action_records,
    validate_decision_record,
    validate_profiling_dataset,
    write_decision_record,
)


TEST_FEATURE_SCHEMA = {"schema_version": "headlight.features/v2", "fields": ["demand_rps"]}
TEST_ADJACENCY = {"schema_version": "headlight.adjacency/v2", "edges": [["a", "b"]]}


def decision_record(*, record_id: str = "decision-301-1", seed: int = 301) -> dict[str, object]:
    return {
        "schema_version": DECISION_RECORD_SCHEMA_VERSION,
        "record_id": record_id,
        "run_id": f"profiling-{seed}",
        "seed": seed,
        "timestamp": 1_700_000_000.0,
        "service": "ts-order-service",
        "observed_state": {
            "demand_rps": 17.5,
            "cpu_utilization": 0.62,
            "p95_latency_ms": 810.0,
            "error_rate": 0.01,
        },
        "masks": {
            "missing": {"demand_rps": False, "cpu_utilization": False, "p95_latency_ms": False, "error_rate": False},
            "stale": {"demand_rps": False, "cpu_utilization": False, "p95_latency_ms": False, "error_rate": False},
        },
        "replicas_by_cluster": {
            "ready": {"cluster2": 1, "cluster3": 1},
            "desired": {"cluster2": 1, "cluster3": 1},
            "pending": {"cluster2": 0, "cluster3": 0},
        },
        "time_since_last_action_seconds": 75.0,
        "forecast_quantiles_by_horizon": {"60": {"q50": 18.0, "q90": 23.0}, "90": {"q50": 19.0, "q90": 25.0}},
        "candidates": [
            {
                "candidate_id": "hold",
                "total_replicas": 2,
                "placement": {"cluster2": 1, "cluster3": 1},
                "prediction_available": True,
                "predicted_p95_latency_ms": 920.0,
                "predicted_sla_probability": 0.03,
                "objective_terms": {"resource": 2.0, "sla_risk": 0.03, "churn": 0.0},
            },
            {
                "candidate_id": "up-one",
                "total_replicas": 3,
                "placement": {"cluster2": 2, "cluster3": 1},
                "prediction_available": True,
                "predicted_p95_latency_ms": 690.0,
                "predicted_sla_probability": 0.005,
                "objective_terms": {"resource": 3.0, "sla_risk": 0.005, "churn": 1.0},
            },
        ],
        "selected_candidate_id": "up-one",
        "selection_reason": "lowest feasible objective",
        "selection_rule": "minimum-objective-then-candidate-id",
        "action_timestamps": {
            "status": "ready",
            "requested_at": 1_700_000_001.0,
            "acknowledged_at": 1_700_000_002.0,
            "ready_at": 1_700_000_031.0,
            "readiness_delay_seconds": 30.0,
        },
        "policy_schema_version": "headlight.policy/v2",
        "policy_version": "2.0.0",
        "policy_id": "headlight-v2-policy",
        "feature_schema_hash": content_sha256(TEST_FEATURE_SCHEMA),
        "adjacency_hash": content_sha256(TEST_ADJACENCY),
        "controller": "headlight-v2-full",
        "bundle_id": "bundle-v2",
        "action_origin": "profiling_isolated",
        "out_of_distribution": False,
        "metadata": {"source": "unit-test"},
    }


def profiling_record(*, record_id: str = "profile-301-1", seed: int = 301) -> dict[str, object]:
    decision = decision_record(record_id=f"{record_id}-decision", seed=seed)
    return {
        "schema_version": PROFILING_RECORD_SCHEMA_VERSION,
        "record_id": record_id,
        "run_id": f"profiling-{seed}",
        "seed": seed,
        "decision": decision,
        "outcomes": {
            "60": {
                "preserve": {
                    "offered_requests": 100,
                    "completed_requests": 99,
                    "end_to_end_p95_latency_ms": 700.0,
                    "sla_violations": 1,
                }
            }
        },
        "clean_action_label": True,
        "exclusion_reasons": [],
    }


class ObservabilityContractTests(unittest.TestCase):
    def _dataset(self, records: list[dict[str, object]]) -> dict[str, object]:
        return build_profiling_dataset(
            records,
            feature_schema=TEST_FEATURE_SCHEMA,
            adjacency=TEST_ADJACENCY,
            fitting_seeds=[301],
            calibration_seeds=[302],
            holdout_seeds=[303],
            evaluation_seeds=[501, 502, 503, 504, 505, 506, 507, 508],
        )

    def test_valid_decision_round_trips_as_canonical_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.jsonl"
            written = write_decision_record(path, decision_record())
            rows = load_action_records(path)
        self.assertEqual(rows, [written])
        self.assertEqual(rows[0]["replicas_by_cluster"]["ready"], {"cluster2": 1, "cluster3": 1})
        self.assertEqual(rows[0]["action_timestamps"]["readiness_delay_seconds"], 30.0)

    def test_existing_unversioned_v3_action_row_remains_loadable(self) -> None:
        legacy = {
            "timestamp": 1_700_000_000,
            "controller": "headlight",
            "service": "ts-order-service",
            "prediction": {"request_q90": 12.0},
            "policy_reason": "forecast-capacity",
            "previous_replicas": {"cluster2": 1, "cluster3": 1},
            "desired_replicas": {"cluster2": 2, "cluster3": 1},
            "batch_successful": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v3.jsonl"
            path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
            self.assertEqual(load_action_records(path), [legacy])

    def test_invalid_candidate_total_and_readiness_delay_fail_clearly(self) -> None:
        invalid_total = decision_record()
        invalid_total["candidates"][0]["total_replicas"] = 99  # type: ignore[index]
        with self.assertRaisesRegex(SchemaValidationError, "total_replicas"):
            validate_decision_record(invalid_total)

        invalid_delay = decision_record()
        invalid_delay["action_timestamps"]["readiness_delay_seconds"] = 31.0  # type: ignore[index]
        with self.assertRaisesRegex(SchemaValidationError, "readiness_delay_seconds"):
            validate_decision_record(invalid_delay)

    def test_decision_requires_at_least_one_forecast_horizon(self) -> None:
        record = decision_record()
        record["forecast_quantiles_by_horizon"] = {}
        with self.assertRaisesRegex(SchemaValidationError, "forecast horizon"):
            validate_decision_record(record)

    def test_action_lifecycle_requires_consistent_timestamps(self) -> None:
        acknowledged_without_timestamp = decision_record()
        acknowledged_without_timestamp["action_timestamps"]["status"] = "acknowledged"  # type: ignore[index]
        acknowledged_without_timestamp["action_timestamps"]["acknowledged_at"] = None  # type: ignore[index]
        acknowledged_without_timestamp["action_timestamps"]["ready_at"] = None  # type: ignore[index]
        acknowledged_without_timestamp["action_timestamps"]["readiness_delay_seconds"] = None  # type: ignore[index]
        with self.assertRaisesRegex(SchemaValidationError, "acknowledged_at"):
            validate_decision_record(acknowledged_without_timestamp)

        requested_with_acknowledgement = decision_record()
        requested_with_acknowledgement["action_timestamps"]["status"] = "requested"  # type: ignore[index]
        requested_with_acknowledgement["action_timestamps"]["ready_at"] = None  # type: ignore[index]
        requested_with_acknowledgement["action_timestamps"]["readiness_delay_seconds"] = None  # type: ignore[index]
        with self.assertRaisesRegex(SchemaValidationError, "requested actions"):
            validate_decision_record(requested_with_acknowledgement)

        failed_with_ready = decision_record()
        failed_with_ready["action_timestamps"]["status"] = "failed"  # type: ignore[index]
        with self.assertRaisesRegex(SchemaValidationError, "failed actions"):
            validate_decision_record(failed_with_ready)

    def test_reactive_emergency_requires_explicit_ood_marker(self) -> None:
        record = decision_record()
        record["action_origin"] = "reactive_emergency"
        with self.assertRaisesRegex(SchemaValidationError, "out_of_distribution"):
            validate_decision_record(record)

    def test_dataset_rejects_duplicate_samples_and_evaluation_leakage(self) -> None:
        record = profiling_record()
        with self.assertRaisesRegex(SchemaValidationError, "duplicate"):
            self._dataset([record, copy.deepcopy(record)])

        duplicate_decision = profiling_record(record_id="profile-301-2")
        duplicate_decision["decision"]["record_id"] = record["decision"]["record_id"]  # type: ignore[index]
        with self.assertRaisesRegex(SchemaValidationError, "duplicate profiling decision"):
            self._dataset([record, duplicate_decision])

        leaked = profiling_record(seed=501)
        with self.assertRaisesRegex(SchemaValidationError, "evaluation seed leakage"):
            self._dataset([leaked])

    def test_dataset_rejects_missing_outcomes_without_inference(self) -> None:
        record = profiling_record()
        record.pop("outcomes")
        with self.assertRaisesRegex(SchemaValidationError, "outcomes"):
            self._dataset([record])

    def test_profiling_exclusions_require_explicit_consistent_reasons(self) -> None:
        excluded_without_reason = profiling_record()
        excluded_without_reason["clean_action_label"] = False
        with self.assertRaisesRegex(SchemaValidationError, "at least one reason"):
            self._dataset([excluded_without_reason])

        clean_with_reason = profiling_record()
        clean_with_reason["exclusion_reasons"] = ["unexpected"]
        with self.assertRaisesRegex(SchemaValidationError, "must be empty"):
            self._dataset([clean_with_reason])

    def test_dataset_rejects_sample_schema_hash_mismatch(self) -> None:
        record = profiling_record()
        record["decision"]["feature_schema_hash"] = "c" * 64  # type: ignore[index]
        with self.assertRaisesRegex(SchemaValidationError, "dataset feature schema"):
            self._dataset([record])

    def test_frozen_v3_denylist_cannot_be_overridden(self) -> None:
        record = profiling_record(seed=101)
        with self.assertRaisesRegex(SchemaValidationError, "forbidden seeds"):
            build_profiling_dataset(
                [record],
                feature_schema=TEST_FEATURE_SCHEMA,
                adjacency=TEST_ADJACENCY,
                fitting_seeds=[101],
                calibration_seeds=[302],
                holdout_seeds=[303],
                evaluation_seeds=[501],
                forbidden_seeds=[999],
            )

    def test_dataset_stores_and_validates_all_content_hashes(self) -> None:
        dataset = self._dataset([profiling_record()])
        self.assertEqual(len(dataset["feature_schema_hash"]), 64)
        self.assertEqual(len(dataset["adjacency_hash"]), 64)
        self.assertEqual(len(dataset["dataset_hash"]), 64)
        self.assertEqual(validate_profiling_dataset(dataset), dataset)
        tampered = copy.deepcopy(dataset)
        tampered["samples"][0]["outcomes"]["60"]["preserve"]["sla_violations"] = 2
        with self.assertRaisesRegex(SchemaValidationError, "dataset_hash"):
            validate_profiling_dataset(tampered)

    def test_manifest_validator_rejects_empty_and_malformed_exclusions(self) -> None:
        dataset = self._dataset([profiling_record()])
        empty = copy.deepcopy(dataset)
        empty["samples"] = []
        empty["source_record_hashes"] = []
        empty["sample_count"] = 0
        unhashed = dict(empty)
        unhashed.pop("dataset_hash")
        empty["dataset_hash"] = content_sha256(unhashed)
        with self.assertRaisesRegex(SchemaValidationError, "at least one validated sample"):
            validate_profiling_dataset(empty)

        excluded = profiling_record(record_id="profile-302-1", seed=302)
        excluded["clean_action_label"] = False
        excluded["exclusion_reasons"] = ["reactive emergency overlap"]
        dataset_with_exclusion = self._dataset([profiling_record(), excluded])
        malformed = copy.deepcopy(dataset_with_exclusion)
        malformed["excluded_records"][0]["source_record_hash"] = "not-a-hash"
        unhashed = dict(malformed)
        unhashed.pop("dataset_hash")
        malformed["dataset_hash"] = content_sha256(unhashed)
        with self.assertRaisesRegex(SchemaValidationError, "source_record_hash"):
            validate_profiling_dataset(malformed)

    def test_manifest_validator_enforces_schema_and_seed_provenance(self) -> None:
        dataset = self._dataset([profiling_record()])
        incompatible = copy.deepcopy(dataset)
        incompatible["samples"][0]["decision"]["feature_schema_hash"] = "c" * 64
        incompatible["source_record_hashes"] = [content_sha256(incompatible["samples"][0])]
        unhashed = dict(incompatible)
        unhashed.pop("dataset_hash")
        incompatible["dataset_hash"] = content_sha256(unhashed)
        with self.assertRaisesRegex(SchemaValidationError, "manifest feature schema"):
            validate_profiling_dataset(incompatible)

        forbidden_split = copy.deepcopy(dataset)
        forbidden_split["seed_partitions"]["fitting"].append(101)
        unhashed = dict(forbidden_split)
        unhashed.pop("dataset_hash")
        forbidden_split["dataset_hash"] = content_sha256(unhashed)
        with self.assertRaisesRegex(SchemaValidationError, "training partitions include forbidden"):
            validate_profiling_dataset(forbidden_split)

    def test_dataset_cli_writes_only_explicit_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            feature_schema = root / "feature.json"
            adjacency = root / "adjacency.json"
            output = root / "dataset.json"
            records.write_text(json.dumps(profiling_record()) + "\n", encoding="utf-8")
            feature_schema.write_text(json.dumps(TEST_FEATURE_SCHEMA), encoding="utf-8")
            adjacency.write_text(json.dumps(TEST_ADJACENCY), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/scripts/build_headlight_v2_profiling_dataset.py",
                    "--records", str(records),
                    "--feature-schema", str(feature_schema),
                    "--adjacency", str(adjacency),
                    "--fitting-seeds", "301",
                    "--calibration-seeds", "302",
                    "--holdout-seeds", "303",
                    "--evaluation-seeds", "501,502,503,504,505,506,507,508",
                    "--output", str(output),
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
