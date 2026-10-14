from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.validate_artifacts import (
    ArtifactValidationError,
    ZERO_GUARD_FIELDS,
    assert_artifacts_not_stale,
    _summary_has_invalid_zeroes,
    validated_summary_rows,
)
from experiments.lib.attempt_registry import append_record, build_record
from experiments.lib.protocol import protocol_reference


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2) + "\n")


def measured_bundle(root: Path, name: str = "run-a", seed: int = 1, *, zero_summary: bool = False) -> Path:
    bundle = root / name
    manifest = {
        "apiVersion": "headlight.experiments/v1alpha1",
        "kind": "ResolvedExperimentManifest",
        "metadata": {"name": "hpa__gradual-ramp__multi-cluster-headlight"},
        "spec": {
            "controller": {"controller": "hpa"},
            "workload": {"pattern": "gradual-ramp", "sampling_interval_seconds": 5},
            "topology": {"namespace": "trainticket", "active_clusters": ["cluster2"]},
            "run": {
                "seeds": [seed],
                "duration_seconds": 10,
                "sampling_interval_seconds": 5,
                "protocol": {"reference": protocol_reference()},
                "sla": {"p95_latency_ms": 1000, "minimum_success_rate": 0.90},
            },
        },
    }
    write_text(bundle / "manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
    write_json(bundle / "environment.json", {"dry_run": False, "git_commit": "abc123"})
    seed_dir = bundle / f"seed-{seed}"
    write_text(seed_dir / "raw/load_profile.csv", "elapsed_seconds,users,spawn_rate\n0,1,1\n")
    write_json(seed_dir / "raw/protocol.json", {"protocol": protocol_reference()})
    write_json(seed_dir / "raw/topology.json", {"common_topology": {"namespace": "trainticket"}})
    write_json(seed_dir / "raw/workload_manifest_hashes.json", {"data_plane_no_policies_sha256": "abc"})
    write_json(seed_dir / "raw/phase_timestamps.json", {"measurement": {"started_at_unix": 0, "finished_at_unix": 10}})
    write_json(seed_dir / "raw/prometheus_queries.json", {"pod_count": "query"})
    write_json(seed_dir / "raw/model_bundle_provenance.json", {})
    write_json(seed_dir / "raw/policy_config.json", {})
    write_json(seed_dir / "raw/static_plan.json", {})
    write_json(
        seed_dir / "raw/hpa_controller_metadata.json",
        {"scale_up_behavior": "kubernetes-default", "sync_period_seconds": {"expected": 15}},
    )
    write_text(seed_dir / "raw/hpa/hpa-cluster2.yaml", "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\n")
    write_json(
        seed_dir / "raw/resource_calculation.json",
        {
            "cpu_seconds": {
                "method": "counter_delta_with_reset_handling",
                "query": "container_cpu_usage_seconds_total",
                "measurement_window": {"started_at_unix": 0, "finished_at_unix": 10},
            },
            "pod_seconds": {
                "method": "stepwise_gauge_integral_exact_window",
                "query": "kube_pod_status_ready",
                "measurement_window": {"started_at_unix": 0, "finished_at_unix": 10},
            },
        },
    )
    write_text(
        seed_dir / "raw/request_events.csv",
        "timestamp,method,endpoint,status,response_time_ms,exception\n"
        "0,GET,/,200,100,\n"
        "10,GET,/,200,200,\n",
    )
    write_text(
        seed_dir / "raw/request_events_measurement.csv",
        "timestamp,method,endpoint,status,response_time_ms,exception\n"
        "0,GET,/,200,100,\n"
        "10,GET,/,200,200,\n",
    )
    write_json(seed_dir / "raw/kubernetes_events.json", {})
    write_json(seed_dir / "raw/crp_transitions.json", [])
    write_json(seed_dir / "raw/actions_handler_batches.json", [])
    write_text(seed_dir / "raw/ai_inference_times.csv", "timestamp,controller,model_variant,service_count,duration_ms,applicable\n,hpa,native,,,false\n")
    prometheus = {
        name: {"response": {"data": {"result": [{"values": [[0, "1"], [5, "1"], [10, "1"]]}]}}}
        for name in ("cpu_seconds", "memory_working_set_bytes", "pod_count")
    }
    write_json(seed_dir / "raw/prometheus_range_data.json", prometheus)
    write_json(seed_dir / "raw/ceiling_binding.json", {"ceiling_hit_invalidates_run": False})
    write_json(seed_dir / "raw/measurement_window.json", {"started_at_unix": 0, "finished_at_unix": 10})
    write_json(seed_dir / "raw/pods_before.json", {})
    write_json(seed_dir / "raw/pods_after.json", {})
    write_json(seed_dir / "raw/deployments_after.json", {})
    write_json(seed_dir / "raw/crps_before.json", {"items": []})
    write_json(seed_dir / "raw/crps_after.json", {"items": []})
    write_text(seed_dir / "logs/runner.log", "ok\n")
    header = [
        "seed",
        "controller",
        "workload",
        "sla_latency_ms",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "success_rate",
        "successful_throughput_rps",
        "offered_throughput_rps",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "sla_violation_rate",
        "timeout_count",
        "exception_count",
        "http_status_distribution",
        "excess_response_time_seconds",
        "cpu_seconds",
        "mean_pod_count",
        "pod_seconds",
        "action_churn",
        "technical_validity",
        "performance_outcome",
        "invalidity_reason",
        "retry_allowed",
        "retry_rule",
    ]
    if zero_summary:
        row = [seed, "hpa", "gradual-ramp", 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "{}", 0, 0, 0, 0, 0, "valid", "measured", "", "False", ""]
    else:
        row = [seed, "hpa", "gradual-ramp", 1000, 2, 2, 0, 1.0, 0.2, 0.2, 150, 190, 198, 0.0, 0, 0, '{"200": 2}', 0.0, 3.0, 1.0, 15.0, 0.0, "valid", "measured", "", "False", ""]
    write_text(
        seed_dir / "summaries/summary.csv",
        ",".join(header) + "\n" + ",".join(str(value) for value in row) + "\n",
    )
    return bundle


class ArtifactValidationTests(unittest.TestCase):
    def test_accepts_complete_measured_singleton_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root)
            rows, validation = validated_summary_rows(
                root,
                controllers={"hpa"},
                workloads={"gradual-ramp"},
                seeds={1},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["validation_status"], "validated")
            self.assertEqual(rows[0]["commit_hash"], "abc123")
            self.assertEqual(validation["validation_status"], "validated")

    def test_rejects_invalid_zero_valued_measured_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root, zero_summary=True)
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertIn("invalid_zero_run", {issue["code"] for issue in ctx.exception.issues})

    def test_zero_summary_guard_uses_tolerance(self) -> None:
        near_zero = {field: "1e-13" for field in ZERO_GUARD_FIELDS}
        meaningful = {field: "1e-13" for field in ZERO_GUARD_FIELDS}
        meaningful["p95_latency_ms"] = "1e-6"
        self.assertTrue(_summary_has_invalid_zeroes(near_zero))
        self.assertFalse(_summary_has_invalid_zeroes(meaningful))

    def test_rejects_duplicate_measured_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root, "run-a")
            measured_bundle(root, "run-b")
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertIn("duplicate_run", {issue["code"] for issue in ctx.exception.issues})

    def test_registry_consumes_only_technically_valid_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = measured_bundle(root, "valid-attempt")
            registry = root / "attempt-registry.jsonl"
            append_record(
                registry,
                build_record(
                    attempt_id="invalid-first",
                    controller="hpa",
                    workload="gradual-ramp",
                    seed=1,
                    started_at_unix=1,
                    finished_at_unix=2,
                    technical_status="invalid",
                    performance_status="not_measured",
                    failure_category="prometheus_empty",
                    evidence_paths=[str(root / "invalid-attempt" / "seed-1")],
                    retry_allowed=True,
                    retry_rule="technical_invalidity_retry",
                    retry_decision_value="retry_same_cell",
                ),
            )
            append_record(
                registry,
                build_record(
                    attempt_id="valid-second",
                    controller="hpa",
                    workload="gradual-ramp",
                    seed=1,
                    started_at_unix=3,
                    finished_at_unix=4,
                    technical_status="valid",
                    performance_status="measured",
                    evidence_paths=[str(bundle / "seed-1")],
                ),
            )
            rows, validation = validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attempt_id"], "valid-second")
            self.assertEqual(validation["attempt_accounting"]["total_attempts"], 2)
            self.assertEqual(validation["attempt_accounting"]["technically_invalid_attempts"], 1)
            self.assertEqual(validation["attempt_accounting"]["invalid_attempts_by_category"]["prometheus_empty"], 1)

    def test_registry_rejects_duplicate_valid_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = measured_bundle(root, "attempt-a")
            second = measured_bundle(root, "attempt-b")
            registry = root / "attempt-registry.jsonl"
            for attempt_id, bundle in [("valid-a", first), ("valid-b", second)]:
                append_record(
                    registry,
                    build_record(
                        attempt_id=attempt_id,
                        controller="hpa",
                        workload="gradual-ramp",
                        seed=1,
                        started_at_unix=1,
                        finished_at_unix=2,
                        technical_status="valid",
                        performance_status="measured",
                        evidence_paths=[str(bundle / "seed-1")],
                    ),
                )
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertIn("duplicate_run", {issue["code"] for issue in ctx.exception.issues})

    def test_registry_allows_superseded_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = measured_bundle(root, "attempt-a")
            second = measured_bundle(root, "attempt-b")
            registry = root / "attempt-registry.jsonl"
            append_record(
                registry,
                build_record(
                    attempt_id="valid-a",
                    controller="hpa",
                    workload="gradual-ramp",
                    seed=1,
                    started_at_unix=1,
                    finished_at_unix=2,
                    technical_status="valid",
                    performance_status="measured",
                    evidence_paths=[str(first / "seed-1")],
                    superseding_attempt_id="valid-b",
                ),
            )
            append_record(
                registry,
                build_record(
                    attempt_id="valid-b",
                    controller="hpa",
                    workload="gradual-ramp",
                    seed=1,
                    started_at_unix=3,
                    finished_at_unix=4,
                    technical_status="valid",
                    performance_status="measured",
                    evidence_paths=[str(second / "seed-1")],
                ),
            )
            rows, _validation = validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertEqual([row["attempt_id"] for row in rows], ["valid-b"])

    def test_registry_rejects_stale_configuration_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = measured_bundle(root, "attempt-a")
            registry = root / "attempt-registry.jsonl"
            record = build_record(
                attempt_id="valid-stale",
                controller="hpa",
                workload="gradual-ramp",
                seed=1,
                started_at_unix=1,
                finished_at_unix=2,
                technical_status="valid",
                performance_status="measured",
                evidence_paths=[str(bundle / "seed-1")],
            )
            record["protocol_hash"] = "stale"
            append_record(registry, record)
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1})
            self.assertIn("stale_configuration", {issue["code"] for issue in ctx.exception.issues})

    def test_rejects_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured_bundle(root)
            with self.assertRaises(ArtifactValidationError) as ctx:
                validated_summary_rows(root, controllers={"hpa"}, workloads={"gradual-ramp"}, seeds={1, 2})
            self.assertIn("incomplete_matrix", {issue["code"] for issue in ctx.exception.issues})

    def test_rejects_stale_existing_generated_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_json(output / "artifact_validation_manifest.json", {"source_digest": "old"})
            with self.assertRaises(ArtifactValidationError):
                assert_artifacts_not_stale(output, "new", replace_stale=False)






if __name__ == "__main__":
    unittest.main()
