from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import csv
import json
import os
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from experiments.lib.config import REPO_ROOT, resolve_run
from experiments.lib.capacity_contract import (
    build_capacity_contract,
    validate_capacity_contract,
    write_capacity_contract,
)
from experiments.lib.attempt_registry import retry_decision
from experiments.lib.metrics import (
    compute_summary_metrics,
    counter_seconds,
    excess_response_time_seconds,
    mean_pod_count,
    percentile,
    pod_seconds,
    request_metrics,
    sla_violation_rate,
)
from experiments.lib.policies import ThresholdState, static_split_replicas, threshold_decision
from experiments.lib.prometheus import export_queries
from experiments.lib.protocol import (
    capacity_seeds,
    evaluation_seeds,
    load_protocol,
    primary_controllers,
    primary_workloads,
    protocol_hash,
    protocol_reference,
)
from experiments.lib.quality import QualityIssue, performance_outcome, technical_validity_payload
from experiments.lib.static_plan import StaticPlanError, build_static_plan, validate_static_plan
from experiments.lib.workloads import generate_shape
from experiments.run_offline_ablations import ABLATION_VARIANTS, DEFAULT_ABLATION_SEEDS, parse_seeds, plan_payload
from experiments.run_headlight_grid import candidates as headlight_grid_candidates
from experiments.scripts.render_minimal_trainticket import render
from experiments.scripts.render_keda_scaledobjects import render as render_keda_scaledobjects, scaled_object
from experiments.scripts.render_hpa_autoscalers import hpa_max_replicas_for_cluster, render as render_hpa_autoscalers
from experiments.scripts.build_calibration_dataset import build_dataset
from experiments.summarize_capacity_results import best_headlight, collect_rows, write_csv
from experiments.write_mixed_score_artifacts import read_rows, score_rows, filter_rows, DEFAULT_WEIGHTS
from experiments import run_experiment
from experiments.run_experiment import (
    _measurement_load_phases,
    _merge_prometheus_responses,
    _observation_tail_mode,
    _prometheus_pod_count_samples,
)
from experiments.validate_configs import validate_all, validate_resolved_manifests

sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))
from headlight.policy import PolicyConfig

try:
    from train_model import build_train_validation_loaders, _workload_shaped_history
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "torch"}:
        raise
    build_train_validation_loaders = None
    _workload_shaped_history = None


class ExperimentTests(unittest.TestCase):
    def test_measurement_metadata_records_equal_multiphase_boundaries_and_steady_tail(self) -> None:
        workload = {
            "pattern": "multi-phase",
            "continuation_mode": "hold_endpoint",
            "phases": [
                {"pattern": "gradual-ramp", "duration_seconds": 405},
                {"pattern": "flash-crowd", "duration_seconds": 390},
                {"pattern": "diurnal", "duration_seconds": 405},
            ],
        }

        phases = _measurement_load_phases(workload, 1000)

        self.assertEqual(phases[0]["started_at_unix"], 1000)
        self.assertEqual(phases[1]["started_at_unix"], 1405)
        self.assertEqual(phases[2]["started_at_unix"], 1795)
        self.assertEqual(phases[2]["finished_at_unix"], 2200)
        self.assertEqual(_observation_tail_mode(workload, 180), "hold_final_load")

    def test_forecast_qualification_is_not_treated_as_a_primary_evaluation(self) -> None:
        manifest = {
            "spec": {
                "controller": {"controller": "headlight"},
                "workload": {"pattern": "gradual-ramp"},
                "run": {"qualification_overrides": {"kind": "forecast-qualification"}},
            }
        }

        self.assertFalse(run_experiment._is_primary_evaluation(manifest))

    def test_headlight_grid_has_exactly_48_unique_candidates(self) -> None:
        candidates = headlight_grid_candidates()
        self.assertEqual(len(candidates), 48)
        self.assertEqual(len({candidate["candidate"] for candidate in candidates}), 48)
        self.assertEqual({candidate["lookback"] for candidate in candidates}, {12, 24, 36})
        self.assertEqual({candidate["graph_blocks"] for candidate in candidates}, {1, 2})
        self.assertEqual({candidate["hidden_width"] for candidate in candidates}, {16, 32})
        self.assertEqual({candidate["learning_rate"] for candidate in candidates}, {0.001, 0.0003})
        self.assertEqual({candidate["residual_bound"] for candidate in candidates}, {0.1, 0.3})

    def test_resolve_run(self) -> None:
        manifest = resolve_run(Path("experiments/configs/runs/headlight-diurnal-multi-cluster-headlight.yaml"))
        self.assertEqual(manifest["metadata"]["name"], "headlight__diurnal__multi-cluster-headlight")
        self.assertEqual(manifest["spec"]["run"]["seeds"], [101, 102, 103, 104, 105])
        self.assertEqual(manifest["spec"]["run"]["protocol"], protocol_reference())

    def test_research_v3_protocol_contract(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["spec"]["status"], "pre-execution")
        self.assertEqual(protocol["spec"]["upstream_base_commit"], "367e833458312fd2021e02ebbc3f70e9fca02ecc")
        self.assertEqual(primary_controllers(protocol), ("headlight", "lstm", "threshold"))
        self.assertEqual(primary_workloads(protocol), ("gradual-ramp", "flash-crowd", "diurnal"))
        self.assertEqual(capacity_seeds(protocol), (11,))
        self.assertEqual(evaluation_seeds(protocol), (101, 102, 103, 104, 105))
        self.assertIn("keda", protocol["spec"]["controllers"]["excluded_from_primary"])
        self.assertEqual(protocol["spec"]["primary_matrix"]["expected_cell_count"], 45)
        self.assertTrue(protocol["spec"]["primary_matrix"]["no_data_dependent_seed_promotion"])
        self.assertEqual(protocol["spec"]["offline_ablations"]["variants"], list(ABLATION_VARIANTS))
        self.assertEqual(protocol["spec"]["statistics"]["independent_unit"], "seed_run")
        self.assertEqual(protocol_reference(protocol)["hash"], protocol_hash())



    def test_offline_ablation_plan_is_separate_and_seed_disjoint(self) -> None:
        args = type(
            "Args",
            (),
            {
                "dataset": Path("results/calibration/train-ticket-calibration.csv"),
                "service_profile": Path("ai-module/configs/service_profiles/advanced-ticket.json"),
                "adjacency": Path("results/calibration/static-adjacency.json"),
                "epochs": 100,
                "batch_size": 32,
                "lookback": 24,
                "horizon": 12,
                "learning_rate": 0.0003,
                "hidden_width": 32,
                "graph_blocks": 2,
                "residual_bound": 0.3,
                "expected_seconds_per_training": 900,
            },
        )()
        seeds = parse_seeds(",".join(str(seed) for seed in DEFAULT_ABLATION_SEEDS))
        plan = plan_payload(args, seeds)
        self.assertEqual(plan["variants"], list(ABLATION_VARIANTS))
        self.assertEqual(plan["cell_count"], 40)
        self.assertTrue(plan["separate_from_live_matrix"])
        self.assertEqual(set(plan["seeds"]) & set(evaluation_seeds()), set())
        self.assertEqual(plan["paired_design"]["replicate_definition"], "one training seed; services and samples within a seed are not independent replicates")
        with self.assertRaises(ValueError):
            parse_seeds("21,22,23,24,101")

    def test_forecast_policy_config_is_frozen_and_seed_separated(self) -> None:
        policy = PolicyConfig.load()
        self.assertEqual(policy.payload["policy_id"], "research-v3-forecast-policy")
        self.assertEqual(policy.payload["status"], "frozen-pre-execution")
        self.assertEqual(policy.configuration_hash, policy.payload["policy_configuration_hash"])
        self.assertEqual(policy.payload["provenance"]["calibration_trace"]["seeds"], [])
        self.assertFalse(set(policy.payload["forbidden_seed_set"]) & set(policy.payload["provenance"]["calibration_trace"]["seeds"]))

    def test_all_run_configs_reference_research_v3_protocol(self) -> None:
        manifests, _messages = validate_all()
        for manifest in manifests:
            self.assertEqual(manifest["spec"]["run"]["protocol"], protocol_reference())

    def test_checked_in_resolved_manifests_match_canonical_runs(self) -> None:
        manifests, _messages = validate_all()
        self.assertEqual(validate_resolved_manifests(manifests), len(manifests))

    def test_resolve_keda_run(self) -> None:
        manifest = resolve_run(Path("experiments/configs/runs/keda-diurnal-multi-cluster-headlight.yaml"))
        self.assertEqual(manifest["metadata"]["name"], "keda__diurnal__multi-cluster-headlight")
        self.assertEqual(manifest["spec"]["controller"]["controller"], "keda")
        self.assertEqual(manifest["spec"]["topology"]["active_clusters"], ["cluster2", "cluster3"])

    def test_workload_shape_determinism(self) -> None:
        manifest = resolve_run(Path("experiments/configs/runs/headlight-flash-crowd-multi-cluster-headlight.yaml"))
        workload = manifest["spec"]["workload"]
        first = generate_shape(workload, 101, 15)
        second = generate_shape(workload, 101, 15)
        self.assertEqual(first, second)
        self.assertEqual(min(point.users for point in first), 2)
        self.assertEqual(max(point.users for point in first), 10)

    def test_workload_shape_honors_run_duration_override(self) -> None:
        manifest = resolve_run(Path("experiments/configs/runs/headlight-flash-crowd-multi-cluster-headlight.yaml"))
        workload = manifest["spec"]["workload"]
        points = generate_shape(
            workload,
            101,
            15,
            warmup_seconds=360,
            duration_seconds=360,
        )
        self.assertEqual(points[-1].elapsed_seconds, 720)
        self.assertEqual(len(points), 49)

    def test_capacity_normalized_shapes_and_diurnal_cycles(self) -> None:
        gradual = resolve_run(Path("experiments/configs/runs/headlight-gradual-ramp-multi-cluster-headlight.yaml"))["spec"]["workload"]
        flash = resolve_run(Path("experiments/configs/runs/headlight-flash-crowd-multi-cluster-headlight.yaml"))["spec"]["workload"]
        diurnal = resolve_run(Path("experiments/configs/runs/headlight-diurnal-multi-cluster-headlight.yaml"))["spec"]["workload"]
        old_capacity = os.environ.get("HEADLIGHT_EVALUATION_CAPACITY")
        try:
            os.environ["HEADLIGHT_EVALUATION_CAPACITY"] = "25"
            gradual_points = generate_shape(gradual, 101, 15)
            flash_points = generate_shape(flash, 101, 15)
            diurnal_points = generate_shape(diurnal, 101, 15)
        finally:
            if old_capacity is None:
                os.environ.pop("HEADLIGHT_EVALUATION_CAPACITY", None)
            else:
                os.environ["HEADLIGHT_EVALUATION_CAPACITY"] = old_capacity
        self.assertEqual(gradual_points[0].users, 5)
        self.assertEqual(max(point.users for point in gradual_points), 25)
        self.assertEqual(min(point.users for point in flash_points), 5)
        self.assertEqual(max(point.users for point in flash_points), 25)
        self.assertEqual(diurnal["duration_seconds"], 2400)
        self.assertEqual(diurnal["locust"]["cycle_seconds"], 1200)
        self.assertEqual(diurnal_points[0].users, diurnal_points[80].users)
        self.assertEqual(diurnal_points[0].users, diurnal_points[160].users)

    def test_threshold_decision(self) -> None:
        up = threshold_decision(
            p95_latency_ms=301,
            cpu_utilization=0.2,
            state=ThresholdState(replicas=2),
            min_replicas=1,
            max_replicas=4,
        )
        self.assertEqual(up.replicas, 3)
        stable = threshold_decision(
            p95_latency_ms=100,
            cpu_utilization=0.1,
            state=ThresholdState(replicas=3, stable_windows=1),
            min_replicas=1,
            max_replicas=4,
        )
        self.assertEqual(stable.replicas, 2)

    def test_static_split(self) -> None:
        self.assertEqual(static_split_replicas(8), {"cluster2": 3, "cluster3": 5})
        self.assertEqual(static_split_replicas(10), {"cluster2": 4, "cluster3": 6})

    def test_metric_formulas(self) -> None:
        latencies = [100, 250, 301, 500]
        self.assertEqual(sla_violation_rate(latencies, 300), 0.5)
        self.assertAlmostEqual(excess_response_time_seconds(latencies, 300), 0.201)
        self.assertEqual(mean_pod_count([1, 2, 3]), 2)
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)

    def test_compute_summary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            request_csv = tmpdir / "request_events.csv"
            request_csv.write_text(
                "timestamp,method,endpoint,status,response_time_ms,exception\n"
                "1,GET,/a,200,100,\n"
                "2,GET,/a,200,500,\n",
                encoding="utf-8",
            )
            prometheus = tmpdir / "prometheus.json"
            prometheus.write_text(
                '{"cpu_seconds":{"response":{"data":{"result":[{"values":[[1,"1.5"],[2,"2.5"]]}]}}},'
                '"pod_count":{"response":{"data":{"result":[{"values":[[1,"2"],[2,"4"]]}]}}}}',
                encoding="utf-8",
            )
            metrics = compute_summary_metrics(request_csv, prometheus, 300)
            self.assertEqual(metrics["sla_violation_rate"], 0.5)
            self.assertEqual(metrics["cpu_seconds"], 1.0)
            self.assertEqual(metrics["mean_pod_count"], 3.0)
            self.assertEqual(metrics["successful_requests"], 2)
            self.assertEqual(metrics["failed_requests"], 0)

    def test_mean_pod_count_sums_labeled_series_by_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            request_csv = tmpdir / "request_events.csv"
            request_csv.write_text(
                "timestamp,method,endpoint,status,response_time_ms,exception\n"
                "1,GET,/a,200,100,\n"
                "2,GET,/a,200,100,\n",
                encoding="utf-8",
            )
            prometheus = tmpdir / "prometheus.json"
            prometheus.write_text(
                '{"pod_count":{"response":{"data":{"result":['
                '{"metric":{"deployment":"a"},"values":[[1,"2"],[2,"4"]]},'
                '{"metric":{"deployment":"b"},"values":[[1,"3"],[2,"5"]]}'
                ']}}}}',
                encoding="utf-8",
            )
            metrics = compute_summary_metrics(request_csv, prometheus, 300)
            self.assertEqual(metrics["mean_pod_count"], 7.0)

    def test_request_metrics_latency_uses_successful_requests_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request_csv = Path(tmp) / "request_events.csv"
            request_csv.write_text(
                "timestamp,method,endpoint,status,response_time_ms,exception\n"
                "1,GET,/a,200,100,\n"
                "2,GET,/a,500,1,\n"
                "3,GET,/a,504,2,timeout\n"
                "4,GET,/a,302,300,\n",
                encoding="utf-8",
            )
            metrics = request_metrics(request_csv, 200, started_at=1, finished_at=4)
            self.assertEqual(metrics["total_requests"], 4)
            self.assertEqual(metrics["successful_requests"], 2)
            self.assertEqual(metrics["failed_requests"], 2)
            self.assertEqual(metrics["timeout_count"], 1)
            self.assertEqual(metrics["exception_count"], 1)
            self.assertEqual(metrics["p95_latency_ms"], 290.0)
            self.assertEqual(json.loads(str(metrics["http_status_distribution"]))["500"], 1)

    def test_cpu_counter_delta_handles_resets_and_multi_pod(self) -> None:
        payload = {
            "cpu_seconds": {
                "response": {
                    "data": {
                        "result": [
                            {"values": [[0, "1"], [5, "4"], [10, "1"], [15, "3"]]},
                            {"values": [[0, "2"], [5, "5"], [10, "7"], [15, "11"]]},
                        ]
                    }
                }
            }
        }
        self.assertEqual(counter_seconds(payload, started_at=0, finished_at=15), 15.0)

    def test_pod_seconds_integrates_exact_window_with_boundary_samples(self) -> None:
        payload = {
            "pod_count": {
                "response": {
                    "data": {
                        "result": [
                            {"values": [[0, "1"], [5, "2"], [10, "2"]]},
                            {"values": [[2, "1"], [7, "3"], [10, "3"]]},
                        ]
                    }
                }
            }
        }
        self.assertEqual(pod_seconds(payload, started_at=0, finished_at=10), 29.0)

    def test_pod_seconds_missing_start_sample_does_not_backfill_future(self) -> None:
        payload = {
            "pod_count": {
                "response": {
                    "data": {
                        "result": [
                            {"values": [[5, "4"], [10, "4"]]},
                        ]
                    }
                }
            }
        }
        self.assertEqual(pod_seconds(payload, started_at=0, finished_at=10), 20.0)

    def test_poor_performance_is_valid_measured_outcome(self) -> None:
        run = {"sla": {"p95_latency_ms": 1000, "minimum_success_rate": 0.99}}
        summary = {"p95_latency_ms": 2500, "success_rate": 0.25}
        self.assertEqual(performance_outcome(summary, run), "poor_but_valid")
        self.assertEqual(technical_validity_payload([])["technical_validity"], "valid")

    def test_technical_invalidity_retry_rules_do_not_retry_performance(self) -> None:
        records: list[dict[str, object]] = []
        allowed = retry_decision(
            records=records,
            controller="hpa",
            workload="gradual-ramp",
            seed=101,
            technical_status="invalid",
            performance_status="not_measured",
            failure_category="prometheus_empty",
        )
        self.assertTrue(allowed["retry_allowed"])
        disallowed = retry_decision(
            records=records,
            controller="hpa",
            workload="gradual-ramp",
            seed=101,
            technical_status="valid",
            performance_status="poor_but_valid",
            failure_category="",
        )
        self.assertFalse(disallowed["retry_allowed"])
        exhausted = retry_decision(
            records=[
                {
                    "controller": "hpa",
                    "workload": "gradual-ramp",
                    "seed": 101,
                    "technical_status": "invalid",
                }
            ],
            controller="hpa",
            workload="gradual-ramp",
            seed=101,
            technical_status="invalid",
            performance_status="not_measured",
            failure_category="prometheus_empty",
        )
        self.assertFalse(exhausted["retry_allowed"])
        self.assertEqual(exhausted["retry_rule"], "max_attempts_exhausted")
        payload = technical_validity_payload([QualityIssue("prometheus_empty", "missing telemetry")])
        self.assertEqual(payload["technical_validity"], "invalid")
        self.assertTrue(payload["retry_allowed"])

    def test_full_trainticket_renderer(self) -> None:
        docs = render("full", "paper-ns", "source")
        deployments = [doc for doc in docs if doc.get("kind") == "Deployment"]
        services = [doc for doc in docs if doc.get("kind") == "Service"]
        self.assertGreaterEqual(len(deployments), 60)
        self.assertGreaterEqual(len(services), 60)
        self.assertTrue(
            all(
                doc["spec"]["template"]["metadata"]["annotations"]["sidecar.istio.io/inject"] == "false"
                for doc in deployments
                if doc["metadata"]["name"].endswith(("-mongo", "-mysql"))
            )
        )

    def test_advanced_ticket_renderer_includes_ticketinfo_dependency(self) -> None:
        docs = render("advanced-ticket", "paper-ns", "capacity-vm")
        names = {doc["metadata"]["name"] for doc in docs if "metadata" in doc}
        self.assertIn("ts-basic", names)
        self.assertIn("ts-basic-service", names)
        self.assertIn("ts-ticketinfo", names)
        self.assertIn("ts-ticketinfo-service", names)
        self.assertNotIn("ts-voucher", names)
        self.assertNotIn("ts-voucher-service", names)

    def test_prometheus_queries(self) -> None:
        queries = export_queries("trainticket", "15s")
        self.assertIn("container_cpu_usage_seconds_total", queries["cpu_seconds"])
        self.assertNotIn("increase(", queries["cpu_seconds"])
        self.assertIn('destination_workload_namespace="trainticket"', queries["request_rate"])
        self.assertIn("label_replace", queries["pod_count"])
        self.assertIn("deployment", queries["pod_count"])

    def test_ceiling_pod_count_ignores_unlabeled_aggregate_series(self) -> None:
        payload = {
            "pod_count": {
                "response": {
                    "data": {
                        "result": [
                            {"metric": {}, "values": [[1, "24"]]},
                            {"metric": {"deployment": "ts-auth", "cluster": "cluster2"}, "values": [[1, "6"]]},
                            {"metric": {"deployment": "ts-auth", "cluster": "cluster3"}, "values": [[1, "8"]]},
                        ]
                    }
                }
            }
        }
        self.assertEqual(
            _prometheus_pod_count_samples(payload),
            {"ts-auth": [(1.0, 14.0)]},
        )

    def test_prometheus_cluster_responses_are_merged_with_provenance(self) -> None:
        response = {"status": "success", "data": {"resultType": "matrix", "result": [{"metric": {"pod": "p"}, "values": [[1, "1"]]}]}}
        merged = _merge_prometheus_responses([("cluster2", response), ("cluster3", response)])
        self.assertEqual(
            [series["metric"]["cluster"] for series in merged["data"]["result"]],
            ["cluster2", "cluster3"],
        )

    def test_runner_dry_run_bundle(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "results") as tmp:
            output = Path(tmp) / "dry-run-bundle"
            env = os.environ.copy()
            env["HEADLIGHT_EXPERIMENT_OUTPUT_DIR"] = str(output)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "experiments" / "run_experiment.py"),
                    "--run",
                    "experiments/configs/runs/headlight-diurnal-multi-cluster-headlight.yaml",
                    "--seeds",
                    "11",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )
            for relative in [
                "manifest.yaml",
                "environment.json",
                "logs",
                "raw",
                "summaries",
                "figures",
                "seed-11/raw/load_profile.csv",
                "seed-11/raw/protocol.json",
                "seed-11/raw/topology.json",
                "seed-11/raw/workload_manifest_hashes.json",
                "seed-11/raw/phase_timestamps.json",
                "seed-11/raw/measurement_window.json",
                "seed-11/raw/resource_calculation.json",
                "seed-11/summaries/summary.csv",
            ]:
                self.assertTrue((output / relative).exists(), relative)
            phases = json.loads((output / "seed-11/raw/phase_timestamps.json").read_text(encoding="utf-8"))
            self.assertEqual(phases["planned"]["warmup_seconds"], 480)
            self.assertEqual(phases["planned"]["measurement_seconds"], 2400)
            self.assertEqual(phases["planned"]["cooldown_seconds"], 60)
            self.assertEqual(phases["planned"]["locust_run_seconds"], 2880)

    def test_capacity_contract_is_read_only_and_seed_separated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capacity-contract.json"
            payload = build_capacity_contract(
                selected_capacity=8,
                seed=11,
                qualification_inputs={"pilot_levels": [2, 4, 6, 8, 10]},
                environment={"dry_run": True},
                pilot_results=[{"users": 8, "healthy": True}],
                workload_manifest_hashes={
                    "trainticket_path": "advanced-ticket",
                    "resource_profile": "capacity-vm",
                    "data_plane_no_policies_sha256": "abc",
                },
            )
            write_capacity_contract(path, payload)
            loaded, warnings = validate_capacity_contract(
                path,
                expected_trainticket_path="advanced-ticket",
                expected_resource_profile="capacity-vm",
            )
            self.assertEqual(loaded["selected_capacity"], 8)
            self.assertEqual(warnings, [])
            self.assertFalse(path.stat().st_mode & 0o222)
            replacement = dict(payload)
            replacement["selected_capacity"] = 7
            write_capacity_contract(path, replacement)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["selected_capacity"], 7)
            self.assertFalse(path.stat().st_mode & 0o222)

    def test_static_plan_is_calibration_derived_and_rejects_evaluation_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "capacity-contract.json"
            contract = build_capacity_contract(
                selected_capacity=8,
                seed=11,
                qualification_inputs={"pilot_levels": [2, 4, 6, 8]},
                environment={"dry_run": True},
                pilot_results=[{"users": 8, "healthy": True}],
                workload_manifest_hashes={
                    "trainticket_path": "advanced-ticket",
                    "resource_profile": "capacity-vm",
                    "data_plane_no_policies_sha256": "abc",
                },
            )
            write_capacity_contract(contract_path, contract)
            plan = build_static_plan(
                service_names=["ts-auth"],
                capacity_contract_path=contract_path,
                calibration_trace_name="calibration.csv",
                calibration_trace_hash="trace",
                calibration_seeds=[12, 13],
            )
            plan_path = root / "static-plan.json"
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loaded = validate_static_plan(plan_path)
            self.assertEqual(loaded["assignments"]["ts-auth"], {"cluster2": 3, "cluster3": 5})
            loaded["calibration_trace"]["seeds"] = [101]
            loaded["plan_hash"] = plan["plan_hash"]
            plan_path.write_text(json.dumps(loaded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(StaticPlanError):
                validate_static_plan(plan_path)

    def test_capacity_result_summarizer_selects_best_headlight(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "results") as tmp:
            root = Path(tmp)

            def bundle(controller: str, name: str, excess: float, violation: float, candidate: str = "") -> None:
                bundle_root = root / name / "qualification" / controller
                (root / name).mkdir(parents=True, exist_ok=True)
                (root / name / "campaign_status.json").write_text('{"status": "success"}\n', encoding="utf-8")
                (bundle_root / "seed-101" / "summaries").mkdir(parents=True)
                (bundle_root / "seed-101" / "raw").mkdir(parents=True)
                (bundle_root / "seed-101" / "raw" / "measurement_window.json").write_text(
                    '{"started_at_unix": 1, "finished_at_unix": 2}\n',
                    encoding="utf-8",
                )
                (bundle_root / "environment.json").write_text('{"dry_run": false}\n', encoding="utf-8")
                headlight_candidate = {"candidate": candidate} if candidate else {}
                (bundle_root / "manifest.yaml").write_text(
                    json.dumps(
                        {
                            "spec": {
                                "run": {"qualification_overrides": {"headlight_candidate": headlight_candidate}},
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (bundle_root / "seed-101" / "summaries" / "summary.csv").write_text(
                    "seed,controller,workload,sla_latency_ms,sla_violation_rate,"
                    "excess_response_time_seconds,cpu_seconds,mean_pod_count,p95_latency_ms\n"
                    f"101,{controller},gradual-ramp,1000,{violation},{excess},10,2,{900 + excess}\n",
                    encoding="utf-8",
                )

            bundle("headlight", "run-h0", 1.0, 0.2, "h0")
            bundle("headlight", "run-h1", 0.5, 0.3, "h1")
            bundle("headlight", "run-failed", 0.1, 0.1, "bad")
            (root / "run-failed" / "campaign_status.json").write_text('{"status": "failure"}\n', encoding="utf-8")
            bundle("hpa", "run-baselines", 2.0, 0.4)
            rows = collect_rows([root])
            selected = best_headlight(rows)
            self.assertEqual(selected["candidate"], "h1")
            self.assertNotIn("bad", {row["candidate"] for row in rows})
            output = root / "single_seed_reduced_slice_summary.csv"
            write_csv(output, [selected])
            with output.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written[0]["controller"], "headlight")
            self.assertEqual(written[0]["p95_ms"], "900.5")

    def test_mixed_score_keeps_excess_as_primary_signal(self) -> None:
        rows = [
            {
                "candidate": "low-p95-bad-excess",
                "controller": "headlight",
                "p95_ms": "700",
                "sla_violation_rate": "0.010",
                "excess_response_time_seconds": "56",
                "cpu_seconds": "10",
                "mean_pod_count": "2",
                "result_root": "a",
            },
            {
                "candidate": "balanced",
                "controller": "headlight",
                "p95_ms": "750",
                "sla_violation_rate": "0.012",
                "excess_response_time_seconds": "40",
                "cpu_seconds": "10",
                "mean_pod_count": "2",
                "result_root": "b",
            },
            {
                "candidate": "baseline",
                "controller": "hpa",
                "p95_ms": "710",
                "sla_violation_rate": "0.009",
                "excess_response_time_seconds": "33",
                "cpu_seconds": "10",
                "mean_pod_count": "2",
                "result_root": "c",
            },
        ]
        selected = filter_rows(rows, "headlight")
        scored = score_rows(selected, DEFAULT_WEIGHTS)
        self.assertEqual(scored[0]["candidate"], "balanced")
        self.assertLess(float(scored[0]["mixed_score"]), float(scored[1]["mixed_score"]))

    def test_mixed_score_artifact_reader_deduplicates_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "results") as tmp:
            path = Path(tmp) / "rows.csv"
            path.write_text(
                "candidate,headlight_step_num_in,headlight_num_blocks,headlight_learning_rate,"
                "controller,p95_ms,sla_violation_rate,excess_response_time_seconds,"
                "cpu_seconds,mean_pod_count,result_root\n"
                "h0,2,2,0.0001,headlight,750,0.01,40,10,2,run-a\n"
                "h0,2,2,0.0001,headlight,750,0.01,40,10,2,run-a\n",
                encoding="utf-8",
            )
            rows = read_rows([path])
            self.assertEqual(len(rows), 1)



    def test_headlight_image_default_matches_local_docker_image(self) -> None:
        runner = (REPO_ROOT / "experiments" / "run_experiment.py").read_text(encoding="utf-8")
        self.assertIn('"local/headlight-ai-module:dev"', runner)
        self.assertIn('"HEADLIGHT_AI_IMAGE": os.environ.get("HEADLIGHT_AI_IMAGE", "local/headlight-ai-module:dev")', runner)

    def test_recorded_runtime_image_defaults_match_execution(self) -> None:
        runner = (REPO_ROOT / "experiments" / "run_experiment.py").read_text(encoding="utf-8")
        for source in (runner,):
            self.assertIn('"ghcr.io/vidbregar/actions-handler:0.2.0"', source)
            self.assertIn('"dstefanov46/locust-test:0.66.0"', source)

    def test_trainticket_deploy_refreshes_stale_crp_dispatch(self) -> None:
        deploy = (REPO_ROOT / "experiments" / "deploy_minimal_trainticket.sh").read_text(encoding="utf-8")
        self.assertIn("refresh_crp_dispatch", deploy)
        self.assertIn("headlight.nephele-project.eu/reconcile-at", deploy)
        self.assertIn("headlight.nephele-project.eu/dispatch-at", deploy)

    def test_metrics_server_installer_targets_experiment_clusters(self) -> None:
        text = (REPO_ROOT / "experiments" / "install_metrics_server.sh").read_text(encoding="utf-8")
        self.assertIn('contexts="${HEADLIGHT_METRICS_SERVER_CONTEXTS:-cluster2 cluster3}"', text)
        self.assertIn("--kubelet-insecure-tls", text)
        self.assertIn("--kubelet-preferred-address-types=InternalIP,Hostname,ExternalIP", text)
        self.assertIn("--kubelet-request-timeout=25s", text)
        self.assertIn("--metric-resolution=30s", text)
        self.assertIn("/livenessProbe/failureThreshold", text)
        self.assertIn('"value": {"type": "Recreate"}', text)
        self.assertIn("top nodes", text)

    def test_keda_installs_into_explicit_context(self) -> None:
        text = (REPO_ROOT / "experiments" / "scripts" / "install_keda_baseline.sh").read_text(encoding="utf-8")
        self.assertIn('context="${HEADLIGHT_KEDA_CONTEXT:-cluster3}"', text)
        self.assertIn('keda_chart_version="${HEADLIGHT_KEDA_CHART_VERSION:-2.19.0}"', text)
        self.assertIn('helm --kube-context "$context" upgrade --install keda', text)
        self.assertIn('--version "$keda_chart_version"', text)
        self.assertIn('kubectl --context="$context" -n keda rollout status deployment/keda-operator', text)

    def test_keda_cpu_trigger_uses_current_metric_type_schema(self) -> None:
        doc = scaled_object(
            "ts-auth",
            "trainticket",
            "http://prometheus.monitoring.svc:9090",
            min_replicas=1,
            max_replicas=8,
            cpu_target=70,
            request_rate_threshold="10",
        )
        cpu_trigger = doc["spec"]["triggers"][0]
        self.assertEqual(cpu_trigger["type"], "cpu")
        self.assertEqual(cpu_trigger["metricType"], "Utilization")
        self.assertEqual(cpu_trigger["metadata"]["value"], "70")
        self.assertNotIn("type", cpu_trigger["metadata"])
        self.assertNotIn("metricType", cpu_trigger["metadata"])

    def test_keda_scaledobjects_follow_advanced_ticket_slice(self) -> None:
        args = type(
            "Args",
            (),
            {
                "source": None,
                "trainticket_path": "advanced-ticket",
                "resource_profile": "capacity-vm",
                "namespace": "trainticket",
                "prometheus_address": "http://prometheus.monitoring.svc:9090",
                "min_replicas": 1,
                "max_replicas": 8,
                "cpu_target": 70,
                "request_rate_threshold": "10",
            },
        )()
        names = {doc["spec"]["scaleTargetRef"]["name"] for doc in render_keda_scaledobjects(args)}
        self.assertIn("ts-auth", names)
        self.assertIn("ts-basic", names)
        self.assertIn("ts-ticketinfo", names)
        self.assertNotIn("ts-voucher", names)

    def test_hpa_autoscalers_follow_advanced_ticket_slice(self) -> None:
        args = type(
            "Args",
            (),
            {
                "source": REPO_ROOT / "ai-module" / "kubernetes-autoscaler" / "hpa.yaml",
                "trainticket_path": "advanced-ticket",
                "resource_profile": "capacity-vm",
                "namespace": "trainticket",
                "max_replicas": 3,
            },
        )()
        targets = {doc["spec"]["scaleTargetRef"]["name"] for doc in render_hpa_autoscalers(args)}
        self.assertIn("ts-auth", targets)
        self.assertNotIn("ts-auth-mongo", targets)
        self.assertIn("ts-basic", targets)
        self.assertIn("ts-ticketinfo", targets)
        self.assertNotIn("ts-voucher", targets)
        self.assertNotIn("ts-user", targets)
        self.assertTrue(all(doc["spec"]["maxReplicas"] == 3 for doc in render_hpa_autoscalers(args)))
        self.assertTrue(
            all("scaleUp" not in doc.get("spec", {}).get("behavior", {}) for doc in render_hpa_autoscalers(args))
        )

    def test_hpa_protocol_split_sums_to_replica_ceiling(self) -> None:
        split = {
            "cluster2": hpa_max_replicas_for_cluster("cluster2"),
            "cluster3": hpa_max_replicas_for_cluster("cluster3"),
        }
        self.assertEqual(split, {"cluster2": 6, "cluster3": 8})
        self.assertEqual(sum(split.values()), 14)

    def test_controller_setup_scripts_do_not_mutate_current_context(self) -> None:
        for relative in [
            "ai-module/ai_module_setup.sh",
            "ai-module/threshold_controller_setup.sh",
            "ai-module/kubernetes_hpa_setup.sh",
            "experiments/install_metrics_server.sh",
            "experiments/scripts/install_keda_baseline.sh",
        ]:
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("kubectl config use-context", text, relative)

    def test_calibration_converter_writes_histories(self) -> None:
        if np is None:
            self.skipTest("NumPy is not installed in this test environment")
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            services = tmpdir / "services.json"
            services.write_text('{"service_names":["ts-auth-service","ts-auth-mongo"]}', encoding="utf-8")
            prometheus = tmpdir / "prometheus.json"
            values = [[100, "1"], [115, "2"], [130, "3"]]
            prometheus.write_text(
                json.dumps(
                    {
                        "cluster2:pod_count": {
                            "cluster": "cluster2",
                            "metric": "pod_count",
                            "response": {
                                "data": {
                                    "result": [
                                        {"metric": {"deployment": "ts-auth-service"}, "values": values},
                                        {"metric": {"deployment": "ts-auth-mongo"}, "values": values},
                                    ]
                                }
                            },
                        },
                        "cluster2:cpu_usage_cores": {
                            "cluster": "cluster2",
                            "metric": "cpu_usage_cores",
                            "response": {
                                "data": {
                                    "result": [
                                        {"metric": {"pod": "ts-auth-service-abc"}, "values": [[100, "0.1"], [115, "0.2"], [130, "0.3"]]},
                                        {"metric": {"pod": "ts-auth-mongo-abc"}, "values": [[100, "0.4"], [115, "0.5"], [130, "0.6"]]},
                                    ]
                                }
                            },
                        },
                        "cluster2:request_rate": {
                            "cluster": "cluster2",
                            "metric": "request_rate",
                            "response": {"data": {"result": [{"metric": {"destination_workload": "ts-auth-service"}, "values": [[100, "7"], [115, "8"], [130, "9"]]}]}},
                        },
                        "cluster2:response_time_sum_ms": {
                            "cluster": "cluster2",
                            "metric": "response_time_sum_ms",
                            "response": {"data": {"result": [{"metric": {"destination_workload": "ts-auth-service"}, "values": [[100, "100"], [115, "200"], [130, "300"]]}]}},
                        },
                        "cluster2:response_time_count": {
                            "cluster": "cluster2",
                            "metric": "response_time_count",
                            "response": {"data": {"result": [{"metric": {"destination_workload": "ts-auth-service"}, "values": [[100, "10"], [115, "10"], [130, "10"]]}]}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = build_dataset(prometheus, services, tmpdir / "calibration.csv", tmpdir / "feature_history", 15)
            self.assertEqual(summary["history_files"], 2)
            auth_history = np.load(tmpdir / "feature_history" / "ts-auth_cluster2_history.npy")
            mongo_history = np.load(tmpdir / "feature_history" / "ts-auth-mongo_cluster2_history.npy")
            self.assertEqual(auth_history.shape, (2, 4))
            self.assertEqual(mongo_history.shape, (2, 4))
            np.testing.assert_allclose(auth_history[0], [2.0, 0.02, 8.0, 0.2])
            np.testing.assert_allclose(mongo_history[0], [2.0, 0.0, 0.0, 0.5])

    def test_training_split_is_chronological(self) -> None:
        if np is None:
            self.skipTest("NumPy is not installed in this test environment")
        if build_train_validation_loaders is None:
            self.skipTest("PyTorch is not installed in this test environment")
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            service_names = ["svc_cluster2", "svc_cluster3"]
            for index, service in enumerate(service_names):
                data = np.zeros((10, 4), dtype=np.float32)
                data[:, 0] = np.arange(10, dtype=np.float32) + index
                data[:, 1] = np.arange(10, dtype=np.float32)
                data[:, 2] = np.arange(10, dtype=np.float32) * 2
                data[:, 3] = np.arange(10, dtype=np.float32) * 3
                np.save(tmpdir / f"{service}_history.npy", data)
            train_loader, validation_loader, train_count, validation_count = build_train_validation_loaders(
                str(tmpdir),
                service_names,
                1,
                4,
                2,
                1,
                64,
                0,
                0.8,
            )
            self.assertEqual((train_count, validation_count), (6, 2))
            train_x, _train_y = next(iter(train_loader))
            validation_x, validation_y = next(iter(validation_loader))
            self.assertEqual(train_x.shape[0], 6)
            self.assertEqual(validation_x.shape[0], 2)
            self.assertEqual(validation_y[0, 0, 0].item(), 8.0)

    def test_headlight_bootstrap_history_follows_workload_shape(self) -> None:
        if np is None:
            self.skipTest("NumPy is not installed in this test environment")
        if _workload_shaped_history is None:
            self.skipTest("PyTorch train_model dependencies are not installed")
        old_env = os.environ.copy()
        try:
            os.environ.update(
                {
                    "HEADLIGHT_BOOTSTRAP_TRAINTICKET_MODE": "advanced-ticket",
                    "HEADLIGHT_BOOTSTRAP_DURATION_SECONDS": "300",
                    "HEADLIGHT_BOOTSTRAP_START_USERS": "5",
                    "HEADLIGHT_BOOTSTRAP_END_USERS": "20",
                    "HEADLIGHT_BOOTSTRAP_RAMP_SECONDS": "300",
                    "HEADLIGHT_BOOTSTRAP_HOLD_SECONDS": "0",
                    "HEADLIGHT_BOOTSTRAP_USERS_PER_POD": "2",
                }
            )
            history = _workload_shaped_history(12, 4, "ts-auth_cluster3")
            self.assertEqual(history.shape, (18, 4))
            self.assertTrue(np.all(history[:6, 0] == 1))
            self.assertGreater(history[-1, 2], history[6, 2])
            self.assertGreaterEqual(history[-1, 0], history[6, 0])
            repeat = _workload_shaped_history(12, 4, "ts-auth_cluster3")
            np.testing.assert_allclose(history, repeat)
        finally:
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
