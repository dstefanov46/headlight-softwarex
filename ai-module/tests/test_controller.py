from __future__ import annotations

import json
import io
import sys
import tempfile
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"optional AI test dependency is not installed: {exc.name}") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))
sys.path.insert(0, str(REPO_ROOT))

from headlight.bundle import ModelBundle, create_bundle, load_checkpoint
from headlight.features import (
    BASE_FEATURES,
    MODEL_FEATURES,
    MetricSample,
    NodeFeatureScaler,
    PrometheusMetricCollector,
    AlignedSeries,
    bounded_forward_fill,
    chronological_split,
    feature_schema,
    prometheus_queries,
)
from headlight.graph import build_static_adjacency, residual_edge_mask
from headlight.models import build_model, decode_forecasts, multitask_loss
from headlight.policy import (
    ForecastPlacementPolicy,
    PlacementActuator,
    PolicyConfig,
    PolicyDecision,
    ServiceForecast,
    ServiceRuntimeState,
    largest_remainder_split,
)
from headlight.profile import load_service_profile
from headlight.runtime import ControllerRuntime, KubectlStateReader
from headlight.training import seed_everything


PROFILE_DIR = REPO_ROOT / "ai-module" / "configs" / "service_profiles"


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _Session:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = list(statuses)
        self.calls: list[dict[str, object]] = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _Response(self.statuses.pop(0))


class ControllerTests(unittest.TestCase):
    def test_partial_member_timeout_holds_instead_of_proposing_from_partial_state(self) -> None:
        from dataclasses import replace
        def partial(args, **kwargs):
            if "cluster3" in args:
                raise subprocess.TimeoutExpired(args, 15)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"items": [{
                "metadata": {"name": "ts-auth"}, "spec": {"replicas": 2}, "status": {"readyReplicas": 2},
            }]}))
        reader = KubectlStateReader("trainticket", command_runner=partial)
        with redirect_stdout(io.StringIO()):
            state = reader.read(self.advanced)["ts-auth"]
        state = replace(state, reactive_fresh=True, current_cpu_utilization=0.95)
        policy = ForecastPlacementPolicy(self.advanced, self._capacity())
        decision = policy.decide("ts-auth", None, state, now=123)
        self.assertEqual(decision.desired_placements, {"cluster2": 2, "cluster3": 0})

    def test_state_reader_timeout_does_not_treat_missing_state_as_ready(self) -> None:
        def unavailable(args, **kwargs):
            raise subprocess.TimeoutExpired(args, 15)
        reader = KubectlStateReader("trainticket", command_runner=unavailable)
        output = io.StringIO()
        with redirect_stdout(output):
            states = reader.read(self.advanced)
        self.assertTrue(all(not state.exists for state in states.values()))
        self.assertTrue(all(sum(state.ready_replicas.values()) == 0 for state in states.values()))
        self.assertIn("HEADLIGHT_STATE_READ_ERROR", output.getvalue())

    def setUp(self) -> None:
        self.advanced = load_service_profile(PROFILE_DIR / "advanced-ticket.json")

    def test_profiles_have_exact_membership_and_fixed_databases(self) -> None:
        expected_apps = {
            "ts-auth",
            "ts-basic",
            "ts-price",
            "ts-route",
            "ts-route-plan",
            "ts-station",
            "ts-ticketinfo",
            "ts-train",
            "ts-travel",
            "ts-travel-plan",
            "ts-travel2",
        }
        expected_databases = {
            "ts-auth-mongo",
            "ts-price-mongo",
            "ts-route-mongo",
            "ts-station-mongo",
            "ts-train-mongo",
            "ts-travel-mongo",
            "ts-travel2-mongo",
        }
        self.assertEqual({node.name for node in self.advanced.scalable_nodes}, expected_apps)
        self.assertEqual({node.name for node in self.advanced.database_nodes}, expected_databases)
        self.assertEqual(len(load_service_profile(PROFILE_DIR / "login-route.json").nodes), 4)
        full = load_service_profile(PROFILE_DIR / "full.json")
        self.assertEqual((len(full.nodes), len(full.scalable_nodes), len(full.database_nodes)), (63, 41, 22))

    def test_scaler_is_node_by_feature_serializable_and_ignores_missing(self) -> None:
        values = np.arange(5 * 2 * len(BASE_FEATURES), dtype=np.float32).reshape(
            5, 2, len(BASE_FEATURES)
        )
        missing = np.zeros_like(values, dtype=bool)
        missing[4, 1, 2] = True
        values[4, 1, 2] = 1_000_000
        scaler = NodeFeatureScaler.fit(values, missing, [0, 15, 30, 45, 60])
        restored = NodeFeatureScaler.from_dict(scaler.to_dict())
        self.assertEqual(restored.mean.shape, (2, len(BASE_FEATURES)))
        self.assertAlmostEqual(restored.mean[1, 2], np.mean(values[:4, 1, 2]))
        transformed = restored.transform(values[:1])
        self.assertEqual(transformed.shape, values[:1].shape)

    def test_chronological_split_has_180_second_purges(self) -> None:
        timestamps = np.arange(100, dtype=np.int64) * 15
        split = chronological_split(timestamps, purge_seconds=180)
        self.assertLess(timestamps[split.train[-1]], split.first_boundary - 180)
        self.assertGreaterEqual(timestamps[split.validation[0]], split.first_boundary)
        self.assertLess(timestamps[split.validation[-1]], split.second_boundary - 180)
        self.assertGreaterEqual(timestamps[split.holdout[0]], split.second_boundary)

    def test_metric_forward_fill_expires_after_30_seconds(self) -> None:
        def query(_query: str, timestamp: int):
            if timestamp != 100:
                return []
            return [
                {
                    "metric": {
                        "deployment": "ts-auth",
                        "destination_workload": "ts-auth",
                        "pod": "ts-auth-abc-def",
                    },
                    "value": [timestamp, "1.5"],
                }
            ]

        collector = PrometheusMetricCollector(
            "http://prometheus.invalid",
            self.advanced,
            query_function=query,
        )
        auth = self.advanced.node_order.index("ts-auth")
        first = collector.collect(100)
        filled = collector.collect(115)
        stale = collector.collect(131)
        self.assertFalse(np.any(first.missing[auth]))
        self.assertTrue(np.all(filled.missing[auth]))
        self.assertFalse(filled.stale_nodes[auth])
        self.assertTrue(stale.stale_nodes[auth])
        self.assertTrue(np.isnan(stale.values[auth, BASE_FEATURES.index("request_rate")]))

    def test_metric_diagnostics_distinguish_query_error_from_observed_value(self) -> None:
        cpu_query = prometheus_queries(self.advanced.namespace)["cpu_utilization"]

        def query(promql: str, timestamp: int):
            if timestamp == 115 and promql == cpu_query:
                raise TimeoutError("synthetic CPU query timeout")
            if promql == cpu_query:
                return [{"metric": {"pod": "ts-auth-abc-def"}, "value": [timestamp, "1.2"]}]
            return []

        collector = PrometheusMetricCollector(
            "http://prometheus.invalid",
            self.advanced,
            query_function=query,
            diagnostic_logging=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            first = collector.collect(100)
            second = collector.collect(115)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        cpu_events = [event for event in events if event["feature"] == "cpu_utilization"]
        self.assertEqual(len(cpu_events), 2)
        self.assertEqual(cpu_events[0]["status"], "ok")
        self.assertEqual(cpu_events[0]["observed_values"]["ts-auth"], first.values[
            self.advanced.node_order.index("ts-auth"), BASE_FEATURES.index("cpu_utilization")
        ])
        self.assertEqual(cpu_events[1]["status"], "error")
        self.assertEqual(cpu_events[1]["error_type"], "TimeoutError")
        self.assertEqual(cpu_events[1]["endpoints"][0]["status"], "error")
        self.assertIn("ts-auth", cpu_events[1]["missing_nodes"])
        self.assertTrue(second.missing[self.advanced.node_order.index("ts-auth"), BASE_FEATURES.index("cpu_utilization")])

    def test_v2_prometheus_query_options_are_decision_semantic(self) -> None:
        queries = prometheus_queries(
            self.advanced.namespace,
            window="60s",
            cpu_window="90s",
            observed_zero_error_rate=True,
        )
        self.assertIn("[90s]", queries["cpu_usage_cores"])
        self.assertIn("[90s]", queries["cpu_utilization"])
        self.assertIn("[60s]", queries["request_rate"])
        self.assertIn("[60s]", queries["p95_latency_ms"])
        self.assertIn("or on (destination_workload)", queries["error_rate"])
        self.assertIn("0 * sum by (destination_workload)", queries["error_rate"])

        legacy = prometheus_queries(self.advanced.namespace)
        self.assertIn("[60s]", legacy["cpu_utilization"])
        self.assertNotIn("or on (destination_workload)", legacy["error_rate"])

    def test_cluster_local_observations_do_not_forward_fill_or_infer_zero(self) -> None:
        class Response:
            def __init__(self, result):
                self.result = result

            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "success", "data": {"result": self.result}}

        collector = PrometheusMetricCollector(
            "http://cluster2.invalid,http://cluster3.invalid",
            self.advanced,
            cluster_names=("cluster2", "cluster3"),
        )
        def get(url, timeout):
            if "cluster2.invalid" in url and "istio_requests_total" in url and "time=100" in url:
                return Response([{"metric": {"destination_workload": "ts-auth"}, "value": [100, "12"]}])
            return Response([])
        collector.session.get = get
        first = collector.collect(100)
        second = collector.collect(115)
        self.assertEqual(first.cluster_values["request_rate"]["ts-auth"], {"cluster2": 12.0})
        self.assertEqual(second.cluster_values, {})
        with self.assertRaises(ValueError):
            PrometheusMetricCollector(
                "http://cluster2.invalid,http://cluster3.invalid", self.advanced,
                cluster_names=("cluster2",),
            )

    def test_offline_forward_fill_matches_runtime_and_preserves_masks(self) -> None:
        values = np.full((4, 1, 1), np.nan, dtype=np.float32)
        missing = np.ones_like(values, dtype=bool)
        values[0, 0, 0] = 7.0
        missing[0, 0, 0] = False
        filled = bounded_forward_fill(
            AlignedSeries(
                timestamps=np.asarray([0, 15, 30, 45], dtype=np.int64),
                values=values,
                missing=missing,
            )
        )
        np.testing.assert_allclose(filled.values[:3, 0, 0], [7.0, 7.0, 7.0])
        self.assertTrue(np.isnan(filled.values[3, 0, 0]))
        self.assertEqual(filled.missing[:, 0, 0].tolist(), [False, True, True, True])

    def test_static_adjacency_is_weighted_normalized_and_residual_is_local(self) -> None:
        payload = build_static_adjacency(
            ("a", "b", "c", "d"),
            [
                {"source": "a", "destination": "b", "weight": 3},
                {"source": "a", "destination": "c", "weight": 1},
                {"source": "b", "destination": "c", "weight": 2},
            ],
        )
        matrix = np.asarray(payload["matrix"])
        self.assertTrue(np.allclose(matrix.sum(axis=1), 1))
        self.assertGreater(matrix[0, 1], matrix[0, 2])
        allowed = residual_edge_mask(matrix, np.asarray(payload["observed_mask"], dtype=bool))
        self.assertTrue(allowed[0, 2])
        self.assertFalse(allowed[0, 3])
        self.assertTrue(np.all(np.diag(allowed)))

    def test_graph_variants_and_lstm_have_real_forward_paths(self) -> None:
        adjacency = build_static_adjacency(
            self.advanced.node_order,
            [{"source": "ts-auth", "destination": "ts-route", "weight": 1}],
        )
        features = torch.randn(2, 24, len(self.advanced.nodes), len(MODEL_FEATURES))
        for variant in ("full", "static_adjacency", "no_graph", "no_attention"):
            config = {
                "model_type": "headlight",
                "num_nodes": len(self.advanced.nodes),
                "input_size": len(MODEL_FEATURES),
                "hidden_width": 16,
                "graph_blocks": 1,
                "variant": variant,
                "residual_bound": 0.3,
            }
            model = build_model(config, adjacency, torch.device("cpu"))
            self.assertEqual(model(features).shape, (2, len(self.advanced.nodes), 5))
            residual = model.adjacency.residual().detach().numpy()
            self.assertLessEqual(float(np.max(np.abs(residual))), 0.3)
        lstm_config = {
            "model_type": "lstm",
            "num_nodes": len(self.advanced.nodes),
            "input_size": len(MODEL_FEATURES),
            "hidden_width": 16,
        }
        lstm = build_model(lstm_config, adjacency, torch.device("cpu"))
        first = lstm(features)
        second = lstm(features + 1)
        self.assertFalse(torch.allclose(first, second))

    def test_training_is_deterministic_for_same_seed(self) -> None:
        adjacency = build_static_adjacency(
            self.advanced.node_order,
            [{"source": "ts-auth", "destination": "ts-route", "weight": 1}],
        )
        config = {
            "model_type": "lstm",
            "num_nodes": len(self.advanced.nodes),
            "input_size": len(MODEL_FEATURES),
            "hidden_width": 16,
        }
        features = torch.linspace(
            0,
            1,
            4 * 12 * len(self.advanced.nodes) * len(MODEL_FEATURES),
        ).reshape(4, 12, len(self.advanced.nodes), len(MODEL_FEATURES))
        targets = torch.ones(4, len(self.advanced.nodes), 4)

        def train_once():
            seed_everything(41)
            model = build_model(config, adjacency, torch.device("cpu"))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            losses = []
            for _ in range(3):
                optimizer.zero_grad()
                raw = model(features)
                loss, _ = multitask_loss(raw, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            return losses, decode_forecasts(model(features)).detach()

        losses_a, predictions_a = train_once()
        losses_b, predictions_b = train_once()
        self.assertEqual(losses_a, losses_b)
        torch.testing.assert_close(predictions_a, predictions_b, rtol=0, atol=0)

    def _capacity(self) -> dict[str, object]:
        return {
            "schema_version": "headlight.capacity/v1",
            "services": {
                node.name: {
                    "request_rate_per_pod": 2.0,
                    "cpu_cores_per_pod": 0.5,
                    "healthy_samples": 10,
                }
                for node in self.advanced.nodes
            },
        }

    def _training_metadata(self) -> dict[str, object]:
        return {
            "seed": 41,
            "dataset_sha256": "dataset",
            "adjacency_sha256": "adjacency",
            "profile_hash": self.advanced.profile_hash,
            "sources": {
                "dataset": "results/calibration/train-ticket-calibration.csv",
                "adjacency": "results/calibration/static-adjacency.json",
                "service_profile": "ai-module/configs/service_profiles/advanced-ticket.json",
            },
            "selection_procedure": "unit-test calibration split",
            "created_before_evaluation": True,
        }

    def test_policy_rounding_readiness_stale_and_scale_down(self) -> None:
        self.assertEqual(
            largest_remainder_split(8, self.advanced.cluster_limits),
            {"cluster2": 3, "cluster3": 5},
        )
        self.assertEqual(
            largest_remainder_split(4, self.advanced.cluster_limits),
            {"cluster2": 2, "cluster3": 2},
        )
        policy = ForecastPlacementPolicy(self.advanced, self._capacity())
        self.assertEqual(policy.policy_config.payload["policy_id"], "research-v3-forecast-policy")
        forecast = ServiceForecast(8, 12, 1, 500, 0.1)
        ready = ServiceRuntimeState(
            {"cluster2": 1, "cluster3": 1},
            {"cluster2": 1, "cluster3": 1},
            500,
            0.4,
            True,
            True,
        )
        up = policy.decide("ts-auth", forecast, ready, now=1000)
        self.assertEqual(up.desired_total, 4)
        pending = ServiceRuntimeState(
            up.desired_placements,
            {"cluster2": 2, "cluster3": 1},
            500,
            0.4,
            True,
            True,
        )
        self.assertEqual(policy.decide("ts-auth", forecast, pending, now=1015).reason, "pending-readiness")
        stale = ServiceRuntimeState(
            {"cluster2": 1, "cluster3": 1},
            {"cluster2": 1, "cluster3": 1},
            None,
            None,
            False,
            False,
        )
        self.assertEqual(policy.decide("ts-auth", None, stale, now=1030).reason, "all-inputs-stale")

        low = ServiceForecast(0, 0, 0, 100, 0.0)
        current = ServiceRuntimeState(
            {"cluster2": 2, "cluster3": 2},
            {"cluster2": 2, "cluster3": 2},
            100,
            0.2,
            True,
            True,
        )
        decisions = [policy.decide("ts-auth", low, current, now=2000 + 15 * index) for index in range(12)]
        self.assertEqual(decisions[-1].desired_total, 3)
        database = policy.decide("ts-auth-mongo", low, current, now=3000)
        self.assertEqual(database.reason, "fixed-database")
        self.assertFalse(database.changed)

    def test_actuator_batches_only_deltas_and_retries_failure(self) -> None:
        session = _Session([500, 202])
        actuator = PlacementActuator(
            "http://actions/placements",
            "trainticket",
            session=session,
        )
        decision = PolicyDecision(
            service="ts-auth",
            prediction=None,
            reason="forecast-capacity",
            previous_placements={"cluster2": 1, "cluster3": 1},
            desired_placements={"cluster2": 1, "cluster3": 3},
            previous_total=2,
            desired_total=4,
            changed=True,
            readiness_delay=False,
        )
        current = {"ts-auth": {"cluster2": 1, "cluster3": 1}}
        failed = actuator.apply([decision], current, self.advanced)
        retried = actuator.apply([decision], current, self.advanced)
        self.assertFalse(failed.successful)
        self.assertTrue(retried.successful)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(session.calls[0]["json"]), 1)
        self.assertEqual(session.calls[0]["json"][0]["clusterName"], "cluster3")
        waiting = actuator.apply([decision], current, self.advanced)
        self.assertFalse(waiting.attempted)

    def test_actuator_pending_readiness_clears_after_ready_replicas(self) -> None:
        session = _Session([202])
        actuator = PlacementActuator(
            "http://actions/placements",
            "trainticket",
            session=session,
        )
        decision = PolicyDecision(
            service="ts-auth",
            prediction=None,
            reason="reactive-guard",
            previous_placements={"cluster2": 1, "cluster3": 1},
            desired_placements={"cluster2": 1, "cluster3": 3},
            previous_total=2,
            desired_total=4,
            changed=True,
            readiness_delay=False,
        )
        actuator.apply([decision], {"ts-auth": {"cluster2": 1, "cluster3": 1}}, self.advanced)
        acknowledged = {"ts-auth": {"cluster2": 1, "cluster3": 3}}
        self.assertEqual(actuator.pending_services(acknowledged), {"ts-auth"})
        states = {
            "ts-auth": ServiceRuntimeState(
                {"cluster2": 1, "cluster3": 3},
                {"cluster2": 1, "cluster3": 3},
                500,
                0.4,
                True,
                True,
            )
        }
        completed = actuator.observe_readiness(states, 1234)
        self.assertIn("ts-auth", completed)
        self.assertEqual(actuator.pending_services(acknowledged), set())

    def test_bundle_is_complete_hash_verified_and_profile_locked(self) -> None:
        adjacency = build_static_adjacency(self.advanced.node_order, [])
        config = {
            "model_type": "lstm",
            "num_nodes": len(self.advanced.nodes),
            "input_size": len(MODEL_FEATURES),
            "lookback": 24,
            "horizon": 12,
            "sampling_interval_seconds": 15,
            "forecast_horizon_seconds": 180,
            "hidden_width": 16,
            "graph_blocks": 0,
            "variant": "per-service-lstm",
            "residual_bound": 0.0,
        }
        model = build_model(config, adjacency, torch.device("cpu"))
        scaler = NodeFeatureScaler(
            np.zeros((len(self.advanced.nodes), len(BASE_FEATURES)), dtype=np.float32),
            np.ones((len(self.advanced.nodes), len(BASE_FEATURES)), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "bundle"
            create_bundle(
                bundle_path,
                model.state_dict(),
                scaler,
                self.advanced,
                adjacency,
                self._capacity(),
                config,
                {"schema_version": "headlight.learned-edges/v1", "edges": []},
                self._training_metadata(),
            )
            loaded = ModelBundle.load(
                bundle_path,
                runtime_profile=self.advanced,
                expected_model_type="lstm",
            )
            checkpoint = load_checkpoint(loaded, torch.device("cpu"))
            self.assertTrue(checkpoint)
            self.assertTrue(loaded.metadata["frozen"])
            self.assertFalse(loaded.metadata["runtime_reload"])
            incompatible_profile = load_service_profile(PROFILE_DIR / "login-route.json")
            with self.assertRaises(ValueError):
                ModelBundle.load(bundle_path, runtime_profile=incompatible_profile)
            scaler_path = bundle_path / "scaler.json"
            scaler_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ModelBundle.load(bundle_path)

    def test_lstm_runtime_consumes_rolling_samples_and_emits_shared_policy_action(self) -> None:
        adjacency = build_static_adjacency(self.advanced.node_order, [])
        config = {
            "model_type": "lstm",
            "num_nodes": len(self.advanced.nodes),
            "input_size": len(MODEL_FEATURES),
            "lookback": 24,
            "horizon": 12,
            "sampling_interval_seconds": 15,
            "forecast_horizon_seconds": 180,
            "hidden_width": 16,
            "graph_blocks": 0,
            "variant": "per-service-lstm",
            "residual_bound": 0.0,
        }
        model = build_model(config, adjacency, torch.device("cpu"))
        scaler = NodeFeatureScaler(
            np.zeros((len(self.advanced.nodes), len(BASE_FEATURES)), dtype=np.float32),
            np.ones((len(self.advanced.nodes), len(BASE_FEATURES)), dtype=np.float32),
        )
        low_capacity = self._capacity()
        for value in low_capacity["services"].values():
            value["request_rate_per_pod"] = 0.01
            value["cpu_cores_per_pod"] = 0.01

        class Collector:
            def __init__(self):
                self.calls = 0

            def collect(self, timestamp):
                self.calls += 1
                values = np.ones(
                    (len(self_profile.nodes), len(BASE_FEATURES)), dtype=np.float32
                )
                values[:, BASE_FEATURES.index("request_rate")] = self.calls
                values[:, BASE_FEATURES.index("p95_latency_ms")] = 200
                values[:, BASE_FEATURES.index("cpu_utilization")] = 0.4
                return MetricSample(
                    int(timestamp),
                    values,
                    np.zeros_like(values, dtype=bool),
                    np.zeros_like(values, dtype=np.int64),
                    np.zeros(len(self_profile.nodes), dtype=bool),
                )

        class Reader:
            def read(self, profile):
                return {
                    node.name: ServiceRuntimeState(
                        {"cluster2": 1, "cluster3": 1},
                        {"cluster2": 1, "cluster3": 1},
                        None,
                        None,
                        False,
                        False,
                    )
                    for node in profile.nodes
                }

        self_profile = self.advanced
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = create_bundle(
                root / "bundle",
                model.state_dict(),
                scaler,
                self.advanced,
                adjacency,
                low_capacity,
                config,
                {"schema_version": "headlight.learned-edges/v1", "edges": []},
                self._training_metadata(),
            )
            session = _Session([202])
            actuator = PlacementActuator(
                "http://actions/placements",
                "trainticket",
                session=session,
            )
            runtime = ControllerRuntime(
                bundle,
                model,
                torch.device("cpu"),
                Collector(),
                Reader(),
                ForecastPlacementPolicy(self.advanced, low_capacity),
                actuator,
                "lstm",
                root / "actions.jsonl",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                for sample_index in range(24):
                    result = runtime.run_once(1000 + sample_index * 15)
            self.assertIsNotNone(result["inference_duration_ms"])
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(runtime.buffers["ts-auth"][-1][BASE_FEATURES.index("request_rate")], 24)
            rows = [
                json.loads(line)
                for line in (root / "actions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(row["prediction"] for row in rows))
            self.assertTrue(any(row["desired_replicas"] != row["previous_replicas"] for row in rows))
            streamed = [json.loads(line.split("HEADLIGHT_ACTION_JSON ", 1)[1])
                        for line in output.getvalue().splitlines() if line.startswith("HEADLIGHT_ACTION_JSON ")]
            self.assertEqual(streamed, rows)
            failed_output = io.StringIO()
            with redirect_stdout(failed_output), patch.object(actuator, "apply", side_effect=RuntimeError("actuation disconnected")):
                with self.assertRaisesRegex(RuntimeError, "actuation disconnected"):
                    runtime.run_once(1360)
            proposals = [json.loads(line.split("HEADLIGHT_PROPOSAL_JSON ", 1)[1])
                         for line in failed_output.getvalue().splitlines() if line.startswith("HEADLIGHT_PROPOSAL_JSON ")]
            self.assertEqual(len(proposals), 18)
            self.assertTrue(all(record["timestamp"] == 1360 for record in proposals))
            self.assertTrue(all("desired_replicas" in record and "prediction" in record for record in proposals))

    def test_headlight_and_lstm_load_same_frozen_policy_configuration(self) -> None:
        headlight_policy = ForecastPlacementPolicy(self.advanced, self._capacity())
        lstm_policy = ForecastPlacementPolicy(self.advanced, self._capacity())
        self.assertEqual(headlight_policy.policy_config.configuration_hash, lstm_policy.policy_config.configuration_hash)
        self.assertEqual(headlight_policy.policy_config.payload["status"], "frozen-pre-execution")
        self.assertFalse(set(PolicyConfig.load().payload["forbidden_seed_set"]) & set(PolicyConfig.load().payload["provenance"]["calibration_trace"]["seeds"]))


if __name__ == "__main__":
    unittest.main()
