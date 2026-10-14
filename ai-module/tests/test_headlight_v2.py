from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"optional AI test dependency is not installed: {exc.name}") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.bundle import HeadlightV2Bundle, create_v2_bundle, load_v2_checkpoint
from headlight.graph import build_static_adjacency
from headlight.models import build_model
from headlight.models_v2 import (
    DERIVED_REQUEST_FEATURES,
    REQUIRED_EDGE_FEATURES,
    REQUIRED_NODE_FEATURES,
    DirectionalGraphBlockV2,
    HeadlightV2Model,
    aggregate_directional_messages,
    estimate_peak_training_memory_bytes,
    model_parameter_count,
)
from headlight.observability import content_sha256
from headlight.profile import load_service_profile
from headlight.policy_v2 import PolicyV2Config, ServiceRuntimeStateV2
from headlight.runtime_v2 import HeadlightV2Runtime
from headlight.training import seed_everything
from headlight.training_v2 import (
    heldout_allocation_boundary_metrics,
    v2_evaluation_metrics,
    v2_multitask_loss,
    validate_v2_training_seed_partitions,
)


NODES = ("caller", "callee", "database")


def model_config(model_type: str = "headlight-v2") -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": "headlight.model-config/v2",
        "model_type": model_type,
        "num_nodes": len(NODES),
        "node_order": list(NODES),
        "node_feature_names": list(REQUIRED_NODE_FEATURES),
        "edge_feature_names": list(REQUIRED_EDGE_FEATURES),
        "service_type_ids": [0, 0, 1],
        "num_service_types": 2,
        "horizons_seconds": [30, 60],
        "endpoint_names": ["preserve", "search"],
        "sampling_interval_seconds": 15,
        "hidden_width": 16,
        "tcn_dilations": [1, 2],
        "tcn_kernel_size": 3,
        "graph_blocks": 2,
        "service_embedding_dim": 4,
        "service_type_embedding_dim": 3,
        "position_encoding_dim": 6,
        "short_slope_window": 3,
        "long_slope_window": 5,
        "num_clusters": 2,
        "readiness_feature_size": 2,
        "readiness_feature_names": ["ready_fraction", "pending_age_seconds"],
        "residual_bound": 0.1,
        "dropout": 0.0,
    }
    if model_type == "headlight-v2-no-action":
        config["capacity_risk_calibration"] = {
            "schema_version": "headlight.capacity-risk-calibration/v2",
            "service_capacity_rps": [10.0, 8.0, 6.0],
            "cluster_capacity_weights": [1.0, 0.9],
            "latency_floor_ms": 50.0,
            "latency_scale_ms": 250.0,
            "risk_temperature": 6.0,
            "risk_bias": -1.0,
            "readiness_penalty": 2.0,
            "endpoint_risk_bias": [0.0, 0.2],
        }
    return config


def feature_schema() -> dict[str, object]:
    return {
        "schema_version": "headlight.features/v2",
        "node_feature_names": list(REQUIRED_NODE_FEATURES),
        "edge_feature_names": list(REQUIRED_EDGE_FEATURES),
        "derived_request_features": list(DERIVED_REQUEST_FEATURES),
        "missing_values": "zero-with-explicit-mask",
    }


def adjacency() -> dict[str, object]:
    return build_static_adjacency(
        NODES,
        [
            {"source": "caller", "destination": "callee", "weight": 4.0},
            {"source": "callee", "destination": "database", "weight": 2.0},
        ],
    )


def training_metadata(
    schema: dict[str, object], graph: dict[str, object]
) -> dict[str, object]:
    return {
        "dataset_hash": "d" * 64,
        "feature_schema_hash": content_sha256(schema),
        "adjacency_hash": content_sha256(graph),
        "fitting_seeds": [301, 302, 303],
        "calibration_seeds": [304],
        "holdout_seeds": [305],
        "initialization_seed": 601,
        "selection_procedure": "synthetic unit-test checkpoint",
        "created_before_evaluation": True,
    }


def inputs(batch: int = 2, steps: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    node = torch.rand(
        batch,
        steps,
        len(NODES),
        len(REQUIRED_NODE_FEATURES),
        generator=generator,
    )
    request_index = REQUIRED_NODE_FEATURES.index("request_rate")
    node[..., request_index] = torch.arange(steps, dtype=torch.float32)[None, :, None]
    edge = torch.rand(
        batch,
        steps,
        len(NODES),
        len(NODES),
        len(REQUIRED_EDGE_FEATURES),
        generator=generator,
    )
    return node, edge


def candidates(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]],
        ]
    )
    placement = first[None].expand(batch, -1, -1, -1).clone()
    replicas = placement.sum(dim=-1)
    readiness = torch.ones(batch, 2, len(NODES), 2)
    readiness[:, 1, :, 0] = 0.5
    return replicas, placement, readiness


class HeadlightV2ModelTests(unittest.TestCase):
    def _model(self, model_type: str = "headlight-v2") -> HeadlightV2Model:
        seed_everything(601)
        model = build_model(model_config(model_type), adjacency(), torch.device("cpu"))
        self.assertIsInstance(model, HeadlightV2Model)
        model.eval()
        return model

    def test_reversing_temporal_order_changes_predictions(self) -> None:
        model = self._model()
        node, edge = inputs()
        forward = model(node, edge)["demand_q90"]
        reversed_time = model(torch.flip(node, dims=(1,)), torch.flip(edge, dims=(1,)))["demand_q90"]
        self.assertFalse(torch.allclose(forward, reversed_time))

    def test_directional_graph_paths_aggregate_opposite_ends(self) -> None:
        caller_messages = torch.zeros(1, 2, 2, 1)
        backpressure_messages = torch.zeros(1, 2, 2, 1)
        caller_messages[0, 0, 1, 0] = 2.0
        backpressure_messages[0, 0, 1, 0] = 3.0
        directed = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        caller_to_callee, callee_to_caller = aggregate_directional_messages(
            caller_messages,
            backpressure_messages,
            directed,
        )
        self.assertEqual(caller_to_callee[0, :, 0].tolist(), [0.0, 2.0])
        self.assertEqual(callee_to_caller[0, :, 0].tolist(), [3.0, 0.0])

    def test_edge_missing_stale_masks_route_to_both_correct_paths(self) -> None:
        block = DirectionalGraphBlockV2(8, REQUIRED_EDGE_FEATURES, 0.0)
        values = torch.arange(len(REQUIRED_EDGE_FEATURES), dtype=torch.float32)
        caller, backpressure = block.split_edge_features(values)
        self.assertEqual(
            caller.tolist(),
            [
                float(REQUIRED_EDGE_FEATURES.index("call_rate")),
                float(REQUIRED_EDGE_FEATURES.index("call_ratio")),
                float(REQUIRED_EDGE_FEATURES.index("cross_cluster")),
                float(REQUIRED_EDGE_FEATURES.index("missing")),
                float(REQUIRED_EDGE_FEATURES.index("stale")),
            ],
        )
        self.assertEqual(
            backpressure.tolist(),
            [
                float(REQUIRED_EDGE_FEATURES.index("latency_ms")),
                float(REQUIRED_EDGE_FEATURES.index("error_rate")),
                float(REQUIRED_EDGE_FEATURES.index("cross_cluster")),
                float(REQUIRED_EDGE_FEATURES.index("missing")),
                float(REQUIRED_EDGE_FEATURES.index("stale")),
            ],
        )
        model = self._model()
        node, edge = inputs()
        baseline = model(node, edge)["demand_q90"]
        masked_node = node.clone()
        masked_node[:, -1, :, REQUIRED_NODE_FEATURES.index("request_rate_missing")] = 1.0
        masked_node[:, -1, :, REQUIRED_NODE_FEATURES.index("request_rate_stale")] = 1.0
        self.assertFalse(torch.allclose(baseline, model(masked_node, edge)["demand_q90"]))
        masked_edge = edge.clone()
        masked_edge[:, -1, :, :, REQUIRED_EDGE_FEATURES.index("missing")] = 1.0
        masked_edge[:, -1, :, :, REQUIRED_EDGE_FEATURES.index("stale")] = 1.0
        self.assertFalse(torch.allclose(baseline, model(node, masked_edge)["demand_q90"]))

    def test_observed_graph_prior_bounds_a_sparse_local_residual(self) -> None:
        model = self._model()
        self.assertEqual(float(model.residual_sparsity_loss().detach()), 0.0)
        self.assertEqual(float(model.adjacency.allowed_mask[2, 0]), 0.0)
        with torch.no_grad():
            model.adjacency.residual_logits[2, 0] = 100.0
            model.adjacency.residual_logits[0, 1] = 1.0
        residual = model.adjacency.residual().detach()
        self.assertEqual(float(residual[2, 0]), 0.0)
        self.assertGreater(float(residual[0, 1]), 0.0)
        torch.testing.assert_close(
            model.effective_adjacency().sum(dim=1),
            torch.ones(len(NODES)),
        )

    def test_candidate_actions_enter_only_the_full_learned_model(self) -> None:
        full = self._model("headlight-v2")
        ablation = self._model("headlight-v2-no-action")
        node, edge = inputs()
        replica, placement, readiness = candidates()
        full_output = full(node, edge, replica, placement, readiness)
        self.assertEqual(full_output["candidate_performance"]["source"], "learned-action-conditioned")
        self.assertEqual(
            full_output["candidate_performance"]["success_probability"].shape,
            full_output["candidate_performance"]["sla_violation_probability"].shape,
        )
        first = full_output["candidate_performance"]["predicted_p95_latency_ms"][:, 0]
        second = full_output["candidate_performance"]["predicted_p95_latency_ms"][:, 1]
        self.assertFalse(torch.allclose(first, second))

        full_reordered = full(node, edge, replica.flip(1), placement.flip(1), readiness.flip(1))
        torch.testing.assert_close(full_output["demand_q90"], full_reordered["demand_q90"])
        with self.assertRaisesRegex(ValueError, "excludes candidates"):
            ablation(node, edge, replica, placement, readiness)
        ablation_output = ablation(node, edge)
        self.assertIsNone(ablation_output["candidate_performance"])
        surrogate = ablation.score_candidates_with_capacity_surrogate(
            ablation_output["demand_q90"], replica, placement, readiness
        )
        self.assertEqual(surrogate["source"], "calibrated-capacity-surrogate")
        self.assertFalse(
            torch.allclose(
                surrogate["sla_violation_probability"][:, 0],
                surrogate["sla_violation_probability"][:, 1],
            )
        )

    def test_output_scales_preserve_units_without_changing_backbone(self) -> None:
        seed_everything(601)
        unscaled = HeadlightV2Model(
            model_config(),
            torch.tensor(adjacency()["matrix"], dtype=torch.float32),
            torch.tensor(adjacency()["observed_mask"], dtype=torch.float32),
        )
        scaled_config = model_config()
        scaled_config["latency_output_scale_ms"] = 300.0
        seed_everything(601)
        scaled = HeadlightV2Model(
            scaled_config,
            torch.tensor(adjacency()["matrix"], dtype=torch.float32),
            torch.tensor(adjacency()["observed_mask"], dtype=torch.float32),
        )
        node, edge = inputs()
        replica, placement, readiness = candidates()
        plain = unscaled(node, edge, replica, placement, readiness)
        adjusted = scaled(node, edge, replica, placement, readiness)
        torch.testing.assert_close(adjusted["demand_q50"], plain["demand_q50"])
        torch.testing.assert_close(
            adjusted["candidate_performance"]["predicted_p95_latency_ms"],
            plain["candidate_performance"]["predicted_p95_latency_ms"] * 300.0,
        )

        demand_config = model_config()
        demand_config["demand_output_scale_rps"] = 6.0
        seed_everything(601)
        demand_scaled = HeadlightV2Model(
            demand_config,
            torch.tensor(adjacency()["matrix"], dtype=torch.float32),
            torch.tensor(adjacency()["observed_mask"], dtype=torch.float32),
        )
        demand_output = demand_scaled(node, edge, replica, placement, readiness)
        torch.testing.assert_close(demand_output["demand_q50"], plain["demand_q50"] * 6.0)

    def test_latency_weights_emphasize_selected_candidate_targets(self) -> None:
        model = self._model("headlight-v2")
        node, edge = inputs()
        replica, placement, readiness = candidates()
        output = model(node, edge, replica, placement, readiness)
        latency = output["candidate_performance"]["predicted_p95_latency_ms"].detach()
        target = latency.clone()
        target[..., 0, 0] += 1000.0
        target[..., 0, 1] += 100.0
        weights = torch.ones_like(target)
        weights[..., 0, 0] = 10.0
        demand_target = output["demand_q50"].detach()
        sla_target = torch.zeros_like(target)
        _, plain = v2_multitask_loss(
            model,
            output,
            demand_target,
            latency_target=target,
            sla_target=sla_target,
            latency_scale_ms=1000.0,
        )
        _, weighted = v2_multitask_loss(
            model,
            output,
            demand_target,
            latency_target=target,
            latency_weights=weights,
            sla_target=sla_target,
            latency_scale_ms=1000.0,
        )
        self.assertGreater(weighted["p95_latency"], plain["p95_latency"])

    def test_candidate_batches_cannot_broadcast_observations(self) -> None:
        full = self._model("headlight-v2")
        ablation = self._model("headlight-v2-no-action")
        node, edge = inputs(batch=1)
        replica, placement, readiness = candidates(batch=2)
        with self.assertRaisesRegex(ValueError, "candidate batch size"):
            full(node, edge, replica, placement, readiness)
        demand = ablation(node, edge)["demand_q90"]
        with self.assertRaisesRegex(ValueError, "candidate batch size"):
            ablation.score_candidates_with_capacity_surrogate(
                demand, replica, placement, readiness
            )

    def test_v2_build_rejects_misordered_adjacency(self) -> None:
        graph = adjacency()
        graph["node_order"] = list(reversed(NODES))
        with self.assertRaisesRegex(ValueError, "adjacency node order"):
            build_model(model_config(), graph, torch.device("cpu"))

    def test_non_crossing_metrics_and_boundary_evaluation(self) -> None:
        model = self._model()
        node, edge = inputs()
        replica, placement, readiness = candidates()
        output = model(node, edge, replica, placement, readiness)
        self.assertTrue(torch.all(output["demand_q90"] >= output["demand_q50"]))
        demand_target = 0.5 * (output["demand_q50"] + output["demand_q90"])
        latency_target = output["candidate_performance"]["predicted_p95_latency_ms"] + 5.0
        sla_target = (
            output["candidate_performance"]["sla_violation_probability"] > 0.5
        ).to(torch.float32)
        reliability_target = output["candidate_performance"]["success_probability"].detach()
        loss, parts = v2_multitask_loss(
            model,
            output,
            demand_target,
            latency_target=latency_target,
            sla_target=sla_target,
            reliability_target=reliability_target,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("demand_wis", parts)
        self.assertIn("reliability_bce", parts)
        self.assertIn("candidate_ranking", parts)
        self.assertTrue(torch.isfinite(parts["candidate_ranking"]))
        metrics = v2_evaluation_metrics(
            output,
            demand_target,
            latency_target=latency_target,
            sla_target=sla_target,
            calibration_bins=5,
        )
        self.assertEqual(metrics["quantile_crossing_rate"], 0.0)
        self.assertIn("sla_brier_score", metrics)
        self.assertIn("sla_expected_calibration_error", metrics)
        boundary = torch.zeros_like(latency_target, dtype=torch.bool)
        boundary[:, 0] = True
        heldout = heldout_allocation_boundary_metrics(
            output,
            latency_target,
            sla_target,
            boundary,
            calibration_bins=5,
        )
        self.assertGreater(heldout["boundary_samples"], 0)

    def test_seeded_construction_is_deterministic(self) -> None:
        node, edge = inputs()
        def train_once() -> torch.Tensor:
            model = self._model()
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            output = model(node, edge)
            target = torch.ones_like(output["demand_q50"])
            loss, _ = v2_multitask_loss(model, output, target)
            loss.backward()
            optimizer.step()
            model.eval()
            return model(node, edge)["demand_q90"].detach()

        first = train_once()
        second = train_once()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(
            validate_v2_training_seed_partitions(
                fitting_seeds=[301, 302],
                calibration_seeds=[304],
                holdout_seeds=[305],
                initialization_seed=601,
            )["initialization"],
            601,
        )
        with self.assertRaisesRegex(ValueError, "evaluation seeds"):
            validate_v2_training_seed_partitions(
                fitting_seeds=[501],
                calibration_seeds=[304],
                holdout_seeds=[305],
                initialization_seed=601,
            )

    def test_bundle_save_load_preserves_outputs_and_compatibility(self) -> None:
        model = self._model()
        node, edge = inputs(batch=1)
        replica, placement, readiness = candidates(batch=1)
        expected = model(node, edge, replica, placement, readiness)
        schema = feature_schema()
        graph = adjacency()
        training = training_metadata(schema, graph)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            created = create_v2_bundle(
                root,
                checkpoint_state=model.state_dict(),
                model_config=model_config(),
                feature_schema=schema,
                adjacency=graph,
                calibration={"schema_version": "headlight.calibration/v2", "method": "synthetic"},
                training_metadata=training,
            )
            loaded = HeadlightV2Bundle.load(
                root,
                expected_model_type="headlight-v2",
                expected_feature_schema_hash=content_sha256(schema),
                expected_adjacency_hash=content_sha256(graph),
            )
            self.assertEqual(created.bundle_id, loaded.bundle_id)
            restored = build_model(loaded.model_config, loaded.adjacency, torch.device("cpu"))
            restored.load_state_dict(load_v2_checkpoint(loaded, torch.device("cpu")), strict=True)
            restored.eval()
            actual = restored(node, edge, replica, placement, readiness)
            torch.testing.assert_close(expected["demand_q50"], actual["demand_q50"], rtol=0, atol=0)
            torch.testing.assert_close(
                expected["candidate_performance"]["sla_violation_probability"],
                actual["candidate_performance"]["sla_violation_probability"],
                rtol=0,
                atol=0,
            )
            with self.assertRaisesRegex(ValueError, "feature schema is incompatible"):
                HeadlightV2Bundle.load(root, expected_feature_schema_hash="0" * 64)

    def test_offline_verifier_loads_bundle_and_emits_shadow_decisions(self) -> None:
        profile = load_service_profile(
            REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json"
        )
        config = model_config()
        config.update(
            num_nodes=len(profile.nodes),
            node_order=list(profile.node_order),
            service_type_ids=[int(node.service_type == "database") for node in profile.nodes],
            lookback_steps=8,
        )
        graph = build_static_adjacency(profile.node_order, [])
        schema = feature_schema()
        model = build_model(config, graph, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_v2_bundle(
                root / "bundle",
                checkpoint_state=model.state_dict(),
                model_config=config,
                feature_schema=schema,
                adjacency=graph,
                calibration={
                    "schema_version": "headlight.calibration/identity-v1",
                    "calibrated": True,
                },
                training_metadata=training_metadata(schema, graph),
            )
            np.savez(
                root / "holdout.npz",
                sample_indices=np.array([7]),
                node_features=np.zeros((8, len(profile.nodes), len(REQUIRED_NODE_FEATURES)), dtype=np.float32),
                edge_source=np.array([0]),
                edge_destination=np.array([1]),
                edge_features=np.zeros((8, 1, len(REQUIRED_EDGE_FEATURES)), dtype=np.float32),
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "config": {"lookback_steps": 8},
                "runs": [{"split": "holdout", "path": "holdout.npz"}],
            }), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "experiments/verify_headlight_v2_live_bundle.py"),
                 "--bundle", str(root / "bundle"), "--dataset-manifest", str(manifest)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            line = next(line for line in result.stdout.splitlines() if line.startswith("HEADLIGHT_V2_OFFLINE_VERIFY_OK "))
            verified = json.loads(line.split(" ", 1)[1])
            self.assertEqual(verified["decision_count"], len(profile.nodes))
            self.assertTrue(verified["shadow_mode"])
            self.assertFalse(verified["actuator_called"])

    def test_diagnostics_score_emergencies_preserve_actions_and_store_replay_inputs(self) -> None:
        # A real checked-in bundle is also exercised in deployment; this small
        # fixture keeps the behavioral regression independent of local artifacts.
        profile = load_service_profile(REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json")
        config = model_config()
        config.update(num_nodes=len(profile.nodes), node_order=list(profile.node_order),
                      service_type_ids=[int(n.service_type == "database") for n in profile.nodes], lookback_steps=8)
        graph = build_static_adjacency(profile.node_order, [])
        schema = feature_schema()
        model = build_model(config, graph, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_v2_bundle(root / "bundle", checkpoint_state=model.state_dict(), model_config=config,
                             feature_schema=schema, adjacency=graph,
                             calibration={"schema_version": "headlight.calibration/identity-v1", "calibrated": True},
                             training_metadata=training_metadata(schema, graph))
            def runtime(directory=None):
                result = HeadlightV2Runtime.from_bundle(
                    root / "bundle", runtime_profile=profile,
                    prometheus_url="http://cluster2.invalid,http://cluster3.invalid",
                    prometheus_clusters=("cluster2", "cluster3"), action_handler_url="http://actions.invalid",
                    policy_config=PolicyV2Config(), run_id="diagnostics-regression", seed=401,
                    diagnostic_dir=directory, device=torch.device("cpu"),
                )
                result.prime(np.zeros((8, 18, len(REQUIRED_NODE_FEATURES)), dtype=np.float32),
                             np.zeros((8, 18, 18, len(REQUIRED_EDGE_FEATURES)), dtype=np.float32))
                return result
            baseline, observed = runtime(), runtime(root / "diagnostics")
            placement = {"cluster2": 1, "cluster3": 1}
            states = {
                n.name: ServiceRuntimeStateV2(
                    desired_replicas=placement, ready_replicas=placement,
                    pending_replicas=dict.fromkeys(placement, 0), pending_age_seconds=dict.fromkeys(placement, 0.0),
                    demand_rps=1.0, cpu_utilization=0.95 if n.name == "ts-auth" else 0.3,
                    queue_depth=0.0, p95_latency_ms=100.0, error_rate=0.0,
                    predictive_fresh=True, time_since_last_action_seconds=600,
                    cluster_cpu_utilization={"cluster2": 0.95 if n.name == "ts-auth" else 0.3},
                ) for n in profile.nodes
            }
            expected = baseline.infer_states(states, timestamp=1010)
            actual = observed.infer_states(states, timestamp=1010)
            self.assertEqual(actual["decisions"], expected["decisions"])
            self.assertEqual(observed.policy.memory, baseline.policy.memory)
            records = [json.loads(line) for line in (root / "diagnostics/decisions.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), len(profile.nodes))
            auth = next(r for r in records if r["decision"]["service"] == "ts-auth")
            self.assertEqual(auth["model_status"], "ok")
            self.assertTrue(auth["decision"]["forecast_quantiles_by_horizon"])
            self.assertTrue(all(c["prediction_available"] for c in auth["decision"]["candidates"]))
            self.assertEqual(auth["final_action"]["action_origin"], "reactive_emergency")
            self.assertEqual(auth["forecast_units"], "requests_per_second")
            self.assertIn("total_objectives", auth)
            with np.load(root / "diagnostics" / auth["replay"]["window"]["path"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["node_window"], np.stack(observed.node_frames)[None])
            with np.load(root / "diagnostics" / auth["replay"]["candidates"]["path"], allow_pickle=False) as saved:
                self.assertEqual(saved["placement"].shape[1], len(auth["decision"]["candidates"]))
            # Diagnostic I/O must not change actions, even when the cap is hit.
            observed.diagnostic_store.max_bytes = observed.diagnostic_store.used_bytes
            with contextlib.redirect_stdout(io.StringIO()) as output:
                actual = observed.infer_states(states, timestamp=1025)
            expected = baseline.infer_states(states, timestamp=1025)
            self.assertEqual(actual["decisions"], expected["decisions"])
            self.assertIn("record_storage_error", output.getvalue())
            self.assertEqual(observed.policy.memory, baseline.policy.memory)
            # A real forward hook injects a model failure; the reactive controller
            # keeps operating and diagnostics explicitly label inference failure.
            def fail(_module, _inputs):
                raise RuntimeError("fixture inference unavailable")
            baseline.model.register_forward_pre_hook(fail)
            observed.model.register_forward_pre_hook(fail)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                actual = observed.infer_states(states, timestamp=1040)
            expected = baseline.infer_states(states, timestamp=1040)
            self.assertEqual(actual["decisions"], expected["decisions"])
            self.assertIn("fixture inference unavailable", output.getvalue())
            self.assertEqual(observed.policy.memory, baseline.policy.memory)

            # Network errors remain errors, but the interval and ambiguous
            # actuator outcome must survive for diagnosis.
            class FailedTransport:
                def post(self, *_args, **_kwargs):
                    raise TimeoutError("fixture actuator timeout")
            observed.session = FailedTransport()
            observed.executor.shadow_mode = False
            observed.shadow_mode = False
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(TimeoutError, "fixture actuator timeout"):
                    observed.infer_states(states, timestamp=1055)
            failed_records = [json.loads(line) for line in output.getvalue().splitlines()
                              if '"event":"HEADLIGHT_V2_DIAGNOSTIC"' in line]
            self.assertEqual(len(failed_records), len(profile.nodes))
            failed_auth = next(r for r in failed_records if r["decision"]["service"] == "ts-auth")
            self.assertEqual(failed_auth["actuator"]["lifecycle"]["status"], "failed")
            self.assertFalse(failed_auth["actuator"]["deployment_confirmed"])
            observed.node_frames.clear()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                unready = observed.infer_states(states, timestamp=1070)
            self.assertFalse(unready["ready"])
            incomplete = [json.loads(line) for line in output.getvalue().splitlines()
                          if '"event":"HEADLIGHT_V2_DIAGNOSTIC"' in line]
            self.assertEqual(len(incomplete), len(profile.nodes))
            self.assertTrue(all(r["model_status"] == "lookback-incomplete" for r in incomplete))

    def test_bundle_rejects_checkpoint_buffers_from_other_components(self) -> None:
        model = self._model()
        schema = feature_schema()
        graph = adjacency()
        mismatched_graph = build_static_adjacency(
            NODES,
            [{"source": "database", "destination": "caller", "weight": 1.0}],
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "derived buffer .* differs"):
                create_v2_bundle(
                    Path(temporary) / "bundle",
                    checkpoint_state=model.state_dict(),
                    model_config=model_config(),
                    feature_schema=schema,
                    adjacency=mismatched_graph,
                    calibration={"schema_version": "headlight.calibration/v2"},
                    training_metadata=training_metadata(schema, mismatched_graph),
                )

    def test_bundle_rejects_evaluation_seeds_in_nested_provenance(self) -> None:
        model = self._model()
        schema = feature_schema()
        graph = adjacency()
        training = training_metadata(schema, graph)
        training["checkpoint_selection"] = {"tuning_seeds": [501]}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "references evaluation seeds"):
                create_v2_bundle(
                    Path(temporary) / "bundle",
                    checkpoint_state=model.state_dict(),
                    model_config=model_config(),
                    feature_schema=schema,
                    adjacency=graph,
                    calibration={"schema_version": "headlight.calibration/v2"},
                    training_metadata=training,
                )

    def test_resource_estimate_is_small_for_synthetic_configuration(self) -> None:
        model = self._model()
        self.assertLess(model_parameter_count(model), 1_000_000)
        estimate = estimate_peak_training_memory_bytes(
            model,
            batch_size=16,
            lookback=24,
            candidate_count=8,
        )
        self.assertLess(estimate, 2 * 1024**3)


if __name__ == "__main__":
    unittest.main()
