#!/usr/bin/env python3
"""Bounded, reproducible Headlight synthetic model smoke.

This development-only driver writes bundles to temporary directories, prints a
single JSON result to stdout, and never consumes research evaluation data.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.bundle import (  # noqa: E402
    HeadlightV2Bundle,
    create_v2_bundle,
    load_v2_checkpoint,
)
from headlight.graph import build_static_adjacency  # noqa: E402
from headlight.models import build_model  # noqa: E402
from headlight.models_v2 import (  # noqa: E402
    DERIVED_REQUEST_FEATURES,
    REQUIRED_EDGE_FEATURES,
    REQUIRED_NODE_FEATURES,
    estimate_peak_training_memory_bytes,
    model_parameter_count,
)
from headlight.observability import content_sha256  # noqa: E402
from headlight.training import seed_everything  # noqa: E402
from headlight.training_v2 import (  # noqa: E402
    validate_v2_training_seed_partitions,
    v2_evaluation_metrics,
    v2_multitask_loss,
)


NODES = ("caller", "callee", "database")
MODEL_TYPES = ("headlight-v2", "headlight-v2-no-action")


def model_config(model_type: str) -> dict[str, Any]:
    config: dict[str, Any] = {
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


def feature_schema() -> dict[str, Any]:
    return {
        "schema_version": "headlight.features/v2",
        "node_feature_names": list(REQUIRED_NODE_FEATURES),
        "edge_feature_names": list(REQUIRED_EDGE_FEATURES),
        "derived_request_features": list(DERIVED_REQUEST_FEATURES),
        "missing_values": "zero-with-explicit-mask",
    }


def adjacency() -> dict[str, Any]:
    return build_static_adjacency(
        NODES,
        [
            {"source": "caller", "destination": "callee", "weight": 4.0},
            {"source": "callee", "destination": "database", "weight": 2.0},
        ],
    )


def synthetic_fixture(device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(4242)
    node = torch.rand(4, 8, 3, len(REQUIRED_NODE_FEATURES), generator=generator)
    request_index = REQUIRED_NODE_FEATURES.index("request_rate")
    node[..., request_index] = torch.arange(8, dtype=torch.float32)[None, :, None]
    edge = torch.rand(
        4,
        8,
        3,
        3,
        len(REQUIRED_EDGE_FEATURES),
        generator=generator,
    )
    placement = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]],
        ]
    )[None].expand(4, -1, -1, -1).clone()
    replicas = placement.sum(dim=-1)
    readiness = torch.ones(4, 2, 3, 2)
    readiness[:, 1, :, 0] = 0.5
    demand = torch.rand(4, 3, 2, generator=generator) * 5.0 + 1.0
    latency = torch.rand(4, 2, 2, 2, generator=generator) * 800.0 + 100.0
    sla = (torch.rand(4, 2, 2, 2, generator=generator) > 0.7).float()
    return tuple(
        value.to(device)
        for value in (node, edge, replicas, placement, readiness, demand, latency, sla)
    )


def forward(
    model: torch.nn.Module,
    model_type: str,
    fixture: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    node, edge, replicas, placement, readiness, *_targets = fixture
    if model_type == "headlight-v2":
        return model(node, edge, replicas, placement, readiness)
    return model(node, edge)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_once(
    model_type: str,
    device: torch.device,
    fixture: tuple[torch.Tensor, ...],
    epochs: int,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    seed_everything(601)
    torch.use_deterministic_algorithms(True)
    model = build_model(model_config(model_type), adjacency(), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    *_, demand_target, latency_target, sla_target = fixture
    losses: list[float] = []
    synchronize(device)
    started = time.perf_counter()
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = forward(model, model_type, fixture)
        if model_type == "headlight-v2":
            loss, _parts = v2_multitask_loss(
                model,
                output,
                demand_target,
                latency_target=latency_target,
                sla_target=sla_target,
            )
        else:
            loss, _parts = v2_multitask_loss(model, output, demand_target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{model_type} produced a non-finite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    output = forward(model, model_type, fixture)
    for name in ("demand_q50", "demand_q90"):
        if not torch.isfinite(output[name]).all():
            raise RuntimeError(f"{model_type} produced non-finite {name}")
    if not bool(torch.all(output["demand_q90"] >= output["demand_q50"])):
        raise RuntimeError(f"{model_type} produced crossing quantiles")
    performance = output.get("candidate_performance")
    if performance is not None:
        for name in ("predicted_p95_latency_ms", "sla_violation_probability"):
            if not torch.isfinite(performance[name]).all():
                raise RuntimeError(f"{model_type} produced non-finite {name}")
        metrics = v2_evaluation_metrics(
            output,
            demand_target,
            latency_target=latency_target,
            sla_target=sla_target,
        )
    else:
        metrics = v2_evaluation_metrics(output, demand_target)
    details = {
        "training_seconds": elapsed,
        "losses": losses,
        "metrics": metrics,
        "parameters": model_parameter_count(model),
        "estimated_peak_training_memory_bytes": estimate_peak_training_memory_bytes(
            model,
            batch_size=4,
            lookback=8,
            candidate_count=2,
        ),
    }
    return model, output, details


def state_dicts_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name].detach().cpu(), right_state[name].detach().cpu())
        for name in left_state
    )


def outputs_equal(
    expected: dict[str, Any],
    actual: dict[str, Any],
    model_type: str,
) -> bool:
    names = ["demand_q50", "demand_q90"]
    if model_type == "headlight-v2":
        names.extend(
            (
                "candidate_performance.predicted_p95_latency_ms",
                "candidate_performance.sla_violation_probability",
            )
        )
    for name in names:
        if name.startswith("candidate_performance."):
            leaf = name.split(".", maxsplit=1)[1]
            left = expected["candidate_performance"][leaf]
            right = actual["candidate_performance"][leaf]
        else:
            left = expected[name]
            right = actual[name]
        if not torch.equal(left, right):
            return False
    return True


def benchmark(
    model: torch.nn.Module,
    model_type: str,
    fixture: tuple[torch.Tensor, ...],
    device: torch.device,
    iterations: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _index in range(20):
            forward(model, model_type, fixture)
        synchronize(device)
        latencies = []
        for _index in range(iterations):
            started = time.perf_counter_ns()
            forward(model, model_type, fixture)
            synchronize(device)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    latencies.sort()
    return {
        "iterations": iterations,
        "mean_ms": sum(latencies) / len(latencies),
        "p50_ms": latencies[max(0, int(0.50 * len(latencies)) - 1)],
        "p95_ms": latencies[max(0, int(0.95 * len(latencies)) - 1)],
    }


def bundle_round_trip(
    model: torch.nn.Module,
    output: dict[str, Any],
    model_type: str,
    fixture: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[bool, str, dict[str, str]]:
    schema = feature_schema()
    graph = adjacency()
    training = {
        "dataset_hash": content_sha256({"synthetic_fixture": "stage-5", "seed": 4242}),
        "feature_schema_hash": content_sha256(schema),
        "adjacency_hash": content_sha256(graph),
        "fitting_seeds": [301, 302, 303],
        "calibration_seeds": [304],
        "holdout_seeds": [305],
        "initialization_seed": 601,
        "selection_procedure": "stage-5 bounded synthetic smoke only",
        "created_before_evaluation": True,
    }
    with tempfile.TemporaryDirectory(prefix="headlight-v2-smoke-") as temporary:
        root = Path(temporary) / "bundle"
        created = create_v2_bundle(
            root,
            checkpoint_state=model.state_dict(),
            model_config=model_config(model_type),
            feature_schema=schema,
            adjacency=graph,
            calibration={
                "schema_version": "headlight.calibration/v2",
                "method": "synthetic-smoke",
            },
            training_metadata=training,
        )
        loaded = HeadlightV2Bundle.load(
            root,
            expected_model_type=model_type,
            expected_feature_schema_hash=content_sha256(schema),
            expected_adjacency_hash=content_sha256(graph),
        )
        restored = build_model(loaded.model_config, loaded.adjacency, device)
        restored.load_state_dict(load_v2_checkpoint(loaded, device), strict=True)
        restored.eval()
        actual = forward(restored, model_type, fixture)
        return (
            outputs_equal(output, actual, model_type),
            created.bundle_id,
            dict(created.metadata["component_hashes"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, choices=(2, 3), default=3)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fixture = synthetic_fixture(device)
    result: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "epochs": args.epochs,
        "variants": {},
        "leakage_guard": {},
    }
    for model_type in MODEL_TYPES:
        model, output, details = train_once(model_type, device, fixture, args.epochs)
        repeat, _repeat_output, _repeat_details = train_once(
            model_type,
            device,
            fixture,
            args.epochs,
        )
        details["deterministic_repeat_exact"] = state_dicts_equal(model, repeat)
        if not details["deterministic_repeat_exact"]:
            raise RuntimeError(f"{model_type} deterministic repeat differs")
        details["inference"] = benchmark(
            model,
            model_type,
            fixture,
            device,
            args.iterations,
        )
        exact, bundle_id, component_hashes = bundle_round_trip(
            model,
            output,
            model_type,
            fixture,
            device,
        )
        if not exact:
            raise RuntimeError(f"{model_type} bundle output differs after reload")
        details["bundle_roundtrip_exact"] = exact
        details["bundle_id"] = bundle_id
        details["bundle_component_hashes"] = component_hashes
        result["variants"][model_type] = details
    result["leakage_guard"]["allowed_partitions"] = validate_v2_training_seed_partitions(
        fitting_seeds=[301, 302, 303],
        calibration_seeds=[304],
        holdout_seeds=[305],
        initialization_seed=601,
    )
    try:
        validate_v2_training_seed_partitions(
            fitting_seeds=[501],
            calibration_seeds=[304],
            holdout_seeds=[305],
            initialization_seed=601,
        )
    except ValueError as exc:
        result["leakage_guard"]["evaluation_seed_rejected"] = str(exc)
    else:
        raise RuntimeError("evaluation seed 501 was accepted")
    result["peak_process_ram_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    result["peak_vram_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
