"""Deterministic, one-shot training for immutable Headlight/LSTM bundles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .bundle import create_bundle
from .features import (
    MODEL_FEATURES,
    apply_scaler_to_sequences,
    build_sequences,
    chronological_split,
    estimate_capacities,
    fit_training_scaler,
    load_calibration_csv,
)
from .graph import validate_adjacency
from .models import build_model, decode_forecasts, learned_edge_artifact, multitask_loss
from .profile import load_service_profile


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features[indices], dtype=torch.float32),
        torch.tensor(targets[indices], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals = {"request": 0.0, "cpu": 0.0, "p95_latency": 0.0, "sla": 0.0}
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for features, targets in loader:
            raw = model(features.to(device))
            loss, parts = multitask_loss(raw, targets.to(device))
            total_loss += float(loss.item())
            for name, value in parts.items():
                totals[name] += float(value.item())
            batches += 1
    model.train()
    divisor = max(1, batches)
    return total_loss / divisor, {name: value / divisor for name, value in totals.items()}


def _holdout_metrics(
    model: torch.nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(features[indices], dtype=torch.float32, device=device))
        decoded = decode_forecasts(raw).cpu().numpy()
    actual = targets[indices]

    def mae(prediction: np.ndarray, target: np.ndarray) -> float:
        valid = np.isfinite(target)
        return float(np.mean(np.abs(prediction[valid] - target[valid]))) if np.any(valid) else 0.0

    valid_sla = np.isfinite(actual[..., 3])
    classification = (
        float(
            np.mean(
                (decoded[..., 4][valid_sla] >= 0.5)
                == (actual[..., 3][valid_sla] >= 0.5)
            )
        )
        if np.any(valid_sla)
        else 0.0
    )
    return {
        "request_q50_mae": mae(decoded[..., 0], actual[..., 0]),
        "request_q90_mae": mae(decoded[..., 1], actual[..., 0]),
        "cpu_q90_mae": mae(decoded[..., 2], actual[..., 1]),
        "p95_latency_q90_mae": mae(decoded[..., 3], actual[..., 2]),
        "sla_classification_accuracy": classification,
    }


def _measure_inference(
    model: torch.nn.Module,
    example: torch.Tensor,
    device: torch.device,
    repetitions: int = 20,
) -> dict[str, float]:
    model.eval()
    durations = []
    with torch.no_grad():
        model(example)
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(example)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.quantile(durations, 0.95)),
        "samples": repetitions,
    }


def _train_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, torch.Tensor], float, int, list[dict[str, Any]]]:
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epoch_metrics: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_batches = 0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            raw = model(features.to(device))
            loss, _ = multitask_loss(raw, targets.to(device))
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())
            train_batches += 1
        validation_loss, validation_parts = _evaluate(model, validation_loader, device)
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_total / max(1, train_batches),
            "validation_loss": validation_loss,
            "validation_components": validation_parts,
        })
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    return best_state, best_validation_loss, best_epoch, epoch_metrics


def train_offline(args: argparse.Namespace) -> dict[str, Any]:
    if not args.once:
        raise ValueError("training is offline-only; --once is required")
    if args.train_split is not None and not np.isclose(args.train_split, 0.70):
        raise ValueError("the repaired pipeline uses a fixed 70/15/15 chronological split")
    seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    profile = load_service_profile(args.service_profile)
    if args.sampling_interval_seconds != profile.sampling_interval_seconds:
        raise ValueError("training sampling interval differs from the selected service profile")
    dataset_path = Path(args.dataset)
    adjacency_path = Path(args.adjacency)
    series = load_calibration_csv(dataset_path, profile)
    sequences = build_sequences(
        series,
        profile,
        lookback=args.lookback,
        horizon=args.horizon,
        sla_latency_ms=args.sla_latency_ms,
    )
    split = chronological_split(
        sequences.target_timestamps,
        purge_seconds=args.purge_seconds,
    )
    scaler = fit_training_scaler(series, sequences, split.train)
    scaled_features = apply_scaler_to_sequences(sequences, scaler)
    adjacency = json.loads(adjacency_path.read_text(encoding="utf-8"))
    validate_adjacency(adjacency, profile.node_order)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but is unavailable")
    variant = args.variant if args.controller == "headlight" else "per-service-lstm"
    model_config = {
        "model_type": args.controller,
        "num_nodes": len(profile.nodes),
        "input_size": len(MODEL_FEATURES),
        "lookback": args.lookback,
        "horizon": args.horizon,
        "sampling_interval_seconds": args.sampling_interval_seconds,
        "forecast_horizon_seconds": args.horizon * args.sampling_interval_seconds,
        "hidden_width": args.hidden_width,
        "graph_blocks": args.graph_blocks if args.controller == "headlight" else 0,
        "variant": variant,
        "residual_bound": (
            args.residual_bound
            if args.controller == "headlight" and args.variant in {"full", "no_attention"}
            else 0.0
        ),
    }
    model = build_model(model_config, adjacency, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    train_loader = _loader(
        scaled_features,
        sequences.targets,
        split.train,
        args.batch_size,
        True,
        args.seed,
    )
    validation_loader = _loader(
        scaled_features,
        sequences.targets,
        split.validation,
        args.batch_size,
        False,
        args.seed,
    )
    holdout_loader = _loader(
        scaled_features,
        sequences.targets,
        split.holdout,
        args.batch_size,
        False,
        args.seed,
    )
    best_state, best_validation_loss, best_epoch, epoch_metrics = _train_epochs(
        model, optimizer, train_loader, validation_loader, device, args.epochs
    )
    model.load_state_dict(best_state)
    holdout_loss, holdout_parts = _evaluate(model, holdout_loader, device)
    holdout = {
        "loss": holdout_loss,
        "components": holdout_parts,
        **_holdout_metrics(
            model,
            scaled_features,
            sequences.targets,
            split.holdout,
            device,
        ),
    }
    example = torch.tensor(
        scaled_features[split.holdout[-1] : split.holdout[-1] + 1],
        dtype=torch.float32,
        device=device,
    )
    inference = _measure_inference(model, example, device)
    if holdout_loss > args.max_holdout_loss:
        raise RuntimeError(
            f"holdout loss {holdout_loss:.6f} exceeds quality limit {args.max_holdout_loss:.6f}"
        )
    if inference["p95_ms"] > args.max_inference_ms:
        raise RuntimeError(
            f"p95 inference latency {inference['p95_ms']:.3f}ms exceeds {args.max_inference_ms:.3f}ms"
        )
    training_frames = sorted(
        {
            int(frame)
            for sample_index in split.train
            for frame in sequences.input_frame_indices[int(sample_index)]
        }
    )
    capacity = estimate_capacities(
        series,
        profile,
        training_frames,
        sla_latency_ms=args.sla_latency_ms,
    )
    learned_edges = learned_edge_artifact(model, profile.node_order)
    metrics = {
        "seed": args.seed,
        "deterministic_algorithms": True,
        "dataset_sha256": _sha256(dataset_path),
        "adjacency_sha256": _sha256(adjacency_path),
        "profile_hash": profile.profile_hash,
        "samples": {
            "total": len(sequences.target_timestamps),
            "train": len(split.train),
            "validation": len(split.validation),
            "holdout": len(split.holdout),
        },
        "split": {
            "ratios": [0.70, 0.15, 0.15],
            "purge_seconds": split.purge_seconds,
            "first_boundary": split.first_boundary,
            "second_boundary": split.second_boundary,
            "maximum_scaler_timestamp": max(scaler.fitted_frame_timestamps),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "holdout": holdout,
        "inference": inference,
        "epochs": epoch_metrics,
        "model_config": model_config,
    }
    output = Path(args.bundle_output)
    bundle = create_bundle(
        output,
        checkpoint_state=best_state,
        scaler=scaler,
        profile=profile,
        adjacency=adjacency,
        capacity=capacity,
        model_config=model_config,
        learned_edges=learned_edges,
        training_metadata={
            key: value
            for key, value in metrics.items()
            if key not in {"epochs", "inference"}
        },
        overwrite=args.overwrite,
    )
    metrics["bundle_id"] = bundle.bundle_id
    metrics["bundle_path"] = str(output)
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="required: train one frozen bundle and exit")
    parser.add_argument("--controller", choices=["headlight", "lstm"], default="headlight")
    parser.add_argument("--dataset", required=True, help="aligned calibration CSV")
    parser.add_argument("--service-profile", required=True)
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--bundle-output", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--lookback", type=int, choices=[12, 24, 36], default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--sampling-interval-seconds", type=int, default=15)
    parser.add_argument("--purge-seconds", type=int, default=180)
    parser.add_argument("--sla-latency-ms", type=float, default=1000.0)
    parser.add_argument("--graph-blocks", type=int, choices=[1, 2], default=2)
    parser.add_argument("--hidden-width", type=int, choices=[16, 32], default=32)
    parser.add_argument("--learning-rate", type=float, choices=[0.001, 0.0003], default=0.0003)
    parser.add_argument("--residual-bound", type=float, choices=[0.1, 0.3], default=0.3)
    parser.add_argument(
        "--variant",
        choices=["full", "static_adjacency", "no_graph", "no_attention"],
        default="full",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-holdout-loss", type=float, default=float("inf"))
    parser.add_argument("--max-inference-ms", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-split", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--service-file", help=argparse.SUPPRESS)
    parser.add_argument("--cluster-file", help=argparse.SUPPRESS)
    parser.add_argument("--feature-history-folder", help=argparse.SUPPRESS)
    parser.add_argument("--model-folder", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", help=argparse.SUPPRESS)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.once:
        parser.error("--once is required; online retraining is disabled")
    metrics = train_offline(args)
    print(
        json.dumps(
            {
                "bundle_id": metrics["bundle_id"],
                "bundle_path": metrics["bundle_path"],
                "best_validation_loss": metrics["best_validation_loss"],
                "holdout_loss": metrics["holdout"]["loss"],
                "inference_p95_ms": metrics["inference"]["p95_ms"],
            },
            sort_keys=True,
        )
    )
    return 0
