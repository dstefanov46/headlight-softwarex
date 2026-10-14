#!/usr/bin/env python3
"""Train and gate Headlight on the synthetic benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific message
    raise SystemExit(
        "PyTorch is required for synthetic training; install ai-module/ai-module/requirements.txt"
    ) from exc

from headlight.models_v2 import build_headlight_v2_model, model_parameter_count  # noqa: E402
from headlight.synthetic_v2 import (  # noqa: E402
    NODE_ORDER,
    iter_window_batches,
    load_manifest,
    split_records,
    validate_dataset_files,
)
from headlight.training import seed_everything  # noqa: E402
from headlight.training_v2 import (  # noqa: E402
    v2_multitask_loss,
    validate_v2_training_seed_partitions,
)


VARIANTS = ("full", "graph-free")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_checkpoint(path_value: str, report_path: Path) -> Path:
    path = Path(path_value)
    for candidate in (path, REPO_ROOT / path, report_path.parent / path.name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint referenced by report does not exist: {path_value}")


def _load_run(root: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = root / str(record["path"])
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files if name != "metadata_json"}


def _model(
    root: Path,
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = _load_json(root / "model-config.json")
    graph = _load_json(root / "adjacency.json")
    matrix = np.asarray(graph["matrix"], dtype=np.float32)
    observed = np.asarray(graph["observed_mask"], dtype=np.float32)
    if variant == "graph-free":
        matrix = np.eye(len(matrix), dtype=np.float32)
        observed = np.eye(len(matrix), dtype=np.float32)
    model = build_headlight_v2_model(
        config,
        torch.tensor(matrix, dtype=torch.float32, device=device),
        torch.tensor(observed, dtype=torch.float32, device=device),
    ).to(device)
    return model, config


def _tensor_batch(
    batch: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    graph_enabled: bool,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, array in batch.items():
        if name == "boundary_mask":
            values[name] = torch.tensor(array, dtype=torch.bool, device=device)
        else:
            values[name] = torch.tensor(array, dtype=torch.float32, device=device)
    if not graph_enabled:
        values["edge_features"].zero_()
    return values


def _forward(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return model(
        batch["node_features"],
        batch["edge_features"],
        batch["candidate_replicas"],
        batch["candidate_placement"],
        batch["readiness"],
    )


def _loss(
    model: torch.nn.Module,
    output: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return v2_multitask_loss(
        model,
        dict(output),
        batch["demand_target"],
        latency_target=batch["latency_target"],
        latency_weights=1.0 + 64.0 * batch["boundary_mask"].to(dtype=torch.float32),
        sla_target=batch["sla_target"],
        reliability_target=batch["reliability_target"],
        demand_weight=1.0,
        performance_weight=1.0,
        sla_weight=0.5,
        reliability_weight=0.5,
        ranking_weight=0.5,
        residual_l1_weight=1e-4,
        demand_scale_rps=6.0,
        latency_scale_ms=1000.0,
    )


class MetricTotals:
    def __init__(self) -> None:
        self.samples = 0
        self.demand_absolute = 0.0
        self.demand_count = 0
        self.q90_covered = 0
        self.latency_absolute = 0.0
        self.latency_target_total = 0.0
        self.latency_count = 0
        self.sla_squared = 0.0
        self.sla_count = 0
        self.reliability_squared = 0.0
        self.reliability_count = 0
        self.ranking_correct = 0
        self.ranking_count = 0
        self.boundary_absolute = 0.0
        self.boundary_count = 0
        self.selected_unsafe = 0
        self.selected_count = 0
        self.selected_replicas = 0.0
        self.selected_ceiling = 0
        self.threshold_unsafe = 0
        self.threshold_replicas = 0.0
        self.loss_total = 0.0

    def update(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            q50 = output["demand_q50"]
            q90 = output["demand_q90"]
            demand = batch["demand_target"]
            performance = output["candidate_performance"]
            latency = performance["predicted_p95_latency_ms"]
            probability = performance["sla_violation_probability"]
            success_probability = performance["success_probability"]
            actual_latency = batch["latency_target"]
            actual_sla = batch["sla_target"]
            actual_reliability = batch["reliability_target"]
            boundary = batch["boundary_mask"]
            count = int(q50.shape[0])
            self.samples += count
            self.loss_total += float(loss.item()) * count
            self.demand_absolute += float(torch.abs(q50 - demand).sum().item())
            self.demand_count += demand.numel()
            self.q90_covered += int((demand <= q90).sum().item())
            self.latency_absolute += float(torch.abs(latency - actual_latency).sum().item())
            self.latency_target_total += float(torch.abs(actual_latency).sum().item())
            self.latency_count += actual_latency.numel()
            self.sla_squared += float(torch.square(probability - actual_sla).sum().item())
            self.sla_count += actual_sla.numel()
            self.reliability_squared += float(
                torch.square(success_probability - actual_reliability).sum().item()
            )
            self.reliability_count += actual_reliability.numel()
            if torch.any(boundary):
                self.boundary_absolute += float(torch.abs(latency[boundary] - actual_latency[boundary]).sum().item())
                self.boundary_count += int(boundary.sum().item())

            # Policy-v2-like candidate scoring balances resource use and calibrated
            # SLA risk. Oracle labels are used only after selection.
            replicas = batch["candidate_replicas"].mean(dim=-1)
            max_probability = probability.amax(dim=(-1, -2))
            reliability_penalty = 1.0 - success_probability.amin(dim=(-1, -2))
            score = 0.25 * replicas / 14.0 + max_probability + 0.5 * reliability_penalty
            selected = torch.argmin(score, dim=1)
            rows = torch.arange(count, device=selected.device)
            selected_sla = actual_sla[rows, selected].amax(dim=(-1, -2))
            selected_replicas = batch["candidate_replicas"][rows, selected]
            self.selected_unsafe += int((selected_sla >= 0.5).sum().item())
            self.selected_count += count
            self.selected_replicas += float(selected_replicas.mean(dim=-1).sum().item())
            ceiling_fraction = (selected_replicas >= 13.5).to(dtype=torch.float32).mean(dim=-1)
            self.selected_ceiling += int((ceiling_fraction >= 0.25).sum().item())
            actual_cost = (
                actual_sla.amax(dim=(-1, -2))
                + 0.25 * (actual_latency / 1000.0).amax(dim=(-1, -2))
            )
            self.ranking_correct += int(
                (selected == torch.argmin(actual_cost, dim=1)).sum().item()
            )
            self.ranking_count += count

            latest = batch["node_features"][:, -1]
            # Input features are scaled; p95 index 4 and CPU-utilization index 2.
            emergency = (latest[..., 4].amax(dim=-1) > 0.30) | (latest[..., 2].amax(dim=-1) > 0.70)
            current = torch.zeros_like(selected)
            totals = batch["candidate_replicas"].mean(dim=-1)
            larger = totals > totals[:, :1] + 0.05
            larger_score = torch.where(larger, totals, torch.full_like(totals, math.inf))
            scale_up = torch.argmin(larger_score, dim=1)
            threshold_selected = torch.where(emergency, scale_up, current)
            threshold_sla = actual_sla[rows, threshold_selected].amax(dim=(-1, -2))
            threshold_replicas = batch["candidate_replicas"][rows, threshold_selected]
            self.threshold_unsafe += int((threshold_sla >= 0.5).sum().item())
            self.threshold_replicas += float(threshold_replicas.mean(dim=-1).sum().item())

    def result(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "loss": self.loss_total / max(1, self.samples),
            "demand_q50_mae_rps": self.demand_absolute / max(1, self.demand_count),
            "demand_q50_normalized_mae": self.demand_absolute / max(1, self.demand_count) / 6.0,
            "demand_q90_coverage": self.q90_covered / max(1, self.demand_count),
            "candidate_p95_mae_ms": self.latency_absolute / max(1, self.latency_count),
            "candidate_p95_normalized_mae": self.latency_absolute
            / max(1.0, self.latency_target_total),
            "sla_brier_score": self.sla_squared / max(1, self.sla_count),
            "reliability_brier_score": self.reliability_squared
            / max(1, self.reliability_count),
            "candidate_top1_accuracy": self.ranking_correct
            / max(1, self.ranking_count),
            "boundary_p95_mae_ms": self.boundary_absolute / max(1, self.boundary_count),
            "boundary_labels": self.boundary_count,
            "policy_selected_sla_rate": self.selected_unsafe / max(1, self.selected_count),
            "policy_mean_replicas_per_service": self.selected_replicas / max(1, self.selected_count),
            "policy_ceiling_run_rate": self.selected_ceiling / max(1, self.selected_count),
            "threshold_selected_sla_rate": self.threshold_unsafe / max(1, self.selected_count),
            "threshold_mean_replicas_per_service": self.threshold_replicas / max(1, self.selected_count),
        }


def _evaluate(
    model: torch.nn.Module,
    root: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    lookback_steps: int,
    batch_size: int,
    device: torch.device,
    graph_enabled: bool,
    max_batches_per_run: int | None,
) -> dict[str, float | int]:
    model.eval()
    totals = MetricTotals()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for record in records:
            run = _load_run(root, record)
            for batch_index, arrays in enumerate(
                iter_window_batches(
                    run,
                    lookback_steps=lookback_steps,
                    batch_size=batch_size,
                    rng=rng,
                    shuffle=False,
                )
            ):
                if max_batches_per_run is not None and batch_index >= max_batches_per_run:
                    break
                batch = _tensor_batch(arrays, device, graph_enabled=graph_enabled)
                output = _forward(model, batch)
                loss, _ = _loss(model, output, batch)
                totals.update(output, batch, loss)
    model.train()
    return totals.result()


def _train_one(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    variant: str,
    initialization_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    max_training_batches_per_run: int | None,
    initial_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    seed_everything(initialization_seed)
    model, config = _model(root, variant, device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    graph_enabled = variant == "full"
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    fitting = split_records(manifest, "fitting")
    calibration = split_records(manifest, "calibration")
    rng = np.random.default_rng(initialization_seed)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    if initial_state is not None:
        initial_metrics = _evaluate(
            model,
            root,
            calibration,
            lookback_steps=int(manifest["config"]["lookback_steps"]),
            batch_size=batch_size,
            device=device,
            graph_enabled=graph_enabled,
            max_batches_per_run=None,
        )
        best_loss = float(initial_metrics["loss"])
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        history.append(
            {"epoch": 0, "training_loss": None, "calibration": initial_metrics}
        )
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(rng.permutation(len(fitting)))
        total_loss = 0.0
        total_samples = 0
        for record_index in order:
            run = _load_run(root, fitting[int(record_index)])
            for batch_index, arrays in enumerate(
                iter_window_batches(
                    run,
                    lookback_steps=int(manifest["config"]["lookback_steps"]),
                    batch_size=batch_size,
                    rng=rng,
                    shuffle=True,
                )
            ):
                if (
                    max_training_batches_per_run is not None
                    and batch_index >= max_training_batches_per_run
                ):
                    break
                batch = _tensor_batch(arrays, device, graph_enabled=graph_enabled)
                optimizer.zero_grad(set_to_none=True)
                output = _forward(model, batch)
                loss, _ = _loss(model, output, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                count = int(batch["node_features"].shape[0])
                total_loss += float(loss.item()) * count
                total_samples += count
        calibration_metrics = _evaluate(
            model,
            root,
            calibration,
            lookback_steps=int(manifest["config"]["lookback_steps"]),
            batch_size=batch_size,
            device=device,
            graph_enabled=graph_enabled,
            max_batches_per_run=None,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / max(1, total_samples),
                "calibration": calibration_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "event": "epoch-complete",
                    "variant": variant,
                    "initialization_seed": initialization_seed,
                    "epoch": epoch,
                    "training_loss": history[-1]["training_loss"],
                    "calibration_loss": calibration_metrics["loss"],
                    "calibration_demand_nmae": calibration_metrics[
                        "demand_q50_normalized_mae"
                    ],
                    "calibration_latency_nmae": calibration_metrics[
                        "candidate_p95_normalized_mae"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        calibration_loss = float(calibration_metrics["loss"])
        if calibration_loss < best_loss - 1e-5:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    holdout = _evaluate(
        model,
        root,
        split_records(manifest, "holdout"),
        lookback_steps=int(manifest["config"]["lookback_steps"]),
        batch_size=batch_size,
        device=device,
        graph_enabled=graph_enabled,
        max_batches_per_run=None,
    )
    return (
        {
            "variant": variant,
            "initialization_seed": initialization_seed,
            "best_epoch": best_epoch,
            "best_calibration_loss": best_loss,
            "epochs_completed": len(history),
            "parameter_count": model_parameter_count(model, trainable_only=True),
            "duration_seconds": time.perf_counter() - started,
            "model_config": config,
            "history": history,
            "holdout": holdout,
        },
        best_state,
    )


def _acceptance(full: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    full_metrics = full["holdout"]
    baseline_metrics = baseline["holdout"]
    gates = {
        "demand_normalized_mae_at_most_0_20": float(full_metrics["demand_q50_normalized_mae"]) <= 0.20,
        "latency_normalized_mae_at_most_0_35": float(full_metrics["candidate_p95_normalized_mae"]) <= 0.35,
        "sla_brier_at_most_0_20": float(full_metrics["sla_brier_score"]) <= 0.20,
        "q90_coverage_between_0_75_and_0_98": 0.75 <= float(full_metrics["demand_q90_coverage"]) <= 0.98,
        "boundary_mae_at_most_300ms": float(full_metrics["boundary_p95_mae_ms"]) <= 300.0,
        "graph_beats_graph_free_demand_by_3_percent": float(full_metrics["demand_q50_mae_rps"]) <= 0.97 * float(baseline_metrics["demand_q50_mae_rps"]),
        "policy_sla_rate_at_most_threshold": float(full_metrics["policy_selected_sla_rate"]) <= float(full_metrics["threshold_selected_sla_rate"]),
        "policy_ceiling_rate_below_0_10": float(full_metrics["policy_ceiling_run_rate"]) < 0.10,
    }
    return {"passed": all(gates.values()), "gates": gates}


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("initialization seeds must not be empty")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic-headlight-v2" / "large-v1" / "manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic-headlight-v2" / "training-v1",
    )
    parser.add_argument("--initialization-seeds", type=_parse_seeds, default=(601, 602, 603))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reuse-checkpoints-from",
        type=Path,
        help="re-evaluate selected checkpoints from an earlier training output directory",
    )
    parser.add_argument(
        "--initialize-from-report",
        type=Path,
        help="fine-tune from selected checkpoints in a compatible training report",
    )
    parser.add_argument(
        "--max-training-batches-per-run",
        type=int,
        default=2,
        help="sample at most this many shuffled batches from each fitting run per epoch",
    )
    parser.add_argument("--require-acceptance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.dataset.resolve()
    validate_dataset_files(manifest_path)
    manifest = load_manifest(manifest_path)
    fitting_seeds = [int(record["seed"]) for record in split_records(manifest, "fitting")]
    calibration_seeds = [int(record["seed"]) for record in split_records(manifest, "calibration")]
    holdout_seeds = [int(record["seed"]) for record in split_records(manifest, "holdout")]
    for seed in args.initialization_seeds:
        validate_v2_training_seed_partitions(
            fitting_seeds=fitting_seeds,
            calibration_seeds=calibration_seeds,
            holdout_seeds=holdout_seeds,
            initialization_seed=seed,
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_checkpoints_from is not None:
        source_dir = args.reuse_checkpoints_from.resolve()
        source = _load_json(source_dir / "training-report.json")
        if source.get("dataset_hash") != manifest["dataset_hash"]:
            raise ValueError("checkpoint report dataset hash does not match the requested dataset")
        selected: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            prior = copy.deepcopy(source["selected"][variant])
            checkpoint = Path(prior["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = REPO_ROOT / checkpoint
            model, _ = _model(manifest_path.parent, variant, device)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            prior["holdout"] = _evaluate(
                model,
                manifest_path.parent,
                split_records(manifest, "holdout"),
                lookback_steps=int(manifest["config"]["lookback_steps"]),
                batch_size=args.batch_size,
                device=device,
                graph_enabled=variant == "full",
                max_batches_per_run=None,
            )
            output_checkpoint = args.output_dir / checkpoint.name
            torch.save({name: value.detach().cpu() for name, value in state.items()}, output_checkpoint)
            prior["checkpoint"] = str(output_checkpoint)
            prior["reevaluated_from"] = str(checkpoint)
            selected[variant] = prior
        acceptance = _acceptance(selected["full"], selected["graph-free"])
        final = {
            "schema_version": "headlight.synthetic-training-report/v1",
            "dataset": str(manifest_path),
            "dataset_hash": manifest["dataset_hash"],
            "device": str(device),
            "initialization_seeds": source["initialization_seeds"],
            "runs": source["runs"],
            "selected": selected,
            "acceptance": acceptance,
            "reevaluated_from": str(source_dir),
        }
        report_path = args.output_dir / "training-report.json"
        report_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), "acceptance": acceptance}, sort_keys=True))
        return 0 if acceptance["passed"] or not args.require_acceptance else 2
    runs: list[dict[str, Any]] = []
    states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    initialization_report = (
        _load_json(args.initialize_from_report.resolve())
        if args.initialize_from_report is not None
        else None
    )
    if initialization_report is not None and initialization_report.get("dataset_hash") != manifest["dataset_hash"]:
        raise ValueError("initialization report dataset hash does not match the requested dataset")
    for variant in VARIANTS:
        for seed in args.initialization_seeds:
            print(
                json.dumps(
                    {
                        "event": "training-started",
                        "variant": variant,
                        "initialization_seed": seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            report, state = _train_one(
                manifest_path.parent,
                manifest,
                variant=variant,
                initialization_seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                device=device,
                max_training_batches_per_run=args.max_training_batches_per_run,
                initial_state=(
                    torch.load(
                        _resolve_checkpoint(
                            str(initialization_report["selected"][variant]["checkpoint"]),
                            args.initialize_from_report.resolve(),
                        ),
                        map_location=device,
                        weights_only=True,
                    )
                    if initialization_report is not None
                    else None
                ),
            )
            if initialization_report is not None:
                report["initialized_from_report"] = str(
                    args.initialize_from_report.resolve()
                )
            runs.append(report)
            states[(variant, seed)] = state
    selected: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        candidates = [run for run in runs if run["variant"] == variant]
        winner = min(candidates, key=lambda run: (run["best_calibration_loss"], run["initialization_seed"]))
        selected[variant] = copy.deepcopy(winner)
        checkpoint = args.output_dir / f"{variant}-seed-{winner['initialization_seed']}.pt"
        torch.save(states[(variant, int(winner["initialization_seed"]))], checkpoint)
        selected[variant]["checkpoint"] = str(checkpoint)
    acceptance = _acceptance(selected["full"], selected["graph-free"])
    final = {
        "schema_version": "headlight.synthetic-training-report/v1",
        "dataset": str(manifest_path),
        "dataset_hash": manifest["dataset_hash"],
        "device": str(device),
        "initialization_seeds": list(args.initialization_seeds),
        "runs": runs,
        "selected": selected,
        "acceptance": acceptance,
    }
    report_path = args.output_dir / "training-report.json"
    report_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "acceptance": acceptance}, sort_keys=True))
    return 0 if acceptance["passed"] or not args.require_acceptance else 2


if __name__ == "__main__":
    raise SystemExit(main())
