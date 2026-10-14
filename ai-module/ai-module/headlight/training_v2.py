"""Losses, diagnostics, and leak guards for deterministic Headlight training."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as functional

from .models_v2 import HeadlightV2Model


RESEARCH_V3_EVALUATION_SEEDS = frozenset({101, 102, 103, 104, 105})
RESEARCH_V4_EVALUATION_SEEDS = frozenset(range(501, 509))
V2_TRAINING_INITIALIZATION_SEEDS = frozenset(range(601, 609))


def _masked_values(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(target)
    return prediction[valid], target[valid]


def pinball_metric(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    predicted, actual = _masked_values(prediction, target)
    if actual.numel() == 0:
        return prediction.sum() * 0.0
    error = actual - predicted
    return torch.mean(torch.maximum(quantile * error, (quantile - 1.0) * error))


def weighted_interval_score_q50_q90(
    q50: torch.Tensor,
    q90: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """WIS using a symmetric q10 estimate around q50 and the modeled q90."""

    valid = torch.isfinite(target)
    if not torch.any(valid):
        return q50.sum() * 0.0
    median = q50[valid]
    upper = q90[valid]
    actual = target[valid]
    lower = torch.clamp(2.0 * median - upper, min=0.0)
    alpha = 0.20
    interval_score = (
        upper
        - lower
        + (2.0 / alpha) * torch.clamp(lower - actual, min=0.0)
        + (2.0 / alpha) * torch.clamp(actual - upper, min=0.0)
    )
    return torch.mean((0.5 * torch.abs(actual - median) + (alpha / 2.0) * interval_score) / 1.5)


def q90_empirical_coverage(q90: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted, actual = _masked_values(q90, target)
    if actual.numel() == 0:
        return q90.sum() * 0.0
    return torch.mean((actual <= predicted).to(dtype=q90.dtype))


def brier_score(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted, actual = _masked_values(probability, target)
    if actual.numel() == 0:
        return probability.sum() * 0.0
    return torch.mean(torch.square(predicted - actual))


def expected_calibration_error(
    probability: torch.Tensor,
    target: torch.Tensor,
    bins: int = 10,
) -> torch.Tensor:
    if bins <= 1:
        raise ValueError("calibration bins must be greater than one")
    predicted, actual = _masked_values(probability, target)
    if actual.numel() == 0:
        return probability.sum() * 0.0
    total = predicted.numel()
    error = probability.new_tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=predicted.device)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        included = (predicted >= lower) & (
            predicted <= upper if index == bins - 1 else predicted < upper
        )
        if torch.any(included):
            weight = included.sum().to(dtype=predicted.dtype) / total
            error = error + weight * torch.abs(
                predicted[included].mean() - actual[included].mean()
            )
    return error


def v2_multitask_loss(
    model: HeadlightV2Model,
    output: dict[str, Any],
    demand_target: torch.Tensor,
    *,
    latency_target: torch.Tensor | None = None,
    latency_weights: torch.Tensor | None = None,
    sla_target: torch.Tensor | None = None,
    reliability_target: torch.Tensor | None = None,
    demand_weight: float = 1.0,
    performance_weight: float = 1.0,
    sla_weight: float = 1.0,
    reliability_weight: float = 0.5,
    ranking_weight: float = 0.5,
    residual_l1_weight: float = 1e-4,
    demand_scale_rps: float = 1.0,
    latency_scale_ms: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q50 = output["demand_q50"]
    q90 = output["demand_q90"]
    if q50.shape != demand_target.shape or q90.shape != demand_target.shape:
        raise ValueError("demand target shape must match multi-horizon predictions")
    if demand_scale_rps <= 0 or latency_scale_ms <= 0:
        raise ValueError("training target scales must be positive")
    demand_q50_loss = pinball_metric(q50 / demand_scale_rps, demand_target / demand_scale_rps, 0.50)
    demand_q90_loss = pinball_metric(q90 / demand_scale_rps, demand_target / demand_scale_rps, 0.90)
    demand_loss = 0.5 * (demand_q50_loss + demand_q90_loss)
    performance_loss = q50.sum() * 0.0
    latency_loss = q50.sum() * 0.0
    sla_loss = q50.sum() * 0.0
    reliability_loss = q50.sum() * 0.0
    ranking_loss = q50.sum() * 0.0
    performance = output.get("candidate_performance")
    if latency_target is not None or sla_target is not None:
        if performance is None or latency_target is None or sla_target is None:
            raise ValueError("latency and SLA targets require learned candidate performance outputs")
        latency_prediction = performance["predicted_p95_latency_ms"]
        sla_probability = performance["sla_violation_probability"]
        if latency_prediction.shape != latency_target.shape or sla_probability.shape != sla_target.shape:
            raise ValueError("candidate performance target shapes do not match predictions")
        valid_latency = torch.isfinite(latency_target)
        predicted_latency = latency_prediction[valid_latency]
        actual_latency = latency_target[valid_latency]
        if latency_weights is not None and latency_weights.shape != latency_target.shape:
            raise ValueError("latency weights must match the latency target shape")
        latency_loss = (
            functional.smooth_l1_loss(
                predicted_latency / latency_scale_ms,
                actual_latency / latency_scale_ms,
                reduction="none",
            )
            if actual_latency.numel()
            else latency_prediction.sum() * 0.0
        )
        if actual_latency.numel():
            if latency_weights is None:
                latency_loss = latency_loss.mean()
            else:
                weights = latency_weights[valid_latency]
                latency_loss = torch.sum(latency_loss * weights) / torch.clamp(
                    weights.sum(), min=1.0
                )
        predicted_sla, actual_sla = _masked_values(sla_probability, sla_target)
        sla_loss = (
            functional.binary_cross_entropy(predicted_sla.clamp(1e-6, 1.0 - 1e-6), actual_sla)
            if actual_sla.numel()
            else sla_probability.sum() * 0.0
        )
        if reliability_target is not None:
            success_probability = performance["success_probability"]
            predicted_reliability, actual_reliability = _masked_values(
                success_probability, reliability_target
            )
            reliability_loss = (
                functional.binary_cross_entropy(
                    predicted_reliability.clamp(1e-6, 1.0 - 1e-6),
                    actual_reliability,
                )
                if actual_reliability.numel()
                else success_probability.sum() * 0.0
            )
        valid_candidates = torch.isfinite(latency_target).all(dim=(-1, -2))
        if torch.any(valid_candidates):
            predicted_candidate_cost = (
                sla_probability.amax(dim=(-1, -2))
                + 0.25
                * (latency_prediction / max(float(latency_scale_ms), 1.0)).amax(dim=(-1, -2))
            )
            actual_candidate_cost = (
                torch.nan_to_num(sla_target, nan=1.0).amax(dim=(-1, -2))
                + 0.25
                * torch.nan_to_num(
                    latency_target / max(float(latency_scale_ms), 1.0), nan=1e6
                ).amax(dim=(-1, -2))
            )
            actual_candidate_cost = torch.where(
                valid_candidates, actual_candidate_cost, torch.full_like(actual_candidate_cost, 1e6)
            )
            best_candidate = torch.argmin(actual_candidate_cost, dim=1)
            ranking_loss = functional.cross_entropy(-predicted_candidate_cost, best_candidate)
        performance_loss = (
            latency_loss
            + float(sla_weight) * sla_loss
            + float(reliability_weight) * reliability_loss
            + float(ranking_weight) * ranking_loss
        )
    residual_penalty = model.residual_sparsity_loss()
    total = (
        float(demand_weight) * demand_loss
        + float(performance_weight) * performance_loss
        + float(residual_l1_weight) * residual_penalty
    )
    return total, {
        "demand_q50_pinball": demand_q50_loss,
        "demand_q90_pinball": demand_q90_loss,
        "demand_wis": weighted_interval_score_q50_q90(q50, q90, demand_target),
        "demand_q90_coverage": q90_empirical_coverage(q90, demand_target),
        "p95_latency": latency_loss,
        "sla_bce": sla_loss,
        "reliability_bce": reliability_loss,
        "candidate_ranking": ranking_loss,
        "residual_l1": residual_penalty,
    }


def v2_evaluation_metrics(
    output: dict[str, Any],
    demand_target: torch.Tensor,
    *,
    latency_target: torch.Tensor | None = None,
    sla_target: torch.Tensor | None = None,
    calibration_bins: int = 10,
) -> dict[str, float]:
    q50 = output["demand_q50"]
    q90 = output["demand_q90"]
    metrics = {
        "demand_q50_pinball": float(pinball_metric(q50, demand_target, 0.50).item()),
        "demand_q90_pinball": float(pinball_metric(q90, demand_target, 0.90).item()),
        "demand_wis": float(weighted_interval_score_q50_q90(q50, q90, demand_target).item()),
        "demand_q90_empirical_coverage": float(q90_empirical_coverage(q90, demand_target).item()),
        "quantile_crossing_rate": float(torch.mean((q50 > q90).to(dtype=q50.dtype)).item()),
    }
    performance = output.get("candidate_performance")
    if performance is not None and latency_target is not None and sla_target is not None:
        p95 = performance["predicted_p95_latency_ms"]
        probability = performance["sla_violation_probability"]
        predicted_latency, actual_latency = _masked_values(p95, latency_target)
        metrics.update(
            {
                "candidate_p95_mae": float(
                    torch.mean(torch.abs(predicted_latency - actual_latency)).item()
                    if actual_latency.numel()
                    else 0.0
                ),
                "sla_brier_score": float(brier_score(probability, sla_target).item()),
                "sla_expected_calibration_error": float(
                    expected_calibration_error(probability, sla_target, calibration_bins).item()
                ),
            }
        )
    return metrics


def heldout_allocation_boundary_metrics(
    output: dict[str, Any],
    latency_target: torch.Tensor,
    sla_target: torch.Tensor,
    boundary_mask: torch.Tensor,
    *,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    performance = output.get("candidate_performance")
    if performance is None:
        raise ValueError("held-out boundary evaluation requires candidate performance outputs")
    if boundary_mask.shape != latency_target.shape or sla_target.shape != latency_target.shape:
        raise ValueError("boundary mask and target shapes must match")
    selected = boundary_mask.to(dtype=torch.bool) & torch.isfinite(latency_target) & torch.isfinite(sla_target)
    if not torch.any(selected):
        raise ValueError("held-out allocation-boundary mask selects no valid samples")
    latency = performance["predicted_p95_latency_ms"][selected]
    probability = performance["sla_violation_probability"][selected]
    actual_latency = latency_target[selected]
    actual_sla = sla_target[selected]
    return {
        "boundary_samples": int(selected.sum().item()),
        "boundary_p95_mae": float(torch.mean(torch.abs(latency - actual_latency)).item()),
        "boundary_sla_brier_score": float(torch.mean(torch.square(probability - actual_sla)).item()),
        "boundary_sla_calibration_error": float(
            expected_calibration_error(probability, actual_sla, calibration_bins).item()
        ),
    }


def validate_v2_training_seed_partitions(
    *,
    fitting_seeds: Iterable[int],
    calibration_seeds: Iterable[int],
    holdout_seeds: Iterable[int],
    initialization_seed: int,
) -> dict[str, tuple[int, ...] | int]:
    partitions = {
        "fitting": tuple(sorted({int(value) for value in fitting_seeds})),
        "calibration": tuple(sorted({int(value) for value in calibration_seeds})),
        "holdout": tuple(sorted({int(value) for value in holdout_seeds})),
    }
    if any(not values for values in partitions.values()):
        raise ValueError("v2 fitting, calibration, and holdout seed sets must be non-empty")
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(set(partitions[left]) & set(partitions[right]))
            if overlap:
                raise ValueError(f"v2 {left} and {right} seeds overlap: {overlap}")
    forbidden = RESEARCH_V3_EVALUATION_SEEDS | RESEARCH_V4_EVALUATION_SEEDS
    used = set().union(*partitions.values())
    if leaked := sorted(used & forbidden):
        raise ValueError(f"v2 training metadata references evaluation seeds: {leaked}")
    initialization = int(initialization_seed)
    if initialization not in V2_TRAINING_INITIALIZATION_SEEDS:
        raise ValueError("v2 initialization seed must be one of 601–608")
    return {**partitions, "initialization": initialization}
