#!/usr/bin/env python3
"""Run the paired Headlight closed-loop synthetic controller matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "PyTorch is required for the HeadLight closed-loop matrix; "
        "install ai-module/ai-module/requirements.txt"
    ) from exc

from headlight.closed_loop_v2 import (  # noqa: E402
    CASE_NAMES,
    CLOSED_LOOP_SCHEMA_VERSION,
    PRIMARY_CONTROLLER_IDS,
    CandidateActions,
    ClosedLoopConfig,
    ClosedLoopController,
    ClosedLoopEnvironment,
    ClosedLoopObservation,
    ControllerAction,
    ControllerDecision,
    HpaCpuEmulation,
    KedaRequestRateEmulation,
    OracleBoundController,
    Preview,
    ThresholdReactiveController,
    run_episode,
)
from headlight.models_v2 import build_headlight_v2_model  # noqa: E402
from headlight.synthetic_v2 import (  # noqa: E402
    DATABASE_NODES,
    ENDPOINT_PATHS,
    ENDPOINTS,
    NODE_ORDER,
    build_adjacency,
)


DEFAULT_SEEDS = (701, 702, 703, 704, 705)


@dataclass(frozen=True)
class NeuralPolicyWeights:
    resource: float = 0.25
    sla_risk: float = 1.0
    migration: float = 0.10
    reliability: float = 0.50

    @classmethod
    def from_json(cls, path: Path | None) -> "NeuralPolicyWeights":
        if path is None:
            return cls()
        value = _load_json(path)
        weights = value.get("weights", value)
        return cls(**{name: float(weights[name]) for name in cls.__dataclass_fields__})
SUMMARY_METRICS = (
    "total_requests",
    "successful_requests",
    "failed_requests",
    "success_rate",
    "timeout_count",
    "exception_count",
    "http_200_count",
    "http_500_count",
    "http_504_count",
    "request_p50_latency_ms",
    "request_p95_latency_ms",
    "request_p99_latency_ms",
    "request_sla_violation_rate",
    "excess_response_time_seconds",
    "p95_latency_ms",
    "sla_violation_rate",
    "endpoint_p95_sla_violation_rate",
    "sla_violation_severity",
    "longest_sla_violation_seconds",
    "successful_throughput_rps",
    "offered_throughput_rps",
    "mean_successful_throughput_rps",
    "mean_offered_throughput_rps",
    "cpu_seconds",
    "mean_pod_count",
    "pod_seconds",
    "replica_changes",
    "action_churn",
    "placement_changes",
    "migration_proxy",
    "mean_desired_replicas_cluster2",
    "mean_desired_replicas_cluster3",
    "mean_ready_replicas_cluster2",
    "mean_ready_replicas_cluster3",
    "mean_placement_count_cluster2",
    "mean_placement_count_cluster3",
    "final_desired_replicas_cluster2",
    "final_desired_replicas_cluster3",
    "final_ready_replicas_cluster2",
    "final_ready_replicas_cluster3",
    "final_placement_count_cluster2",
    "final_placement_count_cluster3",
    "ceiling_time_fraction",
    "pending_pod_seconds",
    "readiness_disruption_rate",
    "rejected_replicas",
    "decision_mean_ms",
    "decision_p95_ms",
    "inference_mean_ms",
    "forecast_q50_mae_rps",
    "guard_modified_decision_rate",
    "mean_guard_modified_services",
    "mean_hysteresis_modified_services",
    "model_graph_free_disagreement_rate",
    "model_graph_free_sla_probability_mae",
    "graph_edge_sensitivity",
    "mean_selected_resource_term",
    "mean_selected_migration_term",
    "mean_selected_reliability_penalty",
    "mean_selected_objective",
)
PAIRED_METRICS = (
    "success_rate",
    "failed_requests",
    "timeout_count",
    "exception_count",
    "request_p50_latency_ms",
    "request_p95_latency_ms",
    "request_p99_latency_ms",
    "request_sla_violation_rate",
    "excess_response_time_seconds",
    "p95_latency_ms",
    "sla_violation_rate",
    "successful_throughput_rps",
    "offered_throughput_rps",
    "mean_successful_throughput_rps",
    "cpu_seconds",
    "mean_pod_count",
    "pod_seconds",
    "action_churn",
    "replica_changes",
    "placement_changes",
    "migration_proxy",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_checkpoint(path_value: str, report_path: Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((REPO_ROOT / path, report_path.parent / path.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint referenced by report does not exist: {path_value}")


class NeuralClosedLoopController:
    def __init__(
        self,
        *,
        controller_id: str,
        model: torch.nn.Module,
        device: torch.device,
        graph_enabled: bool,
        sampling_interval_seconds: int,
        policy_weights: NeuralPolicyWeights | None = None,
        shadow_model: torch.nn.Module | None = None,
    ) -> None:
        self.controller_id = controller_id
        self.model = model
        self.device = device
        self.graph_enabled = graph_enabled
        self.policy_weights = policy_weights or NeuralPolicyWeights()
        self.shadow_model = shadow_model
        self.horizon_steps = tuple(
            int(value) // int(sampling_interval_seconds)
            for value in model.horizons_seconds
        )

    def reset(self) -> None:
        self.model.eval()
        self.healthy_windows = np.zeros(len(NODE_ORDER), dtype=np.int64)

    def decide(
        self,
        observation: ClosedLoopObservation,
        history: Sequence[ClosedLoopObservation],
        candidates: CandidateActions,
        preview: Preview,
    ) -> ControllerDecision:
        if len(history) < 24:
            raise ValueError("Headlight requires a complete closed-loop lookback")
        node = np.stack([item.node_features for item in history], axis=0)[None]
        edge = np.stack([item.edge_features for item in history], axis=0)[None]
        if not self.graph_enabled:
            edge = np.zeros_like(edge)
        node_tensor = torch.tensor(node, dtype=torch.float32, device=self.device)
        edge_tensor = torch.tensor(edge, dtype=torch.float32, device=self.device)
        replica_tensor = torch.tensor(
            candidates.replicas[None], dtype=torch.float32, device=self.device
        )
        placement_tensor = torch.tensor(
            candidates.placements[None], dtype=torch.float32, device=self.device
        )
        readiness_tensor = torch.tensor(
            candidates.readiness[None], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            output = self.model(
                node_tensor, edge_tensor, replica_tensor, placement_tensor, readiness_tensor
            )
        performance = output["candidate_performance"]
        probability = performance["sla_violation_probability"][0]
        replicas = torch.tensor(candidates.replicas, dtype=torch.float32, device=self.device)
        placements = torch.tensor(
            candidates.placements, dtype=torch.float32, device=self.device
        )
        current_placement = torch.tensor(
            observation.desired_placement, dtype=torch.float32, device=self.device
        )
        churn = torch.abs(placements - current_placement[None]).sum(dim=(-1, -2))
        churn = churn / (2.0 * 14.0 * len(NODE_ORDER))
        resource_term = replicas.mean(dim=-1) / 14.0
        risk_term = probability.amax(dim=(-1, -2))
        reliability_term = 1.0 - performance["success_probability"][0].amin(
            dim=(-1, -2)
        )
        score = (
            self.policy_weights.resource * resource_term
            + self.policy_weights.sla_risk * risk_term
            + self.policy_weights.migration * churn
            + self.policy_weights.reliability * reliability_term
        )
        selected = int(torch.argmin(score).item())
        shadow_selected: int | None = None
        shadow_probability_delta: float | None = None
        edge_sensitivity: float | None = None
        with torch.no_grad():
            if self.shadow_model is not None:
                shadow = self.shadow_model(
                    node_tensor,
                    torch.zeros_like(edge_tensor),
                    replica_tensor,
                    placement_tensor,
                    readiness_tensor,
                )["candidate_performance"]
                shadow_risk = shadow["sla_violation_probability"][0].amax(dim=(-1, -2))
                shadow_reliability = 1.0 - shadow["success_probability"][0].amin(
                    dim=(-1, -2)
                )
                shadow_score = (
                    self.policy_weights.resource * resource_term
                    + self.policy_weights.sla_risk * shadow_risk
                    + self.policy_weights.migration * churn
                    + self.policy_weights.reliability * shadow_reliability
                )
                shadow_selected = int(torch.argmin(shadow_score).item())
                shadow_probability_delta = float(torch.mean(torch.abs(probability - shadow["sla_violation_probability"][0])).item())
            if self.graph_enabled:
                edge_ablated = self.model(
                    node_tensor,
                    torch.zeros_like(edge_tensor),
                    replica_tensor,
                    placement_tensor,
                    readiness_tensor,
                )["candidate_performance"]["sla_violation_probability"]
                edge_sensitivity = float(
                    torch.mean(torch.abs(output["candidate_performance"]["sla_violation_probability"] - edge_ablated)).item()
                )
        base = candidates.actions[selected]
        desired = base.desired_replicas.copy()
        emergency = (
            (observation.cpu_utilization >= 0.85)
            | (observation.service_p95_latency_ms >= 1000.0)
            | (observation.error_rate >= 0.05)
        )
        node_index = {name: index for index, name in enumerate(NODE_ORDER)}
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            if observation.endpoint_p95_latency_ms[endpoint_index] >= 1000.0:
                emergency[
                    [node_index[name] for name in ENDPOINT_PATHS[endpoint]]
                ] = True
        healthy = (
            (observation.cpu_utilization <= 0.40)
            & (observation.service_p95_latency_ms <= 800.0)
            & (observation.error_rate <= 0.01)
            & (observation.pending_replicas == 0)
        )
        self.healthy_windows = np.where(healthy, self.healthy_windows + 1, 0)
        premature_scale_in = (desired < observation.desired_replicas) & (
            self.healthy_windows < 3
        )
        desired[premature_scale_in] = observation.desired_replicas[premature_scale_in]
        reactive_target = np.minimum(observation.desired_replicas + 2, 14)
        desired[emergency] = np.maximum(desired[emergency], reactive_target[emergency])
        desired[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
        placement = base.placement.copy()
        changed_by_guard = desired != base.desired_replicas
        if np.any(changed_by_guard):
            left = np.clip(np.rint(desired * 0.5), 1, desired - 1).astype(np.int64)
            placement[changed_by_guard, 0] = left[changed_by_guard]
            placement[changed_by_guard, 1] = (
                desired[changed_by_guard] - left[changed_by_guard]
            )
        action = ControllerAction(
            desired,
            placement,
            (
                f"Headlight candidate {selected} with reactive guard"
                if np.any(changed_by_guard)
                else f"Headlight candidate {selected}"
            ),
        )
        diagnostics = {
            "selected_candidate": selected,
            "selected_predicted_p95_latency_ms": float(
                performance["predicted_p95_latency_ms"][0, selected].amax().item()
            ),
            "selected_sla_probability": float(probability[selected].amax().item()),
            "selected_churn_cost": float(churn[selected].item()),
            "selected_resource_term": float(resource_term[selected].item()),
            "selected_reliability_penalty": float(reliability_term[selected].item()),
            "selected_objective": float(score[selected].item()),
            "shadow_selected_candidate": shadow_selected,
            "shadow_disagrees": shadow_selected is not None and shadow_selected != selected,
            "shadow_sla_probability_mae": shadow_probability_delta,
            "graph_edge_sensitivity": edge_sensitivity,
            "reactive_guard_services": int(changed_by_guard.sum()),
            "scale_in_hysteresis_services": int(premature_scale_in.sum()),
            "demand_q50": output["demand_q50"][0].detach().cpu().numpy(),
            "demand_q90": output["demand_q90"][0].detach().cpu().numpy(),
            "horizon_steps": self.horizon_steps,
            "forecast_origin_step": observation.step,
        }
        return ControllerDecision(action, diagnostics)


def _build_neural_controllers(
    report_path: Path,
    device: torch.device,
    policy_weights: NeuralPolicyWeights | None = None,
    *,
    artifact_root: Path | None = None,
    relocation_map: Mapping[str, str] | None = None,
) -> dict[str, Callable[[], ClosedLoopController]]:
    report = _load_json(report_path)
    if artifact_root is not None:
        from experiments.lib.matched_baselines import resolve_artifact_path
        dataset_root = resolve_artifact_path(report['dataset'], artifact_root, relocation_map or {}).parent
    else:
        dataset_root = Path(report["dataset"]).resolve().parent
    config = _load_json(dataset_root / "model-config.json")
    graph = build_adjacency()
    matrix = np.asarray(graph["matrix"], dtype=np.float32)
    observed = np.asarray(graph["observed_mask"], dtype=np.float32)
    factories: dict[str, Callable[[], ClosedLoopController]] = {}
    models: dict[str, torch.nn.Module] = {}
    variants = {
        "headlight-v2": ("full", True),
        "headlight-v2-graph-free": ("graph-free", False),
    }
    for controller_id, (variant, graph_enabled) in variants.items():
        selected = report["selected"][variant]
        checkpoint = (
            resolve_artifact_path(str(selected['checkpoint']), artifact_root, relocation_map or {})
            if artifact_root is not None else _resolve_checkpoint(str(selected["checkpoint"]), report_path)
        )
        adjacency = matrix if graph_enabled else np.eye(len(matrix), dtype=np.float32)
        mask = observed if graph_enabled else np.eye(len(matrix), dtype=np.float32)
        model = build_headlight_v2_model(
            config,
            torch.tensor(adjacency, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        models[variant] = model

    for controller_id, (variant, graph_enabled) in variants.items():
        model = models[variant]

        def factory(
            controller_id: str = controller_id,
            model: torch.nn.Module = model,
            graph_enabled: bool = graph_enabled,
            shadow_model: torch.nn.Module | None = models.get("graph-free") if graph_enabled else models.get("full"),
        ) -> ClosedLoopController:
            return NeuralClosedLoopController(
                controller_id=controller_id,
                model=model,
                device=device,
                graph_enabled=graph_enabled,
                sampling_interval_seconds=int(config["sampling_interval_seconds"]),
                policy_weights=policy_weights,
                shadow_model=shadow_model,
            )

        factories[controller_id] = factory
    return factories


def controller_factories(
    report_path: Path,
    device: torch.device,
    policy_weights: NeuralPolicyWeights | None = None,
) -> dict[str, Callable[[], ClosedLoopController]]:
    result = _build_neural_controllers(report_path, device, policy_weights)
    result.update(
        {
            "hpa-cpu-emulation": HpaCpuEmulation,
            "keda-request-rate-emulation": KedaRequestRateEmulation,
            "threshold-reactive": ThresholdReactiveController,
            "oracle-bound": OracleBoundController,
        }
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_std(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(finite))
    return {
        "count": len(finite),
        "mean": mean,
        "median": statistics.median(finite),
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in episodes if not row.get("failed")]
    groups: dict[str, dict[str, Any]] = {}
    for controller in sorted({str(row["controller"]) for row in completed}):
        controller_rows = [row for row in completed if row["controller"] == controller]
        groups[controller] = {
            "overall": {
                metric: _mean_std(
                    [row[metric] for row in controller_rows if row.get(metric) is not None]
                )
                for metric in SUMMARY_METRICS
            },
            "cases": {
                case: {
                    metric: _mean_std(
                        [
                            row[metric]
                            for row in controller_rows
                            if row["case"] == case and row.get(metric) is not None
                        ]
                    )
                    for metric in SUMMARY_METRICS
                }
                for case in CASE_NAMES
            },
        }
    trace_groups: dict[tuple[str, int], set[str]] = {}
    for row in completed:
        trace_groups.setdefault((str(row["case"]), int(row["seed"])), set()).add(
            str(row["trace_hash"])
        )
    pairing_valid = all(len(hashes) == 1 for hashes in trace_groups.values())
    return {
        "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
        "episodes": len(episodes),
        "completed_episodes": len(completed),
        "failed_episodes": len(episodes) - len(completed),
        "paired_trace_integrity": pairing_valid,
        "controllers": groups,
    }


def paired_rows(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in episodes if not row.get("failed")]
    by_key = {
        (str(row["controller"]), str(row["case"]), int(row["seed"])): row
        for row in complete
    }
    rows: list[dict[str, Any]] = []
    comparators = [
        controller
        for controller in PRIMARY_CONTROLLER_IDS
        if controller != "headlight-v2"
    ]
    for comparator in comparators:
        for workload in ("overall", *CASE_NAMES):
            for metric in PAIRED_METRICS:
                differences = []
                cases = CASE_NAMES if workload == "overall" else (workload,)
                for case in cases:
                    seeds = sorted(
                        seed
                        for controller, candidate_case, seed in by_key
                        if controller == "headlight-v2" and candidate_case == case
                    )
                    for seed in seeds:
                        primary = by_key.get(("headlight-v2", case, seed))
                        other = by_key.get((comparator, case, seed))
                        if primary is not None and other is not None:
                            differences.append(
                                float(other[metric]) - float(primary[metric])
                            )
                if not differences:
                    continue
                mean = statistics.fmean(differences)
                median = statistics.median(differences)
                std = statistics.stdev(differences) if len(differences) > 1 else 0.0
                half_width = 1.96 * std / math.sqrt(len(differences))
                rows.append(
                    {
                        "primary": "headlight-v2",
                        "comparator": comparator,
                        "workload": workload,
                        "metric": metric,
                        "pairs": len(differences),
                        "difference_definition": "comparator_minus_headlight",
                        "mean_difference": mean,
                        "median_difference": median,
                        "std_difference": std,
                        "ci95_low": mean - half_width,
                        "ci95_high": mean + half_width,
                    }
                )
    return rows


def render_report(summary: Mapping[str, Any]) -> str:
    def value(controller: str, metric: str) -> str:
        item = summary["controllers"][controller]["overall"][metric]["mean"]
        return "n/a" if item is None else f"{float(item):.4f}"

    lines = [
        "# Headlight closed-loop synthetic evaluation",
        "",
        "> Engineering evidence only. This is an offline deterministic emulator, not",
        "> Kubernetes, Karmada, native HPA, or native KEDA evidence.",
        "",
        f"- Episodes: {summary['completed_episodes']} completed, {summary['failed_episodes']} failed",
        f"- Paired trace integrity: {summary['paired_trace_integrity']}",
        f"- Valid matrix: {summary['valid']}",
        "",
        "## Primary controllers and baselines",
        "",
            "### Synthetic request outcomes",
        "",
        "| Controller | Success rate | Request SLA violation rate | p50 (ms) | p95 (ms) | p99 (ms) | Successful throughput (rps) | Excess response (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for controller in summary["protocol"]["primary_controllers"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    controller,
                    value(controller, "success_rate"),
                    value(controller, "request_sla_violation_rate"),
                    value(controller, "request_p50_latency_ms"),
                    value(controller, "request_p95_latency_ms"),
                    value(controller, "request_p99_latency_ms"),
                    value(controller, "successful_throughput_rps"),
                    value(controller, "excess_response_time_seconds"),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### Synthetic request accounting",
            "",
            "Counts are means per episode; HTTP outcomes are generated by the emulator.",
            "",
            "| Controller | Total | Successful/HTTP 200 | Failed | HTTP 500/exceptions | HTTP 504/timeouts |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for controller in summary["protocol"]["primary_controllers"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    controller,
                    value(controller, "total_requests"),
                    value(controller, "successful_requests"),
                    value(controller, "failed_requests"),
                    value(controller, "http_500_count"),
                    value(controller, "http_504_count"),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### Resources and control activity",
            "",
            "| Controller | CPU-seconds | Mean pods | Pod-seconds | Action churn | Placement changes | Migration proxy | Inference mean (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for controller in summary["protocol"]["primary_controllers"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    controller,
                    value(controller, "cpu_seconds"),
                    value(controller, "mean_pod_count"),
                    value(controller, "pod_seconds"),
                    value(controller, "action_churn"),
                    value(controller, "placement_changes"),
                    value(controller, "migration_proxy"),
                    value(controller, "inference_mean_ms"),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Oracle sanity bound",
            "",
            "The oracle previews simulator state and is not a deployable controller. It is",
            "excluded from primary paired comparisons.",
            "",
            "| Request SLA violation rate | Request p95 latency (ms) | Pod-seconds |",
            "| ---: | ---: | ---: |",
            "| "
            + " | ".join(
                (
                    value("oracle-bound", "request_sla_violation_rate"),
                    value("oracle-bound", "request_p95_latency_ms"),
                    value("oracle-bound", "pod_seconds"),
                )
            )
            + " |",
            "",
            "## Limitations",
            "",
        )
    )
    lines.extend(f"- {item}" for item in summary["protocol"]["limitations"])
    return "\n".join(lines) + "\n"


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one paired seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("paired seeds must be unique")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-report",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "synthetic-headlight-v2"
        / "training-final-seed601-margin"
        / "training-report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic-headlight-v2" / "closed-loop-v1",
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--policy-config",
        type=Path,
        help="frozen JSON policy weights selected using development seeds",
    )
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        raise SystemExit(f"output directory is not empty: {output_dir}; pass --replace")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.replace:
        for name in (
            "episodes.jsonl",
            "episode-results.csv",
            "summary.json",
            "paired-comparisons.csv",
            "report.md",
        ):
            (output_dir / name).unlink(missing_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = ClosedLoopConfig(steps=args.steps)
    policy_weights = NeuralPolicyWeights.from_json(
        args.policy_config.resolve() if args.policy_config else None
    )
    factories = controller_factories(
        args.checkpoint_report.resolve(), device, policy_weights
    )
    expected = len(CASE_NAMES) * len(args.seeds) * len(factories)
    episodes: list[dict[str, Any]] = []
    for case in CASE_NAMES:
        for seed in args.seeds:
            for controller_id, factory in factories.items():
                print(
                    json.dumps(
                        {
                            "event": "episode-started",
                            "controller": controller_id,
                            "case": case,
                            "seed": seed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                try:
                    environment = ClosedLoopEnvironment(case=case, seed=seed, config=config)
                    result = run_episode(environment, factory())
                except Exception as exc:  # preserve failed-run accounting
                    result = {
                        "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
                        "controller": controller_id,
                        "case": case,
                        "seed": seed,
                        "failed": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                episodes.append(result)
                print(
                    json.dumps(
                        {
                            "event": "episode-complete",
                            "controller": controller_id,
                            "case": case,
                            "seed": seed,
                            "failed": result["failed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    episode_path = output_dir / "episodes.jsonl"
    episode_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes),
        encoding="utf-8",
    )
    episode_fields = sorted({key for row in episodes for key in row})
    with (output_dir / "episode-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            for row in episodes
        )
    summary = summarize(episodes)
    summary["protocol"] = {
        "engineering_evidence_only": True,
        "checkpoint_report": str(args.checkpoint_report.resolve()),
        "policy_weights": asdict(policy_weights),
        "policy_config": str(args.policy_config.resolve()) if args.policy_config else None,
        "policy_config_sha256": _file_sha256(args.policy_config.resolve())
        if args.policy_config
        else None,
        "cases": list(CASE_NAMES),
        "paired_seeds": list(args.seeds),
        "controllers": list(factories),
        "primary_controllers": list(PRIMARY_CONTROLLER_IDS),
        "oracle_reported_separately": True,
        "steps": args.steps,
        "sampling_interval_seconds": config.sampling_interval_seconds,
        "lookback_steps": config.lookback_steps,
        "measurement_contract": {
            "request_outcomes_are_synthetic": True,
            "successful_status": 200,
            "exception_status": 500,
            "timeout_status": 504,
            "endpoint_request_shares": {
                "login": 0.20,
                "search": 0.50,
                "booking": 0.30,
            },
            "request_latency_distribution": (
                "deterministic log-normal quantile grid calibrated from each "
                "endpoint's synthetic p50 and p95"
            ),
            "request_sla_latency_ms": config.sla_latency_ms,
            "legacy_sla_violation_rate_semantics": (
                "fraction of endpoint-window p95 observations above the SLA; "
                "use request_sla_violation_rate for request-level comparisons"
            ),
            "action_churn_semantics": (
                "number of measured decisions with at least one applied replica change"
            ),
            "placement_snapshot_semantics": (
                "time-mean cluster totals plus final per-service desired placements"
            ),
        },
        "limitations": [
            "offline emulation, not Kubernetes or Karmada actuation",
            "HPA and KEDA formulas are explicit emulations, not deployed controllers",
            "request outcomes, HTTP statuses, exceptions, and timeouts are synthetic",
            "no RBAC, API failure, WAN, bandwidth, or controller-conflict evidence",
        ],
        "environment": asdict(config),
        "action_executor": {
            "same_for_every_controller": True,
            "max_scale_out_step": config.max_scale_out_step,
            "max_scale_in_step": config.max_scale_in_step,
            "cooldown_steps": config.cooldown_steps,
            "cluster_capacity_checked": True,
        },
        "controller_emulations": {
            "hpa-cpu-emulation": {
                "target_cpu_utilization": 0.70,
                "tolerance_ratio": [0.90, 1.10],
                "downscale_stabilization_seconds": 300,
            },
            "keda-request-rate-emulation": {
                "target_rps_per_replica": 0.80,
                "activation_rps": 0.10,
                "cooldown_seconds": 300,
            },
            "threshold-reactive": {
                "cpu_scale_out": 0.75,
                "service_p95_scale_out_ms": 1000.0,
                "error_rate_scale_out": 0.05,
                "healthy_windows_before_scale_in": 3,
            },
        },
        "checkpoint_report_sha256": _file_sha256(args.checkpoint_report.resolve()),
    }
    summary["validity"] = {
        "expected_episodes": expected,
        "episode_count_complete": len(episodes) == expected,
        "zero_failed_episodes": summary["failed_episodes"] == 0,
        "paired_trace_integrity": summary["paired_trace_integrity"],
    }
    summary["valid"] = all(summary["validity"].values())
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparisons = paired_rows(episodes)
    with (output_dir / "paired-comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]) if comparisons else [])
        if comparisons:
            writer.writeheader()
            writer.writerows(comparisons)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(output_dir / "summary.json"),
                "episodes": len(episodes),
                "valid": summary["valid"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
