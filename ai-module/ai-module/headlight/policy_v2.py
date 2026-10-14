"""Headlight candidate enumeration, scoring, and safe action serialization.

This module is intentionally separate from :mod:`headlight.policy`, whose
research-v3 behavior and frozen configuration remain unchanged.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol

from .observability import DECISION_RECORD_SCHEMA_VERSION, validate_decision_record
from .profile import ServiceNode, ServiceProfile


POLICY_V2_SCHEMA_VERSION = "headlight.policy/v2"
POLICY_V2_VERSION = "2.0.0"
POLICY_V2_SELECTION_RULE = "minimum-objective-then-churn-total-placement"

HEADLIGHT_V2_FULL_CONTROLLER = "headlight-v2-full"
HEADLIGHT_V2_NO_ACTION_CONTROLLER = "headlight-v2-no-action"

DEVELOPMENT_SCALE_DOWN_HEALTHY_WINDOWS = (3, 4)
DEVELOPMENT_SCALE_DOWN_COOLDOWN_SECONDS = (60, 75, 90)
RESEARCH_V4_DEVELOPMENT_SEEDS = frozenset({401, 403, 407})
FORBIDDEN_POLICY_TUNING_SEEDS = frozenset({101, 102, 103, 104, 105, *range(501, 509)})

TELEMETRY_FIELDS = (
    "demand_rps",
    "cpu_utilization",
    "queue_depth",
    "p95_latency_ms",
    "error_rate",
)
SCALE_IN_TELEMETRY_FIELDS = frozenset({
    "cpu_utilization", "queue_depth", "p95_latency_ms", "error_rate",
})


@dataclass(frozen=True)
class PolicyV2ControllerRegistration:
    controller_id: str
    model_type: str
    risk_source: str
    action_conditioning: bool


POLICY_V2_CONTROLLER_REGISTRATIONS = {
    HEADLIGHT_V2_FULL_CONTROLLER: PolicyV2ControllerRegistration(
        controller_id=HEADLIGHT_V2_FULL_CONTROLLER,
        model_type="headlight-v2",
        risk_source="learned-action-conditioned",
        action_conditioning=True,
    ),
    HEADLIGHT_V2_NO_ACTION_CONTROLLER: PolicyV2ControllerRegistration(
        controller_id=HEADLIGHT_V2_NO_ACTION_CONTROLLER,
        model_type="headlight-v2-no-action",
        risk_source="calibrated-capacity-surrogate",
        action_conditioning=False,
    ),
}


def policy_v2_registration(controller_id: str) -> PolicyV2ControllerRegistration:
    try:
        return POLICY_V2_CONTROLLER_REGISTRATIONS[controller_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Headlight controller {controller_id!r}") from exc


@dataclass(frozen=True)
class PolicyV2Config:
    """Configured development parameters; Phase 6 must tune and freeze them."""

    policy_id: str = "headlight-v2-policy-development"
    resource_weight: float = 1.0
    sla_risk_weight: float = 1.0
    churn_weight: float = 0.1
    reliability_weight: float = 0.0
    sla_latency_ms: float = 1000.0
    predictive_scale_out_step: int = 1
    reactive_scale_out_step: int = 1
    reactive_emergency_enabled: bool = True
    reactive_pending_wait_seconds: int = 240
    scale_in_step: int = 1
    scale_in_sla_probability: float = 0.01
    reactive_cpu_utilization: float = 0.85
    reactive_queue_depth: float = 1.0
    reactive_latency_ms: float = 1000.0
    reactive_error_rate: float = 0.05
    scale_in_cpu_utilization: float = 0.40
    scale_in_queue_depth: float = 0.0
    scale_in_latency_ms: float = 800.0
    scale_in_error_rate: float = 0.01
    scale_in_telemetry_fields: tuple[str, ...] = (
        "cpu_utilization",
        "p95_latency_ms",
        "error_rate",
    )
    scale_down_healthy_windows: int = 3
    cooldown_seconds: int = 60
    missing_prediction_risk: float = 1.0
    max_candidates_per_service: int = 4096
    cluster_resource_weights: Mapping[str, float] | None = None
    tuning_seeds: tuple[int, ...] = (401, 403, 407)

    def validate(self, profile: ServiceProfile, *, controller_id: str = HEADLIGHT_V2_FULL_CONTROLLER) -> None:
        if not isinstance(self.reactive_emergency_enabled, bool):
            raise ValueError("reactive_emergency_enabled must be an explicit boolean")
        if not self.policy_id.strip():
            raise ValueError("Policy-v2 policy_id must be non-empty")
        nonnegative = {
            "resource_weight": self.resource_weight,
            "sla_risk_weight": self.sla_risk_weight,
            "churn_weight": self.churn_weight,
            "reliability_weight": self.reliability_weight,
            "sla_latency_ms": self.sla_latency_ms,
            "reactive_cpu_utilization": self.reactive_cpu_utilization,
            "reactive_queue_depth": self.reactive_queue_depth,
            "reactive_latency_ms": self.reactive_latency_ms,
            "reactive_error_rate": self.reactive_error_rate,
            "scale_in_cpu_utilization": self.scale_in_cpu_utilization,
            "scale_in_queue_depth": self.scale_in_queue_depth,
            "scale_in_latency_ms": self.scale_in_latency_ms,
            "scale_in_error_rate": self.scale_in_error_rate,
        }
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in nonnegative.values()):
            raise ValueError("Policy-v2 weights and thresholds must be finite and non-negative")
        for name in ("predictive_scale_out_step", "reactive_scale_out_step", "scale_in_step"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"Policy-v2 {name} must be positive")
        if int(self.reactive_pending_wait_seconds) <= 0:
            raise ValueError("Policy-v2 reactive pending wait must be positive")
        if not 0 <= self.scale_in_sla_probability <= 1:
            raise ValueError("Policy-v2 scale-in SLA probability must be in [0, 1]")
        if not 0 <= self.missing_prediction_risk <= 1:
            raise ValueError("Policy-v2 missing-prediction risk must be in [0, 1]")
        if self.scale_in_cpu_utilization >= self.reactive_cpu_utilization:
            raise ValueError("Policy-v2 scale-in CPU threshold must be below the reactive guard")
        if self.scale_down_healthy_windows not in DEVELOPMENT_SCALE_DOWN_HEALTHY_WINDOWS:
            raise ValueError("Policy-v2 healthy windows must be one of the preregistered values 3 or 4")
        if self.cooldown_seconds not in DEVELOPMENT_SCALE_DOWN_COOLDOWN_SECONDS:
            raise ValueError("Policy-v2 cooldown must be one of 60, 75, or 90 seconds")
        if self.max_candidates_per_service <= 0:
            raise ValueError("Policy-v2 candidate limit must be positive")
        scale_in_fields = tuple(self.scale_in_telemetry_fields)
        if not scale_in_fields or len(set(scale_in_fields)) != len(scale_in_fields):
            raise ValueError("Policy-v2 scale-in telemetry fields must be non-empty and unique")
        unknown_scale_in_fields = set(scale_in_fields) - SCALE_IN_TELEMETRY_FIELDS
        if unknown_scale_in_fields:
            raise ValueError(
                "Policy-v2 scale-in telemetry fields are unsupported: "
                f"{sorted(unknown_scale_in_fields)}"
            )
        seeds = {int(seed) for seed in self.tuning_seeds}
        allowed_seeds = frozenset({711, 712, 713}) if controller_id == 'lstm-matched' else RESEARCH_V4_DEVELOPMENT_SEEDS
        if not seeds or not seeds <= allowed_seeds:
            message = 'Policy-v2 tuning seeds must come only from 711, 712, and 713' if controller_id == 'lstm-matched' else 'Policy-v2 tuning seeds must come only from 401, 403, and 407'
            raise ValueError(message)
        if seeds & FORBIDDEN_POLICY_TUNING_SEEDS:
            raise ValueError("Policy-v2 tuning cannot reference research-v3 or v4 evaluation seeds")
        if self.cluster_resource_weights is not None:
            if set(self.cluster_resource_weights) != set(profile.cluster_limits):
                raise ValueError("Policy-v2 resource weights must name exactly the profile clusters")
            if any(
                not math.isfinite(float(weight)) or float(weight) <= 0
                for weight in self.cluster_resource_weights.values()
            ):
                raise ValueError("Policy-v2 resource weights must be finite and positive")


def _finite_nonnegative(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
        raise ValueError(f"{name} must be finite and non-negative when present")


def _canonical_placement(
    placement: Mapping[str, int], clusters: tuple[str, ...]
) -> tuple[tuple[str, int], ...]:
    if set(placement) != set(clusters):
        raise ValueError("placement must name exactly the service-profile clusters")
    result = tuple((cluster, int(placement[cluster])) for cluster in clusters)
    if any(replicas < 0 for _cluster, replicas in result):
        raise ValueError("replica placements must be non-negative")
    return result


@dataclass(frozen=True)
class ServiceRuntimeStateV2:
    desired_replicas: Mapping[str, int]
    ready_replicas: Mapping[str, int]
    pending_replicas: Mapping[str, int]
    pending_age_seconds: Mapping[str, float]
    demand_rps: float | None
    cpu_utilization: float | None
    queue_depth: float | None
    p95_latency_ms: float | None
    error_rate: float | None
    predictive_fresh: bool
    missing_fields: frozenset[str] = field(default_factory=frozenset)
    stale_fields: frozenset[str] = field(default_factory=frozenset)
    time_since_last_action_seconds: float | None = None
    exists: bool = True
    cluster_request_rate: Mapping[str, float] = field(default_factory=dict)
    cluster_cpu_utilization: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in TELEMETRY_FIELDS:
            _finite_nonnegative(getattr(self, name), name)
        if self.time_since_last_action_seconds is not None:
            _finite_nonnegative(
                self.time_since_last_action_seconds,
                "time_since_last_action_seconds",
            )
        unknown_masks = (set(self.missing_fields) | set(self.stale_fields)) - set(TELEMETRY_FIELDS)
        if unknown_masks:
            raise ValueError(f"Policy-v2 masks contain unsupported fields: {sorted(unknown_masks)}")
        for label, placement in (
            ("desired", self.desired_replicas),
            ("ready", self.ready_replicas),
            ("pending", self.pending_replicas),
        ):
            if any(int(value) < 0 for value in placement.values()):
                raise ValueError(f"Policy-v2 {label} replicas must be non-negative")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in self.pending_age_seconds.values()):
            raise ValueError("Policy-v2 pending ages must be finite and non-negative")
        for label, observations in (("cluster request rate", self.cluster_request_rate), ("cluster CPU utilization", self.cluster_cpu_utilization)):
            if set(observations) - set(self.desired_replicas):
                raise ValueError(f"Policy-v2 {label} has unknown clusters")
            for value in observations.values():
                _finite_nonnegative(value, label)

    @property
    def total_desired(self) -> int:
        return sum(int(value) for value in self.desired_replicas.values())

    @property
    def total_pending(self) -> int:
        return sum(int(value) for value in self.pending_replicas.values())

    def is_fresh(self, field_name: str) -> bool:
        return (
            field_name not in self.missing_fields
            and field_name not in self.stale_fields
            and getattr(self, field_name) is not None
        )


@dataclass(frozen=True)
class CandidateReadinessV2:
    ready_replicas: tuple[tuple[str, int], ...]
    pending_replicas: tuple[tuple[str, int], ...]
    existing_pending_replicas: tuple[tuple[str, int], ...]
    ready_fraction: float
    pending_age_seconds: float

    def ready_dict(self) -> dict[str, int]:
        return dict(self.ready_replicas)

    def pending_dict(self) -> dict[str, int]:
        return dict(self.pending_replicas)


@dataclass(frozen=True)
class ReplicaPlacementCandidateV2:
    candidate_id: str
    total_replicas: int
    placement: tuple[tuple[str, int], ...]
    readiness: CandidateReadinessV2

    def placement_dict(self) -> dict[str, int]:
        return dict(self.placement)


@dataclass(frozen=True)
class CandidateEnumerationV2:
    service: str
    candidates: tuple[ReplicaPlacementCandidateV2, ...]
    placements_examined: int
    theoretical_candidate_count: int


def estimate_candidate_enumeration_count(profile: ServiceProfile) -> int:
    return math.prod(
        int(limits["max_replicas"]) - int(limits["min_replicas"]) + 1
        for limits in profile.cluster_limits.values()
    )


def _candidate_id(placement: tuple[tuple[str, int], ...]) -> str:
    total = sum(value for _cluster, value in placement)
    return "total=" + str(total) + "|" + "|".join(
        f"{cluster}={replicas}" for cluster, replicas in placement
    )


def _candidate_readiness(
    placement: tuple[tuple[str, int], ...],
    state: ServiceRuntimeStateV2,
) -> CandidateReadinessV2:
    ready: list[tuple[str, int]] = []
    pending: list[tuple[str, int]] = []
    existing_pending: list[tuple[str, int]] = []
    retained_pending_ages: list[float] = []
    for cluster, desired in placement:
        available_ready = min(desired, int(state.ready_replicas.get(cluster, 0)))
        candidate_pending = max(0, desired - available_ready)
        retained = min(candidate_pending, int(state.pending_replicas.get(cluster, 0)))
        ready.append((cluster, available_ready))
        pending.append((cluster, candidate_pending))
        existing_pending.append((cluster, retained))
        if retained:
            retained_pending_ages.append(float(state.pending_age_seconds.get(cluster, 0.0)))
    total = sum(value for _cluster, value in placement)
    return CandidateReadinessV2(
        ready_replicas=tuple(ready),
        pending_replicas=tuple(pending),
        existing_pending_replicas=tuple(existing_pending),
        ready_fraction=(sum(value for _cluster, value in ready) / total if total else 1.0),
        pending_age_seconds=max(retained_pending_ages, default=0.0),
    )


def enumerate_feasible_candidates(
    profile: ServiceProfile,
    service_name: str,
    state: ServiceRuntimeStateV2,
    *,
    max_candidates: int = 4096,
) -> CandidateEnumerationV2:
    """Enumerate every profile-valid complete placement in deterministic order."""

    node = profile.node(service_name)
    clusters = tuple(profile.cluster_limits)
    for mapping_name, mapping in (
        ("desired", state.desired_replicas),
        ("ready", state.ready_replicas),
        ("pending", state.pending_replicas),
        ("pending_age", state.pending_age_seconds),
    ):
        if set(mapping) != set(clusters):
            raise ValueError(
                f"Policy-v2 {mapping_name} state must name exactly the profile clusters"
            )
    current = _canonical_placement(state.desired_replicas, clusters)
    if not node.scalable or not state.exists:
        candidate = ReplicaPlacementCandidateV2(
            candidate_id=_candidate_id(current),
            total_replicas=sum(value for _cluster, value in current),
            placement=current,
            readiness=_candidate_readiness(current, state),
        )
        return CandidateEnumerationV2(service_name, (candidate,), 1, 1)

    theoretical = estimate_candidate_enumeration_count(profile)
    if theoretical > max_candidates:
        raise ValueError(
            f"Policy-v2 candidate space {theoretical} exceeds configured limit {max_candidates}"
        )
    ranges = tuple(
        range(int(limits["min_replicas"]), int(limits["max_replicas"]) + 1)
        for limits in profile.cluster_limits.values()
    )
    candidates = []
    for counts in itertools.product(*ranges):
        placement = tuple(zip(clusters, counts))
        candidates.append(
            ReplicaPlacementCandidateV2(
                candidate_id=_candidate_id(placement),
                total_replicas=sum(counts),
                placement=placement,
                readiness=_candidate_readiness(placement, state),
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.total_replicas,
            tuple(value for _cluster, value in candidate.placement),
            candidate.candidate_id,
        )
    )
    return CandidateEnumerationV2(
        service=service_name,
        candidates=tuple(candidates),
        placements_examined=theoretical,
        theoretical_candidate_count=theoretical,
    )


@dataclass(frozen=True)
class CandidatePredictionV2:
    predicted_p95_latency_ms: float
    sla_violation_probability: float
    success_probability: float | None = None
    calibrated: bool = True

    def __post_init__(self) -> None:
        _finite_nonnegative(self.predicted_p95_latency_ms, "predicted_p95_latency_ms")
        if not math.isfinite(float(self.sla_violation_probability)) or not (
            0 <= self.sla_violation_probability <= 1
        ):
            raise ValueError("candidate SLA probability must be finite and in [0, 1]")
        if self.success_probability is not None and (
            not math.isfinite(float(self.success_probability))
            or not 0 <= float(self.success_probability) <= 1
        ):
            raise ValueError("candidate success probability must be finite and in [0, 1]")
        if not self.calibrated:
            raise ValueError("Policy-v2 accepts only calibrated candidate SLA probabilities")


class CandidateScorerV2(Protocol):
    def __call__(
        self,
        service_name: str,
        candidates: tuple[ReplicaPlacementCandidateV2, ...],
    ) -> Mapping[str, CandidatePredictionV2]: ...


@dataclass(frozen=True)
class ScoredCandidateV2:
    candidate: ReplicaPlacementCandidateV2
    prediction_available: bool
    predicted_p95_latency_ms: float | None
    predicted_sla_probability: float | None
    predicted_success_probability: float | None
    resource_cost: float
    sla_risk: float
    churn_cost: float
    reliability_cost: float
    objective: float

    def to_decision_candidate(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "total_replicas": self.candidate.total_replicas,
            "placement": self.candidate.placement_dict(),
            "prediction_available": self.prediction_available,
            "predicted_p95_latency_ms": self.predicted_p95_latency_ms,
            "predicted_sla_probability": self.predicted_sla_probability,
            "predicted_success_probability": self.predicted_success_probability,
            "objective_terms": {
                "resource": self.resource_cost,
                "sla_risk": self.sla_risk,
                "churn": self.churn_cost,
                "reliability": self.reliability_cost,
            },
        }


@dataclass(frozen=True)
class PolicyV2Decision:
    service: str
    controller_id: str
    policy_id: str
    resource_weight: float
    sla_risk_weight: float
    churn_weight: float
    reliability_weight: float
    previous_placement: tuple[tuple[str, int], ...]
    candidates: tuple[ScoredCandidateV2, ...]
    selected_candidate_id: str
    reason: str
    action_origin: str
    out_of_distribution: bool
    objective_improvement: float
    readiness_delay: bool

    @property
    def selected(self) -> ScoredCandidateV2:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate.candidate_id == self.selected_candidate_id
        )

    @property
    def changed(self) -> bool:
        return self.selected.candidate.placement != self.previous_placement

    @property
    def desired_placement(self) -> dict[str, int]:
        return self.selected.candidate.placement_dict()

    @property
    def previous_placement_dict(self) -> dict[str, int]:
        return dict(self.previous_placement)


@dataclass
class _PolicyV2Memory:
    healthy_windows: int = 0
    pending_timeout_buckets: dict[str, int] = field(default_factory=dict)


class HeadlightPolicyV2:
    """Shared full/ablation controller policy with no joint-action modeling."""

    def __init__(
        self,
        profile: ServiceProfile,
        *,
        controller_id: str = HEADLIGHT_V2_FULL_CONTROLLER,
        config: PolicyV2Config | None = None,
    ) -> None:
        self.profile = profile
        self.registration = policy_v2_registration(controller_id)
        self.config = config or PolicyV2Config()
        self.config.validate(profile, controller_id=controller_id)
        self.clusters = tuple(profile.cluster_limits)
        self.memory = {node.name: _PolicyV2Memory() for node in profile.nodes}

    def _resource_weights(self) -> dict[str, float]:
        if self.config.cluster_resource_weights is None:
            return dict.fromkeys(self.clusters, 1.0)
        return {
            cluster: float(self.config.cluster_resource_weights[cluster])
            for cluster in self.clusters
        }

    def _score(
        self,
        enumeration: CandidateEnumerationV2,
        state: ServiceRuntimeStateV2,
        predictions: Mapping[str, CandidatePredictionV2],
    ) -> tuple[ScoredCandidateV2, ...]:
        known_ids = {candidate.candidate_id for candidate in enumeration.candidates}
        unknown = sorted(set(predictions) - known_ids)
        if unknown:
            raise ValueError(f"candidate scorer returned unknown IDs: {unknown}")
        resource_weights = self._resource_weights()
        resource_denominator = sum(
            int(self.profile.cluster_limits[cluster]["max_replicas"])
            * resource_weights[cluster]
            for cluster in self.clusters
        )
        churn_denominator = max(
            1,
            2
            * sum(
                int(limits["max_replicas"])
                for limits in self.profile.cluster_limits.values()
            ),
        )
        current = dict(state.desired_replicas)
        scored = []
        for candidate in enumeration.candidates:
            placement = candidate.placement_dict()
            resource = sum(
                placement[cluster] * resource_weights[cluster]
                for cluster in self.clusters
            ) / resource_denominator
            churn = sum(
                abs(placement[cluster] - int(current[cluster]))
                for cluster in self.clusters
            ) / churn_denominator
            prediction = predictions.get(candidate.candidate_id)
            risk = (
                float(prediction.sla_violation_probability)
                if prediction is not None
                else self.config.missing_prediction_risk
            )
            reliability = (
                1.0 - float(prediction.success_probability)
                if prediction is not None and prediction.success_probability is not None
                else 0.0
            )
            objective = (
                self.config.resource_weight * resource
                + self.config.sla_risk_weight * risk
                + self.config.churn_weight * churn
                + self.config.reliability_weight * reliability
            )
            scored.append(
                ScoredCandidateV2(
                    candidate=candidate,
                    prediction_available=prediction is not None,
                    predicted_p95_latency_ms=(
                        float(prediction.predicted_p95_latency_ms)
                        if prediction is not None
                        else None
                    ),
                    predicted_sla_probability=(
                        float(prediction.sla_violation_probability)
                        if prediction is not None
                        else None
                    ),
                    predicted_success_probability=(
                        float(prediction.success_probability)
                        if prediction is not None and prediction.success_probability is not None
                        else None
                    ),
                    resource_cost=float(resource),
                    sla_risk=float(risk),
                    churn_cost=float(churn),
                    reliability_cost=float(reliability),
                    objective=float(objective),
                )
            )
        return tuple(scored)

    @staticmethod
    def _selection_key(candidate: ScoredCandidateV2) -> tuple[Any, ...]:
        return (
            candidate.objective,
            candidate.churn_cost,
            candidate.candidate.total_replicas,
            tuple(value for _cluster, value in candidate.candidate.placement),
            candidate.candidate.candidate_id,
        )

    @staticmethod
    def _hold_candidate(
        scored: tuple[ScoredCandidateV2, ...],
        current: tuple[tuple[str, int], ...],
    ) -> ScoredCandidateV2:
        try:
            return next(candidate for candidate in scored if candidate.candidate.placement == current)
        except StopIteration as exc:
            raise ValueError("current placement is outside the profile's valid candidate space") from exc

    def _reactive_triggers(self, state: ServiceRuntimeStateV2) -> tuple[str, ...]:
        thresholds = {
            "cpu_utilization": self.config.reactive_cpu_utilization,
            "queue_depth": self.config.reactive_queue_depth,
            "p95_latency_ms": self.config.reactive_latency_ms,
            "error_rate": self.config.reactive_error_rate,
        }
        return tuple(
            name
            for name, threshold in thresholds.items()
            if state.is_fresh(name) and float(getattr(state, name)) >= threshold
        )

    def _healthy_for_scale_in(
        self,
        state: ServiceRuntimeStateV2,
        candidate: ScoredCandidateV2,
    ) -> bool:
        thresholds = {
            "cpu_utilization": self.config.scale_in_cpu_utilization,
            "queue_depth": self.config.scale_in_queue_depth,
            "p95_latency_ms": self.config.scale_in_latency_ms,
            "error_rate": self.config.scale_in_error_rate,
        }
        telemetry_healthy = all(
            state.is_fresh(name) and float(getattr(state, name)) <= threshold
            for name, threshold in thresholds.items()
            if name in self.config.scale_in_telemetry_fields
        )
        prediction_healthy = bool(
            candidate.prediction_available
            and candidate.predicted_sla_probability is not None
            and candidate.predicted_sla_probability <= self.config.scale_in_sla_probability
            and candidate.predicted_p95_latency_ms is not None
            and candidate.predicted_p95_latency_ms <= self.config.sla_latency_ms
        )
        return state.total_pending == 0 and telemetry_healthy and prediction_healthy

    def _decision(
        self,
        node: ServiceNode,
        current: tuple[tuple[str, int], ...],
        scored: tuple[ScoredCandidateV2, ...],
        selected: ScoredCandidateV2,
        reason: str,
        *,
        action_origin: str = "model_predictive",
        out_of_distribution: bool = False,
        readiness_delay: bool = False,
    ) -> PolicyV2Decision:
        hold = self._hold_candidate(scored, current)
        return PolicyV2Decision(
            service=node.name,
            controller_id=self.registration.controller_id,
            policy_id=self.config.policy_id,
            resource_weight=self.config.resource_weight,
            sla_risk_weight=self.config.sla_risk_weight,
            churn_weight=self.config.churn_weight,
            reliability_weight=self.config.reliability_weight,
            previous_placement=current,
            candidates=scored,
            selected_candidate_id=selected.candidate.candidate_id,
            reason=reason,
            action_origin=action_origin,
            out_of_distribution=out_of_distribution,
            objective_improvement=max(0.0, hold.objective - selected.objective),
            readiness_delay=readiness_delay,
        )

    def _evaluate_service(
        self,
        service_name: str,
        state: ServiceRuntimeStateV2,
        scorer: CandidateScorerV2 | None,
        *,
        evaluate_emergency: bool = True,
    ) -> PolicyV2Decision:
        node = self.profile.node(service_name)
        enumeration = enumerate_feasible_candidates(
            self.profile,
            service_name,
            state,
            max_candidates=self.config.max_candidates_per_service,
        )
        current = _canonical_placement(state.desired_replicas, self.clusters)
        memory = self.memory[service_name]
        for cluster in self.clusters:
            if int(state.pending_replicas.get(cluster, 0)) == 0:
                memory.pending_timeout_buckets.pop(cluster, None)

        if not node.scalable:
            memory.healthy_windows = 0
            scored = self._score(enumeration, state, {})
            hold = self._hold_candidate(scored, current)
            return self._decision(node, current, scored, hold, "fixed-service")
        if not state.exists:
            memory.healthy_windows = 0
            scored = self._score(enumeration, state, {})
            hold = self._hold_candidate(scored, current)
            return self._decision(node, current, scored, hold, "absent-service")

        reactive_triggers = self._reactive_triggers(state)
        if reactive_triggers and evaluate_emergency and self.config.reactive_emergency_enabled:
            memory.healthy_windows = 0
            predictions: dict[str, CandidatePredictionV2] = {}
            scored = self._score(enumeration, state, predictions)
            hold = self._hold_candidate(scored, current)
            current_by_cluster = dict(current)
            maximum = sum(
                int(limits["max_replicas"])
                for limits in self.profile.cluster_limits.values()
            )
            target = min(maximum, state.total_desired + self.config.reactive_scale_out_step)
            emergency = [
                candidate
                for candidate in scored
                if candidate.candidate.total_replicas == target
                and all(
                    replicas >= current_by_cluster[cluster]
                    for cluster, replicas in candidate.candidate.placement
                )
            ]
            # Emergency replicas must follow an observed hot or receiving cluster.
            # A global maximum alone cannot identify where a pod can serve traffic.
            preferred_cluster = None
            if "cpu_utilization" in reactive_triggers:
                hot = {
                    cluster: value for cluster, value in state.cluster_cpu_utilization.items()
                    if value >= self.config.reactive_cpu_utilization
                }
                if hot:
                    preferred_cluster = max(hot, key=lambda cluster: (hot[cluster], cluster))
            if preferred_cluster is None:
                receiving = {cluster: value for cluster, value in state.cluster_request_rate.items() if value > 0}
                if receiving:
                    preferred_cluster = max(receiving, key=lambda cluster: (receiving[cluster], cluster))
            local_candidates = [
                candidate for candidate in emergency
                if preferred_cluster is not None
                and dict(candidate.candidate.placement)[preferred_cluster] > current_by_cluster[preferred_cluster]
            ]
            pending_guard_reason = None
            pending_timeout_bucket = None
            if preferred_cluster is not None and state.pending_replicas[preferred_cluster] > 0:
                pending_age = float(state.pending_age_seconds[preferred_cluster])
                if pending_age < self.config.reactive_pending_wait_seconds:
                    local_candidates = []
                    pending_guard_reason = "pending-readiness"
                else:
                    pending_timeout_bucket = int(
                        pending_age // self.config.reactive_pending_wait_seconds
                    )
                    last_bucket = memory.pending_timeout_buckets.get(preferred_cluster, 0)
                    if pending_timeout_bucket <= last_bucket:
                        local_candidates = []
                        pending_guard_reason = "pending-timeout-cooldown"
                    else:
                        pending_guard_reason = "pending-timeout"
            selected = min(
                local_candidates,
                key=lambda candidate: (-dict(candidate.candidate.placement)[preferred_cluster], self._selection_key(candidate)),
            ) if local_candidates else hold
            if (
                pending_guard_reason == "pending-timeout"
                and selected.candidate.placement != current
                and pending_timeout_bucket is not None
                and preferred_cluster is not None
            ):
                memory.pending_timeout_buckets[preferred_cluster] = pending_timeout_bucket
            reason = (
                "reactive-emergency:"
                + ",".join(reactive_triggers)
                + ":route-aware-fallback"
            )
            if pending_guard_reason is not None:
                reason += f":{pending_guard_reason}:{preferred_cluster}"
            if selected.candidate.placement == current and pending_guard_reason is None:
                if state.total_desired >= maximum:
                    reason += ":at-maximum"
                else:
                    reason += ":placement-unknown" if preferred_cluster is None else ":local-capacity-unavailable"
            return self._decision(
                node,
                current,
                scored,
                selected,
                reason,
                action_origin="reactive_emergency",
                out_of_distribution=True,
                readiness_delay=state.total_pending > 0,
            )

        predictions = {}
        scorer_failed = False
        if scorer is not None and state.predictive_fresh:
            try:
                predictions = dict(scorer(service_name, enumeration.candidates))
                scored = self._score(enumeration, state, predictions)
            except Exception:
                predictions = {}
                scored = self._score(enumeration, state, predictions)
                scorer_failed = True
        else:
            scored = self._score(enumeration, state, predictions)
        hold = self._hold_candidate(scored, current)

        if state.total_pending > 0:
            memory.healthy_windows = 0
            return self._decision(
                node,
                current,
                scored,
                hold,
                "pending-readiness",
                readiness_delay=True,
            )
        if not state.predictive_fresh:
            memory.healthy_windows = 0
            return self._decision(node, current, scored, hold, "predictive-input-stale")
        if scorer_failed:
            memory.healthy_windows = 0
            return self._decision(node, current, scored, hold, "candidate-scorer-failed")
        if not predictions or any(not candidate.prediction_available for candidate in scored):
            memory.healthy_windows = 0
            return self._decision(node, current, scored, hold, "incomplete-candidate-predictions")

        minimum_total = max(0, state.total_desired - self.config.scale_in_step)
        maximum_total = state.total_desired + self.config.predictive_scale_out_step
        admissible = [
            candidate
            for candidate in scored
            if minimum_total <= candidate.candidate.total_replicas <= maximum_total
        ]
        selected = min(admissible, key=self._selection_key)
        selected_total = selected.candidate.total_replicas
        if selected_total > state.total_desired:
            memory.healthy_windows = 0
            return self._decision(node, current, scored, selected, "predictive-scale-out")
        if selected_total == state.total_desired:
            memory.healthy_windows = 0
            reason = "hold-minimum-objective" if selected.candidate.placement == current else "predictive-placement"
            return self._decision(node, current, scored, selected, reason)

        healthy = self._healthy_for_scale_in(state, selected)
        memory.healthy_windows = memory.healthy_windows + 1 if healthy else 0
        if not healthy:
            return self._decision(node, current, scored, hold, "scale-in-safety-hold")
        if memory.healthy_windows < self.config.scale_down_healthy_windows:
            return self._decision(
                node,
                current,
                scored,
                hold,
                "scale-in-healthy-windows-"
                f"{memory.healthy_windows}/{self.config.scale_down_healthy_windows}",
            )
        recency = state.time_since_last_action_seconds
        if recency is None or recency < self.config.cooldown_seconds:
            return self._decision(node, current, scored, hold, "scale-in-cooldown")
        return self._decision(node, current, scored, selected, "utilization-aware-scale-in")

    def decide(
        self,
        service_name: str,
        state: ServiceRuntimeStateV2,
        scorer: CandidateScorerV2 | None,
    ) -> PolicyV2Decision:
        decision = self._evaluate_service(service_name, state, scorer)
        if decision.changed:
            self.memory[service_name].healthy_windows = 0
        return decision

    def diagnose_interval(
        self,
        states: Mapping[str, ServiceRuntimeStateV2],
        scorer: CandidateScorerV2 | None,
        *,
        initial_memory: Mapping[str, _PolicyV2Memory] | None = None,
    ) -> dict[str, PolicyV2Decision]:
        """Evaluate local model proposals on isolated memory, with no actuation.

        This intentionally retains every proposal before interval serialization
        or another service's emergency can suppress it. All predictive safety
        constraints are evaluated by the same implementation as real control.
        """
        observer = copy.copy(self)
        observer.memory = copy.deepcopy(initial_memory if initial_memory is not None else self.memory)
        return {
            node.name: observer._evaluate_service(
                node.name, states[node.name], scorer, evaluate_emergency=False,
            )
            for node in self.profile.nodes
        }

    @staticmethod
    def _replace_with_hold(decision: PolicyV2Decision, reason: str) -> PolicyV2Decision:
        hold = HeadlightPolicyV2._hold_candidate(
            decision.candidates,
            decision.previous_placement,
        )
        return replace(
            decision,
            selected_candidate_id=hold.candidate.candidate_id,
            reason=reason,
            action_origin="model_predictive",
            out_of_distribution=False,
            objective_improvement=0.0,
            readiness_delay=False,
        )

    def decide_interval(
        self,
        states: Mapping[str, ServiceRuntimeStateV2],
        scorer: CandidateScorerV2 | None,
    ) -> tuple[PolicyV2Decision, ...]:
        missing = sorted(set(self.profile.node_order) - set(states))
        if missing:
            raise ValueError(f"Policy-v2 interval is missing service state: {missing}")
        decisions = [
            self._evaluate_service(node.name, states[node.name], scorer)
            for node in self.profile.nodes
        ]
        emergencies = [
            decision
            for decision in decisions
            if decision.action_origin == "reactive_emergency"
        ]
        predictive = [
            decision
            for decision in decisions
            if decision.changed and decision.action_origin == "model_predictive"
        ]
        winner: PolicyV2Decision | None = None
        if not emergencies and predictive:
            winner = min(
                predictive,
                key=lambda decision: (-decision.objective_improvement, decision.service),
            )
        final = []
        for decision in decisions:
            if decision in predictive and decision is not winner:
                suffix = "emergency-priority" if emergencies else f"serialized-behind:{winner.service}"
                final.append(self._replace_with_hold(decision, suffix))
            else:
                final.append(decision)
                if decision.changed:
                    self.memory[decision.service].healthy_windows = 0
        return tuple(final)


@dataclass(frozen=True)
class PolicyV2RecordContext:
    run_id: str
    seed: int
    feature_schema_hash: str
    adjacency_hash: str
    controller: str
    bundle_id: str


def _masked_value(state: ServiceRuntimeStateV2, field_name: str) -> float:
    value = getattr(state, field_name)
    return float(value) if value is not None and math.isfinite(float(value)) else 0.0


def build_policy_v2_decision_record(
    decision: PolicyV2Decision,
    state: ServiceRuntimeStateV2,
    context: PolicyV2RecordContext,
    *,
    timestamp: float,
    forecast_quantiles_by_horizon: Mapping[str, Mapping[str, float]],
    action_timestamps: Mapping[str, Any],
    shadow_mode: bool,
) -> dict[str, Any]:
    """Emit a strict v2 record without inferring action lifecycle timestamps."""

    if shadow_mode and action_timestamps.get("status") != "not_requested":
        raise ValueError("shadow Policy-v2 decisions cannot contain requested action timestamps")
    if context.controller != decision.controller_id:
        raise ValueError("decision-record controller differs from the Policy-v2 registration")
    candidate_readiness = {
        scored.candidate.candidate_id: {
            "ready": scored.candidate.readiness.ready_dict(),
            "pending": scored.candidate.readiness.pending_dict(),
            "existing_pending": dict(scored.candidate.readiness.existing_pending_replicas),
            "ready_fraction": scored.candidate.readiness.ready_fraction,
            "pending_age_seconds": scored.candidate.readiness.pending_age_seconds,
            "weighted_objective": scored.objective,
        }
        for scored in decision.candidates
    }
    record = {
        "schema_version": DECISION_RECORD_SCHEMA_VERSION,
        "record_id": f"{context.run_id}:{int(timestamp * 1000)}:{decision.service}",
        "run_id": context.run_id,
        "seed": int(context.seed),
        "timestamp": float(timestamp),
        "service": decision.service,
        "observed_state": {
            "demand_rps": _masked_value(state, "demand_rps"),
            "cpu_utilization": _masked_value(state, "cpu_utilization"),
            "p95_latency_ms": _masked_value(state, "p95_latency_ms"),
            "error_rate": _masked_value(state, "error_rate"),
        },
        "masks": {
            "missing": {
                field_name: field_name in state.missing_fields
                or getattr(state, field_name) is None
                for field_name in ("demand_rps", "cpu_utilization", "p95_latency_ms", "error_rate")
            },
            "stale": {
                field_name: field_name in state.stale_fields
                for field_name in ("demand_rps", "cpu_utilization", "p95_latency_ms", "error_rate")
            },
        },
        "replicas_by_cluster": {
            "ready": {cluster: int(state.ready_replicas[cluster]) for cluster in decision.previous_placement_dict},
            "desired": decision.previous_placement_dict,
            "pending": {cluster: int(state.pending_replicas[cluster]) for cluster in decision.previous_placement_dict},
        },
        "time_since_last_action_seconds": state.time_since_last_action_seconds,
        "forecast_quantiles_by_horizon": {
            str(horizon): dict(quantiles)
            for horizon, quantiles in forecast_quantiles_by_horizon.items()
        },
        "candidates": [candidate.to_decision_candidate() for candidate in decision.candidates],
        "selected_candidate_id": decision.selected_candidate_id,
        "selection_reason": decision.reason,
        "selection_rule": POLICY_V2_SELECTION_RULE,
        "action_timestamps": dict(action_timestamps),
        "policy_schema_version": POLICY_V2_SCHEMA_VERSION,
        "policy_version": POLICY_V2_VERSION,
        "policy_id": decision.policy_id,
        "feature_schema_hash": context.feature_schema_hash,
        "adjacency_hash": context.adjacency_hash,
        "controller": context.controller,
        "bundle_id": context.bundle_id,
        "action_origin": decision.action_origin,
        "out_of_distribution": decision.out_of_distribution,
        "metadata": {
            "shadow_mode": bool(shadow_mode),
            "queue_depth": _masked_value(state, "queue_depth"),
            "queue_missing": "queue_depth" in state.missing_fields or state.queue_depth is None,
            "queue_stale": "queue_depth" in state.stale_fields,
            "candidate_readiness": candidate_readiness,
            "objective": {
                "formula": "alpha*resource + lambda*sla_risk + gamma*churn + rho*(1-success)",
                "alpha": decision.resource_weight,
                "lambda": decision.sla_risk_weight,
                "gamma": decision.churn_weight,
                "rho": decision.reliability_weight,
            },
        },
    }
    return validate_decision_record(record)


def _as_nested_list(value: Any, name: str) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a tensor-like nested array")
    return value


def _nested_shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        raise ValueError(f"{name} must not contain empty axes")
    child_shapes = [_nested_shape(child, name) for child in value]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError(f"{name} must be rectangular")
    return (len(value), *child_shapes[0])


def calibrated_predictions_from_model_output(
    controller_id: str,
    candidates: tuple[ReplicaPlacementCandidateV2, ...],
    performance: Mapping[str, Any],
    *,
    expected_endpoint_names: tuple[str, ...],
    expected_horizons_seconds: tuple[int, ...],
) -> dict[str, CandidatePredictionV2]:
    """Conservatively aggregate calibrated endpoint/horizon model output."""

    registration = policy_v2_registration(controller_id)
    if performance.get("source") != registration.risk_source:
        raise ValueError("candidate-performance source does not match the controller registration")
    if performance.get("calibrated") is not True:
        raise ValueError("Policy-v2 requires explicitly calibrated candidate performance")
    if not expected_endpoint_names or not expected_horizons_seconds:
        raise ValueError("Policy-v2 expected endpoint and horizon axes must be non-empty")
    if tuple(performance.get("endpoint_names", ())) != expected_endpoint_names:
        raise ValueError("candidate-performance endpoint identity differs from the bundle")
    if tuple(performance.get("horizons_seconds", ())) != expected_horizons_seconds:
        raise ValueError("candidate-performance horizon identity differs from the bundle")
    latency_batches = _as_nested_list(
        performance.get("predicted_p95_latency_ms"),
        "predicted_p95_latency_ms",
    )
    risk_batches = _as_nested_list(
        performance.get("sla_violation_probability"),
        "sla_violation_probability",
    )
    success_batches = (
        _as_nested_list(performance.get("success_probability"), "success_probability")
        if performance.get("success_probability") is not None
        else None
    )
    expected_shape = (
        1,
        len(candidates),
        len(expected_endpoint_names),
        len(expected_horizons_seconds),
    )
    latency_shape = _nested_shape(latency_batches, "predicted_p95_latency_ms")
    risk_shape = _nested_shape(risk_batches, "sla_violation_probability")
    if latency_shape != risk_shape:
        raise ValueError(
            "candidate latency and SLA-risk shapes differ: "
            f"latency={latency_shape}, risk={risk_shape}"
        )
    if latency_shape != expected_shape:
        raise ValueError(
            "candidate performance must have exact "
            f"[batch,candidate,endpoint,horizon] shape {expected_shape}; "
            f"got latency={latency_shape}, risk={risk_shape}"
        )
    latencies = latency_batches[0]
    risks = risk_batches[0]
    successes = success_batches[0] if success_batches is not None else None
    if success_batches is not None and _nested_shape(success_batches, "success_probability") != expected_shape:
        raise ValueError(f"success_probability must have exact shape {expected_shape}")

    predictions: dict[str, CandidatePredictionV2] = {}
    for index, candidate in enumerate(candidates):
        try:
            candidate_latencies = [
                float(value)
                for endpoint in latencies[index]
                for value in endpoint
            ]
            candidate_risks = [
                float(value)
                for endpoint in risks[index]
                for value in endpoint
            ]
            candidate_successes = (
                [float(value) for endpoint in successes[index] for value in endpoint]
                if successes is not None
                else []
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate performance contains a non-numeric value") from exc
        if any(not math.isfinite(value) for value in candidate_latencies + candidate_risks + candidate_successes):
            raise ValueError("candidate performance must contain only finite values")
        if any(value < 0 for value in candidate_latencies):
            raise ValueError("predicted_p95_latency_ms must be non-negative")
        if any(value < 0 or value > 1 for value in candidate_risks):
            raise ValueError("sla_violation_probability must be in [0, 1]")
        if any(value < 0 or value > 1 for value in candidate_successes):
            raise ValueError("success_probability must be in [0, 1]")
        predictions[candidate.candidate_id] = CandidatePredictionV2(
            predicted_p95_latency_ms=max(candidate_latencies),
            sla_violation_probability=max(candidate_risks),
            success_probability=min(candidate_successes) if candidate_successes else None,
        )
    return predictions


@dataclass(frozen=True)
class PolicyV2ExecutionResult:
    decisions: tuple[PolicyV2Decision, ...]
    actionable: tuple[PolicyV2Decision, ...]
    shadow_mode: bool
    actuator_called: bool


class PolicyV2Executor:
    """Record every interval decision and optionally invoke an external actuator."""

    def __init__(self, policy: HeadlightPolicyV2, *, shadow_mode: bool = False) -> None:
        self.policy = policy
        self.shadow_mode = bool(shadow_mode)

    def execute(
        self,
        states: Mapping[str, ServiceRuntimeStateV2],
        scorer: CandidateScorerV2 | None,
        *,
        recorder: Callable[[PolicyV2Decision], None],
        actuator: Callable[[tuple[PolicyV2Decision, ...]], None] | None = None,
    ) -> PolicyV2ExecutionResult:
        decisions = self.policy.decide_interval(states, scorer)
        for decision in decisions:
            recorder(decision)
        actionable = tuple(decision for decision in decisions if decision.changed)
        actuator_called = False
        if actionable and not self.shadow_mode:
            if actuator is None:
                raise ValueError("non-shadow Policy-v2 execution requires an actuator")
            actuator(actionable)
            actuator_called = True
        return PolicyV2ExecutionResult(
            decisions=decisions,
            actionable=actionable,
            shadow_mode=self.shadow_mode,
            actuator_called=actuator_called,
        )
