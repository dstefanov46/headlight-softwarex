from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.policy_v2 import (  # noqa: E402
    DEVELOPMENT_SCALE_DOWN_COOLDOWN_SECONDS,
    DEVELOPMENT_SCALE_DOWN_HEALTHY_WINDOWS,
    HEADLIGHT_V2_FULL_CONTROLLER,
    HEADLIGHT_V2_NO_ACTION_CONTROLLER,
    CandidatePredictionV2,
    HeadlightPolicyV2,
    PolicyV2Config,
    PolicyV2Executor,
    PolicyV2RecordContext,
    ServiceRuntimeStateV2,
    build_policy_v2_decision_record,
    calibrated_predictions_from_model_output,
    enumerate_feasible_candidates,
    estimate_candidate_enumeration_count,
    policy_v2_registration,
)
from headlight.profile import load_service_profile  # noqa: E402


PROFILE_DIR = REPO_ROOT / "ai-module" / "configs" / "service_profiles"


class PolicyV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_service_profile(PROFILE_DIR / "advanced-ticket.json")

    def state(
        self,
        *,
        desired: dict[str, int] | None = None,
        ready: dict[str, int] | None = None,
        pending: dict[str, int] | None = None,
        pending_age: dict[str, float] | None = None,
        demand: float = 100.0,
        cpu: float = 0.3,
        queue: float = 0.0,
        latency: float = 300.0,
        error: float = 0.0,
        predictive_fresh: bool = True,
        missing: frozenset[str] = frozenset(),
        stale: frozenset[str] = frozenset(),
        recency: float | None = 600.0,
        exists: bool = True,
        cluster_requests: dict[str, float] | None = None,
        cluster_cpu: dict[str, float] | None = None,
    ) -> ServiceRuntimeStateV2:
        desired = desired or {"cluster2": 1, "cluster3": 1}
        ready = ready or dict(desired)
        pending = pending or {"cluster2": 0, "cluster3": 0}
        pending_age = pending_age or {"cluster2": 0.0, "cluster3": 0.0}
        return ServiceRuntimeStateV2(
            desired_replicas=desired,
            ready_replicas=ready,
            pending_replicas=pending,
            pending_age_seconds=pending_age,
            demand_rps=demand,
            cpu_utilization=cpu,
            queue_depth=queue,
            p95_latency_ms=latency,
            error_rate=error,
            predictive_fresh=predictive_fresh,
            missing_fields=missing,
            stale_fields=stale,
            time_since_last_action_seconds=recency,
            exists=exists,
            cluster_request_rate={"cluster2": demand} if cluster_requests is None else cluster_requests,
            cluster_cpu_utilization={"cluster2": cpu} if cluster_cpu is None else cluster_cpu,
        )

    @staticmethod
    def scorer_for(risk_for_total, latency_for_total=lambda _total: 500.0):
        def score(_service, candidates):
            return {
                candidate.candidate_id: CandidatePredictionV2(
                    predicted_p95_latency_ms=float(latency_for_total(candidate.total_replicas)),
                    sla_violation_probability=float(risk_for_total(candidate.total_replicas)),
                )
                for candidate in candidates
            }

        return score

    def test_enumerates_every_feasible_total_and_placement(self) -> None:
        enumeration = enumerate_feasible_candidates(
            self.profile,
            "ts-auth",
            self.state(),
        )
        self.assertEqual(len(enumeration.candidates), 6 * 8)
        self.assertEqual(
            {candidate.total_replicas for candidate in enumeration.candidates},
            set(range(2, 15)),
        )
        self.assertEqual(
            len({candidate.placement for candidate in enumeration.candidates}),
            len(enumeration.candidates),
        )
        for candidate in enumeration.candidates:
            placement = candidate.placement_dict()
            self.assertIn(placement["cluster2"], range(1, 7))
            self.assertIn(placement["cluster3"], range(1, 9))
            self.assertEqual(sum(placement.values()), candidate.total_replicas)

    def test_diagnostic_model_proposals_survive_emergencies_without_changing_memory(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        states = {node.name: self.state() for node in self.profile.nodes}
        states["ts-auth"] = self.state(cpu=0.95)
        score = self.scorer_for(lambda total: 0.8 if total == 2 else 0.01)
        initial = copy.deepcopy(policy.memory)
        proposals = policy.diagnose_interval(states, score)
        self.assertEqual(policy.memory, initial)
        self.assertEqual(proposals["ts-auth"].reason, "predictive-scale-out")
        self.assertTrue(proposals["ts-route"].changed)
        actual = policy.decide_interval(states, score)
        self.assertEqual(next(d for d in actual if d.service == "ts-auth").action_origin, "reactive_emergency")
        self.assertEqual(next(d for d in actual if d.service == "ts-route").reason, "emergency-priority")

    def test_diagnostic_evaluation_preserves_hybrid_sequence_and_scale_in_memory(self) -> None:
        baseline = HeadlightPolicyV2(self.profile)
        observed = HeadlightPolicyV2(self.profile)
        score = self.scorer_for(lambda total: 0.01 if total == 2 else 0.2)
        for cpu in (0.2, 0.2, 0.95, 0.2, 0.2, 0.2):
            states = {node.name: self.state(desired={"cluster2": 2, "cluster3": 1}, cpu=cpu) for node in self.profile.nodes}
            observed.diagnose_interval(states, score)
            self.assertEqual(observed.decide_interval(states, score), baseline.decide_interval(states, score))
            self.assertEqual(observed.memory, baseline.memory)

    def test_diagnostic_missing_inputs_and_failure_hold_without_touching_memory(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        states = {node.name: self.state(cpu=0.95, predictive_fresh=False) for node in self.profile.nodes}
        def fail(_service, _candidates):
            raise RuntimeError("fixture inference failure")
        initial = copy.deepcopy(policy.memory)
        stale = policy.diagnose_interval(states, fail)
        self.assertEqual(stale["ts-auth"].reason, "predictive-input-stale")
        states["ts-auth"] = self.state(cpu=0.95)
        failed = policy.diagnose_interval(states, fail)
        self.assertEqual(failed["ts-auth"].reason, "candidate-scorer-failed")
        self.assertEqual(policy.memory, initial)

    def test_emergency_disabled_bypasses_only_guard_and_preserves_predictive_serialization(self) -> None:
        hybrid = HeadlightPolicyV2(self.profile)
        ablated = HeadlightPolicyV2(self.profile, config=PolicyV2Config(reactive_emergency_enabled=False))
        states = {node.name: self.state() for node in self.profile.nodes}
        states["ts-auth"] = self.state(cpu=0.95)
        score = self.scorer_for(lambda total: 0.8 if total == 2 else 0.01)
        a, b = hybrid.decide_interval(states, score), ablated.decide_interval(states, score)
        self.assertTrue(any(d.action_origin == "reactive_emergency" for d in a))
        self.assertFalse(any(d.action_origin == "reactive_emergency" or d.reason == "emergency-priority" for d in b))
        self.assertEqual(sum(d.changed for d in b), 1)
        self.assertTrue(any(d.reason.startswith("serialized-behind:") for d in b))
        self.assertEqual(ablated._reactive_triggers(states["ts-auth"]), ("cpu_utilization",))

    def test_emergency_disabled_preserves_constraints_default_mode_and_nonemergency_sequences(self) -> None:
        default = HeadlightPolicyV2(self.profile)
        explicit = HeadlightPolicyV2(self.profile, config=PolicyV2Config(reactive_emergency_enabled=True))
        ablated = HeadlightPolicyV2(self.profile, config=PolicyV2Config(reactive_emergency_enabled=False))
        score = self.scorer_for(lambda total: 0.01 if total == 2 else 0.2)
        cases = (
            self.state(desired={"cluster2": 2, "cluster3": 1}),
            self.state(desired={"cluster2": 2, "cluster3": 1}, recency=0),
            self.state(predictive_fresh=False),
            self.state(desired={"cluster2": 2, "cluster3": 1}, ready={"cluster2": 1, "cluster3": 1},
                       pending={"cluster2": 1, "cluster3": 0}),
            self.state(),
        )
        for state in (*cases, *cases):
            states = {node.name: state for node in self.profile.nodes}
            result = default.decide_interval(states, score)
            self.assertEqual(result, explicit.decide_interval(states, score))
            self.assertEqual(result, ablated.decide_interval(states, score))
            self.assertEqual(default.memory, ablated.memory)
        hot_pending = self.state(cpu=0.95, desired={"cluster2": 2, "cluster3": 1},
                                 ready={"cluster2": 1, "cluster3": 1}, pending={"cluster2": 1, "cluster3": 0})
        self.assertEqual(ablated.decide("ts-auth", hot_pending, score).reason, "pending-readiness")

    def test_boundary_constraints_and_candidate_cost_guard(self) -> None:
        self.assertEqual(estimate_candidate_enumeration_count(self.profile), 48)
        enumeration = enumerate_feasible_candidates(
            self.profile,
            "ts-auth",
            self.state(),
            max_candidates=48,
        )
        self.assertEqual(enumeration.placements_examined, 48)
        self.assertEqual(enumeration.theoretical_candidate_count, 48)
        with self.assertRaisesRegex(ValueError, "exceeds configured limit"):
            enumerate_feasible_candidates(
                self.profile,
                "ts-auth",
                self.state(),
                max_candidates=47,
            )
        invalid = self.state(desired={"cluster2": 7, "cluster3": 1})
        policy = HeadlightPolicyV2(self.profile)
        with self.assertRaisesRegex(ValueError, "outside the profile"):
            policy.decide("ts-auth", invalid, self.scorer_for(lambda _total: 0.1))

    def test_candidate_readiness_reflects_ready_pending_and_pending_age(self) -> None:
        state = self.state(
            desired={"cluster2": 2, "cluster3": 1},
            ready={"cluster2": 1, "cluster3": 1},
            pending={"cluster2": 1, "cluster3": 0},
            pending_age={"cluster2": 45.0, "cluster3": 0.0},
        )
        enumeration = enumerate_feasible_candidates(self.profile, "ts-auth", state)
        hold = next(
            candidate
            for candidate in enumeration.candidates
            if candidate.placement_dict() == {"cluster2": 2, "cluster3": 1}
        )
        self.assertEqual(hold.readiness.ready_dict(), {"cluster2": 1, "cluster3": 1})
        self.assertEqual(hold.readiness.pending_dict(), {"cluster2": 1, "cluster3": 0})
        self.assertEqual(hold.readiness.pending_age_seconds, 45.0)
        policy = HeadlightPolicyV2(self.profile)
        decision = policy.decide("ts-auth", state, self.scorer_for(lambda _total: 0.0))
        self.assertFalse(decision.changed)
        self.assertEqual(decision.reason, "pending-readiness")
        self.assertTrue(decision.readiness_delay)

    def test_stale_reactive_telemetry_cannot_trigger_scale_out(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        state = self.state(
            cpu=0.99,
            predictive_fresh=False,
            stale=frozenset({"cpu_utilization"}),
        )
        decision = policy.decide("ts-auth", state, None)
        self.assertFalse(decision.changed)
        self.assertEqual(decision.reason, "predictive-input-stale")
        self.assertEqual(decision.action_origin, "model_predictive")

    def test_reactive_guard_forces_emergency_scale_out(self) -> None:
        policy = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(reactive_scale_out_step=2),
        )
        state = self.state(error=0.2, predictive_fresh=False)
        decision = policy.decide("ts-auth", state, None)
        self.assertEqual(decision.selected.candidate.total_replicas, 4)
        self.assertTrue(decision.changed)
        self.assertEqual(decision.action_origin, "reactive_emergency")
        self.assertTrue(decision.out_of_distribution)
        self.assertIn("error_rate", decision.reason)

    def test_emergency_never_reduces_a_cluster_and_bypasses_scorer(self) -> None:
        policy = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(reactive_scale_out_step=1, sla_risk_weight=10.0),
        )
        state = self.state(
            desired={"cluster2": 2, "cluster3": 1},
            ready={"cluster2": 2, "cluster3": 1},
            cpu=0.99,
        )

        def favors_unsafe_move(_service, candidates):
            return {
                candidate.candidate_id: CandidatePredictionV2(
                    100.0,
                    0.0
                    if candidate.placement_dict() == {"cluster2": 1, "cluster3": 3}
                    else 1.0,
                )
                for candidate in candidates
            }

        decision = policy.decide("ts-auth", state, favors_unsafe_move)
        self.assertEqual(decision.selected.candidate.total_replicas, 4)
        self.assertGreaterEqual(decision.desired_placement["cluster2"], 2)
        self.assertGreaterEqual(decision.desired_placement["cluster3"], 1)

        scorer_calls = []

        def failing_scorer(_service, _candidates):
            scorer_calls.append(True)
            raise RuntimeError("synthetic inference failure")

        fallback = HeadlightPolicyV2(self.profile).decide(
            "ts-auth",
            state,
            failing_scorer,
        )
        self.assertTrue(fallback.changed)
        self.assertEqual(fallback.selected.candidate.total_replicas, 4)
        self.assertEqual(fallback.action_origin, "reactive_emergency")
        self.assertIn("route-aware-fallback", fallback.reason)
        self.assertEqual(scorer_calls, [])

    def test_emergency_placement_follows_cluster_local_hotspot(self) -> None:
        for hot_cluster, expected in (
            ("cluster2", {"cluster2": 2, "cluster3": 1}),
            ("cluster3", {"cluster2": 1, "cluster3": 2}),
        ):
            with self.subTest(hot_cluster=hot_cluster):
                state = self.state(
                    cpu=0.95,
                    cluster_cpu={hot_cluster: 0.95},
                    cluster_requests={hot_cluster: 50.0},
                )
                decision = HeadlightPolicyV2(self.profile).decide("ts-auth", state, None)
                self.assertEqual(decision.desired_placement, expected)
                self.assertTrue(decision.changed)

    def test_emergency_waits_for_pending_replica_in_hot_cluster(self) -> None:
        state = self.state(
            desired={"cluster2": 2, "cluster3": 1},
            ready={"cluster2": 1, "cluster3": 1},
            pending={"cluster2": 1, "cluster3": 0},
            pending_age={"cluster2": 30.0, "cluster3": 0.0},
            cpu=0.99,
            cluster_cpu={"cluster2": 0.99},
            cluster_requests={"cluster2": 50.0},
        )
        decision = HeadlightPolicyV2(self.profile).decide("ts-auth", state, None)
        self.assertFalse(decision.changed)
        self.assertEqual(decision.desired_placement, state.desired_replicas)
        self.assertEqual(
            decision.reason,
            "reactive-emergency:cpu_utilization:route-aware-fallback:pending-readiness:cluster2",
        )
        self.assertTrue(decision.readiness_delay)

    def test_pending_in_other_cluster_does_not_block_hot_cluster_scaleout(self) -> None:
        state = self.state(
            desired={"cluster2": 1, "cluster3": 2},
            ready={"cluster2": 1, "cluster3": 1},
            pending={"cluster2": 0, "cluster3": 1},
            pending_age={"cluster2": 0.0, "cluster3": 30.0},
            cpu=0.99,
            cluster_cpu={"cluster2": 0.99},
            cluster_requests={"cluster2": 50.0},
        )
        decision = HeadlightPolicyV2(self.profile).decide("ts-auth", state, None)
        self.assertEqual(decision.desired_placement, {"cluster2": 2, "cluster3": 2})
        self.assertTrue(decision.changed)

    def test_pending_timeout_allows_one_guarded_scaleout_per_wait_window(self) -> None:
        policy = HeadlightPolicyV2(self.profile)

        def state_at(age: float) -> ServiceRuntimeStateV2:
            return self.state(
                desired={"cluster2": 2, "cluster3": 1},
                ready={"cluster2": 1, "cluster3": 1},
                pending={"cluster2": 1, "cluster3": 0},
                pending_age={"cluster2": age, "cluster3": 0.0},
                cpu=0.99,
                cluster_cpu={"cluster2": 0.99},
                cluster_requests={"cluster2": 50.0},
            )

        before_timeout = policy.decide("ts-auth", state_at(239.0), None)
        self.assertFalse(before_timeout.changed)
        self.assertIn("pending-readiness:cluster2", before_timeout.reason)

        after_timeout = policy.decide("ts-auth", state_at(240.0), None)
        self.assertTrue(after_timeout.changed)
        self.assertIn("pending-timeout:cluster2", after_timeout.reason)

        within_timeout_bucket = policy.decide("ts-auth", state_at(255.0), None)
        self.assertFalse(within_timeout_bucket.changed)
        self.assertIn("pending-timeout-cooldown:cluster2", within_timeout_bucket.reason)

        next_timeout_bucket = policy.decide("ts-auth", state_at(480.0), None)
        self.assertTrue(next_timeout_bucket.changed)
        self.assertIn("pending-timeout:cluster2", next_timeout_bucket.reason)

    def test_emergency_without_cluster_evidence_holds(self) -> None:
        state = self.state(cpu=0.95, cluster_cpu={}, cluster_requests={})
        decision = HeadlightPolicyV2(self.profile).decide("ts-auth", state, None)
        self.assertFalse(decision.changed)
        self.assertEqual(decision.desired_placement, {"cluster2": 1, "cluster3": 1})
        self.assertIn("placement-unknown", decision.reason)

    def test_latency_emergency_follows_receiving_cluster(self) -> None:
        state = self.state(
            cpu=0.2, latency=1200.0,
            cluster_cpu={"cluster2": 0.2, "cluster3": 0.2},
            cluster_requests={"cluster3": 30.0},
        )
        decision = HeadlightPolicyV2(self.profile).decide("ts-auth", state, None)
        self.assertEqual(decision.desired_placement, {"cluster2": 1, "cluster3": 2})
        self.assertIn("p95_latency_ms", decision.reason)

    def test_hot_cluster_at_capacity_does_not_use_unroutable_cluster(self) -> None:
        maximum = int(self.profile.cluster_limits["cluster2"]["max_replicas"])
        state = self.state(
            desired={"cluster2": maximum, "cluster3": 1},
            ready={"cluster2": maximum, "cluster3": 1},
            cpu=0.95, cluster_cpu={"cluster2": 0.95},
            cluster_requests={"cluster2": 30.0},
        )
        decision = HeadlightPolicyV2(self.profile).decide("ts-auth", state, None)
        self.assertFalse(decision.changed)
        self.assertIn("local-capacity-unavailable", decision.reason)

    def test_predictive_scale_out_uses_calibrated_candidate_risk(self) -> None:
        policy = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(sla_risk_weight=2.0, churn_weight=0.1),
        )
        scorer = self.scorer_for(lambda total: 0.8 if total <= 2 else 0.01)
        decision = policy.decide("ts-auth", self.state(), scorer)
        self.assertTrue(decision.changed)
        self.assertEqual(decision.reason, "predictive-scale-out")
        self.assertEqual(decision.selected.candidate.total_replicas, 3)
        for candidate in decision.candidates:
            self.assertAlmostEqual(
                candidate.objective,
                candidate.resource_cost
                + 2.0 * candidate.sla_risk
                + 0.1 * candidate.churn_cost,
            )

    def test_safe_scale_in_releases_surplus_despite_high_forecast(self) -> None:
        policy = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(
                sla_risk_weight=2.0,
                scale_down_healthy_windows=3,
                cooldown_seconds=60,
            ),
        )
        state = self.state(
            desired={"cluster2": 2, "cluster3": 3},
            ready={"cluster2": 2, "cluster3": 3},
            demand=10_000.0,
            cpu=0.15,
            queue=0.0,
            latency=250.0,
            error=0.0,
            recency=600.0,
        )
        scorer = self.scorer_for(
            lambda total: 0.001 if total >= 4 else 0.9,
            lambda total: 400.0 if total >= 4 else 1500.0,
        )
        decisions = [policy.decide("ts-auth", state, scorer) for _ in range(3)]
        self.assertFalse(decisions[0].changed)
        self.assertFalse(decisions[1].changed)
        self.assertEqual(decisions[2].reason, "utilization-aware-scale-in")
        self.assertEqual(decisions[2].selected.candidate.total_replicas, 4)

    def test_live_scale_in_does_not_require_uncollected_queue_depth(self) -> None:
        policy = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(
                scale_down_healthy_windows=3,
                cooldown_seconds=60,
            ),
        )
        state = self.state(
            desired={"cluster2": 2, "cluster3": 3},
            ready={"cluster2": 2, "cluster3": 3},
            cpu=0.15,
            queue=None,
            latency=250.0,
            error=0.0,
            missing=frozenset({"queue_depth"}),
            recency=600.0,
        )
        scorer = self.scorer_for(
            lambda total: 0.001 if total >= 4 else 0.9,
            lambda total: 400.0 if total >= 4 else 1500.0,
        )
        decisions = [policy.decide("ts-auth", state, scorer) for _ in range(3)]
        self.assertEqual(decisions[-1].reason, "utilization-aware-scale-in")

        queue_required = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(
                scale_in_telemetry_fields=(
                    "cpu_utilization",
                    "queue_depth",
                    "p95_latency_ms",
                    "error_rate",
                ),
            ),
        )
        held = [queue_required.decide("ts-auth", state, scorer) for _ in range(3)]
        self.assertEqual(held[-1].reason, "scale-in-safety-hold")

    def test_scale_in_rejects_fields_without_health_thresholds(self) -> None:
        for fields in (("demand_rps",), ("cpu_utilization", "demand_rps")):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "scale-in telemetry fields"):
                    HeadlightPolicyV2(
                        self.profile,
                        config=PolicyV2Config(scale_in_telemetry_fields=fields),
                    )

    def test_scale_in_requires_cooldown_and_no_pending(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        scorer = self.scorer_for(
            lambda total: 0.001 if total >= 3 else 0.9,
            lambda total: 400.0 if total >= 3 else 1500.0,
        )
        cooling = self.state(
            desired={"cluster2": 2, "cluster3": 2},
            ready={"cluster2": 2, "cluster3": 2},
            recency=30.0,
        )
        decisions = [policy.decide("ts-auth", cooling, scorer) for _ in range(3)]
        self.assertEqual(decisions[-1].reason, "scale-in-cooldown")
        pending = self.state(
            desired={"cluster2": 2, "cluster3": 2},
            ready={"cluster2": 1, "cluster3": 2},
            pending={"cluster2": 1, "cluster3": 0},
            pending_age={"cluster2": 10.0, "cluster3": 0.0},
        )
        self.assertEqual(policy.decide("ts-auth", pending, scorer).reason, "pending-readiness")

    def test_deterministic_tie_uses_placement_order(self) -> None:
        policy = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(sla_risk_weight=2.0, churn_weight=0.0),
        )

        def tied(_service, candidates):
            return {
                candidate.candidate_id: CandidatePredictionV2(
                    500.0,
                    0.0 if candidate.total_replicas == 3 else 1.0,
                )
                for candidate in candidates
            }

        first = policy.decide("ts-auth", self.state(), tied)
        second = HeadlightPolicyV2(
            self.profile,
            config=PolicyV2Config(sla_risk_weight=2.0, churn_weight=0.0),
        ).decide("ts-auth", self.state(), tied)
        self.assertEqual(first.selected_candidate_id, second.selected_candidate_id)
        self.assertEqual(first.desired_placement, {"cluster2": 1, "cluster3": 2})

    def test_interval_serializes_predictive_actions_but_not_emergencies(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        states = {node.name: self.state() for node in self.profile.nodes}
        scorer = self.scorer_for(lambda total: 0.9 if total == 2 else 0.0)
        decisions = policy.decide_interval(states, scorer)
        predictive_changes = [
            decision
            for decision in decisions
            if decision.changed and decision.action_origin == "model_predictive"
        ]
        self.assertEqual(len(predictive_changes), 1)

        emergency_states = dict(states)
        emergency_states["ts-auth"] = self.state(cpu=0.99, predictive_fresh=False)
        emergency_states["ts-basic"] = self.state(error=0.2, predictive_fresh=False)
        emergencies = policy.decide_interval(emergency_states, scorer)
        emergency_changes = [
            decision
            for decision in emergencies
            if decision.changed and decision.action_origin == "reactive_emergency"
        ]
        self.assertEqual({decision.service for decision in emergency_changes}, {"ts-auth", "ts-basic"})
        self.assertTrue(all(decision.out_of_distribution for decision in emergency_changes))

    def test_emergency_at_maximum_suppresses_every_predictive_action(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        states = {node.name: self.state() for node in self.profile.nodes}
        states["ts-auth"] = self.state(
            desired={"cluster2": 6, "cluster3": 8},
            ready={"cluster2": 6, "cluster3": 8},
            cpu=0.99,
        )
        scorer = self.scorer_for(lambda total: 0.9 if total == 2 else 0.0)
        decisions = policy.decide_interval(states, scorer)
        emergency = next(decision for decision in decisions if decision.service == "ts-auth")
        self.assertEqual(emergency.action_origin, "reactive_emergency")
        self.assertFalse(emergency.changed)
        self.assertIn("at-maximum", emergency.reason)
        self.assertFalse(
            any(
                decision.changed and decision.action_origin == "model_predictive"
                for decision in decisions
            )
        )
        self.assertTrue(
            any(decision.reason == "emergency-priority" for decision in decisions)
        )

    def test_shadow_records_without_actuation(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        states = {node.name: self.state() for node in self.profile.nodes}
        scorer = self.scorer_for(lambda total: 0.9 if total == 2 else 0.0)
        recorded = []
        actuated = []
        result = PolicyV2Executor(policy, shadow_mode=True).execute(
            states,
            scorer,
            recorder=recorded.append,
            actuator=actuated.append,
        )
        self.assertEqual(len(recorded), len(self.profile.nodes))
        self.assertEqual(len(result.actionable), 1)
        self.assertFalse(result.actuator_called)
        self.assertEqual(actuated, [])

    def test_emits_every_candidate_through_strict_decision_schema(self) -> None:
        policy = HeadlightPolicyV2(self.profile)
        state = self.state()
        decision = policy.decide(
            "ts-auth",
            state,
            self.scorer_for(lambda total: 0.9 if total == 2 else 0.0),
        )
        record = build_policy_v2_decision_record(
            decision,
            state,
            PolicyV2RecordContext(
                run_id="shadow-401",
                seed=401,
                feature_schema_hash="a" * 64,
                adjacency_hash="b" * 64,
                controller=HEADLIGHT_V2_FULL_CONTROLLER,
                bundle_id="synthetic-bundle",
            ),
            timestamp=1_700_000_000.0,
            forecast_quantiles_by_horizon={"60": {"q50": 90.0, "q90": 120.0}},
            action_timestamps={
                "status": "not_requested",
                "requested_at": None,
                "acknowledged_at": None,
                "ready_at": None,
                "readiness_delay_seconds": None,
            },
            shadow_mode=True,
        )
        self.assertEqual(len(record["candidates"]), 48)
        self.assertEqual(record["selected_candidate_id"], decision.selected_candidate_id)
        self.assertTrue(record["metadata"]["shadow_mode"])
        self.assertEqual(len(record["metadata"]["candidate_readiness"]), 48)
        self.assertEqual(record["policy_id"], policy.config.policy_id)
        self.assertEqual(record["metadata"]["objective"]["lambda"], 1.0)

    def test_model_output_adapter_requires_calibration_and_registered_source(self) -> None:
        candidates = enumerate_feasible_candidates(
            self.profile,
            "ts-auth",
            self.state(),
        ).candidates[:2]
        performance = {
            "source": "learned-action-conditioned",
            "calibrated": True,
            "endpoint_names": ["preserve", "search"],
            "horizons_seconds": [60, 90],
            "predicted_p95_latency_ms": [
                [[[100.0, 200.0], [150.0, 180.0]], [[80.0, 90.0], [70.0, 85.0]]]
            ],
            "sla_violation_probability": [
                [[[0.01, 0.02], [0.03, 0.01]], [[0.001, 0.002], [0.004, 0.003]]]
            ],
        }
        predictions = calibrated_predictions_from_model_output(
            HEADLIGHT_V2_FULL_CONTROLLER,
            candidates,
            performance,
            expected_endpoint_names=("preserve", "search"),
            expected_horizons_seconds=(60, 90),
        )
        self.assertEqual(predictions[candidates[0].candidate_id].predicted_p95_latency_ms, 200.0)
        self.assertEqual(predictions[candidates[0].candidate_id].sla_violation_probability, 0.03)
        uncalibrated = dict(performance)
        uncalibrated["calibrated"] = False
        with self.assertRaisesRegex(ValueError, "explicitly calibrated"):
            calibrated_predictions_from_model_output(
                HEADLIGHT_V2_FULL_CONTROLLER,
                candidates,
                uncalibrated,
                expected_endpoint_names=("preserve", "search"),
                expected_horizons_seconds=(60, 90),
            )
        with self.assertRaisesRegex(ValueError, "source"):
            calibrated_predictions_from_model_output(
                HEADLIGHT_V2_NO_ACTION_CONTROLLER,
                candidates,
                performance,
                expected_endpoint_names=("preserve", "search"),
                expected_horizons_seconds=(60, 90),
            )

        truncated = copy.deepcopy(performance)
        truncated["predicted_p95_latency_ms"] = [
            [[[100.0, 200.0]], [[80.0, 90.0]]]
        ]
        truncated["sla_violation_probability"] = [
            [[[0.01, 0.02]], [[0.001, 0.002]]]
        ]
        with self.assertRaisesRegex(ValueError, "exact .* shape"):
            calibrated_predictions_from_model_output(
                HEADLIGHT_V2_FULL_CONTROLLER,
                candidates,
                truncated,
                expected_endpoint_names=("preserve", "search"),
                expected_horizons_seconds=(60, 90),
            )

        scalar = copy.deepcopy(performance)
        scalar["sla_violation_probability"] = [0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            calibrated_predictions_from_model_output(
                HEADLIGHT_V2_FULL_CONTROLLER,
                candidates,
                scalar,
                expected_endpoint_names=("preserve", "search"),
                expected_horizons_seconds=(60, 90),
            )

        wrong_horizons = copy.deepcopy(performance)
        wrong_horizons["horizons_seconds"] = [30, 60]
        with self.assertRaisesRegex(ValueError, "horizon identity"):
            calibrated_predictions_from_model_output(
                HEADLIGHT_V2_FULL_CONTROLLER,
                candidates,
                wrong_horizons,
                expected_endpoint_names=("preserve", "search"),
                expected_horizons_seconds=(60, 90),
            )

    def test_full_and_no_action_controller_registrations_share_policy(self) -> None:
        full = policy_v2_registration(HEADLIGHT_V2_FULL_CONTROLLER)
        ablation = policy_v2_registration(HEADLIGHT_V2_NO_ACTION_CONTROLLER)
        self.assertTrue(full.action_conditioning)
        self.assertFalse(ablation.action_conditioning)
        self.assertNotEqual(full.risk_source, ablation.risk_source)
        self.assertEqual(
            HeadlightPolicyV2(
                self.profile,
                controller_id=HEADLIGHT_V2_NO_ACTION_CONTROLLER,
            ).config,
            HeadlightPolicyV2(self.profile).config,
        )
        self.assertEqual(DEVELOPMENT_SCALE_DOWN_HEALTHY_WINDOWS, (3, 4))
        self.assertEqual(DEVELOPMENT_SCALE_DOWN_COOLDOWN_SECONDS, (60, 75, 90))
        with self.assertRaisesRegex(ValueError, "healthy windows"):
            HeadlightPolicyV2(
                self.profile,
                config=PolicyV2Config(scale_down_healthy_windows=12),
            )
        with self.assertRaisesRegex(ValueError, "cooldown"):
            HeadlightPolicyV2(
                self.profile,
                config=PolicyV2Config(cooldown_seconds=180),
            )
        with self.assertRaisesRegex(ValueError, "tuning seeds"):
            HeadlightPolicyV2(
                self.profile,
                config=PolicyV2Config(tuning_seeds=(501,)),
            )


if __name__ == "__main__":
    unittest.main()
