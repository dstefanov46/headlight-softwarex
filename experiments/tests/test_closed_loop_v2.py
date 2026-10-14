from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.closed_loop_v2 import (  # noqa: E402
    CASE_NAMES,
    CandidateActions,
    ClosedLoopConfig,
    ClosedLoopEnvironment,
    ControllerAction,
    HpaCpuEmulation,
    KedaRequestRateEmulation,
    ThresholdReactiveController,
    _balanced_placement,
    hold_action,
    run_episode,
)
from headlight.synthetic_v2 import DATABASE_NODES, NODE_ORDER  # noqa: E402


class ClosedLoopV2Tests(unittest.TestCase):
    def config(self) -> ClosedLoopConfig:
        return ClosedLoopConfig(steps=52, lookback_steps=24)

    def test_trace_pairing_is_deterministic_and_controller_independent(self) -> None:
        first = ClosedLoopEnvironment(case="flash-crowd", seed=701, config=self.config())
        second = ClosedLoopEnvironment(case="flash-crowd", seed=701, config=self.config())
        self.assertEqual(first.trace_hash, second.trace_hash)
        np.testing.assert_array_equal(first.demand_trace, second.demand_trace)
        first.step(hold_action(first.observe()))
        observation = second.observe()
        target = np.minimum(observation.desired_replicas + 2, 14)
        target[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
        second.step(
            ControllerAction(target, _balanced_placement(target), "test scale-out")
        )
        self.assertEqual(first.trace_hash, second.trace_hash)
        self.assertFalse(
            np.array_equal(first.desired_placement, second.desired_placement)
        )

    def test_selected_actions_change_later_readiness_and_latency(self) -> None:
        hold = ClosedLoopEnvironment(case="gradual-ramp", seed=702, config=self.config())
        scale = ClosedLoopEnvironment(case="gradual-ramp", seed=702, config=self.config())
        hold_observation = hold.observe()
        scale_observation = scale.observe()
        for _ in range(10):
            hold_observation, _ = hold.step(hold_action(hold_observation))
            target = np.minimum(scale_observation.desired_replicas + 2, 14)
            target[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
            scale_observation, _ = scale.step(
                ControllerAction(
                    target,
                    _balanced_placement(target),
                    "repeated scale-out",
                )
            )
        self.assertGreater(
            float(scale_observation.ready_replicas.sum()),
            float(hold_observation.ready_replicas.sum()),
        )
        self.assertLess(
            float(scale_observation.endpoint_p95_latency_ms.max()),
            float(hold_observation.endpoint_p95_latency_ms.max()),
        )

    def test_capacity_and_readiness_failure_rejects_or_delays_actions(self) -> None:
        environment = ClosedLoopEnvironment(
            case="capacity-readiness-failure", seed=703, config=self.config()
        )
        observation = environment.observe()
        target = np.full(len(NODE_ORDER), 14, dtype=np.int64)
        target[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
        placement = _balanced_placement(target, 0.20)
        next_observation, applied = environment.step(
            ControllerAction(target, placement, "capacity pressure")
        )
        self.assertGreater(applied.rejected_replicas + applied.replica_changes, 0)
        self.assertGreater(float(next_observation.pending_replicas.sum()), 0.0)
        self.assertLessEqual(
            int(next_observation.desired_placement[:, 1].sum()),
            environment.config.constrained_cluster_capacity[1],
        )

    def test_cooldown_applies_to_placement_only_changes(self) -> None:
        environment = ClosedLoopEnvironment(
            case="diurnal", seed=706, config=self.config()
        )
        observation = environment.observe()
        target = observation.desired_replicas + 2
        target[[name in DATABASE_NODES for name in NODE_ORDER]] = 2
        observation, _ = environment.step(
            ControllerAction(target, _balanced_placement(target), "first action")
        )
        previous = observation.desired_placement.copy()
        requested = previous.copy()
        scalable = np.asarray([name not in DATABASE_NODES for name in NODE_ORDER])
        requested[scalable, 0] = 1
        requested[scalable, 1] = observation.desired_replicas[scalable] - 1
        observation, applied = environment.step(
            ControllerAction(
                observation.desired_replicas,
                requested,
                "placement-only action during cooldown",
            )
        )
        self.assertGreater(applied.cooldown_holds, 0)
        np.testing.assert_array_equal(applied.applied.placement, previous)

    def test_emulated_controllers_produce_bounded_actions(self) -> None:
        environment = ClosedLoopEnvironment(
            case="flash-crowd", seed=704, config=self.config()
        )
        observation = environment.observe()
        candidates = environment.candidates(observation)
        for controller in (
            HpaCpuEmulation(),
            KedaRequestRateEmulation(),
            ThresholdReactiveController(),
        ):
            controller.reset()
            decision = controller.decide(
                observation, [observation], candidates, environment.preview
            )
            decision.action.validate()
            self.assertTrue(np.all(decision.action.desired_replicas >= 2))
            self.assertTrue(np.all(decision.action.desired_replicas <= 14))

    def test_candidate_generation_preserves_current_cluster_share(self) -> None:
        environment = ClosedLoopEnvironment(
            case="diurnal", seed=709, config=self.config()
        )
        observation = environment.observe()
        first = environment.candidates(observation)
        second = environment.candidates(observation)
        np.testing.assert_array_equal(first.placements, second.placements)
        np.testing.assert_array_equal(
            first.placements[0], observation.desired_placement
        )
        current_share = (
            observation.desired_placement[:, 0]
            / observation.desired_placement.sum(axis=-1)
        )
        for candidate in range(1, len(first.actions)):
            candidate_share = (
                first.placements[candidate, :, 0]
                / first.placements[candidate].sum(axis=-1)
            )
            tolerance = 1.0 / first.replicas[candidate]
            self.assertTrue(np.all(np.abs(candidate_share - current_share) <= tolerance))

    def test_synthetic_request_accounting_is_complete(self) -> None:
        observation = ClosedLoopEnvironment(
            case="flash-crowd", seed=708, config=self.config()
        ).observe()
        self.assertEqual(
            observation.total_requests,
            observation.successful_requests + observation.failed_requests,
        )
        self.assertEqual(
            observation.total_requests,
            sum(observation.http_status_distribution.values()),
        )
        self.assertEqual(
            observation.failed_requests,
            observation.timeout_count + observation.exception_count,
        )
        self.assertEqual(
            observation.successful_requests,
            len(observation.successful_latency_ms),
        )
        self.assertTrue(
            np.all(
                observation.endpoint_p50_latency_ms
                < observation.endpoint_p95_latency_ms
            )
        )
        self.assertTrue(
            np.all(
                observation.endpoint_p95_latency_ms
                < observation.endpoint_p99_latency_ms
            )
        )

    def test_episode_reports_requested_closed_loop_metrics(self) -> None:
        environment = ClosedLoopEnvironment(
            case="diurnal", seed=705, config=self.config()
        )
        result = run_episode(environment, ThresholdReactiveController())
        self.assertFalse(result["failed"])
        self.assertEqual(result["samples"], self.config().steps - self.config().lookback_steps)
        for metric in (
            "total_requests",
            "successful_requests",
            "failed_requests",
            "success_rate",
            "timeout_count",
            "exception_count",
            "http_200_count",
            "http_500_count",
            "http_504_count",
            "http_status_distribution",
            "request_p50_latency_ms",
            "request_p95_latency_ms",
            "request_p99_latency_ms",
            "request_sla_violation_rate",
            "excess_response_time_seconds",
            "p95_latency_ms",
            "sla_violation_rate",
            "endpoint_p95_sla_violation_rate",
            "successful_throughput_rps",
            "offered_throughput_rps",
            "mean_successful_throughput_rps",
            "cpu_seconds",
            "mean_pod_count",
            "pod_seconds",
            "replica_changes",
            "action_churn",
            "placement_changes",
            "migration_proxy",
            "mean_desired_replicas_cluster2",
            "mean_desired_replicas_cluster3",
            "final_service_placement",
            "ceiling_time_fraction",
            "pending_pod_seconds",
            "readiness_disruption_rate",
            "decision_mean_ms",
            "forecast_q50_mae_rps",
        ):
            self.assertIn(metric, result)
        self.assertEqual(
            result["total_requests"],
            result["successful_requests"] + result["failed_requests"],
        )
        self.assertEqual(
            result["total_requests"],
            sum(result["http_status_distribution"].values()),
        )
        self.assertAlmostEqual(
            result["mean_pod_count"],
            result["pod_seconds"] / result["measurement_seconds"],
        )

    def test_episode_metrics_are_deterministic_except_wall_clock_timing(self) -> None:
        results = [
            run_episode(
                ClosedLoopEnvironment(
                    case="flash-crowd", seed=707, config=self.config()
                ),
                ThresholdReactiveController(),
            )
            for _ in range(2)
        ]
        for result in results:
            result.pop("decision_mean_ms")
            result.pop("decision_p95_ms")
            result.pop("inference_mean_ms")
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
