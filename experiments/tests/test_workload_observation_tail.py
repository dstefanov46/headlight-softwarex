from __future__ import annotations

import unittest

from experiments.lib.workloads import generate_shape, workload_phase_windows


class WorkloadObservationTailTests(unittest.TestCase):
    def test_multi_phase_workload_shares_warmup_and_continues_final_diurnal_phase(self) -> None:
        workload = {
            "pattern": "multi-phase",
            "duration_seconds": 120,
            "phases": [
                {
                    "pattern": "gradual-ramp",
                    "duration_seconds": 40,
                    "locust": {
                        "spawn_rate": 5,
                        "start_users": "0.2C",
                        "end_users": "C",
                        "ramp_seconds": 30,
                        "hold_seconds": 10,
                    },
                },
                {
                    "pattern": "flash-crowd",
                    "duration_seconds": 40,
                    "locust": {
                        "spawn_rate": 10,
                        "base_users": "0.2C",
                        "spike_users": "C",
                        "spike_start_seconds": 10,
                        "spike_duration_seconds": 20,
                    },
                },
                {
                    "pattern": "diurnal",
                    "duration_seconds": 40,
                    "locust": {
                        "spawn_rate": 4,
                        "min_users": "0.2C",
                        "max_users": "C",
                        "cycle_seconds": 40,
                    },
                },
            ],
        }

        points = generate_shape(
            workload,
            seed=12,
            sampling_interval_seconds=10,
            warmup_seconds=20,
            duration_seconds=120,
            continuation_seconds=20,
            capacity_users=10,
        )

        self.assertEqual(points[-1].elapsed_seconds, 160)
        self.assertEqual(points[0].users, 2)
        self.assertEqual(points[6].users, 2)  # flash crowd begins at elapsed 60
        self.assertEqual(points[7].users, 10)  # flash crowd spike begins at elapsed 70
        self.assertEqual(points[10].users, 2)  # diurnal phase starts at elapsed 100
        self.assertEqual(points[10].spawn_rate, 4)

    def test_multi_phase_observation_tail_holds_final_endpoint_without_extending_diurnal_cycle(self) -> None:
        workload = {
            "pattern": "multi-phase",
            "duration_seconds": 120,
            "continuation_mode": "hold_endpoint",
            "phases": [
                {
                    "pattern": "gradual-ramp",
                    "duration_seconds": 40,
                    "locust": {
                        "spawn_rate": 5,
                        "start_users": 2,
                        "end_users": 10,
                        "ramp_seconds": 30,
                        "hold_seconds": 10,
                    },
                },
                {
                    "pattern": "flash-crowd",
                    "duration_seconds": 40,
                    "locust": {
                        "spawn_rate": 10,
                        "base_users": 2,
                        "spike_users": 10,
                        "spike_start_seconds": 10,
                        "spike_duration_seconds": 20,
                    },
                },
                {
                    "pattern": "diurnal",
                    "duration_seconds": 40,
                    "locust": {
                        "spawn_rate": 4,
                        "min_users": 2,
                        "max_users": 10,
                        "cycle_seconds": 40,
                    },
                },
            ],
        }

        points = generate_shape(
            workload,
            seed=12,
            sampling_interval_seconds=10,
            duration_seconds=120,
            continuation_seconds=20,
        )

        self.assertEqual([point.users for point in points[-3:]], [2, 2, 2])

    def test_multi_phase_workload_requires_phase_durations_to_match_measurement(self) -> None:
        workload = {
            "pattern": "multi-phase",
            "duration_seconds": 60,
            "phases": [
                {
                    "pattern": "diurnal",
                    "duration_seconds": 40,
                    "locust": {
                        "spawn_rate": 4,
                        "min_users": 2,
                        "max_users": 10,
                        "cycle_seconds": 40,
                    },
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "phase durations must sum to workload duration"):
            generate_shape(workload, 12, 10, duration_seconds=60, capacity_users=10)

    def test_multi_phase_windows_are_contiguous_with_measurement_relative_offsets(self) -> None:
        phases = [
            {"pattern": "gradual-ramp", "duration_seconds": 405},
            {"pattern": "flash-crowd", "duration_seconds": 390},
            {"pattern": "diurnal", "duration_seconds": 405},
        ]

        windows = workload_phase_windows(phases)

        self.assertEqual(
            windows,
            [
                {"pattern": "gradual-ramp", "start_offset_seconds": 0, "end_offset_seconds": 405},
                {"pattern": "flash-crowd", "start_offset_seconds": 405, "end_offset_seconds": 795},
                {"pattern": "diurnal", "start_offset_seconds": 795, "end_offset_seconds": 1200},
            ],
        )

    def test_tail_extends_load_profile_without_extending_scored_ramp(self) -> None:
        workload = {
            "pattern": "gradual-ramp",
            "duration_seconds": 120,
            "locust": {
                "spawn_rate": 10,
                "start_users": "0.2C",
                "end_users": "C",
                "ramp_seconds": 90,
                "hold_seconds": 30,
            },
        }

        points = generate_shape(
            workload,
            seed=12,
            sampling_interval_seconds=15,
            warmup_seconds=30,
            duration_seconds=120,
            continuation_seconds=30,
            capacity_users=10,
        )

        self.assertEqual(points[-1].elapsed_seconds, 180)
        self.assertEqual(points[-1].users, 10)
        self.assertEqual(points[-2].users, 10)

    def test_tail_continues_diurnal_cycle(self) -> None:
        workload = {
            "pattern": "diurnal",
            "duration_seconds": 120,
            "locust": {
                "spawn_rate": 10,
                "min_users": 2,
                "max_users": 10,
                "cycle_seconds": 60,
            },
        }

        points = generate_shape(
            workload,
            seed=12,
            sampling_interval_seconds=15,
            duration_seconds=120,
            continuation_seconds=30,
            capacity_users=10,
        )

        self.assertEqual(points[-1].elapsed_seconds, 150)
        self.assertEqual(points[9].users, points[5].users)


if __name__ == "__main__":
    unittest.main()
