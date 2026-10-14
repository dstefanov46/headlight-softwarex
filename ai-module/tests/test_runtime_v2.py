from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.readiness import PendingAgeTracker  # noqa: E402
from headlight.policy_v2 import PolicyV2Config  # noqa: E402
from headlight.profile import load_service_profile  # noqa: E402
from headlight.runtime_v2 import HeadlightV2Runtime  # noqa: E402


class RuntimeTelemetryConfigurationTests(unittest.TestCase):
    def test_invalid_cluster_mapping_is_rejected_before_loading_bundle(self) -> None:
        profile = load_service_profile(
            REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json"
        )
        cases = (
            ("http://cluster3.invalid", None),
            ("http://cluster3.invalid", ("cluster3",)),
            ("http://cluster3.invalid", ("cluster2", "cluster3")),
            ("http://cluster2.invalid,http://cluster3.invalid", ("cluster2", "unknown")),
            (
                "http://cluster2.invalid,http://cluster3.invalid,http://duplicate.invalid",
                ("cluster2", "cluster3", "cluster3"),
            ),
        )
        with tempfile.TemporaryDirectory() as empty_bundle:
            for urls, clusters in cases:
                with self.subTest(urls=urls, clusters=clusters):
                    with self.assertRaisesRegex(ValueError, "Prometheus.*cluster"):
                        HeadlightV2Runtime.from_bundle(
                            empty_bundle,
                            runtime_profile=profile,
                            prometheus_url=urls,
                            prometheus_clusters=clusters,
                            action_handler_url="http://actions.invalid/placements",
                            policy_config=PolicyV2Config(),
                            run_id="configuration-regression",
                            seed=401,
                        )

    def test_complete_cluster_mappings_reach_bundle_loading_in_either_order(self) -> None:
        profile = load_service_profile(
            REPO_ROOT / "ai-module/configs/service_profiles/advanced-ticket.json"
        )
        with tempfile.TemporaryDirectory() as empty_bundle:
            for clusters in (("cluster2", "cluster3"), ("cluster3", "cluster2")):
                with self.subTest(clusters=clusters):
                    with self.assertRaises(FileNotFoundError):
                        HeadlightV2Runtime.from_bundle(
                            empty_bundle,
                            runtime_profile=profile,
                            prometheus_url=",".join(f"http://{cluster}.invalid" for cluster in clusters),
                            prometheus_clusters=clusters,
                            action_handler_url="http://actions.invalid/placements",
                            policy_config=PolicyV2Config(),
                            run_id="configuration-regression",
                            seed=401,
                        )


class PendingAgeTrackerTests(unittest.TestCase):
    def test_age_is_cluster_local_and_resets_when_pending_clears(self) -> None:
        tracker = PendingAgeTracker()
        self.assertEqual(
            tracker.ages("ts-auth", {"cluster2": 1, "cluster3": 0}, 100.0),
            {"cluster2": 0.0, "cluster3": 0.0},
        )
        self.assertEqual(
            tracker.ages("ts-auth", {"cluster2": 1, "cluster3": 1}, 160.0),
            {"cluster2": 60.0, "cluster3": 0.0},
        )
        self.assertEqual(
            tracker.ages("ts-auth", {"cluster2": 0, "cluster3": 0}, 175.0),
            {"cluster2": 0.0, "cluster3": 0.0},
        )
        self.assertEqual(
            tracker.ages("ts-auth", {"cluster2": 0, "cluster3": 1}, 190.0),
            {"cluster2": 0.0, "cluster3": 0.0},
        )
        self.assertEqual(
            tracker.ages("ts-auth", {"cluster2": 0, "cluster3": 1}, 205.0),
            {"cluster2": 0.0, "cluster3": 15.0},
        )


if __name__ == "__main__":
    unittest.main()
