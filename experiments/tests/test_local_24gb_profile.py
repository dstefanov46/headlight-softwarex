from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from experiments.scripts.render_minimal_trainticket import (
    ADVANCED_APPS,
    ADVANCED_DBS,
    LOGIN_ROUTE_APPS,
    LOGIN_ROUTE_DBS,
    LOCAL_24GB_ADMISSION_MEMORY,
    LOCAL_24GB_ADMISSION_QUOTA_NAME,
    RESOURCE_PROFILES,
    render,
)


PROFILE = "local-24gb-tested"
APPS = LOGIN_ROUTE_APPS | ADVANCED_APPS
DATABASES = LOGIN_ROUTE_DBS | ADVANCED_DBS
SIDECAR_ANNOTATIONS = {
    "sidecar.istio.io/proxyCPU": "10m",
    "sidecar.istio.io/proxyMemory": "64Mi",
    "sidecar.istio.io/proxyCPULimit": "100m",
    "sidecar.istio.io/proxyMemoryLimit": "128Mi",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


class Local24GiBProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.docs = render("advanced-ticket", "local-24gb-test", PROFILE)
        cls.by_kind_name = {
            (doc.get("kind"), doc.get("metadata", {}).get("name")): doc
            for doc in cls.docs
        }

    def test_profile_records_verified_upstream_and_preserved_sidecar(self) -> None:
        metadata = RESOURCE_PROFILES[PROFILE]["metadata"]
        self.assertEqual(metadata["workload_path"], "advanced-ticket")
        self.assertEqual(
            metadata["upstream_revision"],
            "2d1ae2c3e255f464c993b980b207e88dced7f796",
        )
        self.assertEqual(RESOURCE_PROFILES[PROFILE]["sidecar"], SIDECAR_ANNOTATIONS)
        self.assertEqual(
            metadata["admission_requests_memory_per_cluster"],
            LOCAL_24GB_ADMISSION_MEMORY,
        )

    def test_exact_advanced_ticket_subset_and_dependencies_are_rendered(self) -> None:
        deployments = {
            name for kind, name in self.by_kind_name if kind == "Deployment"
        }
        services = {name for kind, name in self.by_kind_name if kind == "Service"}
        self.assertEqual(deployments, APPS | DATABASES)
        self.assertEqual(len(APPS), 11)
        self.assertEqual(len(DATABASES), 7)
        self.assertEqual(
            services,
            {f"{name}-service" for name in APPS} | DATABASES,
        )
        self.assertIn("ts-ticketinfo", deployments)
        self.assertIn("ts-ticketinfo-service", services)

    def test_application_resources_and_istio_sidecars_are_explicit(self) -> None:
        expected_resources = {
            "requests": {"cpu": "25m", "memory": "80Mi"},
            "limits": {"cpu": "200m", "memory": "500Mi"},
        }
        for name in APPS:
            deployment = self.by_kind_name[("Deployment", name)]
            pod_template = deployment["spec"]["template"]
            self.assertEqual(deployment["spec"]["replicas"], 1)
            self.assertEqual(
                pod_template["spec"]["containers"][0]["resources"],
                expected_resources,
            )
            annotations = pod_template["metadata"]["annotations"]
            self.assertEqual(annotations["sidecar.istio.io/inject"], "true")
            self.assertEqual(
                {key: annotations[key] for key in SIDECAR_ANNOTATIONS},
                SIDECAR_ANNOTATIONS,
            )

    def test_database_resources_have_requests_and_no_limits(self) -> None:
        for name in DATABASES:
            deployment = self.by_kind_name[("Deployment", name)]
            pod_template = deployment["spec"]["template"]
            resources = pod_template["spec"]["containers"][0]["resources"]
            self.assertEqual(deployment["spec"]["replicas"], 1)
            self.assertEqual(
                resources,
                {"requests": {"cpu": "25m", "memory": "50Mi"}},
            )
            self.assertEqual(
                pod_template["metadata"]["annotations"]["sidecar.istio.io/inject"],
                "false",
            )

    def test_two_cluster_initial_placement_and_actuation_objects_are_preserved(self) -> None:
        namespace = self.by_kind_name[("Namespace", "local-24gb-test")]
        self.assertEqual(namespace["metadata"]["labels"]["istio-injection"], "enabled")

        for name in APPS | DATABASES:
            crp = self.by_kind_name[("ControlledReplicaPropagation", name)]
            self.assertEqual(
                crp["spec"]["placements"],
                [
                    {"clusterName": "cluster2", "replicas": 1},
                    {"clusterName": "cluster3", "replicas": 1},
                ],
            )

        expected_policy_names = (
            {f"{name}-service" for name in APPS}
            | DATABASES
            | {LOCAL_24GB_ADMISSION_QUOTA_NAME}
        )
        self.assertEqual(
            {
                name
                for kind, name in self.by_kind_name
                if kind == "PropagationPolicy"
            },
            expected_policy_names,
        )

    def test_measured_admission_quota_allows_exactly_one_extra_app_per_cluster(self) -> None:
        quota = self.by_kind_name[("ResourceQuota", LOCAL_24GB_ADMISSION_QUOTA_NAME)]
        self.assertEqual(
            quota["spec"]["hard"],
            {"requests.memory": LOCAL_24GB_ADMISSION_MEMORY},
        )
        policy = self.by_kind_name[("PropagationPolicy", LOCAL_24GB_ADMISSION_QUOTA_NAME)]
        self.assertEqual(
            policy["spec"]["resourceSelectors"],
            [
                {
                    "apiVersion": "v1",
                    "kind": "ResourceQuota",
                    "name": LOCAL_24GB_ADMISSION_QUOTA_NAME,
                    "namespace": "local-24gb-test",
                }
            ],
        )

        without_policies = render(
            "advanced-ticket",
            "local-24gb-test",
            PROFILE,
            include_policies=False,
        )
        self.assertTrue(any(doc.get("kind") == "ResourceQuota" for doc in without_policies))
        self.assertFalse(any(doc.get("kind") == "PropagationPolicy" for doc in without_policies))

    def test_separate_controller_profile_has_measured_two_by_two_ceiling(self) -> None:
        path = (
            REPO_ROOT
            / "ai-module"
            / "configs"
            / "service_profiles"
            / "advanced-ticket-local-24gb-tested.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared_hash = payload.pop("profile_hash")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        self.assertEqual(declared_hash, hashlib.sha256(canonical).hexdigest())
        self.assertEqual(
            payload["cluster_limits"],
            {
                "cluster2": {"min_replicas": 1, "initial_replicas": 1, "max_replicas": 2},
                "cluster3": {"min_replicas": 1, "initial_replicas": 1, "max_replicas": 2},
            },
        )
        self.assertEqual(payload["source"]["resource_profile"], PROFILE)


if __name__ == "__main__":
    unittest.main()
