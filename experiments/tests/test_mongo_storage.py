from __future__ import annotations

import copy
import hashlib
import unittest
from unittest.mock import patch

import yaml

from experiments.scripts.render_minimal_trainticket import render, normalize


class MongoStorageTests(unittest.TestCase):
    def test_default_mongo_storage_is_owned_by_pod_on_node_filesystem(self) -> None:
        docs = render("advanced-ticket", "trainticket", "capacity-vm")
        mongo = [d for d in docs if d.get("kind") == "Deployment" and d["metadata"]["name"].endswith("-mongo")]
        self.assertEqual(len(mongo), 7)
        for doc in mongo:
            with self.subTest(service=doc["metadata"]["name"]):
                pod = doc["spec"]["template"]["spec"]
                mounts = {m["mountPath"]: m["name"] for m in pod["containers"][0].get("volumeMounts", [])}
                self.assertEqual(mounts, {"/data/db": "mongo-data", "/data/configdb": "mongo-configdb"})
                self.assertEqual(pod["volumes"], [
                    {"name": "mongo-data", "emptyDir": {}},
                    {"name": "mongo-configdb", "emptyDir": {}},
                ])
                self.assertEqual(doc["spec"]["template"]["metadata"]["annotations"]["headlight.research/mongo-storage"], "pod-emptydir-v1")

    def test_legacy_mode_reproduces_previous_rendered_manifest(self) -> None:
        legacy = render("advanced-ticket", "trainticket", "capacity-vm", mongo_storage="image-defined")
        text = yaml.safe_dump_all(legacy, sort_keys=False)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), "6283b88c36e5f514264fc43e254b9f1890338129204b12ec505fbae5e0c0c0d0")

    def test_mongo_storage_changes_only_mongo_deployments(self) -> None:
        legacy = render("full", "trainticket", "source", mongo_storage="image-defined")
        fixed = render("full", "trainticket", "source")
        self.assertEqual(len(legacy), len(fixed))
        for old, new in zip(legacy, fixed):
            is_mongo = old.get("kind") == "Deployment" and old["metadata"]["name"].endswith("-mongo")
            if is_mongo:
                self.assertEqual(old["spec"]["template"]["spec"]["containers"][0]["resources"], new["spec"]["template"]["spec"]["containers"][0]["resources"])
            else:
                self.assertEqual(old, new)

    def test_normalizing_explicit_mongo_storage_is_idempotent(self) -> None:
        doc = next(d for d in render("login-route", "trainticket", "capacity-vm") if d.get("kind") == "Deployment" and d["metadata"]["name"] == "ts-auth-mongo")
        expected = copy.deepcopy(doc)
        normalize(doc, "trainticket", {"ts-auth-mongo"}, set(), "capacity-vm")
        self.assertEqual(doc, expected)

    def test_invalid_storage_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render("advanced-ticket", "trainticket", "capacity-vm", mongo_storage="typo")

    def test_existing_database_mount_is_not_overwritten(self) -> None:
        doc = next(d for d in render("login-route", "trainticket", "capacity-vm", mongo_storage="image-defined") if d.get("kind") == "Deployment" and d["metadata"]["name"] == "ts-auth-mongo")
        doc["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = [{"name": "existing-pvc", "mountPath": "/data/db"}]
        with self.assertRaisesRegex(ValueError, "existing"):
            normalize(doc, "trainticket", {"ts-auth-mongo"}, set(), "capacity-vm")

    def test_run_hashes_record_selected_storage_mode(self) -> None:
        from experiments.run_experiment import _rendered_workload_hashes

        manifest = {"spec": {"topology": {"namespace": "trainticket"}}}
        with patch.dict("os.environ", {"HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket", "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm", "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "pod-emptydir"}):
            explicit = _rendered_workload_hashes(manifest)
        with patch.dict("os.environ", {"HEADLIGHT_TRAINTICKET_PATH": "advanced-ticket", "HEADLIGHT_TRAINTICKET_RESOURCE_PROFILE": "capacity-vm", "HEADLIGHT_TRAINTICKET_MONGO_STORAGE": "image-defined"}):
            legacy = _rendered_workload_hashes(manifest)
        self.assertEqual(explicit["mongo_storage"], "pod-emptydir")
        self.assertEqual(legacy["mongo_storage"], "image-defined")
        self.assertNotEqual(explicit["data_plane_no_policies_sha256"], legacy["data_plane_no_policies_sha256"])


if __name__ == "__main__":
    unittest.main()
