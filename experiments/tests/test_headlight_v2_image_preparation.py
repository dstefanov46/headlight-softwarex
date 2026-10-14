"""Offline checks: no Docker daemon or live testbed access is permitted here."""
from __future__ import annotations

import fcntl
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path("experiments/assets/headlight-v2")
CAPSULE = ASSET_DIR / "source-capsule.json"
ASSET = ASSET_DIR / "source-1c1d8ea3fe6d.tar.gz"


class FrozenV2ImageTests(unittest.TestCase):
    def test_offline_verification_accepts_frozen_source_without_reverting_checkout(self):
        source = json.loads((ROOT / CAPSULE).read_text())
        expected_files = set(source["file_sha256"])
        # The archive preserves historical image inputs that are no longer in this checkout.
        intentionally_archived_only = {
            "ai-module/baselines/lstm_autoscaler.py",
            "ai-module/baselines/lstm_train.py",
            "ai-module/baselines/static_autoscaler.py",
            "ai-module/lstm_autoscaler_setup.sh",
        }
        checked_out = {name for name in expected_files if (ROOT / name).is_file()}
        archived_only = expected_files - checked_out
        self.assertEqual(archived_only, intentionally_archived_only)
        before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in checked_out}
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)
            for command in ("docker", "kind"):
                executable = binary / command
                executable.write_text('#!/usr/bin/env bash\nprintf called >"$FORBIDDEN_CALL"\nexit 99\n')
                executable.chmod(0o755)
            sentinel = binary / "forbidden-call"
            environment = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"],
                           "FORBIDDEN_CALL": str(sentinel)}
            result = subprocess.run(["bash", str(ROOT / "experiments/scripts/verify_and_build_headlight_v2_image.sh"),
                                     "--verify-only"], env=environment, capture_output=True, text=True, timeout=20)
            self.assertFalse(sentinel.exists(), "offline verification contacted Docker/kind")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("files=66", result.stdout)
        self.assertEqual(before, {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in checked_out})
        self.assertFalse(any((ROOT / name).exists() for name in archived_only))

    def source_module(self):
        return importlib.import_module("experiments.scripts.frozen_headlight_v2_source")

    def test_clean_build_context_contains_exactly_historical_hashed_inputs(self):
        module = self.source_module()
        files, source, _ = module.verify_assets(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            module.materialize_context(files, destination)
            actual = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in destination.rglob("*") if p.is_file()}
        self.assertEqual(actual, source["file_sha256"])
        self.assertEqual(len(actual), 66)
        self.assertEqual(actual["ai-module/ai-module/headlight/runtime.py"],
                         "10cd64fee8e6705b5548646c344e947ce18675abaf798cf7f5450a0fd219d1a7")

    def test_source_capsule_rejects_changed_missing_extra_and_unsafe_entries(self):
        module = self.source_module()
        original = json.loads((ROOT / CAPSULE).read_text())
        files, _, _ = module.verify_assets(ROOT)
        first = next(iter(files))
        variants = (
            {**files, first: b"changed source"},
            {k: v for k, v in files.items() if k != first},
            {**files, "ai-module/ai-module/untracked.py": b"extra"},
            {**files, "../escaped.py": b"escape"},
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.tar.gz"
            for variant in variants:
                with self.subTest(entries=list(set(variant) - set(files))):
                    with tarfile.open(archive, "w:gz") as output:
                        for name, data in variant.items():
                            info = tarfile.TarInfo(name)
                            info.size = len(data)
                            output.addfile(info, io.BytesIO(data))
                    with self.assertRaises(ValueError):
                        module.read_verified_source(archive, original)

    def test_missing_or_changed_weights_fail_offline_before_build(self):
        module = self.source_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (ASSET_DIR, Path("results/headlight-v2-live/bundle-full-seed-601"),
                         Path("experiments/configs/live-v2"), Path("experiments/configs/models"),
                         ):
                shutil.copytree(ROOT / path, root / path)
            checkpoint = root / "results/headlight-v2-live/bundle-full-seed-601/checkpoint.pt"
            checkpoint.write_bytes(b"wrong weights")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                module.verify_assets(root)
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                module.verify_assets(root)

    def test_build_uses_isolated_context_and_records_actual_image_identity(self):
        # Docker and kind are external mutating boundaries; replace only their executables.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            docker = binary / "docker"
            docker.write_text("""#!/usr/bin/env python3
import hashlib, json, os
from pathlib import Path
import sys
args = sys.argv[1:]
if args[0] == 'build':
    context = Path(args[-1])
    assert context != Path(os.environ['REAL_ROOT']) / 'ai-module'
    source = json.loads((Path(os.environ['REAL_ROOT']) / os.environ['CAPSULE']).read_text())
    actual = {'ai-module/' + p.relative_to(context).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in context.rglob('*') if p.is_file()}
    assert actual == source['file_sha256'], actual
elif args[:2] == ['image', 'inspect']:
    environment = json.loads((Path(os.environ['REAL_ROOT']) / os.environ['ASSET_DIR'] / 'execution-environment.json').read_text())
    print(json.dumps([{'Id': 'sha256:new-build', 'RepoDigests': [],
                      'RootFS': {'Layers': ['sha256:new-layer']},
                      'Config': {'Labels': {'org.opencontainers.image.revision': environment['HEADLIGHT_CODE_REVISION']}}}]))
else:
    raise SystemExit('unexpected Docker invocation: ' + repr(args))
""")
            docker.chmod(0o755)
            kind = binary / "kind"
            kind.write_text("#!/usr/bin/env bash\nset -eu\n[[ \"$*\" == 'load docker-image headlight-ai:v2-diag-ablation-1c1d8ea3fe6d --name cluster3' ]]\n")
            kind.chmod(0o755)
            artifact_dir = root / "build-evidence"
            environment = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"],
                           "REAL_ROOT": str(ROOT), "CAPSULE": str(CAPSULE),
                           "ASSET_DIR": str(ASSET_DIR),
                           "HEADLIGHT_V2_IMAGE_ARTIFACT_DIR": str(artifact_dir)}
            result = subprocess.run(["bash", str(ROOT / "experiments/scripts/verify_and_build_headlight_v2_image.sh")],
                                    env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads((artifact_dir / "image-build-provenance.json").read_text())
            self.assertEqual(evidence["image_id"], "sha256:new-build")
            self.assertFalse(evidence["environment_equivalence_established"])


class V2PreparationTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("experiments.scripts.prepare_headlight_v2_seed104_105")

    def test_active_campaign_blocks_preparation_before_any_testbed_command_or_campaign_write(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared.lock"
            shared.write_text("existing lock contents\n")
            campaign = root / "results/new-campaign"
            with shared.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(module, "REPO_ROOT", root), patch.object(module, "SHARED_LOCK", shared):
                    with patch.object(module.subprocess, "run", side_effect=AssertionError("testbed was touched")):
                        with self.assertRaisesRegex(RuntimeError, "testbed lock"):
                            module.prepare(campaign)
            self.assertFalse(campaign.exists())
            self.assertEqual(shared.read_text(), "existing lock contents\n")

    def test_historical_and_nonempty_campaign_directories_are_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "results/existing"
            existing.mkdir(parents=True)
            artifact = existing / "campaign-status.json"
            artifact.write_text('{"status":"running"}\n')
            with patch.object(module, "REPO_ROOT", root):
                for path in (root / "results/headlight-v2-live/new", root / "results/socc26/new", existing):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        module.prepare(path)
            self.assertEqual(artifact.read_text(), '{"status":"running"}\n')

    def test_preparation_holds_shared_lock_and_requires_fresh_destination_smoke(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared.lock"
            campaign = root / "results/new-campaign"
            calls = []

            def external(command, **kwargs):
                with shared.open("a") as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                calls.append(command)
                environment = kwargs["env"]
                self.assertEqual(environment["HEADLIGHT_TRAINTICKET_MONGO_STORAGE"], "image-defined")
                self.assertEqual(environment["HEADLIGHT_V2_ENABLE_ACTUATION"], "0")
                self.assertEqual(environment["HEADLIGHT_V2_SHADOW_EVIDENCE"],
                                 str(campaign / "shadow-smoke/shadow-smoke.ok"))
                if "prepare_seed102_103_testbed.sh" in command[1]:
                    (campaign / "campaign-env.sh").write_text("# destination environment\n")
                else:
                    smoke = campaign / "shadow-smoke/shadow-smoke.ok"
                    smoke.parent.mkdir()
                    smoke.write_text("bundle_id=c37050a44dd2fc876bf6523aaeabdd364af7608ffeee5e1586a6cb8bbaaaeb4a\nstatus=passed\n")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(module, "REPO_ROOT", root), patch.object(module, "SHARED_LOCK", shared):
                with patch.object(module, "verify_assets", return_value=({}, {}, {})):
                    with patch.object(module.subprocess, "run", side_effect=external):
                        module.prepare(campaign)
            evidence = json.loads((campaign / "preparation.json").read_text())
            self.assertEqual(evidence["status"], "ready")
            self.assertFalse(evidence["environment_equivalence_established"])
            self.assertEqual(len(calls), 2)
            self.assertTrue((campaign / "campaign-env.sh").is_file())
            with shared.open("a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
