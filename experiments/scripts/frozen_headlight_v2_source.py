#!/usr/bin/env python3
"""Verify frozen Headlight source inputs and stage an isolated Docker context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path("experiments/assets/headlight-v2")
BUNDLE_DIR = Path("results/headlight-v2-live/bundle-full-seed-601")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_verified_source(archive: Path, source: dict) -> dict[str, bytes]:
    expected = source["file_sha256"]
    files = {}
    with tarfile.open(archive, "r:gz") as capsule:
        for member in capsule:
            name = member.name
            path = PurePosixPath(name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or name not in expected or name in files or member.size > 10 * 1024 * 1024):
                raise ValueError(f"invalid frozen source entry: {name}")
            with capsule.extractfile(member) as handle:
                data = handle.read()
            if sha256(data) != expected[name]:
                raise ValueError(f"frozen source checksum mismatch: {name}")
            files[name] = data
    missing = set(expected) - set(files)
    if missing:
        raise ValueError(f"missing frozen source inputs: {sorted(missing)}")
    return files


def verify_assets(root: Path) -> tuple[dict[str, bytes], dict, dict]:
    manifest = json.loads((root / ASSET_DIR / "source-capsule.json").read_text())
    source = manifest
    environment = json.loads((root / ASSET_DIR / "execution-environment.json").read_text())
    archive = root / ASSET_DIR / "source-1c1d8ea3fe6d.tar.gz"
    if sha256(archive.read_bytes()) != manifest["archive_sha256"]:
        raise ValueError("frozen source capsule checksum mismatch")
    if manifest["source_sha256"] != source["source_sha256"]:
        raise ValueError("frozen source capsule disagrees with retained source provenance")
    files = read_verified_source(archive, source)
    image = source["image_tag"]
    if environment["HEADLIGHT_V2_IMAGE"] != image or environment["HEADLIGHT_IMAGE_REVISION"] != image:
        raise ValueError("retained image tag disagrees across provenance files")
    lock = json.loads((root / "experiments/configs/live-v2/runtime-lock.json").read_text())
    bundle = root / BUNDLE_DIR
    metadata = json.loads((bundle / "metadata.json").read_text())
    if metadata["bundle_id"] != lock["bundle"]["bundle_id"]:
        raise ValueError("frozen model bundle ID mismatch")
    for name, expected in lock["bundle"]["component_hashes"].items():
        path = bundle / ("checkpoint.pt" if name == "checkpoint" else name + ".json")
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"missing or changed frozen model component: {name}")
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        if name not in {"adjacency.json", "calibration.json", "checkpoint.pt", "feature_schema.json",
                        "metadata.json", "model_config.json"}:
            raise ValueError(f"invalid frozen model checksum entry: {name}")
        if sha256((bundle / name).read_bytes()) != expected:
            raise ValueError(f"frozen model checksum mismatch: {name}")
    policy = root / lock["policy"]["path"]
    if sha256(policy.read_bytes()) != lock["policy"]["sha256"]:
        raise ValueError("frozen policy checksum mismatch")
    return files, source, environment


def materialize_context(files: dict[str, bytes], destination: Path) -> None:
    if any(destination.iterdir()):
        raise ValueError("frozen build context must be empty")
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if name.endswith(".sh") else 0o644)


def record_build(root: Path, inspect_path: Path, output: Path, source: dict, environment: dict) -> None:
    inspected = json.loads(inspect_path.read_text())[0]
    labels = inspected.get("Config", {}).get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != environment["HEADLIGHT_CODE_REVISION"]:
        raise ValueError("built image revision label disagrees with frozen source")
    capsule = json.loads((root / ASSET_DIR / "source-capsule.json").read_text())
    evidence = {
        "schema_version": "headlight.v2-frozen-source-image-build/v1",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_tag": source["image_tag"], "image_id": inspected["Id"],
        "repo_digests": inspected.get("RepoDigests", []),
        "source_sha256": source["source_sha256"], "file_sha256": source["file_sha256"],
        "source_capsule_sha256": capsule["archive_sha256"],
        "build_args": {name: environment[name] for name in ("HEADLIGHT_CODE_REVISION", "HEADLIGHT_IMAGE_REVISION")},
        "environment_equivalence_established": False,
        "limitation": "Exact hashed source, model and policy; base-image digest and apt dependencies were not locked. A source rebuild does not verify original image or VM equivalence.",
        "docker_image_inspect": inspected,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "image-build-provenance.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--record-image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files, source, environment = verify_assets(REPO_ROOT)
    if args.context:
        materialize_context(files, args.context)
    if args.record_image:
        if not args.output:
            parser.error("--record-image requires --output")
        record_build(REPO_ROOT, args.record_image, args.output, source, environment)
    print(source["image_tag"])
    print(environment["HEADLIGHT_CODE_REVISION"])
    print(environment["HEADLIGHT_IMAGE_REVISION"])
    print(len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
