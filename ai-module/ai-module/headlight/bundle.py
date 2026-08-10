"""Immutable, hash-verified model bundle creation and loading."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .features import NodeFeatureScaler, feature_schema
from .graph import validate_adjacency
from .profile import ServiceProfile


BUNDLE_SCHEMA_VERSION = "headlight.model-bundle/v1"
COMPONENT_FILES = {
    "checkpoint": "checkpoint.pt",
    "scaler": "scaler.json",
    "feature_schema": "feature_schema.json",
    "service_profile": "service_profile.json",
    "adjacency": "adjacency.json",
    "capacity": "capacity.json",
    "model_config": "model_config.json",
    "learned_edges": "learned_edges.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bundle_id(component_hashes: dict[str, str], metadata_core: dict[str, Any]) -> str:
    payload = json.dumps(
        {"components": component_hashes, "metadata": metadata_core},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_bundle_integrity(bundle_root: Path, metadata: dict[str, Any]) -> dict[str, str]:
    hashes = metadata.get("component_hashes", {})
    missing = [filename for filename in COMPONENT_FILES.values() if not (bundle_root / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"model bundle {bundle_root} is incomplete: missing {', '.join(missing)}")
    for component, filename in COMPONENT_FILES.items():
        expected_hash = hashes.get(component)
        if not expected_hash:
            raise ValueError(f"model bundle has no hash for component {component}")
        actual_hash = file_sha256(bundle_root / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"model bundle component hash mismatch for {filename}: expected={expected_hash} actual={actual_hash}")
    metadata_core = {key: value for key, value in metadata.items() if key not in {"bundle_id", "component_hashes"}}
    if metadata.get("bundle_id") != _bundle_id(hashes, metadata_core):
        raise ValueError("model bundle ID does not match its metadata and component hashes")
    return hashes


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    metadata: dict[str, Any]
    profile: ServiceProfile
    scaler: NodeFeatureScaler
    feature_schema: dict[str, Any]
    adjacency: dict[str, Any]
    capacity: dict[str, Any]
    model_config: dict[str, Any]
    learned_edges: dict[str, Any]

    @property
    def checkpoint_path(self) -> Path:
        return self.root / COMPONENT_FILES["checkpoint"]

    @property
    def bundle_id(self) -> str:
        return str(self.metadata["bundle_id"])

    @classmethod
    def load(
        cls,
        root: str | Path,
        runtime_profile: ServiceProfile | None = None,
        expected_model_type: str | None = None,
    ) -> "ModelBundle":
        bundle_root = Path(root)
        metadata_path = bundle_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"model bundle is missing metadata.json: {bundle_root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported model bundle schema")
        _validate_bundle_integrity(bundle_root, metadata)
        profile = ServiceProfile.from_dict(
            json.loads((bundle_root / COMPONENT_FILES["service_profile"]).read_text(encoding="utf-8"))
        )
        if runtime_profile is not None:
            profile.assert_compatible(runtime_profile)
        scaler = NodeFeatureScaler.from_dict(
            json.loads((bundle_root / COMPONENT_FILES["scaler"]).read_text(encoding="utf-8"))
        )
        schema = json.loads(
            (bundle_root / COMPONENT_FILES["feature_schema"]).read_text(encoding="utf-8")
        )
        if schema != feature_schema():
            raise ValueError("model bundle feature schema is incompatible with runtime")
        if scaler.mean.shape != (len(profile.nodes), len(schema["base_features"])):
            raise ValueError("model bundle scaler dimensions do not match its service profile")
        adjacency = json.loads(
            (bundle_root / COMPONENT_FILES["adjacency"]).read_text(encoding="utf-8")
        )
        validate_adjacency(adjacency, profile.node_order)
        capacity = json.loads(
            (bundle_root / COMPONENT_FILES["capacity"]).read_text(encoding="utf-8")
        )
        if capacity.get("schema_version") != "headlight.capacity/v1":
            raise ValueError("model bundle capacity schema is incompatible")
        missing_capacity = sorted(set(profile.node_order) - set(capacity.get("services", {})))
        if missing_capacity:
            raise ValueError(
                f"model bundle has no capacity estimates for: {', '.join(missing_capacity)}"
            )
        model_config = json.loads(
            (bundle_root / COMPONENT_FILES["model_config"]).read_text(encoding="utf-8")
        )
        _validate_model_config(model_config, profile)
        if expected_model_type and model_config["model_type"] != expected_model_type:
            raise ValueError(
                f"expected a {expected_model_type} bundle, found {model_config['model_type']}"
            )
        learned_edges = json.loads(
            (bundle_root / COMPONENT_FILES["learned_edges"]).read_text(encoding="utf-8")
        )
        return cls(
            root=bundle_root,
            metadata=metadata,
            profile=profile,
            scaler=scaler,
            feature_schema=schema,
            adjacency=adjacency,
            capacity=capacity,
            model_config=model_config,
            learned_edges=learned_edges,
        )


def _validate_model_config(config: dict[str, Any], profile: ServiceProfile) -> None:
    required = {
        "model_type",
        "num_nodes",
        "input_size",
        "lookback",
        "horizon",
        "sampling_interval_seconds",
        "hidden_width",
        "graph_blocks",
        "variant",
        "residual_bound",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"model config is missing fields: {', '.join(missing)}")
    if config["model_type"] not in {"headlight", "lstm"}:
        raise ValueError("model config has an unsupported model_type")
    if int(config["num_nodes"]) != len(profile.nodes):
        raise ValueError("model config num_nodes differs from the service profile")
    if int(config["input_size"]) != len(feature_schema()["model_features"]):
        raise ValueError("model config input_size differs from the feature schema")
    if int(config["sampling_interval_seconds"]) != profile.sampling_interval_seconds:
        raise ValueError("model sampling interval differs from the service profile")
    if int(config["lookback"]) <= 0 or int(config["horizon"]) <= 0:
        raise ValueError("model lookback and horizon must be positive")


def create_bundle(
    output: str | Path,
    checkpoint_state: dict[str, torch.Tensor],
    scaler: NodeFeatureScaler,
    profile: ServiceProfile,
    adjacency: dict[str, Any],
    capacity: dict[str, Any],
    model_config: dict[str, Any],
    learned_edges: dict[str, Any],
    training_metadata: dict[str, Any],
    overwrite: bool = False,
) -> ModelBundle:
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to replace immutable model bundle {destination}; pass --overwrite explicitly"
        )
    _validate_model_config(model_config, profile)
    validate_adjacency(adjacency, profile.node_order)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        cpu_state = {key: value.detach().cpu() for key, value in checkpoint_state.items()}
        torch.save(cpu_state, temporary / COMPONENT_FILES["checkpoint"])
        json_components = {
            "scaler": scaler.to_dict(),
            "feature_schema": feature_schema(),
            "service_profile": profile.to_dict(),
            "adjacency": adjacency,
            "capacity": capacity,
            "model_config": model_config,
            "learned_edges": learned_edges,
        }
        for component, payload in json_components.items():
            _write_json(temporary / COMPONENT_FILES[component], payload)
        component_hashes = {
            component: file_sha256(temporary / filename)
            for component, filename in COMPONENT_FILES.items()
        }
        metadata_core = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frozen": True,
            "runtime_reload": False,
            "profile_hash": profile.profile_hash,
            "node_order": list(profile.node_order),
            "training": training_metadata,
        }
        metadata = {
            **metadata_core,
            "component_hashes": component_hashes,
            "bundle_id": _bundle_id(component_hashes, metadata_core),
        }
        _write_json(temporary / "metadata.json", metadata)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ModelBundle.load(destination, runtime_profile=profile)


def load_checkpoint(bundle: ModelBundle, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(bundle.checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("model bundle checkpoint is not a non-empty state dict")
    return state
