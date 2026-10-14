"""Immutable, hash-verified bundle contract for Headlight model families."""

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

from .graph import residual_edge_mask, validate_adjacency
from .models_v2 import (
    HEADLIGHT_V2_MODEL_TYPES,
    HEADLIGHT_V2_VARIANTS,
    validate_v2_model_config,
)
from .observability import canonical_json_bytes, content_sha256
from .training_v2 import (
    RESEARCH_V3_EVALUATION_SEEDS,
    RESEARCH_V4_EVALUATION_SEEDS,
    validate_v2_training_seed_partitions,
)


HEADLIGHT_V2_BUNDLE_SCHEMA_VERSION = "headlight.model-bundle/v2"
HEADLIGHT_V2_COMPONENT_FILES = {
    "checkpoint": "checkpoint.pt",
    "model_config": "model_config.json",
    "feature_schema": "feature_schema.json",
    "adjacency": "adjacency.json",
    "calibration": "calibration.json",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bundle_id(component_hashes: dict[str, str], metadata_core: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"components": component_hashes, "metadata": metadata_core})
    ).hexdigest()


def _seed_values(value: Any, path: str) -> set[int]:
    if isinstance(value, bool) or isinstance(value, dict):
        raise ValueError(f"Headlight training metadata has an invalid seed value at {path}")
    if isinstance(value, (list, tuple, set)):
        seeds: set[int] = set()
        for index, item in enumerate(value):
            seeds.update(_seed_values(item, f"{path}[{index}]"))
        return seeds
    try:
        return {int(value)}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Headlight training metadata has an invalid seed value at {path}"
        ) from exc


def _is_seed_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in {"seed", "seeds"} or normalized.endswith(("_seed", "_seeds"))


def _metadata_seeds(value: Any, path: str = "training") -> set[int]:
    seeds: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if _is_seed_key(key):
                seeds.update(_seed_values(item, item_path))
            else:
                seeds.update(_metadata_seeds(item, item_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            seeds.update(_metadata_seeds(item, f"{path}[{index}]"))
    return seeds


def _authoritative_buffers(
    model_config: dict[str, Any],
    adjacency: dict[str, Any],
) -> dict[str, torch.Tensor]:
    matrix = validate_adjacency(adjacency, tuple(model_config["node_order"]))
    observed = adjacency.get("observed_mask")
    allowed = residual_edge_mask(matrix, observed)
    buffers = {
        "adjacency.static_adjacency": torch.tensor(matrix, dtype=torch.float32),
        "adjacency.allowed_mask": torch.tensor(allowed, dtype=torch.float32),
        "service_type_ids": torch.tensor(model_config["service_type_ids"], dtype=torch.long),
    }
    if model_config["model_type"] == "headlight-v2-no-action":
        calibration = model_config["capacity_risk_calibration"]
        buffers.update(
            {
                "capacity_risk_surrogate.service_capacity_rps": torch.tensor(
                    calibration["service_capacity_rps"], dtype=torch.float32
                ),
                "capacity_risk_surrogate.cluster_capacity_weights": torch.tensor(
                    calibration["cluster_capacity_weights"], dtype=torch.float32
                ),
                "capacity_risk_surrogate.endpoint_risk_bias": torch.tensor(
                    calibration["endpoint_risk_bias"], dtype=torch.float32
                ),
            }
        )
    return buffers


def _validate_checkpoint_buffers(
    checkpoint_state: dict[str, torch.Tensor],
    authoritative: dict[str, torch.Tensor],
    *,
    require_all: bool,
) -> None:
    for name, expected in authoritative.items():
        actual = checkpoint_state.get(name)
        if actual is None:
            if require_all:
                raise ValueError(f"Headlight checkpoint is missing derived buffer {name}")
            continue
        if not isinstance(actual, torch.Tensor) or not torch.equal(
            actual.detach().cpu(), expected
        ):
            raise ValueError(
                f"Headlight checkpoint derived buffer {name} differs from bundle components"
            )


def _validate_feature_schema(schema: dict[str, Any], model_config: dict[str, Any]) -> None:
    if schema.get("schema_version") != "headlight.features/v2":
        raise ValueError("Headlight bundle feature schema is incompatible")
    if tuple(schema.get("node_feature_names", [])) != tuple(model_config["node_feature_names"]):
        raise ValueError("Headlight bundle node feature order differs from model config")
    if tuple(schema.get("edge_feature_names", [])) != tuple(model_config["edge_feature_names"]):
        raise ValueError("Headlight bundle edge feature order differs from model config")
    if tuple(schema.get("derived_request_features", [])) != (
        "request_rate_difference",
        "request_rate_short_slope",
        "request_rate_long_slope",
    ):
        raise ValueError("Headlight bundle derived request features are incompatible")


def _validate_training_metadata(
    training: dict[str, Any],
    *,
    feature_schema_hash: str,
    adjacency_hash: str,
) -> None:
    required = {
        "dataset_hash",
        "feature_schema_hash",
        "adjacency_hash",
        "fitting_seeds",
        "calibration_seeds",
        "holdout_seeds",
        "initialization_seed",
        "selection_procedure",
        "created_before_evaluation",
    }
    missing = sorted(required - set(training))
    if missing:
        raise ValueError(f"Headlight training metadata is missing: {', '.join(missing)}")
    dataset_hash = str(training["dataset_hash"])
    if len(dataset_hash) != 64 or any(value not in "0123456789abcdef" for value in dataset_hash):
        raise ValueError("Headlight dataset_hash must be a lowercase SHA-256 digest")
    if training["feature_schema_hash"] != feature_schema_hash:
        raise ValueError("Headlight training feature-schema hash differs from the bundle")
    if training["adjacency_hash"] != adjacency_hash:
        raise ValueError("Headlight training adjacency hash differs from the bundle")
    validate_v2_training_seed_partitions(
        fitting_seeds=training["fitting_seeds"],
        calibration_seeds=training["calibration_seeds"],
        holdout_seeds=training["holdout_seeds"],
        initialization_seed=int(training["initialization_seed"]),
    )
    forbidden = RESEARCH_V3_EVALUATION_SEEDS | RESEARCH_V4_EVALUATION_SEEDS
    if leaked := sorted(_metadata_seeds(training) & forbidden):
        raise ValueError(f"Headlight training metadata references evaluation seeds: {leaked}")
    if training["created_before_evaluation"] is not True:
        raise ValueError("Headlight bundle must be created before evaluation")
    if not isinstance(training["selection_procedure"], str) or not training["selection_procedure"].strip():
        raise ValueError("Headlight selection_procedure must be explicit")


@dataclass(frozen=True)
class HeadlightV2Bundle:
    root: Path
    metadata: dict[str, Any]
    model_config: dict[str, Any]
    feature_schema: dict[str, Any]
    adjacency: dict[str, Any]
    calibration: dict[str, Any]

    @property
    def checkpoint_path(self) -> Path:
        return self.root / HEADLIGHT_V2_COMPONENT_FILES["checkpoint"]

    @property
    def bundle_id(self) -> str:
        return str(self.metadata["bundle_id"])

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_model_type: str | None = None,
        expected_feature_schema_hash: str | None = None,
        expected_adjacency_hash: str | None = None,
    ) -> "HeadlightV2Bundle":
        bundle_root = Path(root)
        metadata_path = bundle_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Headlight bundle is missing metadata.json: {bundle_root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != HEADLIGHT_V2_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported Headlight bundle schema")
        if metadata.get("frozen") is not True or metadata.get("runtime_reload") is not False:
            raise ValueError("Headlight bundle must be frozen with runtime reload disabled")
        component_hashes = metadata.get("component_hashes")
        if not isinstance(component_hashes, dict):
            raise ValueError("Headlight bundle component hashes are missing")
        for component, filename in HEADLIGHT_V2_COMPONENT_FILES.items():
            path = bundle_root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Headlight bundle is missing {filename}")
            actual = _file_sha256(path)
            if component_hashes.get(component) != actual:
                raise ValueError(f"Headlight bundle component hash mismatch for {filename}")
        metadata_core = {
            key: value
            for key, value in metadata.items()
            if key not in {"bundle_id", "component_hashes"}
        }
        if metadata.get("bundle_id") != _bundle_id(component_hashes, metadata_core):
            raise ValueError("Headlight bundle ID does not match its contents")
        model_config = json.loads(
            (bundle_root / HEADLIGHT_V2_COMPONENT_FILES["model_config"]).read_text(encoding="utf-8")
        )
        validate_v2_model_config(model_config)
        if expected_model_type and model_config["model_type"] != expected_model_type:
            raise ValueError(
                f"expected {expected_model_type}, found {model_config['model_type']}"
            )
        feature_schema = json.loads(
            (bundle_root / HEADLIGHT_V2_COMPONENT_FILES["feature_schema"]).read_text(encoding="utf-8")
        )
        _validate_feature_schema(feature_schema, model_config)
        adjacency = json.loads(
            (bundle_root / HEADLIGHT_V2_COMPONENT_FILES["adjacency"]).read_text(encoding="utf-8")
        )
        validate_adjacency(adjacency, tuple(model_config["node_order"]))
        calibration = json.loads(
            (bundle_root / HEADLIGHT_V2_COMPONENT_FILES["calibration"]).read_text(encoding="utf-8")
        )
        if not isinstance(calibration.get("schema_version"), str):
            raise ValueError("Headlight calibration schema version is missing")
        feature_schema_hash = content_sha256(feature_schema)
        adjacency_hash = content_sha256(adjacency)
        if expected_feature_schema_hash and feature_schema_hash != expected_feature_schema_hash:
            raise ValueError("Headlight feature schema is incompatible with the runtime")
        if expected_adjacency_hash and adjacency_hash != expected_adjacency_hash:
            raise ValueError("Headlight adjacency is incompatible with the runtime")
        training = metadata.get("training")
        if not isinstance(training, dict):
            raise ValueError("Headlight training metadata is missing")
        _validate_training_metadata(
            training,
            feature_schema_hash=feature_schema_hash,
            adjacency_hash=adjacency_hash,
        )
        return cls(
            root=bundle_root,
            metadata=metadata,
            model_config=model_config,
            feature_schema=feature_schema,
            adjacency=adjacency,
            calibration=calibration,
        )


def create_v2_bundle(
    output: str | Path,
    *,
    checkpoint_state: dict[str, torch.Tensor],
    model_config: dict[str, Any],
    feature_schema: dict[str, Any],
    adjacency: dict[str, Any],
    calibration: dict[str, Any],
    training_metadata: dict[str, Any],
    overwrite: bool = False,
) -> HeadlightV2Bundle:
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace immutable Headlight bundle {destination}")
    validate_v2_model_config(model_config)
    _validate_feature_schema(feature_schema, model_config)
    validate_adjacency(adjacency, tuple(model_config["node_order"]))
    if model_config["model_type"] not in HEADLIGHT_V2_MODEL_TYPES:
        raise ValueError("Headlight bundle has an unsupported model type")
    if not isinstance(calibration.get("schema_version"), str):
        raise ValueError("Headlight calibration schema version is missing")
    feature_schema_hash = content_sha256(feature_schema)
    adjacency_hash = content_sha256(adjacency)
    _validate_training_metadata(
        training_metadata,
        feature_schema_hash=feature_schema_hash,
        adjacency_hash=adjacency_hash,
    )
    authoritative_buffers = _authoritative_buffers(model_config, adjacency)
    _validate_checkpoint_buffers(
        checkpoint_state,
        authoritative_buffers,
        require_all=True,
    )
    learned_state = {
        name: tensor.detach().cpu()
        for name, tensor in checkpoint_state.items()
        if name not in authoritative_buffers
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        torch.save(
            learned_state,
            temporary / HEADLIGHT_V2_COMPONENT_FILES["checkpoint"],
        )
        for component, payload in {
            "model_config": model_config,
            "feature_schema": feature_schema,
            "adjacency": adjacency,
            "calibration": calibration,
        }.items():
            _write_json(temporary / HEADLIGHT_V2_COMPONENT_FILES[component], payload)
        component_hashes = {
            component: _file_sha256(temporary / filename)
            for component, filename in HEADLIGHT_V2_COMPONENT_FILES.items()
        }
        metadata_core = {
            "schema_version": HEADLIGHT_V2_BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frozen": True,
            "runtime_reload": False,
            "model_type": model_config["model_type"],
            "variant": HEADLIGHT_V2_VARIANTS[model_config["model_type"]],
            "training": dict(training_metadata),
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
    return HeadlightV2Bundle.load(destination, expected_model_type=model_config["model_type"])


def load_v2_checkpoint(
    bundle: HeadlightV2Bundle,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state = torch.load(bundle.checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Headlight checkpoint is not a non-empty state dict")
    authoritative_buffers = _authoritative_buffers(bundle.model_config, bundle.adjacency)
    _validate_checkpoint_buffers(state, authoritative_buffers, require_all=False)
    learned_state = {
        name: tensor
        for name, tensor in state.items()
        if name not in authoritative_buffers
    }
    learned_state.update(
        {name: tensor.to(device=device) for name, tensor in authoritative_buffers.items()}
    )
    return learned_state
