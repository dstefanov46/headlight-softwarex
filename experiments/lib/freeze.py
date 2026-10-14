"""Create and validate the tracked manifest for the external frozen release bundle."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

from experiments.lib.capacity_contract import validate_capacity_contract
from experiments.lib.config import REPO_ROOT
from experiments.lib.protocol import protocol_reference
from experiments.lib.static_plan import validate_static_plan


FREEZE_SCHEMA = "headlight.research-v3-freeze/v1"
FREEZE_MANIFEST_PATH = REPO_ROOT / "experiments" / "configs" / "protocols" / "research-v3-freeze.json"
EVALUATION_SEEDS = {101, 102, 103, 104, 105}
MODEL_COMPONENTS = (
    "checkpoint.pt",
    "scaler.json",
    "feature_schema.json",
    "service_profile.json",
    "adjacency.json",
    "capacity.json",
    "model_config.json",
    "learned_edges.json",
    "metadata.json",
)


class FreezeManifestError(RuntimeError):
    """Raised when the immutable release inputs are absent or inconsistent."""


def validate_threshold_evaluation_inputs() -> dict[str, Any]:
    """Check the retained policy/protocol without requiring retired model baselines."""
    payload = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("protocol") != protocol_reference():
        raise FreezeManifestError("Threshold evaluation requires the frozen protocol reference")
    policy = payload["policy"]
    path = (REPO_ROOT / policy["path"]).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise FreezeManifestError("Threshold policy must be inside the repository")
    if hashlib.sha256(path.read_bytes()).hexdigest() != policy["sha256"]:
        raise FreezeManifestError("Threshold evaluation policy hash differs from the frozen input")
    return {"protocol": payload["protocol"], "policy": policy}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _read_only(path: Path) -> bool:
    return path.exists() and path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0


def _lock_tree(path: Path) -> None:
    for child in path.rglob("*"):
        child.chmod(child.stat().st_mode & ~0o222)
    path.chmod(path.stat().st_mode & ~0o222)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _metadata_has_evaluation_seed(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"seed", "seeds", "capacity_seed", "calibration_seeds", "tuning_seeds"}:
                values = item if isinstance(item, list) else [item]
                try:
                    if EVALUATION_SEEDS.intersection(int(candidate) for candidate in values):
                        return True
                except (TypeError, ValueError):
                    pass
            if _metadata_has_evaluation_seed(item):
                return True
    elif isinstance(value, list):
        return any(_metadata_has_evaluation_seed(item) for item in value)
    return False


def _bundle_record(path: Path, controller: str) -> dict[str, Any]:
    if not path.is_dir():
        raise FreezeManifestError(f"{controller} model bundle is absent: {path}")
    missing = [name for name in MODEL_COMPONENTS if not (path / name).is_file()]
    if missing:
        raise FreezeManifestError(f"{controller} model bundle is incomplete: {', '.join(missing)}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("frozen") is not True or metadata.get("runtime_reload") is not False:
        raise FreezeManifestError(f"{controller} model bundle is not frozen against online retraining")
    if _metadata_has_evaluation_seed(metadata):
        raise FreezeManifestError(f"{controller} model bundle metadata references an evaluation seed")
    if not _read_only(path):
        raise FreezeManifestError(f"{controller} model bundle must be read-only: {path}")
    return {"path": _stored_path(path), "sha256": sha256_tree(path), "bundle_id": metadata.get("bundle_id")}


def create_freeze_manifest(
    *,
    capacity_contract: Path,
    static_plan: Path,
    headlight_bundle: Path,
    lstm_bundle: Path,
    policy: Path,
    output: Path = FREEZE_MANIFEST_PATH,
    release_root: str = "external/research-v3-release",
) -> dict[str, Any]:
    contract, warnings = validate_capacity_contract(capacity_contract)
    if warnings:
        raise FreezeManifestError("capacity contract is too close to the headroom limit: " + "; ".join(warnings))
    plan = validate_static_plan(static_plan)
    _lock_tree(static_plan)
    if not _read_only(static_plan):
        raise FreezeManifestError(f"static plan must be read-only: {static_plan}")
    if not policy.is_file():
        raise FreezeManifestError(f"policy configuration is absent: {policy}")
    policy_payload = json.loads(policy.read_text(encoding="utf-8"))
    policy_without_hash = dict(policy_payload)
    configured_hash = policy_without_hash.pop("policy_configuration_hash", None)
    actual_policy_hash = hashlib.sha256(
        json.dumps(policy_without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if configured_hash != actual_policy_hash:
        raise FreezeManifestError("policy configuration hash does not match its contents")
    if _metadata_has_evaluation_seed(policy_payload.get("provenance", {})):
        raise FreezeManifestError("policy provenance references an evaluation seed")
    _lock_tree(headlight_bundle)
    _lock_tree(lstm_bundle)
    payload: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "status": "frozen",
        "release_root": release_root,
        "protocol": protocol_reference(),
        "capacity_contract": {
            "path": _stored_path(capacity_contract),
            "sha256": sha256_file(capacity_contract),
            "selected_capacity": int(contract["selected_capacity"]),
        },
        "static_plan": {
            "path": _stored_path(static_plan),
            "sha256": sha256_file(static_plan),
            "plan_hash": plan["plan_hash"],
        },
        "policy": {"path": _stored_path(policy), "sha256": sha256_file(policy)},
        "model_bundles": {
            "headlight": _bundle_record(headlight_bundle, "Headlight"),
            "lstm": _bundle_record(lstm_bundle, "LSTM-P"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_freeze_manifest(path: Path = FREEZE_MANIFEST_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FreezeManifestError(f"tracked freeze manifest is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FreezeManifestError(f"tracked freeze manifest is invalid JSON: {path}: {exc}") from exc
    if payload.get("schema_version") != FREEZE_SCHEMA or payload.get("status") != "frozen":
        raise FreezeManifestError("tracked freeze manifest is not a frozen research-v3 release")
    if payload.get("protocol") != protocol_reference():
        raise FreezeManifestError("freeze manifest protocol reference is incompatible")
    for key in ("capacity_contract", "static_plan", "policy"):
        record = payload.get(key, {})
        artifact = _resolve(str(record.get("path", "")))
        if sha256_file(artifact) != record.get("sha256"):
            raise FreezeManifestError(f"{key} hash does not match the tracked freeze manifest")
    bundles = payload.get("model_bundles", {})
    for controller in ("headlight", "lstm"):
        record = bundles.get(controller, {})
        artifact = _resolve(str(record.get("path", "")))
        if sha256_tree(artifact) != record.get("sha256"):
            raise FreezeManifestError(f"{controller} bundle hash does not match the tracked freeze manifest")
        _bundle_record(artifact, controller)
    validate_capacity_contract(_resolve(str(payload["capacity_contract"]["path"])))
    validate_static_plan(_resolve(str(payload["static_plan"]["path"])))
    return payload
