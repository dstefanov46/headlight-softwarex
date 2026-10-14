#!/usr/bin/env python3
"""Validate and resolve experiment contract files.

This is intentionally not a runner. It only enforces the T01 experiment
contract so later tickets can rely on stable, machine-checked config files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.protocol import (
    PROTOCOL_PATH,
    ProtocolError,
    load_protocol,
    primary_controllers,
    primary_workloads,
    protocol_reference,
)


CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
RESOLVED_ROOT = REPO_ROOT / "experiments" / "resolved"
API_VERSION = "headlight.experiments/v1alpha1"

VALID_CONTROLLERS = {"headlight", "hpa", "keda", "threshold"}
VALID_TOPOLOGIES = {"single-cluster", "multi-cluster-static", "multi-cluster-headlight"}
VALID_WORKLOADS = {"diurnal", "flash-crowd", "gradual-ramp", "bursty"}
VALID_IMPLEMENTATION_STATES = {"ready", "planned"}
VALID_ACTUATION_PATHS = {"actions-handler", "keda", "kubernetes-hpa"}
VALID_KINDS = {
    "ControllerProfile",
    "TopologyProfile",
    "WorkloadProfile",
    "ExperimentRun",
    "ResearchProtocol",
}


class ValidationError(Exception):
    """Raised when a config is invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"{path}: top-level YAML value must be a mapping")
    return data


def require_fields(path: Path, obj: dict[str, Any], fields: list[str]) -> None:
    for field in fields:
        cursor: Any = obj
        for part in field.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValidationError(f"{path}: missing required field `{field}`")
            cursor = cursor[part]


def assert_relative_repo_path(path: Path, value: str, field_name: str) -> None:
    target = REPO_ROOT / value
    if Path(value).is_absolute():
        raise ValidationError(f"{path}: `{field_name}` must be repository-relative, got absolute path `{value}`")
    if not target.exists():
        raise ValidationError(f"{path}: `{field_name}` points to missing path `{value}`")


def check_common(path: Path, data: dict[str, Any], expected_kind: str) -> None:
    require_fields(path, data, ["apiVersion", "kind", "metadata.name", "spec"])
    if data["apiVersion"] != API_VERSION:
        raise ValidationError(f"{path}: unsupported apiVersion `{data['apiVersion']}`")
    if data["kind"] not in VALID_KINDS:
        raise ValidationError(f"{path}: unsupported kind `{data['kind']}`")
    if data["kind"] != expected_kind:
        raise ValidationError(f"{path}: expected kind `{expected_kind}`, got `{data['kind']}`")


def _validate_controller_paths(path: Path, spec: dict[str, Any]) -> None:
    assert_relative_repo_path(path, spec["setup_script"], "spec.setup_script")
    assert_relative_repo_path(path, spec["metrics_collector"], "spec.metrics_collector")
    optional_path_fields = [
        "deployment_manifest",
        "hpa_manifest",
        "training_entrypoint",
        "inference_entrypoint",
        "entrypoint",
        "service_profile",
        "model_config",
    ]
    for field in optional_path_fields:
        if field not in spec:
            continue
        if Path(spec[field]).is_absolute():
            raise ValidationError(
                f"{path}: `spec.{field}` must be repository-relative, got absolute path `{spec[field]}`"
            )
        target = REPO_ROOT / spec[field]
        if spec["implementation_state"] == "ready" and not target.exists():
            raise ValidationError(
                f"{path}: `spec.{field}` points to missing path `{spec[field]}` for ready profile"
            )


def _validate_controller_spec(path: Path, name: str, spec: dict[str, Any]) -> None:
    if name not in VALID_CONTROLLERS:
        raise ValidationError(f"{path}: invalid controller profile name `{name}`")
    if spec["controller"] != name:
        raise ValidationError(f"{path}: `spec.controller` must match `metadata.name`")
    if spec["implementation_state"] not in VALID_IMPLEMENTATION_STATES:
        raise ValidationError(
            f"{path}: invalid implementation state `{spec['implementation_state']}`"
        )
    if spec["actuation_path"] not in VALID_ACTUATION_PATHS:
        raise ValidationError(f"{path}: invalid actuation path `{spec['actuation_path']}`")
    if "sampling_interval_seconds" in spec and (
        not isinstance(spec["sampling_interval_seconds"], int) or spec["sampling_interval_seconds"] <= 0
    ):
        raise ValidationError(f"{path}: `spec.sampling_interval_seconds` must be a positive integer")
    _validate_controller_paths(path, spec)


def validate_controller_profile(path: Path, data: dict[str, Any]) -> str:
    check_common(path, data, "ControllerProfile")
    require_fields(
        path,
        data,
        [
            "spec.controller",
            "spec.implementation_state",
            "spec.setup_script",
            "spec.metrics_collector",
            "spec.actuation_path",
        ],
    )

    name = data["metadata"]["name"]
    spec = data["spec"]
    _validate_controller_spec(path, name, spec)
    return name


def validate_topology_profile(path: Path, data: dict[str, Any]) -> str:
    check_common(path, data, "TopologyProfile")
    require_fields(
        path,
        data,
        [
            "spec.mode",
            "spec.namespace",
            "spec.active_clusters",
            "spec.clusters_file",
            "spec.services_file",
            "spec.additional_services_file",
            "spec.placement_source",
        ],
    )

    name = data["metadata"]["name"]
    spec = data["spec"]

    if name not in VALID_TOPOLOGIES:
        raise ValidationError(f"{path}: invalid topology profile name `{name}`")
    if spec["mode"] != name:
        raise ValidationError(f"{path}: `spec.mode` must match `metadata.name`")
    if not isinstance(spec["active_clusters"], list) or not spec["active_clusters"]:
        raise ValidationError(f"{path}: `spec.active_clusters` must be a non-empty list")

    for field in ["clusters_file", "services_file", "additional_services_file"]:
        assert_relative_repo_path(path, spec[field], f"spec.{field}")

    return name


def validate_workload_profile(path: Path, data: dict[str, Any]) -> str:
    check_common(path, data, "WorkloadProfile")
    require_fields(
        path,
        data,
        [
            "spec.pattern",
            "spec.duration_seconds",
            "spec.warmup_seconds",
            "spec.locust.spawn_rate",
        ],
    )

    name = data["metadata"]["name"]
    spec = data["spec"]

    if name not in VALID_WORKLOADS:
        raise ValidationError(f"{path}: invalid workload profile name `{name}`")
    if spec["pattern"] != name:
        raise ValidationError(f"{path}: `spec.pattern` must match `metadata.name`")
    if not isinstance(spec["duration_seconds"], int) or spec["duration_seconds"] <= 0:
        raise ValidationError(f"{path}: `spec.duration_seconds` must be a positive integer")
    if not isinstance(spec["warmup_seconds"], int) or spec["warmup_seconds"] < 0:
        raise ValidationError(f"{path}: `spec.warmup_seconds` must be a non-negative integer")

    locust = spec["locust"]
    if not isinstance(locust, dict):
        raise ValidationError(f"{path}: `spec.locust` must be a mapping")
    if not isinstance(locust["spawn_rate"], int) or locust["spawn_rate"] <= 0:
        raise ValidationError(f"{path}: `spec.locust.spawn_rate` must be a positive integer")
    if "sampling_interval_seconds" in spec and (
        not isinstance(spec["sampling_interval_seconds"], int) or spec["sampling_interval_seconds"] <= 0
    ):
        raise ValidationError(f"{path}: `spec.sampling_interval_seconds` must be a positive integer")

    return name


def validate_protocol_file(path: Path, data: dict[str, Any]) -> str:
    try:
        protocol = load_protocol(path)
    except ProtocolError as exc:
        raise ValidationError(str(exc)) from exc
    if data["metadata"]["name"] != protocol["metadata"]["name"]:
        raise ValidationError(f"{path}: metadata.name changed while loading protocol")
    return str(protocol["metadata"]["name"])


def merge_manifest(
    run_data: dict[str, Any],
    controller: dict[str, Any],
    topology: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    run_spec = run_data["spec"]
    merged = {
        "apiVersion": API_VERSION,
        "kind": "ResolvedExperimentManifest",
        "metadata": {
            "name": run_data["metadata"]["name"],
        },
        "spec": {
            "controller": controller["spec"],
            "topology": topology["spec"],
            "workload": workload["spec"],
            "run": run_spec,
        },
    }
    return merged


def _validate_run_seeds(path: Path, spec: dict[str, Any]) -> None:
    if not isinstance(spec["seeds"], list) or not spec["seeds"]:
        raise ValidationError(f"{path}: `spec.seeds` must be a non-empty list")
    if not all(isinstance(seed, int) for seed in spec["seeds"]):
        raise ValidationError(f"{path}: all `spec.seeds` values must be integers")
    if spec["seeds"] != [101, 102, 103, 104, 105]:
        raise ValidationError(f"{path}: `spec.seeds` must use common paired seeds [101, 102, 103, 104, 105]")


def _validate_run_timing(path: Path, spec: dict[str, Any]) -> None:
    if not isinstance(spec["duration_seconds"], int) or spec["duration_seconds"] <= 0:
        raise ValidationError(f"{path}: `spec.duration_seconds` must be a positive integer")
    if not isinstance(spec["warmup_seconds"], int) or spec["warmup_seconds"] < 0:
        raise ValidationError(f"{path}: `spec.warmup_seconds` must be a non-negative integer")


def _validate_run_outputs(path: Path, spec: dict[str, Any]) -> None:
    if not isinstance(spec["sla"], dict):
        raise ValidationError(f"{path}: `spec.sla` must be a mapping")
    if "p95_latency_ms" not in spec["sla"] or "minimum_success_rate" not in spec["sla"]:
        raise ValidationError(
            f"{path}: `spec.sla` must include `p95_latency_ms` and `minimum_success_rate`"
        )
    if not isinstance(spec["output_dir"], str) or not spec["output_dir"]:
        raise ValidationError(f"{path}: `spec.output_dir` must be a non-empty string")
    if spec.get("sampling_interval_seconds") != 15:
        raise ValidationError(f"{path}: `spec.sampling_interval_seconds` must be 15")
    if "reset_policy" in spec and not isinstance(spec["reset_policy"], dict):
        raise ValidationError(f"{path}: `spec.reset_policy` must be a mapping")


def _validate_run_protocol(path: Path, spec: dict[str, Any], manifest: dict[str, Any]) -> None:
    reference = spec.get("protocol")
    if not isinstance(reference, dict):
        raise ValidationError(f"{path}: `spec.protocol` must reference the approved protocol")
    expected = protocol_reference()
    for field, expected_value in expected.items():
        if reference.get(field) != expected_value:
            raise ValidationError(
                f"{path}: `spec.protocol.{field}` must be `{expected_value}`, got `{reference.get(field)}`"
            )

    protocol = load_protocol()
    controller = str(manifest["spec"]["controller"]["controller"])
    workload = str(manifest["spec"]["workload"]["pattern"])
    if controller in primary_controllers(protocol) and workload not in primary_workloads(protocol):
        raise ValidationError(f"{path}: primary controller `{controller}` uses non-primary workload `{workload}`")
    topology = manifest["spec"]["topology"]
    common = protocol["spec"]["common_topology"]
    if topology["namespace"] != common["namespace"]:
        raise ValidationError(f"{path}: topology namespace must be `{common['namespace']}`")
    if list(topology["active_clusters"]) != list(common["active_clusters"]):
        raise ValidationError(f"{path}: active clusters must match the approved common topology")

    timing = protocol["spec"]["timing"]
    if int(spec["warmup_seconds"]) != int(timing["warmup_seconds"]):
        raise ValidationError(f"{path}: warmup_seconds must be {timing['warmup_seconds']}")
    expected_duration = int(timing["measurement_seconds"].get(workload, spec["duration_seconds"]))
    if int(spec["duration_seconds"]) != expected_duration:
        raise ValidationError(f"{path}: duration_seconds must be {expected_duration} for {workload}")


def _validate_run_values(path: Path, spec: dict[str, Any]) -> None:
    _validate_run_seeds(path, spec)
    _validate_run_timing(path, spec)
    _validate_run_outputs(path, spec)


def _load_run_profiles(
    path: Path,
    spec: dict[str, Any],
    controllers: dict[str, dict[str, Any]],
    topologies: dict[str, dict[str, Any]],
    workloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for ref_field in ["controller_profile", "topology_profile", "workload_profile"]:
        assert_relative_repo_path(path, spec[ref_field], ref_field)
    profiles = (
        controllers.get(str((REPO_ROOT / spec["controller_profile"]).resolve())),
        topologies.get(str((REPO_ROOT / spec["topology_profile"]).resolve())),
        workloads.get(str((REPO_ROOT / spec["workload_profile"]).resolve())),
    )
    labels = ("controller", "topology", "workload")
    for label, profile in zip(labels, profiles):
        if profile is None:
            raise ValidationError(f"{path}: unknown {label} profile")
    return profiles  # type: ignore[return-value]


def _validate_run_name(path: Path, name: str, profiles: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    expected_name = "__".join(
        [
            profiles[0]["metadata"]["name"],
            profiles[2]["metadata"]["name"],
            profiles[1]["metadata"]["name"],
        ]
    )
    if name != expected_name:
        raise ValidationError(
            f"{path}: run name must be `{expected_name}` based on referenced profiles, got `{name}`"
        )


def validate_run_file(
    path: Path,
    data: dict[str, Any],
    controllers: dict[str, dict[str, Any]],
    topologies: dict[str, dict[str, Any]],
    workloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    check_common(path, data, "ExperimentRun")
    require_fields(
        path,
        data,
        [
            "spec.controller_profile",
            "spec.topology_profile",
            "spec.workload_profile",
            "spec.seeds",
            "spec.duration_seconds",
            "spec.warmup_seconds",
            "spec.sla",
            "spec.output_dir",
            "spec.protocol",
        ],
    )

    name = data["metadata"]["name"]
    spec = data["spec"]
    _validate_run_values(path, spec)
    controller, topology, workload = _load_run_profiles(path, spec, controllers, topologies, workloads)
    _validate_run_name(path, name, (controller, topology, workload))
    manifest = merge_manifest(data, controller, topology, workload)
    _validate_run_protocol(path, spec, manifest)
    return manifest


def list_files(subdir: str) -> list[Path]:
    return sorted((CONFIG_ROOT / subdir).glob("*.yaml"))


def validate_all() -> tuple[list[dict[str, Any]], list[str]]:
    manifests: list[dict[str, Any]] = []
    messages: list[str] = []

    controllers: dict[str, dict[str, Any]] = {}
    topologies: dict[str, dict[str, Any]] = {}
    workloads: dict[str, dict[str, Any]] = {}

    for path in list_files("protocols"):
        data = load_yaml(path)
        validate_protocol_file(path, data)
        messages.append(f"validated research protocol: {path.relative_to(REPO_ROOT)}")

    if not PROTOCOL_PATH.exists():
        raise ValidationError(f"missing approved research protocol: {PROTOCOL_PATH.relative_to(REPO_ROOT)}")

    for path in list_files("baselines"):
        data = load_yaml(path)
        validate_controller_profile(path, data)
        controllers[str(path.resolve())] = data
        messages.append(f"validated controller profile: {path.relative_to(REPO_ROOT)}")

    for path in list_files("topologies"):
        data = load_yaml(path)
        validate_topology_profile(path, data)
        topologies[str(path.resolve())] = data
        messages.append(f"validated topology profile: {path.relative_to(REPO_ROOT)}")

    for path in list_files("workloads"):
        data = load_yaml(path)
        validate_workload_profile(path, data)
        workloads[str(path.resolve())] = data
        messages.append(f"validated workload profile: {path.relative_to(REPO_ROOT)}")

    for path in list_files("runs"):
        data = load_yaml(path)
        manifest = validate_run_file(path, data, controllers, topologies, workloads)
        manifests.append(manifest)
        messages.append(f"validated run: {path.relative_to(REPO_ROOT)}")

    return manifests, messages


def validate_resolved_manifests(manifests: list[dict[str, Any]]) -> int:
    """Validate checked-in resolved manifests against canonical run configs.

    Resolved manifests are generated artifacts, but they are tracked and are
    consumed by matrix enumeration. If the directory exists, require an exact
    semantic match with the manifests produced by ``validate_all`` so malformed
    or stale generated files cannot pass the normal config validation command.
    """

    if not RESOLVED_ROOT.exists():
        return 0

    expected = {str(manifest["metadata"]["name"]): manifest for manifest in manifests}
    files = sorted(RESOLVED_ROOT.glob("*.yaml"))
    actual_names: set[str] = set()
    for path in files:
        try:
            data = load_yaml(path)
        except (OSError, yaml.YAMLError) as exc:
            raise ValidationError(f"{path}: invalid resolved YAML: {exc}") from exc
        metadata = data.get("metadata")
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{path}: resolved manifest must contain metadata.name")
        actual_names.add(name)
        if name not in expected:
            raise ValidationError(f"{path}: resolved manifest does not correspond to a canonical run")
        if data != expected[name]:
            raise ValidationError(
                f"{path}: resolved manifest differs from the canonical run; regenerate with --write-resolved"
            )

    missing = sorted(set(expected) - actual_names)
    if missing:
        raise ValidationError(
            "missing resolved manifests: " + ", ".join(missing)
        )
    if len(actual_names) != len(files):
        raise ValidationError("resolved manifest filenames contain duplicate metadata.name values")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment contract files")
    parser.add_argument(
        "--write-resolved",
        action="store_true",
        help="write resolved manifests next to run files under experiments/resolved",
    )
    args = parser.parse_args()

    try:
        manifests, messages = validate_all()
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    for message in messages:
        print(message)

    if args.write_resolved:
        resolved_root = REPO_ROOT / "experiments" / "resolved"
        resolved_root.mkdir(exist_ok=True)
        for manifest in manifests:
            output = resolved_root / f"{manifest['metadata']['name']}.yaml"
            with output.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(manifest, handle, sort_keys=False)
            print(f"wrote resolved manifest: {output.relative_to(REPO_ROOT)}")

    try:
        resolved_count = validate_resolved_manifests(manifests)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    if resolved_count:
        print(f"validated resolved manifests: {resolved_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
