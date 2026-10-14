"""A fixed provisioning vector, independent of workload user count."""
from pathlib import Path
from typing import Mapping, Any
from experiments.lib.matched_baselines import identity, read_json
from headlight.profile import ServiceProfile, load_service_profile

def fixed_placement(total_replicas: int, profile: ServiceProfile) -> dict[str, dict[str, int]]:
    if total_replicas not in (2, 3, 4, 6, 8, 10, 12, 14):
        raise ValueError('unregistered replica total')
    left = min(6, max(1, round(total_replicas * 6 / 14)))
    right = total_replicas - left
    if not 1 <= right <= 8:
        raise ValueError('cluster bounds violated')
    return {node.name: {'cluster2': left, 'cluster3': right} if node.scalable else {c: int(limits['initial_replicas']) for c, limits in profile.cluster_limits.items()} for node in profile.nodes}

def build_matched_static_plan(selection: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict:
    profile = load_service_profile(Path(comparison['service_profile']))
    plan={'schema_version': 'headlight.static-plan/matched-v2', 'profile_hash': profile.profile_hash, 'selection_hash': identity(selection), 'input_hash': identity(comparison), 'workload_capacity_users': comparison['workload_capacity_users'], 'replicas_per_application_service': selection['total_replicas'], 'assignments': fixed_placement(selection['total_replicas'], profile)}
    plan['plan_hash']=identity(plan)
    return plan

def validate_matched_static_plan(path: Path, profile_path: Path) -> dict:
    plan = read_json(path)
    profile = load_service_profile(profile_path)
    if plan.get('plan_hash')!=identity({k:v for k,v in plan.items() if k!='plan_hash'}):
        raise ValueError('static plan hash mismatch')
    if plan['schema_version'] != 'headlight.static-plan/matched-v2' or plan['profile_hash'] != profile.profile_hash:
        raise ValueError('static plan/profile mismatch')
    if plan['assignments'] != fixed_placement(plan['replicas_per_application_service'], profile):
        raise ValueError('static placement mismatch')
    if not all(len(plan.get(k, '')) == 64 for k in ('selection_hash', 'input_hash')):
        raise ValueError('missing plan provenance')
    return plan
