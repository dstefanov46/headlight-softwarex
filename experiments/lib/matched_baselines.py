"""Frozen input ownership and selection for matched-v2 preparation."""
from __future__ import annotations

import hashlib
import json
import math
import itertools
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ai-module/ai-module'))
EXPECTED = {
    'large-v2-policy/manifest.json': 'c7adcaa63849d66a170d5c5b2906a8481ad8381489099bab5e8ca0742d5aa69b',
    'training-v2-policy-accepted/training-report.json': '7c1cd6bcb01b4ef30a520e686b109528df86d203f9cff81c1766247b56e811b4',
    'policy-tuning-v2/selection.json': 'aa40e455a4cd59b0d7e910deb366befd6589d07cf3e98a76843a9decbde1316d',
}
DATASET_HASH = '576cde51cb4f1466d748761eb0f2159b2de77b029de22f64d3e0b76fb504c7ec'
FORBIDDEN = set(range(101, 106)) | set(range(701, 706))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def implementation_hashes() -> dict[str, str]:
    """Include executable code, checks, and the matched deployment contract."""
    paths = set()
    for relative in ('experiments', 'experiments/lib', 'ai-module/ai-module/headlight', 'experiments/tests', 'ai-module/tests'):
        paths.update((ROOT/relative).glob('*.py'))
    for relative in ('experiments/scripts', 'ai-module/baselines'):
        paths.update(p for p in (ROOT/relative).glob('*') if p.is_file())
    for relative in ('ai-module/deployments/lstm-matched-deployment.yaml',
                     'experiments/configs/baselines/static-matched.yaml',
                     'experiments/configs/baselines/lstm-matched.yaml',
                     'experiments/configs/models/lstm-matched-live-policy.json',
                     'experiments/configs/protocols/matched-baselines-v2.json'):
        paths.add(ROOT/relative)
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}

def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected object: {path}')
    return value

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive writes preserve completed evidence and frozen artifacts.
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def resolve_artifact_path(recorded_path: str, artifact_root: Path, relocation: Mapping[str, str]) -> Path:
    path = Path(recorded_path)
    if '..' in path.parts:
        raise ValueError('artifact path traversal')
    if path.is_absolute():
        for old in sorted(relocation, key=len, reverse=True):
            try:
                suffix = path.relative_to(old)
            except ValueError:
                continue
            return (Path(relocation[old]) / suffix).resolve()
        raise ValueError(f'absolute artifact path has no relocation: {path}')
    return (artifact_root / path).resolve()

def validate_comparison_inputs(inputs: Mapping[str, Any]) -> None:
    from headlight.synthetic_v2 import load_manifest, validate_dataset_files, NODE_ORDER, ENDPOINTS
    base = Path(inputs['artifact_root']) / 'results/synthetic-headlight-v2'
    for relative, expected in EXPECTED.items():
        if sha256(base / relative) != expected:
            raise ValueError(f'historical input hash mismatch: {relative}')
    manifest_path = base / 'large-v2-policy/manifest.json'
    manifest = load_manifest(manifest_path)
    if manifest['dataset_hash'] != DATASET_HASH or tuple(manifest['node_order']) != tuple(NODE_ORDER) or tuple(manifest['endpoints']) != tuple(ENDPOINTS):
        raise ValueError('dataset identity/order mismatch')
    for filename, field in [('feature-schema.json','feature_schema_sha256'),('model-config.json','model_config_sha256'),('adjacency.json','adjacency_sha256'),('paper-priors.json','paper_priors_sha256')]:
        if sha256(manifest_path.parent/filename) != manifest[field]:
            raise ValueError(f'changed dataset schema/config: {filename}')
    model_config=read_json(manifest_path.parent/'model-config.json')
    if model_config['horizons_seconds'] != [30,60,180] or model_config['sampling_interval_seconds'] != 15:
        raise ValueError('changed model timing/horizons')
    membership = {}
    for split, start, count in [('fitting', 10000, 128), ('calibration', 10128, 32), ('holdout', 10160, 32)]:
        seeds = [int(r['seed']) for r in manifest['runs'] if r['split'] == split]
        if len(seeds) != count or set(seeds) != set(range(start, start + count)):
            raise ValueError(f'{split} membership mismatch')
        membership[split] = set(seeds)
    owned = set().union(*membership.values(), set(inputs['development_seeds']))
    if owned & FORBIDDEN or inputs['development_seeds'] != [711, 712, 713]:
        raise ValueError('evaluation seed leakage or changed development membership')
    if inputs['static_totals'] != [2, 3, 4, 6, 8, 10, 12, 14]:
        raise ValueError('changed static search')
    if inputs['cases'] != ['gradual-ramp', 'flash-crowd', 'diurnal', 'capacity-readiness-failure']:
        raise ValueError('changed development cases')
    expected_grid={f'weights-{i:02d}':{'resource':r,'sla_risk':1.,'migration':m,'reliability':a} for i,(r,m,a) in enumerate(itertools.product([.25,.5],[.1,1.],[.25,.5]))}
    if inputs['candidate_weights']!=expected_grid:
        raise ValueError('changed preregistered policy grid')
    if inputs['loss_scales']!={'demand_rps':6.,'latency_ms':1000.,'boundary_extra_weight':64.,'sla_bce_weight':.5,'success_bce_weight':.5}:
        raise ValueError('changed archived loss normalization')
    validate_dataset_files(manifest_path, verify_hashes=True)

def load_comparison_inputs(path: Path) -> dict[str, Any]:
    inputs = read_json(path)
    for key in ('artifact_root', 'service_profile', 'live_policy', 'reference_runtime_lock', 'retained_live_metadata', 'retained_live_bundle'):
        if key in inputs:
            p = Path(inputs[key])
            inputs[key] = str((path.resolve().parent / p).resolve() if not p.is_absolute() else p.resolve())
    inputs['relocation_map'] = {'/home/dimi/headlight-jsa': inputs['artifact_root']}
    return inputs

def prepare_inputs(archive: Path | None, output_dir: Path) -> dict[str, Any]:
    """Selectively recover data, rejecting traversal and mismatched existing bytes."""
    if archive is not None:
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                name=PurePosixPath(entry.filename)
                if name.is_absolute() or '..' in name.parts or '\\' in entry.filename:
                    raise ValueError('unsafe archive member')
            required={'large-v2-policy','training-v2-policy-accepted','training-v2-policy-final'}
            pending=['training-v2-policy-accepted','training-v2-policy-final']
            inspected=set()
            while pending:
                family=pending.pop()
                if family in inspected: continue
                inspected.add(family)
                report=json.loads(source.read(f'headlight-jsa/results/synthetic-headlight-v2/{family}/training-report.json'))
                for run in report.get('runs',[]):
                    recorded=run.get('initialized_from_report')
                    if not recorded: continue
                    parts=PurePosixPath(recorded.replace('\\','/')).parts
                    if '..' in parts: raise ValueError('training-chain path traversal')
                    try:
                        index=parts.index('synthetic-headlight-v2')
                        ancestor=parts[index+1]
                    except (ValueError,IndexError) as exc:
                        raise ValueError('training-chain reference outside required results') from exc
                    if not ancestor.startswith('training-'): raise ValueError('unsupported training-chain directory')
                    required.add(ancestor);pending.append(ancestor)
            for entry in source.infolist():
                name = PurePosixPath(entry.filename)
                if name.is_absolute() or '..' in name.parts or '\\' in entry.filename:
                    raise ValueError('unsafe archive member')
                parts = name.parts
                if len(parts) < 4 or parts[:3] != ('headlight-jsa', 'results', 'synthetic-headlight-v2'):
                    continue
                family = parts[3]
                if not (family in required or (family == 'policy-tuning-v2' and name.name == 'selection.json')):
                    continue
                if entry.is_dir():
                    continue
                if entry.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError('archive symlink forbidden')
                target = output_dir.joinpath(*parts[1:])
                if not target.resolve().is_relative_to(output_dir.resolve()):
                    raise ValueError('archive destination escapes recovery root')
                data = source.read(entry)
                if target.exists():
                    if target.read_bytes() != data:
                        raise ValueError(f'refuse mismatched overwrite: {target}')
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(data)
    base = output_dir / 'results/synthetic-headlight-v2'
    for relative, expected in EXPECTED.items():
        if sha256(base / relative) != expected:
            raise ValueError(f'input mismatch: {relative}')
    from headlight.synthetic_v2 import validate_dataset_files
    counts = validate_dataset_files(base / 'large-v2-policy/manifest.json')
    return {'artifact_root': str(output_dir.resolve()), 'hashes': EXPECTED, 'dataset_hash': DATASET_HASH, 'counts': counts}

def select_development_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError('empty candidate set')
    fields = ('success_rate', 'request_sla_violation_rate', 'mean_pod_count', 'migration_proxy')
    if any(not all(math.isfinite(float(row[k])) for k in fields) for row in rows):
        raise ValueError('nonfinite development metrics')
    best_success = max(float(row['success_rate']) for row in rows)
    best_sla = min(float(row['request_sla_violation_rate']) for row in rows)
    eligible = [dict(row) for row in rows if float(row['success_rate']) >= best_success - .002 and float(row['request_sla_violation_rate']) <= best_sla + .01]
    if not eligible:
        raise ValueError('empty joint eligibility set')
    # Historical tuning normalizes within the joint eligible set.
    for row in eligible:
        row['selection_objective'] = sum((float(row[k]) - min(float(r[k]) for r in eligible)) / max(max(float(r[k]) for r in eligible) - min(float(r[k]) for r in eligible), 1e-9) for k in ('mean_pod_count', 'migration_proxy'))
    return min(eligible, key=lambda r: (r['selection_objective'], r['request_sla_violation_rate'], -r['success_rate'], r['candidate_id']))
