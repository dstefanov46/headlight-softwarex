#!/usr/bin/env python3
"""Verify every retained diagnostic artifact hash, array schema and candidate axis."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(records_path: Path) -> dict:
    checked = {}
    records = references = replayable = 0
    for line in records_path.read_text().splitlines():
        record = json.loads(line)
        records += 1
        if not record.get('replay'):
            if record.get('model_status') == 'ok':
                raise ValueError('scored record has no replay artifacts')
            continue
        replayable += 1
        for kind, reference in record['replay'].items():
            references += 1
            path = records_path.parent / reference['path']
            if path.parent.resolve() != records_path.parent.resolve():
                raise ValueError('artifact reference leaves the diagnostic directory')
            if path not in checked:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path, allow_pickle=False) as arrays:
                    schema = {name: {'shape': list(arrays[name].shape), 'dtype': str(arrays[name].dtype)} for name in arrays.files}
                checked[path] = digest, schema
            digest, schema = checked[path]
            if digest != reference['sha256'] or schema != reference['arrays']:
                raise ValueError(f'artifact hash or array schema differs: {path}')
            if kind == 'candidates':
                count = len(record['candidate_axes']['candidate_ids'])
                if any(value['shape'][1] != count for value in schema.values()):
                    raise ValueError('candidate axis differs from candidate IDs')
    return {'records': records, 'replayable_records': replayable, 'references_verified': references,
            'unique_artifacts_verified': len(checked), 'all_hashes_shapes_dtypes_and_candidate_axes_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.records), indent=2))


if __name__ == '__main__':
    main()
