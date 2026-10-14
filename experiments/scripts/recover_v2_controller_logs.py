#!/usr/bin/env python3
"""Archive retained Kind node CRI logs and reconstruct the complete controller stream."""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def lines(path):
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()


def reconstruct(directory: Path, output: Path) -> dict:
    sources=[]
    for path in directory.iterdir():
        if not path.is_file(): continue
        with lines(path) as handle:
            first=next(handle,'')
        if first:
            sources.append((first.split(' ',1)[0],path))
    partial=collections.defaultdict(str)
    events=collections.Counter()
    ready=False
    with output.open('w') as target:
        for _,path in sorted(sources):
            with lines(path) as handle:
                for line in handle:
                    fields=line.rstrip('\n').split(' ',3)
                    if len(fields)!=4 or fields[2] not in ('P','F'):
                        raise ValueError(f'invalid CRI record in {path}')
                    _,stream,flag,payload=fields
                    partial[stream]+=payload
                    if flag=='P': continue
                    text=partial.pop(stream)
                    target.write(text+'\n')
                    ready=ready or text.startswith('HEADLIGHT_V2_RUNTIME_READY ')
                    try: record=json.loads(text)
                    except ValueError: continue
                    events[record.get('event','other')]+=1
    if any(partial.values()):
        raise ValueError('captured CRI logs end in a partial line')
    return {'runtime_ready_retained':ready,'events':dict(events),
            'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for _,path in sorted(sources)},
            'reconstructed_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pod-json',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    pod=json.loads(args.pod_json.read_text())
    pod=pod.get('pod_provenance',pod)
    meta=pod['metadata']
    source=f"/var/log/pods/{meta['namespace']}_{meta['name']}_{meta['uid']}/ai-module-v2"
    args.output_dir.mkdir(parents=True,exist_ok=False)
    subprocess.run(['docker','cp',f"{pod['spec']['nodeName']}:{source}",str(args.output_dir/'cri')],check=True)
    report=reconstruct(args.output_dir/'cri',args.output_dir/'controller-complete.log')
    (args.output_dir/'recovery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
