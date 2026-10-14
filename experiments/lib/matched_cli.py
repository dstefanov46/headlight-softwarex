"""One CLI contract shared by individually executable preparation stages."""
import argparse
import json
import os
import sys
import time
import traceback
import subprocess
from pathlib import Path
from experiments.lib.matched_baselines import ROOT, identity, implementation_hashes, load_comparison_inputs, prepare_inputs, read_json, sha256, write_json, validate_comparison_inputs

STAGES = ('inputs','static','train','calibrate','lstm','reference','verify','freeze')

def stage_artifacts(output, stage, newly_created):
    # Recovered successful episodes predate this attempt, but remain evidence
    # owned by the completed search and must be checked on every future resume.
    owned = set(newly_created)
    if stage in {'static','lstm'}:
        owned.update(p for p in (output/f'{stage}-episode-records').rglob('*') if p.is_file())
    return {str(p.relative_to(output)):sha256(p) for p in sorted(owned)
            if p != output/'status.json'}

def validate_completed_artifacts(output, completed):
    for stage, artifacts in completed.items():
        for name, digest in artifacts.items():
            path=(output/name).resolve()
            if not path.is_relative_to(output) or not path.is_file() or sha256(path)!=digest:
                raise ValueError(f'completed-stage artifact mismatch: {stage}/{name}')

def preparation_state(status):
    completed=status.get('completed',{})
    unresolved_failures=[attempt for attempt in status.get('attempts',[])
                         if attempt.get('exit_status') not in (None,0)
                         and attempt.get('stage') not in completed]
    return 'blocked/configured' if unresolved_failures else 'in-progress'

def main(stage=None, argv=None):
    parser=argparse.ArgumentParser(description='Matched-v2 preparation. Full runs are offline only; evaluation is never launched.')
    parser.add_argument('--comparison',type=Path,default=ROOT/'experiments/configs/protocols/matched-baselines-v2.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--artifact-root',type=Path)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    parser.add_argument('--workers',type=int,default=1,help='Independent CPU workers for static episodes; GPU episodes remain serial')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='List dependency stages only; no training, selection, or evaluation')
    parser.add_argument('--live-readiness',action='store_true',help='Record readiness as pending; use deployment/shadow adapter separately against the assigned testbed')
    args=parser.parse_args(argv)
    import torch
    from experiments.lib import matched_preparation as work
    comparison=load_comparison_inputs(args.comparison)
    if args.artifact_root:
        comparison['artifact_root']=str(args.artifact_root.resolve())
        comparison['relocation_map']={'/home/dimi/headlight-jsa':comparison['artifact_root']}
    comparison['device']=args.device
    comparison['workers']=args.workers
    recover_archive=False
    if args.archive:
        existing=Path(comparison['artifact_root'])/'results/synthetic-headlight-v2/large-v2-policy/manifest.json'
        if existing.exists():
            # A mismatching existing corpus must fail rather than be replaced.
            validate_comparison_inputs(comparison)
        else:
            recover_archive=True
            recovered=args.output_dir.resolve()/'recovered-inputs'
            comparison['artifact_root']=str(recovered)
            comparison['relocation_map']={'/home/dimi/headlight-jsa':str(recovered)}
    selected_stages=[stage] if stage else list(STAGES)
    if args.dry_run:
        print(json.dumps({'stages':selected_stages,'output_dir':str(args.output_dir),'device':args.device,'execution':False},indent=2)); return 0
    if args.workers<1: parser.error('--workers must be positive')
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA requested but unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda': os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    if (output/'freeze.json').exists(): raise ValueError('frozen run cannot be modified')
    logs=output/'logs'; logs.mkdir(exist_ok=True)
    source=implementation_hashes()
    fingerprint=identity({'comparison':comparison,'source':source,'profile_hash':sha256(Path(comparison['service_profile'])),'live_policy_hash':sha256(Path(comparison['live_policy'])),'torch':torch.__version__})
    state_path=output/'status.json'
    if state_path.exists():
        status=read_json(state_path)
        if not args.resume: raise ValueError('existing run requires --resume')
        if status['fingerprint']!=fingerprint: raise ValueError('resume source/config/input identity mismatch')
    else:
        status={'fingerprint':fingerprint,'state':'implementation-ready','completed':{},'attempts':[]}
    started=time.perf_counter()
    attempt=None
    stage_start=started
    try:
        # Individual entry points must enforce the same prerequisite integrity
        # as the complete orchestration command.
        validate_completed_artifacts(output,status['completed'])
        for current in selected_stages:
            if current in status['completed']:
                continue
            attempt={'stage':current,'argv':list(argv if argv is not None else sys.argv[1:]),'started_at_unix':time.time()}
            status['attempts'].append(attempt)
            state_path.write_text(json.dumps(status,indent=2)+'\n')
            before=set(p for p in output.rglob('*') if p.is_file())
            stage_start=time.perf_counter()
            print(json.dumps({'event':'stage-started','stage':current}),flush=True)
            dependencies={'static':['inputs'],'train':['inputs'],'calibrate':['inputs','train'],'lstm':['inputs','calibrate'],'reference':['inputs'],'verify':['inputs','static','train','calibrate','lstm','reference'],'freeze':['verify']}
            if any(required not in status['completed'] for required in dependencies.get(current,[])):
                raise ValueError(f'{current} prerequisites incomplete: {dependencies[current]}')
            if current!='inputs':
                validate_comparison_inputs(comparison)
                for name,digest in read_json(output/'input-verification.json')['historical_files'].items():
                    if sha256(Path(name))!=digest: raise ValueError(f'original artifact changed: {name}')
            if current=='inputs':
                if recover_archive:
                    prepare_inputs(args.archive,Path(comparison['artifact_root']))
                work.audit_inputs(comparison,output)
            elif current=='static': work.search(comparison,output,'static')
            elif current=='train': work.train(comparison,output,torch.device(args.device))
            elif current=='calibrate': work.calibrate(comparison,output,torch.device(args.device))
            elif current=='lstm': work.search(comparison,output,'lstm')
            elif current=='reference': work.replay(comparison,output)
            elif current=='verify':
                from headlight.bundle_lstm_matched import MatchedLSTMBundle
                model=MatchedLSTMBundle.load(output/'lstm-bundle').build_model(torch.device(args.device))
                sample=work.tensor_batch(next(work.batches(comparison,'fitting')),torch.device(args.device))
                with torch.no_grad():
                    prediction=model(sample['node_features'],sample['edge_features'],sample['candidate_replicas'],sample['candidate_placement'],sample['readiness'])
                if any(not torch.isfinite(value).all() for value in prediction['candidate_performance'].values() if isinstance(value,torch.Tensor)):
                    raise ValueError('nonfinite bundle predictions')
                checks=[]
                for directory,pattern in [('experiments/tests','test_matched*.py'),('experiments/tests','test_static_plan_matched.py'),('ai-module/tests','test_lstm*.py'),('ai-module/tests','test_runtime_lstm_matched.py'),('experiments/tests','test_closed_loop_v2*.py'),('ai-module/tests','test_runtime_v2.py'),('ai-module/tests','test_policy_v2.py')]:
                    command=[sys.executable,'-m','unittest','discover','-s',directory,'-p',pattern]
                    check_start=time.perf_counter()
                    result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    log=logs/f'check-{len(checks):02d}.txt'
                    with log.open('x') as stream: stream.write(result.stdout)
                    checks.append({'command':command,'exit_status':result.returncode,'elapsed_seconds':time.perf_counter()-check_start,'log':str(log),'log_hash':sha256(log)})
                    if result.returncode:
                        write_json(output/'implementation-checks.json',{'passed':False,'checks':checks})
                        raise ValueError(f'focused implementation check failed: {pattern}')
                write_json(output/'implementation-checks.json',{'passed':True,'checks':checks})
                write_json(output/'readiness-report.json',{'status':'pending','mocked_checks_are_live_qualification':False,'dependency':'assigned testbed contexts, reachable telemetry endpoints, and executed static/shadow readiness evidence','requested':args.live_readiness})
            elif current=='freeze': work.freeze(comparison,output)
            attempt.update(exit_status=0,elapsed_seconds=time.perf_counter()-stage_start)
            write_json(logs/f'{len(status["attempts"]):03d}-{current}.json',attempt)
            after=set(p for p in output.rglob('*') if p.is_file())
            status['completed'][current]=stage_artifacts(output,current,after-before)
            status['state']='offline preparation complete, live readiness pending' if current=='freeze' else preparation_state(status)
            state_path.write_text(json.dumps(status,indent=2)+'\n')
    except Exception as exc:
        if attempt is None:
            attempt={'stage':'resume-validation','argv':list(argv if argv is not None else sys.argv[1:]),'started_at_unix':time.time()}
            status['attempts'].append(attempt)
        attempt.update(exit_status=1,error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-stage_start)
        status['state']='blocked/configured'
        write_json(logs/f'{len(status["attempts"]):03d}-failed.json',{**attempt,'traceback':traceback.format_exc()})
        state_path.write_text(json.dumps(status,indent=2)+'\n')
        print(attempt['error'],file=sys.stderr); return 1
    finally:
        cumulative=sum(float(a.get('elapsed_seconds',0)) for a in status['attempts'])
        (output/'completion-report.md').write_text('# Matched baseline preparation\n\n'+f'State: {status["state"]}\n\nCumulative recorded stage execution seconds: {cumulative:.3f}\nLatest invocation seconds: '+f'{time.perf_counter()-started:.3f}\n\n'+ '\n'.join(f'- {a["stage"]}: exit {a.get("exit_status","interrupted")}, seconds {a.get("elapsed_seconds","unknown")}' for a in status['attempts'])+'\n\nSee status.json and logs for exact commands and artifact hashes. Final evaluation was not launched.\n')
    return 0
