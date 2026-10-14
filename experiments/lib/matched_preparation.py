"""Executable preparation stages; full runs are explicitly user launched."""
from __future__ import annotations
import itertools
import json
import platform
import shutil
import statistics
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from experiments.lib.matched_baselines import ROOT, EXPECTED, implementation_hashes, read_json, write_json, sha256, identity, validate_comparison_inputs, select_development_candidate, resolve_artifact_path
from headlight.synthetic_v2 import load_manifest, split_records, iter_window_batches
from headlight.models_lstm_matched import MatchedLSTMModel
from headlight.training_v2 import pinball_metric

def dataset(comparison):
    return Path(comparison['artifact_root'])/'results/synthetic-headlight-v2/large-v2-policy'

def runs(comparison, split):
    root = dataset(comparison)
    for record in split_records(load_manifest(root/'manifest.json'), split):
        with np.load(root/record['path'], allow_pickle=False) as source:
            yield {k:source[k] for k in source.files}

def batches(comparison, split, *, seed=601, shuffle=False, limit=None):
    root = dataset(comparison)
    records = split_records(load_manifest(root/'manifest.json'), split)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(records)
    for record in records:
        with np.load(root/record['path'], allow_pickle=False) as source:
            iterator = iter_window_batches(source, lookback_steps=24, batch_size=64, rng=rng, shuffle=shuffle)
            yield from itertools.islice(iterator, limit) if limit else iterator

def tensor_batch(batch, device):
    return {k:torch.as_tensor(v,device=device,dtype=torch.bool if k=='boundary_mask' else torch.float32) for k,v in batch.items()}

def demand_loss(model, batch):
    output = model(batch['node_features'])
    scale = model.demand_output_scale_rps
    loss = sum(pinball_metric(output[k]/scale, batch['demand_target']/scale, q) for k,q in [('demand_q50',.5),('demand_q90',.9)])*.5
    return loss

def training_config(comparison):
    config = read_json(dataset(comparison)/'model-config.json')
    config.update(model_type='lstm-matched-v2', lookback_steps=24, hidden_width=48, dropout=.05, lstm_layers=2)
    return config

def audit_inputs(comparison, output):
    validate_comparison_inputs(comparison)
    from dataclasses import asdict
    from headlight.closed_loop_v2 import ClosedLoopConfig
    from headlight.profile import load_service_profile
    if asdict(ClosedLoopConfig()) != {**comparison['simulator_config'], 'normal_cluster_capacity':tuple(comparison['simulator_config']['normal_cluster_capacity']), 'constrained_cluster_capacity':tuple(comparison['simulator_config']['constrained_cluster_capacity'])}:
        raise ValueError('simulator defaults differ from frozen protocol')
    profile = load_service_profile(comparison['service_profile'])
    if profile.profile_hash != comparison['service_profile_hash']:
        raise ValueError('authoritative profile identity mismatch')
    retained=read_json(Path(comparison['retained_live_metadata']))
    lock=read_json(Path(comparison['reference_runtime_lock']))
    expected_bundle='c37050a44dd2fc876bf6523aaeabdd364af7608ffeee5e1586a6cb8bbaaaeb4a'
    if retained['bundle_id']!=expected_bundle or lock['bundle']['bundle_id']!=expected_bundle:
        raise ValueError('retained reference bundle identity mismatch')
    checkpoint=dataset(comparison).parent/'training-v2-policy-accepted/full-seed-601.pt'
    # Live packaging strips authoritative buffers and serializes learned
    # weights anew. Its file hash is not the source checkpoint's file hash.
    equivalence={'source_checkpoint_hash':sha256(checkpoint),'packaged_checkpoint_hash':retained['component_hashes']['checkpoint'],'verification':'packaged bundle unavailable; tensor equivalence not assessed'}
    live_bundle=Path(comparison['retained_live_bundle'])
    if live_bundle.is_dir():
        from headlight.bundle_v2 import HeadlightV2Bundle, load_v2_checkpoint
        bundled=HeadlightV2Bundle.load(live_bundle)
        if bundled.bundle_id!=expected_bundle: raise ValueError('retained packaged bundle differs')
        original=torch.load(checkpoint,map_location='cpu',weights_only=True)
        packaged=load_v2_checkpoint(bundled,torch.device('cpu'))
        if original.keys()!=packaged.keys() or any(not torch.equal(original[k],packaged[k]) for k in original):
            raise ValueError('accepted weights differ from retained packaged weights')
        equivalence.update(verification='all restored tensors exactly equal',tensor_count=len(original))
    if comparison['cluster_capacity_weights'] != [1.,1.]:
        raise ValueError('this preregistered equal-resource profile requires weights [1,1]')
    from headlight.policy_v2 import PolicyV2Config
    live=read_json(Path(comparison['live_policy']))
    if set(live)!=set(PolicyV2Config.__dataclass_fields__):
        raise ValueError('live fixed settings incomplete')
    PolicyV2Config(**live).validate(profile,controller_id='lstm-matched')
    if tuple(profile.node_order) != tuple(read_json(dataset(comparison)/'model-config.json')['node_order']):
        raise ValueError('profile/schema node mismatch')
    for cluster, maximum in [('cluster2',6),('cluster3',8)]:
        if profile.cluster_limits[cluster]['max_replicas'] != maximum:
            raise ValueError('comparison profile bounds mismatch')
    reports = []
    pending = ['results/synthetic-headlight-v2/training-v2-policy-accepted/training-report.json']
    seen = set()
    while pending:
        recorded = pending.pop()
        path = resolve_artifact_path(recorded, Path(comparison['artifact_root']), comparison['relocation_map'])
        if path in seen:
            continue
        seen.add(path)
        report = read_json(path)
        entries = []
        for run in report.get('runs',[]):
            history = run.get('history',[])
            entries.append({'initialization_seed':run.get('initialization_seed'), 'optimizer_epochs_observed':sum(e.get('training_loss') is not None for e in history) if history else 'unknown', 'validation_entries':len(history), 'budget_fields':{k:run.get(k,'unknown') for k in ('max_epochs','learning_rate','batch_size','max_batches_per_run')}, 'initialized_from_report':run.get('initialized_from_report')})
            if run.get('initialized_from_report'):
                pending.append(run['initialized_from_report'])
        reports.append({'path':str(path),'sha256':sha256(path),'runs':entries})
    source = implementation_hashes()
    for relative in source:
        destination=output/'implementation-source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    import subprocess
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    ancestry=subprocess.run(['git','merge-base','--is-ancestor','2b544d3','HEAD'],cwd=ROOT).returncode
    patch=subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    (output/'implementation.diff').write_text(patch)
    historical_files={str(path):sha256(path) for report in reports for path in [Path(report['path']),*Path(report['path']).parent.glob('*.pt')]}
    historical_files.update({comparison[key]:sha256(Path(comparison[key])) for key in ('service_profile','live_policy','retained_live_metadata','reference_runtime_lock')})
    evidence = {'original_hashes':EXPECTED,'dataset_hash':load_manifest(dataset(comparison)/'manifest.json')['dataset_hash'],'profile_hash':profile.profile_hash,'profile_file_hash':sha256(Path(comparison['service_profile'])),'live_policy_hash':sha256(Path(comparison['live_policy'])),'retained_metadata_hash':sha256(Path(comparison['retained_live_metadata'])),'reference_runtime_lock_hash':sha256(Path(comparison['reference_runtime_lock'])),'accepted_checkpoint_hash':sha256(checkpoint),'historical_files':historical_files,'source_hashes':source,'revision':revision,'historical_revision_is_ancestor':ancestry==0,'implementation_diff_sha256':sha256(output/'implementation.diff'),'python':sys.version,'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,'platform':platform.platform()}
    write_json(output/'input-verification.json',evidence)
    write_json(output/'checkpoint-equivalence.json',equivalence)
    write_json(output/'relocation-map.json',comparison['relocation_map'])
    write_json(output/'preparation-protocol.json',comparison)
    write_json(output/'historical-training-audit.json',{'reports':reports,'new_lstm_budget':comparison['lstm_budget'],'budget_equivalence_claimed':False})
    return evidence

def train(comparison, output, device):
    expected = {'initializations':[601,602,603],'max_epochs':12,'batch_size':64,'learning_rate':.001,'weight_decay':1e-5,'gradient_clip':5.,'patience':4,'improvement_tolerance':1e-5,'max_batches_per_run':2}
    if comparison['lstm_budget'] != expected:
        raise ValueError('changed preregistered training budget')
    config = training_config(comparison)
    target = output/'lstm-training'; target.mkdir(exist_ok=True)
    write_json(target/'model-config.json',config)
    # Tiny fitting-only smoke before the registered independent initializations.
    torch.manual_seed(601)
    smoke = MatchedLSTMModel(config).to(device)
    sample = tensor_batch(next(batches(comparison,'fitting')),device)
    smoke_loss = demand_loss(smoke,sample)
    smoke_loss.backward()
    if not torch.isfinite(smoke_loss):
        raise ValueError('fitting smoke failed')
    torch.nn.utils.clip_grad_norm_(smoke.parameters(),5.)
    torch.optim.AdamW(smoke.parameters(),lr=.001,weight_decay=1e-5).step()
    write_json(target/'smoke.json',{'split':'fitting','loss':float(smoke_loss.detach()),'updates':1,'batches':1,'samples':len(sample['node_features']),'not_a_selection_candidate':True})
    del smoke
    choices = []
    for seed in expected['initializations']:
        started=time.perf_counter(); torch.manual_seed(seed); np.random.seed(seed)
        model=MatchedLSTMModel(config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
        best=float('inf'); stale=0; updates=0; samples=0; history=[]; best_epoch=None
        for epoch in range(1,13):
            epoch_start=time.perf_counter(); model.train(); training=[]
            for batch in batches(comparison,'fitting',seed=seed*100+epoch,shuffle=True,limit=2):
                values=tensor_batch(batch,device); loss=demand_loss(model,values)
                if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
                updates+=1; samples+=len(batch['node_features']); training.append(float(loss.detach()))
            model.eval(); total=0.; count=0
            with torch.no_grad():
                for batch in batches(comparison,'calibration'):
                    values=tensor_batch(batch,device); n=len(batch['node_features'])
                    total+=float(demand_loss(model,values))*n; count+=n
            calibration=total/count
            history.append({'epoch':epoch,'training_loss':statistics.fmean(training),'calibration_loss':calibration,'calibration_samples':count,'elapsed_seconds':time.perf_counter()-epoch_start})
            print(json.dumps({'event':'lstm-epoch-complete','seed':seed,**history[-1]}),flush=True)
            # Exact minimum determines checkpoint; tolerance controls patience only.
            improvement=best-calibration
            if calibration < best:
                best=calibration; best_epoch=epoch
                torch.save(model.state_dict(),target/f'seed-{seed}.pt')
            stale=0 if improvement>1e-5 else stale+1
            if stale>=4: break
        row={'seed':seed,'epoch':best_epoch,'calibration_loss':best,'checkpoint':f'seed-{seed}.pt','updates':updates,'batches':updates,'samples':samples,'epochs':len(history),'early_stopped':stale>=4,'history':history,'parameter_count':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.perf_counter()-started}
        write_json(target/f'seed-{seed}-history.json',row); choices.append(row)
    selected=min(choices,key=lambda r:(r['calibration_loss'],r['seed'],r['epoch']))
    write_json(target/'selection.json',{'selected':selected,'budget':expected,'holdout_assessed':False})
    return selected

def calibrate(comparison, output, device):
    from headlight.lstm_capacity_risk import capacity_coverage, fit_lstm_risk_calibration
    from headlight.bundle_lstm_matched import MatchedLSTMBundle
    if comparison['scorer_budget'] != {'steps':300,'learning_rate':.01,'batch_size':64,'seed':601}:
        raise ValueError('changed scorer budget')
    config=read_json(output/'lstm-training/model-config.json')
    selection=read_json(output/'lstm-training/selection.json')['selected']
    model=MatchedLSTMModel(config).to(device)
    checkpoint=output/'lstm-training'/selection['checkpoint']
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); model.eval()
    coverage=capacity_coverage(runs(comparison,'fitting'),read_json(dataset(comparison)/'feature-schema.json'))
    write_json(output/'capacity-coverage.json',coverage)
    if not coverage['complete']: raise ValueError('required service lacks eligible capacity samples')
    weights=comparison['cluster_capacity_weights']
    if weights != [1.,1.] and not comparison.get('capacity_weight_artifact'):
        raise ValueError('heterogeneous resources require capacity-weight artifact')
    config.update(service_capacity_rps=coverage['service_capacity_rps'],cluster_capacity_weights=weights)
    torch.manual_seed(601)
    fitted=fit_lstm_risk_calibration(model,lambda:runs(comparison,'calibration'),config)
    bundle=output/'lstm-bundle'; bundle.mkdir()
    shutil.copyfile(checkpoint,bundle/'forecaster.pt'); shutil.copyfile(dataset(comparison)/'feature-schema.json',bundle/'feature-schema.json')
    write_json(bundle/'model-config.json',read_json(output/'lstm-training/model-config.json'))
    torch.save(fitted['scorer'].state_dict(),bundle/'scorer.pt')
    write_json(bundle/'scorer-config.json',{'service_capacity_rps':coverage['service_capacity_rps'],'cluster_capacity_weights':weights,'checkpoint_hash':sha256(bundle/'forecaster.pt'),'schema_hash':sha256(bundle/'feature-schema.json')})
    from headlight.profile import load_service_profile
    metadata={'schema_version':'headlight.lstm-bundle/matched-v2','component_hashes':{name:sha256(bundle/name) for name in ['forecaster.pt','scorer.pt','model-config.json','feature-schema.json','scorer-config.json']},'input_hash':identity(comparison),'selected_seed':selection['seed'],'selected_epoch':selection['epoch'],'runtime_reload':False,'training':{'dataset_hash':load_manifest(dataset(comparison)/'manifest.json')['dataset_hash'],'feature_schema_hash':sha256(bundle/'feature-schema.json'),'profile_hash':load_service_profile(comparison['service_profile']).profile_hash,'fitting_seeds':list(range(10000,10128)),'calibration_seeds':list(range(10128,10160)),'holdout_seeds':list(range(10160,10192)),'selection_procedure':'minimum complete calibration pinball loss; tie by initialization then epoch; holdout once after scorer fitting','created_before_evaluation':True}}
    metadata['bundle_id']=identity(metadata); write_json(bundle/'metadata.json',metadata)
    write_json(output/'scorer-fit-report.json',{'history':fitted['history'],'optimizer_steps':300,'coverage':coverage,'membership':'calibration seeds 10128-10159','lstm_checkpoint_hash':sha256(checkpoint),'schema_hash':sha256(bundle/'feature-schema.json'),'budget':comparison['scorer_budget'],'loss_scales':{'latency_ms':1000.,'boundary_extra_weight':64.,'sla_bce_weight':.5,'success_bce_weight':.5}})
    return assess_holdout(comparison, output, MatchedLSTMBundle.load(bundle).build_model(device), device)

def assess_holdout(comparison, output, model, device):
    # Use the historical accumulator, preserving all normalization definitions.
    from experiments.train_headlight_v2_synthetic import MetricTotals
    if (output/'holdout-report.json').exists() or (output/'holdout-started.json').exists():
        raise ValueError('holdout already assessed or attempted; cannot reassess')
    write_json(output/'holdout-started.json',{'bundle_checkpoint':sha256(output/'lstm-bundle/forecaster.pt')})
    accumulator=MetricTotals(); timings=[]
    with torch.no_grad():
        for batch in batches(comparison,'holdout'):
            values=tensor_batch(batch,device); start=time.perf_counter()
            prediction=model(values['node_features'],values['edge_features'],values['candidate_replicas'],values['candidate_placement'],values['readiness'])
            if device.type=='cuda': torch.cuda.synchronize(device)
            timings.append((time.perf_counter()-start)*1000)
            accumulator.update(prediction,values,torch.tensor(0.))
    metrics=accumulator.result()
    gates={'demand':metrics['demand_q50_normalized_mae']<=.2,'coverage':.75<=metrics['demand_q90_coverage']<=.98,'latency':metrics['candidate_p95_normalized_mae']<=.35,'boundary':metrics['boundary_p95_mae_ms']<=300,'sla':metrics['sla_brier_score']<=.2}
    report={'metrics':metrics,'gates':gates,'qualified':all(gates.values()),'inference_batch_mean_ms':statistics.fmean(timings),'holdout_assessments':1}
    write_json(output/'holdout-report.json',report)
    if not report['qualified']: raise ValueError('LSTM holdout gates failed; evidence preserved; no retuning permitted')
    return report

def search(comparison, output, family, *, allow_configured_lstm=False):
    from experiments.lib.matched_closed_loop import run_development_episode
    from experiments.lib.static_plan_matched import build_matched_static_plan
    candidates=[f'static-{n:02d}' for n in comparison['static_totals']] if family=='static' else list(comparison['candidate_weights'])
    if family=='lstm' and not read_json(output/'holdout-report.json')['qualified'] and not allow_configured_lstm:
        raise ValueError('configured LSTM cannot enter policy selection')
    if family=='lstm' and allow_configured_lstm:
        report=read_json(output/'holdout-report.json')
        if report.get('holdout_assessments')!=1 or {k for k,v in report.get('gates',{}).items() if not v} != {'latency','boundary'}:
            raise ValueError('configured LSTM policy selection requires the recorded single assessment with only latency and boundary gates failed')
    comparison={**comparison,'lstm_bundle':str(output/'lstm-bundle')}
    records=[]; summaries=[]
    episode_dir=output/f'{family}-episode-records'; episode_dir.mkdir(exist_ok=True)
    if family=='static' and comparison.get('workers',1)>1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        with ProcessPoolExecutor(max_workers=comparison['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={}
            for candidate,case,seed in itertools.product(candidates,comparison['cases'],comparison['development_seeds']):
                path=episode_dir/f'{candidate}--{case}--{seed}.json'
                if not path.exists():
                    future=pool.submit(_timed_episode,f'{family}-matched',candidate,case,seed,comparison)
                    pending[future]=path
            for future in as_completed(pending):
                path=pending[future]
                try:
                    row=future.result()
                except Exception as exc:
                    failure=episode_dir/'failed-attempts'/(path.stem+f'-failed-{time.time_ns()}.json')
                    write_json(failure,{'technical_error':str(exc),'comparison_hash':identity(comparison),'not_a_completed_episode':True})
                    continue
                write_json(path,{'comparison_hash':identity(comparison),'episode':row})
            if any(not path.exists() for path in pending.values()):
                raise ValueError(f'technical episode failures preserved under {episode_dir / "failed-attempts"}')
    for candidate in candidates:
        rows=[]
        for case,seed in itertools.product(comparison['cases'],comparison['development_seeds']):
            path=episode_dir/f'{candidate}--{case}--{seed}.json'
            if path.exists():
                saved=read_json(path)
                if saved['comparison_hash']!=identity(comparison): raise ValueError('episode resume identity mismatch')
                row=saved['episode']
            else:
                try:
                    row=_timed_episode(f'{family}-matched',candidate,case,seed,comparison)
                except Exception as exc:
                    write_json(episode_dir/'failed-attempts'/(path.stem+f'-failed-{time.time_ns()}.json'),{'technical_error':str(exc),'candidate_id':candidate,'case':case,'seed':seed,'comparison_hash':identity(comparison),'not_a_completed_episode':True})
                    raise
                write_json(path,{'comparison_hash':identity(comparison),'episode':row})
            print(json.dumps({'event':'episode-complete','controller':family,'candidate_id':candidate,'case':case,'seed':seed,'elapsed_seconds':row['elapsed_seconds']}),flush=True)
            rows.append(row); records.append(row)
        summary={'candidate_id':candidate,'episodes':len(rows),**{k:statistics.fmean(r[k] for r in rows) for k in ('success_rate','request_sla_violation_rate','mean_pod_count','migration_proxy')}}
        summary.update({'total_replicas':int(candidate.split('-')[-1])} if family=='static' else {'weights':comparison['candidate_weights'][candidate]})
        summaries.append(summary)
    episodes_path=output/f'{family}-episodes.jsonl'
    episodes_text=''.join(json.dumps(row,allow_nan=False)+'\n' for row in records)
    if episodes_path.exists():
        if episodes_path.read_text()!=episodes_text: raise ValueError('completed episode summary differs from resumed records')
    else:
        with episodes_path.open('x') as stream: stream.write(episodes_text)
    selected=select_development_candidate(summaries)
    result={'selected':selected,'candidates':summaries,'episode_count':len(records)}
    if family=='lstm':
        from headlight.bundle_lstm_matched import MatchedLSTMBundle
        result['bundle_id']=MatchedLSTMBundle.load(output/'lstm-bundle').bundle_id
    selection_path=output/f'{family}-selection.json'
    if selection_path.exists():
        existing=read_json(selection_path)
        if existing.get('selected')!=result['selected'] or existing.get('candidates')!=result['candidates'] or existing.get('bundle_id')!=result.get('bundle_id'):
            raise ValueError('completed selection differs from resumed development records')
    else:
        write_json(selection_path,result)
    if family=='static': write_json(output/'static-plan.json',build_matched_static_plan(selected,comparison))
    return result

def _timed_episode(controller, candidate, case, seed, comparison):
    from experiments.lib.matched_closed_loop import run_development_episode
    torch.set_num_threads(1)
    started=time.perf_counter()
    row=run_development_episode(controller,candidate,case,seed,comparison)
    row['elapsed_seconds']=time.perf_counter()-started
    return row

def replay(comparison, output):
    from experiments.lib.matched_closed_loop import run_development_episode
    selection=read_json(Path(comparison['artifact_root'])/'results/synthetic-headlight-v2/policy-tuning-v2/selection.json')
    weights=selection['weights']
    local={**comparison,'candidate_weights':{'reference':weights}}
    records=[run_development_episode('headlight-v2','reference',case,seed,local) for case,seed in itertools.product(comparison['cases'],comparison['development_seeds'])]
    with (output/'headlight-reference-episodes.jsonl').open('x') as stream:
        for row in records: stream.write(json.dumps(row,allow_nan=False)+'\n')
    expected=selection['selected_development_metrics']; differences={}
    for key,value in expected.items():
        if key not in records[0] or not isinstance(value,(int,float)): continue
        actual=statistics.fmean(row[key] for row in records)
        differences[key]={'expected':value,'actual':actual,'within_tolerance':bool(np.isclose(actual,value,atol=1e-6,rtol=1e-5))}
    import subprocess,hashlib
    source={}
    for name in ['ai-module/ai-module/headlight/closed_loop_v2.py','ai-module/ai-module/headlight/models_v2.py','experiments/run_headlight_v2_closed_loop.py']:
        historical=subprocess.run(['git','show',f'2b544d3:{name}'],cwd=ROOT,capture_output=True)
        old_hash=hashlib.sha256(historical.stdout).hexdigest() if historical.returncode==0 else None
        source[name]={'historical_sha256':old_hash,'current_sha256':sha256(ROOT/name),'byte_identical':old_hash==sha256(ROOT/name)}
    write_json(output/'reference-reproduction-report.json',{'differences':differences,'source_comparison':source,'source_identity_claim':'historical 2b544d3 versus current implementation; loader now supports relocated artifacts; simulator objective and guard reused','exact_reproduction_claimed':all(row['byte_identical'] for row in source.values()) and all(row['within_tolerance'] for row in differences.values()),'absolute_tolerance':1e-6,'relative_tolerance':1e-5})
    return differences

def freeze(comparison, output):
    from experiments.lib.static_plan_matched import validate_matched_static_plan
    from headlight.bundle_lstm_matched import MatchedLSTMBundle
    files=['input-verification.json','checkpoint-equivalence.json','implementation.diff','relocation-map.json','preparation-protocol.json','historical-training-audit.json','static-selection.json','static-plan.json','static-episodes.jsonl','lstm-training/selection.json','scorer-fit-report.json','holdout-report.json','lstm-selection.json','lstm-episodes.jsonl','headlight-reference-episodes.jsonl','reference-reproduction-report.json','readiness-report.json','implementation-checks.json']
    holdout=read_json(output/'holdout-report.json')
    metrics=holdout['metrics']
    gates=[metrics['demand_q50_normalized_mae']<=.2,.75<=metrics['demand_q90_coverage']<=.98,metrics['candidate_p95_normalized_mae']<=.35,metrics['boundary_p95_mae_ms']<=300,metrics['sla_brier_score']<=.2]
    if not holdout['qualified'] or not all(gates) or holdout['holdout_assessments']!=1: raise ValueError('cannot freeze configured model')
    if not read_json(output/'implementation-checks.json')['passed']: raise ValueError('implementation checks did not pass')
    MatchedLSTMBundle.load(output/'lstm-bundle').build_model(torch.device(comparison.get('device','cpu')))
    plan=validate_matched_static_plan(output/'static-plan.json',Path(comparison['service_profile']))
    if plan['selection_hash']!=identity(read_json(output/'static-selection.json')['selected']): raise ValueError('static selection binding mismatch')
    traces={}; counts={}
    for family,name in [('static','static-episodes.jsonl'),('lstm','lstm-episodes.jsonl'),('reference','headlight-reference-episodes.jsonl')]:
        rows=[json.loads(line) for line in (output/name).read_text().splitlines()]
        if len(rows)!=(12 if family=='reference' else 96): raise ValueError('incomplete development records')
        unique=set()
        expected_candidates={f'static-{n:02d}' for n in comparison['static_totals']} if family=='static' else set(comparison['candidate_weights']) if family=='lstm' else {'reference'}
        for row in rows:
            key=(row['case'],row['seed']); cell=(row['candidate_id'],*key)
            if cell in unique or row['candidate_id'] not in expected_candidates or row['seed'] not in comparison['development_seeds'] or row['case'] not in comparison['cases']: raise ValueError('duplicate/unregistered episode')
            if any(not math.isfinite(float(row[k])) for k in ('success_rate','request_sla_violation_rate','mean_pod_count','migration_proxy')): raise ValueError('nonfinite episode metric')
            unique.add(cell)
            if key in traces and traces[key]!=row['exogenous_hash']: raise ValueError('trace parity failure')
            traces[key]=row['exogenous_hash']
        counts[family]=len(rows)
        if family!='reference':
            summary=read_json(output/f'{family}-selection.json')
            if family=='lstm' and summary.get('bundle_id')!=MatchedLSTMBundle.load(output/'lstm-bundle').bundle_id:
                raise ValueError('LSTM policy/bundle binding mismatch')
            if {r['candidate_id'] for r in summary['candidates']}!=expected_candidates or len(summary['candidates'])!=8: raise ValueError('incomplete candidate summary')
            for candidate in summary['candidates']:
                owned=[r for r in rows if r['candidate_id']==candidate['candidate_id']]
                if len(owned)!=12: raise ValueError('incomplete candidate episodes')
                for key in ('success_rate','request_sla_violation_rate','mean_pod_count','migration_proxy'):
                    if candidate[key]!=statistics.fmean(r[key] for r in owned): raise ValueError('candidate aggregate mismatch')
            if select_development_candidate(summary['candidates']) != summary['selected']: raise ValueError('selection result mismatch')
    write_json(output/'trace-parity-report.json',{'counts':counts,'hashes':{f'{k[0]}/{k[1]}':v for k,v in traces.items()},'passed':True})
    files+=['trace-parity-report.json']
    from experiments.run_matched_baseline_matrix import matrix_cells
    matrix={'execution':False,'cells':matrix_cells(output)}
    write_json(output/'future-evaluation-matrix.json',matrix)
    files += ['future-evaluation-matrix.json']
    files += [str(p.relative_to(output)) for folder in ['lstm-bundle','lstm-training','logs','static-episode-records','lstm-episode-records','implementation-source'] for p in (output/folder).rglob('*') if p.is_file()]
    evidence={'schema_version':'headlight.matched-freeze/v2','qualified_offline':True,'live_readiness':read_json(output/'readiness-report.json'),'artifacts':{name:sha256(output/name) for name in files},'comparison_hash':identity(comparison)}
    publish_freeze(output,evidence,matrix['cells'])
    return evidence

def publish_freeze(output, evidence, expected_cells):
    from experiments.run_matched_baseline_matrix import expand_matrix
    # Publishing the qualified marker is the last operation, after every check.
    if expand_matrix(output,freeze=evidence)!=expected_cells:
        raise ValueError('future dry-run matrix validation failed')
    write_json(output/'freeze.json',evidence)
