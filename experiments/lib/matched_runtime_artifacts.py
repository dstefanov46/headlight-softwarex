"""Runner artifact capture for new matched controller identities."""
import json
import os
import hashlib
from pathlib import Path
from dataclasses import asdict
from experiments.lib.matched_baselines import ROOT, read_json, sha256, identity
from experiments.run_matched_baseline_matrix import expand_matrix

def capture_matched_artifacts(controller, raw_dir):
    if controller not in {'static-matched','lstm-matched'}: raise ValueError('not a matched controller')
    root=Path(os.environ['HEADLIGHT_MATCHED_FREEZE_DIR']).resolve()
    profile=Path(os.environ.get('HEADLIGHT_SERVICE_PROFILE_FILE',str(ROOT/'ai-module/configs/service_profiles/advanced-ticket.json')))
    if controller=='static-matched':
        from experiments.lib.static_plan_matched import validate_matched_static_plan
        if (root/'static-amendment.json').is_file():
            from experiments.lib.matched_static_amendment import validate_static_amendment
            amendment=validate_static_amendment(root)
            if sha256(profile)!=amendment['service_profile_sha256']:
                raise ValueError('live profile file differs from static amendment')
            (raw_dir/'matched_static_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
            plan=validate_matched_static_plan(root/'static-plan.json',profile)
            (raw_dir/'static_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
            return
    if controller=='lstm-matched' and (root/'lstm-amendment.json').is_file():
        from experiments.lib.matched_lstm_amendment import validate_lstm_amendment
        amendment=validate_lstm_amendment(root)
        if sha256(profile)!=amendment['service_profile_sha256']:
            raise ValueError('live profile file differs from LSTM amendment')
        from headlight.bundle_lstm_matched import MatchedLSTMBundle
        from headlight.policy_v2 import PolicyV2Config
        bundle=MatchedLSTMBundle.load(root/'lstm-bundle')
        policy_path=Path(os.environ.get('HEADLIGHT_MATCHED_LIVE_POLICY',amendment['live_policy_path']))
        fixed=read_json(policy_path)
        if sha256(policy_path)!=amendment['live_policy_sha256']:
            raise ValueError('live policy fixed settings differ from LSTM amendment')
        if set(fixed)!=set(PolicyV2Config.__dataclass_fields__): raise ValueError('live policy settings incomplete')
        selected=read_json(root/'lstm-selection.json')
        if selected['bundle_id']!=bundle.bundle_id: raise ValueError('policy/bundle binding mismatch')
        weights=selected['selected']['weights']
        fixed.update(resource_weight=weights['resource'],sla_risk_weight=weights['sla_risk'],churn_weight=weights['migration'],reliability_weight=weights['reliability'])
        from headlight.profile import load_service_profile
        config=PolicyV2Config(**fixed);config.validate(load_service_profile(profile),controller_id='lstm-matched')
        policy={'effective_live_settings':asdict(config),'policy_configuration_hash':identity(fixed),'provenance':{'selection_hash':sha256(root/'lstm-selection.json'),'development_seeds':amendment['development_seeds'],'offline_to_live_mapping':{'resource':'resource_weight','sla_risk':'sla_risk_weight','migration':'churn_weight','reliability':'reliability_weight'}}}
        digest=hashlib.sha256()
        for path in sorted(p for p in bundle.root.rglob('*') if p.is_file()):
            digest.update(str(path.relative_to(bundle.root)).encode()); digest.update(path.read_bytes())
        provenance={'controller':controller,'bundle_id':bundle.bundle_id,'bundle_path':str(bundle.root),'bundle_sha256':digest.hexdigest(),'bundle_metadata':bundle.metadata,'no_online_retraining':True,'amendment_id':amendment['amendment_id'],'original_holdout_gates':amendment['original_holdout_gates'],'waived_gates':amendment['waived_gates']}
        for filename,value in [('model_bundle_metadata.json',bundle.metadata),('model_bundle_provenance.json',provenance),('policy_config.json',policy),('matched_lstm_amendment.json',amendment)]:
            (raw_dir/filename).write_text(json.dumps(value,indent=2)+'\n')
        return
    expand_matrix(root)  # Strict validation only; no execution or trace creation.
    freeze=read_json(root/'freeze.json')
    if sha256(profile)!=read_json(root/'input-verification.json')['profile_file_hash']:
        raise ValueError('live profile file differs from frozen comparison')
    (raw_dir/'matched_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    if controller=='static-matched':
        from experiments.lib.static_plan_matched import validate_matched_static_plan
        plan=validate_matched_static_plan(root/'static-plan.json',profile)
        (raw_dir/'static_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        return
    from headlight.bundle_lstm_matched import MatchedLSTMBundle
    from headlight.policy_v2 import PolicyV2Config
    bundle=MatchedLSTMBundle.load(root/'lstm-bundle')
    policy_path=Path(os.environ.get('HEADLIGHT_MATCHED_LIVE_POLICY',str(ROOT/'experiments/configs/models/lstm-matched-live-policy.json')))
    fixed=read_json(policy_path)
    if sha256(policy_path)!=read_json(root/'input-verification.json')['live_policy_hash']:
        raise ValueError('live policy fixed settings differ from frozen comparison')
    if set(fixed)!=set(PolicyV2Config.__dataclass_fields__): raise ValueError('live policy settings incomplete')
    selected=read_json(root/'lstm-selection.json')
    if selected['bundle_id']!=bundle.bundle_id: raise ValueError('policy/bundle binding mismatch')
    weights=selected['selected']['weights']
    fixed.update(resource_weight=weights['resource'],sla_risk_weight=weights['sla_risk'],churn_weight=weights['migration'],reliability_weight=weights['reliability'])
    from headlight.profile import load_service_profile
    config=PolicyV2Config(**fixed);config.validate(load_service_profile(profile),controller_id='lstm-matched')
    policy={'effective_live_settings':asdict(config),'policy_configuration_hash':identity(fixed),'provenance':{'selection_hash':sha256(root/'lstm-selection.json'),'development_seeds':[711,712,713],'offline_to_live_mapping':{'resource':'resource_weight','sla_risk':'sla_risk_weight','migration':'churn_weight','reliability':'reliability_weight'}}}
    digest=hashlib.sha256()
    for path in sorted(p for p in bundle.root.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(bundle.root)).encode()); digest.update(path.read_bytes())
    provenance={'controller':controller,'bundle_id':bundle.bundle_id,'bundle_path':str(bundle.root),'bundle_sha256':digest.hexdigest(),'bundle_metadata':bundle.metadata,'no_online_retraining':True,'freeze_sha256':sha256(root/'freeze.json')}
    for filename,value in [('model_bundle_metadata.json',bundle.metadata),('model_bundle_provenance.json',provenance),('policy_config.json',policy)]:
        (raw_dir/filename).write_text(json.dumps(value,indent=2)+'\n')
