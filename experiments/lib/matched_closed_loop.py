"""Adapters that preserve historical dynamics for dynamic controllers."""
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Any
import numpy as np
import torch
from experiments.lib.matched_baselines import read_json
from experiments.lib.static_plan_matched import fixed_placement
from headlight.profile import load_service_profile
from headlight.synthetic_v2 import NODE_ORDER, DATABASE_NODES
from headlight.closed_loop_v2 import ClosedLoopEnvironment, ClosedLoopConfig, ControllerAction, ControllerDecision, AppliedAction, run_episode
from experiments.run_headlight_v2_closed_loop import NeuralClosedLoopController, NeuralPolicyWeights

class FixedStaticController:
    controller_id = 'static-matched'
    def __init__(self, placement):
        self.placement = np.asarray(placement, dtype=np.int64)
    def reset(self):
        pass
    def decide(self, observation, history, candidates, preview):
        return ControllerDecision(ControllerAction(self.placement.sum(-1), self.placement.copy(), 'fixed static plan'))

class StaticClosedLoopEnvironment(ClosedLoopEnvironment):
    """Fixed desired placement with DB-first, stable-node capacity scheduling.

    Initial pods start pending before shared warmup. Startup timers accrue while
    pending, but admission always requires actual capacity. Dynamic default
    dynamics are untouched.
    """
    def __init__(self, *, placement, **kwargs):
        self.fixed = np.asarray(placement, dtype=np.int64)
        if self.fixed.shape != (len(NODE_ORDER), 2) or np.any(self.fixed < 1):
            raise ValueError('invalid static placement')
        super().__init__(**kwargs)
    def reset(self):
        super().reset()
        self.desired_placement = self.fixed.copy()
        self.ready_placement = np.zeros_like(self.fixed)
        return self.observe()
    def _bounded_action(self, requested):
        requested.validate()
        if not np.array_equal(requested.placement, self.fixed):
            raise ValueError('static desired placement cannot change')
        action = ControllerAction(self.fixed.sum(-1), self.fixed.copy(), 'fixed static plan')
        return AppliedAction(requested, action, 0, 0, 0, 0, 0)
    def _advance_state(self, applied):
        self.steps_since_action += 1
        pending = self.fixed - self.ready_placement
        self.pending_age[pending > 0] += self.config.sampling_interval_seconds
        order = sorted(range(len(NODE_ORDER)), key=lambda i: (NODE_ORDER[i] not in DATABASE_NODES, i))
        capacities = self._cluster_capacities()
        for c, capacity in enumerate(capacities):
            databases=[i for i in order if NODE_ORDER[i] in DATABASE_NODES]
            applications=[i for i in order if NODE_ORDER[i] not in DATABASE_NODES]
            # Reserve DB desired capacity even while those pods start up.
            remaining=int(capacity)-sum(int(self.fixed[i,c]) for i in databases)
            if remaining<0: raise ValueError('capacity cannot reserve fixed databases')
            for i in databases:
                if self.pending_age[i,c]>=self.startup_steps[i,c]*self.config.sampling_interval_seconds:
                    self.ready_placement[i,c]=self.fixed[i,c]
            for i in applications:
                self.ready_placement[i, c] = min(int(self.ready_placement[i, c]), remaining)
                remaining -= int(self.ready_placement[i, c])
            for i in applications:
                if self.pending_age[i, c] >= self.startup_steps[i, c] * self.config.sampling_interval_seconds:
                    admitted = min(int(self.fixed[i, c] - self.ready_placement[i, c]), remaining)
                    self.ready_placement[i, c] += admitted
                    remaining -= admitted
        self.pending_age[self.ready_placement == self.fixed] = 0

def build_lstm_closed_loop_controller(bundle, weights, device):
    return NeuralClosedLoopController(controller_id='lstm-matched', model=bundle.build_model(device), device=device, graph_enabled=False, sampling_interval_seconds=bundle.model_config['sampling_interval_seconds'], policy_weights=weights, shadow_model=None)

def run_development_episode(controller_id: str, candidate_id: str, case: str, seed: int, comparison: Mapping[str, Any]) -> dict:
    if seed not in comparison['development_seeds'] or case not in comparison['cases']:
        raise ValueError('unregistered development episode')
    config = ClosedLoopConfig(**comparison['simulator_config'])
    if controller_id == 'static-matched':
        profile = load_service_profile(Path(comparison['service_profile']))
        placements = fixed_placement(int(candidate_id.split('-')[-1]), profile)
        vector = np.array([[placements[name][c] for c in ('cluster2', 'cluster3')] for name in NODE_ORDER])
        environment = StaticClosedLoopEnvironment(case=case, seed=seed, config=config, placement=vector)
        controller = FixedStaticController(vector)
    else:
        device = torch.device(comparison.get('device', 'cpu'))
        environment = ClosedLoopEnvironment(case=case, seed=seed, config=config)
        weights = NeuralPolicyWeights(**comparison['candidate_weights'][candidate_id])
        if controller_id == 'lstm-matched':
            from headlight.bundle_lstm_matched import MatchedLSTMBundle
            controller = build_lstm_closed_loop_controller(MatchedLSTMBundle.load(Path(comparison['lstm_bundle'])), weights, device)
        elif controller_id == 'headlight-v2':
            from experiments.run_headlight_v2_closed_loop import _build_neural_controllers
            root = Path(comparison['artifact_root'])
            controller = _build_neural_controllers(root / 'results/synthetic-headlight-v2/training-v2-policy-accepted/training-report.json', device, weights, artifact_root=root, relocation_map=comparison['relocation_map'])['headlight-v2']()
        else:
            raise ValueError('unknown controller')
    row = run_episode(environment, controller)
    row.update(candidate_id=candidate_id, exogenous_hash=environment.trace_hash, simulator_config=asdict(config), scheduling='fixed-db-first-stable-node-v1' if controller_id == 'static-matched' else 'historical-dynamic')
    if controller_id == 'static-matched':
        row['initial_provisioning'] = vector.tolist()
    return row
