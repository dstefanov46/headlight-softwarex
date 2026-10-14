# Experiment Config Schema

All configs use `apiVersion: headlight.experiments/v1alpha1`. Paths are repository-root relative.

## Config Kinds

### `ControllerProfile`

Defines the autoscaling or orchestration method being evaluated.

Required fields:

- `metadata.name`
- `spec.controller`
- `spec.implementation_state`
- `spec.setup_script`
- `spec.metrics_collector`
- `spec.actuation_path`

Optional fields include deployment manifests, training or inference entrypoints, and method-specific notes.

Experiment profiles may also define:

- `spec.sampling_interval_seconds`
- `spec.controller_variant`
- `spec.training_trace`
- `spec.image_tag_env`
- `spec.policy`
- `spec.model_config`

Validation rule:

- `implementation_state: ready` means all referenced implementation paths must already exist.
- `implementation_state: planned` means future implementation paths may be reserved before the code lands.

### `TopologyProfile`

Defines where services are allowed to run and how placement decisions are applied.

Required fields:

- `metadata.name`
- `spec.mode`
- `spec.namespace`
- `spec.active_clusters`
- `spec.clusters_file`
- `spec.services_file`
- `spec.additional_services_file`
- `spec.placement_source`

### `WorkloadProfile`

Defines the load shape and its default timing knobs.

Required fields:

- `metadata.name`
- `spec.pattern`
- `spec.duration_seconds`
- `spec.warmup_seconds`
- `spec.locust.spawn_rate`

Each workload profile also includes shape-specific parameters such as `min_users`, `max_users`, `spike_start_seconds`, or `ramp_seconds`.

The paper workloads use deterministic shapes:

- `gradual-ramp`: ramp up, hold, ramp down.
- `flash-crowd`: two short spikes.
- `diurnal`: sinusoidal periodic load.

### `ExperimentRun`

Defines one paper-facing experiment by composing the three profiles above.

Required fields:

- `metadata.name`
- `spec.controller_profile`
- `spec.topology_profile`
- `spec.workload_profile`
- `spec.seeds`
- `spec.duration_seconds`
- `spec.warmup_seconds`
- `spec.sla`
- `spec.output_dir`

Additional paper-run fields include:

- `spec.sampling_interval_seconds`
- `spec.reset_policy`
- `spec.capture`

## Canonical `ExperimentRun` Skeleton

```yaml
apiVersion: headlight.experiments/v1alpha1
kind: ExperimentRun
metadata:
  name: headlight__diurnal__multi-cluster-headlight
spec:
  controller_profile: experiments/configs/baselines/headlight.yaml
  topology_profile: experiments/configs/topologies/multi-cluster-headlight.yaml
  workload_profile: experiments/configs/workloads/diurnal.yaml
  seeds: [101, 102, 103, 104, 105]
  duration_seconds: 3600
  warmup_seconds: 300
  sla:
    p95_latency_ms: 1000
    minimum_success_rate: 0.99
  output_dir: results/socc26/headlight-diurnal-multi
```

### `ModelVariant`

Headlight ablation configs live under `experiments/configs/models/` and define explicit switches for:

- `full`
- `static_adjacency`
- `no_attention`
- `no_graph`

## Inheritance Rules

- Profile files may define defaults.
- The run file wins whenever the same field appears in multiple places.
- A future runner must write the fully resolved configuration to `manifest.yaml` before execution starts.
