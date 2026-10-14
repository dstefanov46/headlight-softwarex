# AI Module

The AI Module collects Prometheus, Istio and cAdvisor observations, assembles
causal service and edge windows, loads a frozen Headlight bundle and proposes
bounded replica and placement actions.

The evaluated input dimensions are `B × 24 × 18 × 19` for node state and
`B × 24 × 18 × 18 × 7` for directional edges. Samples are spaced by 15 s and
forecast horizons are 30, 60 and 180 s. The encoder has width 48 and temporal
dilations 1, 2 and 4, with an adjacency residual bounded by 0.1.

Demand outputs are q50 and q90. Candidate-conditioned endpoint outputs describe
p95 latency, SLA violation risk and success probability. Node features preserve
missing and stale state, and distinguish desired, ready and pending replicas.
The runtime holds decisions when predictive inputs or candidate outputs are
incomplete. Its default shadow mode requires explicit actuation enablement.

The bundle contains checkpoint, model configuration, feature schema, adjacency,
calibration and hash metadata. The service profile and policy are checked before
inference. See [model](../research-v4/headlight-v2-model.md) and
[policy](../research-v4/headlight-v2-policy.md).

```bash
python experiments/generate_headlight_v2_synthetic.py --help
python experiments/train_headlight_v2_synthetic.py --help
python ai-module/ai-module/live_inference_v2.py --help
```

The [evaluation](../experiments.md) reports 60 measured runs. Deployment manifests
and setup helpers live under `ai-module/`. A dry-run checks contracts and wiring.
