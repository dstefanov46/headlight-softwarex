# Headlight model

The Headlight model uses nineteen node features,
seven directional edge features, eighteen Train Ticket services and two workload
clusters. Its lookback is 24 observations sampled every 15 s, and demand horizons
are 30, 60 and 180 s.

Node windows have shape `[B,24,18,19]`. Edge windows have shape `[B,24,18,18,7]`.
The causal temporal encoder uses hidden width 48 and dilations 1, 2 and 4. It
adds service/type embeddings, lag position encoding and causal request
differences/slopes. Directional graph blocks propagate caller demand to callees
and latency/error backpressure to callers. A bounded residual of at most 0.1
refines observed and local two-hop adjacency edges.

The demand head emits q50 and a positive increment to q90. The candidate head
combines the encoded state with proposed replicas, placement and readiness to
predict endpoint p95 latency, violation risk and success probability.

Training uses quantile pinball loss, candidate performance losses, candidate
ranking and graph residual regularization. The deterministic synthetic dataset
has separate fitting, calibration and holdout runs. Initialization seed 601 was
selected by calibration loss among accepted full-model candidates. Evaluation
seeds 101–105 are excluded from training and tuning.

The immutable bundle records checkpoint, model configuration, feature schema,
adjacency, calibration and hashes. The live runtime checks the service/cluster
ordering and calibration contract before inference.

Model code is in `ai-module/ai-module/headlight/models_v2.py`, synthetic data
generation in `synthetic_v2.py`, and losses/metrics in `training_v2.py`.
The [synthetic benchmark](headlight-v2-synthetic-recovery.md) describes the
training checks. The [measured evaluation](../experiments.md) is
separate from model wiring and dry-run checks.
