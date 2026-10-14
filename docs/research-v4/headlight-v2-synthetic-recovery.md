# Headlight synthetic recovery benchmark
Evidence note: Dry-run validation is not measured evidence; any measured results here apply only to the run and scope explicitly described.

This benchmark is engineering evidence for recovering and testing Headlight.
It is not research-v3 or research-v4 evaluation evidence, and its outputs must
not be copied into paper result tables.

## Dataset

The generator creates deterministic, run-owned fitting, calibration, and
holdout partitions. The default large dataset contains 192 runs and 62,400
causal windows: 128 fitting runs, 32 calibration runs, and 32 holdout runs.
Each run contains 18 TrainTicket services, a directed call graph, three workload
families, six fault/capacity scenarios, three prediction horizons, and eight
counterfactual replica/placement/readiness candidates per sample.

The paper aggregate is used only to bound broad workload priors. It is never
used as a label. Generated files are hash-addressed, split by run and seed, and
ignored by Git under `results/synthetic-headlight-v2/`.

Generate and validate the default dataset:

```bash
PYTHONPATH=ai-module/ai-module python experiments/generate_headlight_v2_synthetic.py
```

The validated local dataset has hash
`576cde51cb4f1466d748761eb0f2159b2de77b029de22f64d3e0b76fb504c7ec`,
548,786 SLA-boundary labels, and 2,888,931 positive SLA labels.

## Training and gates

The trainer uses the production Headlight model, normalized multitask loss,
endpoint-specific candidate attention, run-level partitions, early stopping,
candidate-ranking loss, a learned endpoint-success head, and a graph-free
ablation. Calibration selects checkpoints; holdout is evaluated only after
selection. Near-SLA candidate labels receive additional latency weight. The
default fitting loop samples two shuffled batches per fitting run per epoch but
always evaluates every calibration and holdout window.

```bash
python experiments/train_headlight_v2_synthetic.py --require-acceptance
```

For a faster deterministic engineering check, use one initialization seed:

```bash
python experiments/train_headlight_v2_synthetic.py \
  --initialization-seeds 601 \
  --output-dir results/synthetic-headlight-v2/training-seed601 \
  --require-acceptance
```

The acceptance gates require bounded demand and latency error, calibrated q90
coverage and SLA probability, accurate near-SLA latency, a measurable graph
gain over the graph-free control, policy safety no worse than the threshold
heuristic, and fewer than 10% whole-system ceiling events.

## Validated result

Initialization seed 601 passed all eight gates on 10,400
holdout windows:

| Metric | Headlight | Graph-free |
| --- | ---: | ---: |
| Demand q50 normalized MAE | 0.0908 | 0.1031 |
| Candidate p95 relative MAE | 0.2992 | 0.3136 |
| SLA-boundary p95 MAE | 283.4 ms | 281.1 ms |
| SLA Brier score | 0.0803 | 0.0822 |
| Reliability Brier score | 0.0145 | 0.0150 |
| Candidate top-1 accuracy | 0.6652 | 0.6412 |
| q90 empirical coverage | 0.8722 | 0.8585 |
| Policy-selected SLA rate | 0.7559 | 0.7561 |
| Policy ceiling-run rate | 0.0013 | 0.0033 |

The threshold heuristic's SLA rate was 0.8030 and it used 4.34 replicas per
service on average. Headlight used 3.93 while remaining safer on the
synthetic counterfactual labels.
This is a single-initialization engineering result; run the default seeds
601–603 before making a robustness claim, and do not treat synthetic success as
a substitute for closed-loop Kubernetes or frozen research-v4 evaluation.

The deterministic controller-level follow-up, including the distinction
between open-loop and closed-loop trajectories, is documented in
[`headlight-v2-closed-loop-synthetic.md`](headlight-v2-closed-loop-synthetic.md).
