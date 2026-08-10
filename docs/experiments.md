# Experiments and artifacts

Experiment files under `experiments/configs/` are repository-root-relative and
compose one controller profile, topology, workload, and run definition. The
schema and inheritance rules are documented in
[`experiments/configs/schema.md`](../experiments/configs/schema.md).

Four workload profiles exist: diurnal, flash-crowd, gradual-ramp, and bursty.
The measured comparison matrix uses the first three workload profiles with Headlight,
HPA, KEDA, LSTM, static, and threshold controllers. `run_paper_matrix.sh`
defaults to seeds 101, 102, 103, 104, and 105. Run files define the duration,
warmup, SLA thresholds, output directory, and seeds. The seed list, workload
shape, sampling interval, controller variant, and topology must be preserved in
the resolved `manifest.yaml`.

Every measured run should retain raw metrics, logs, events, CRP transitions,
summaries, figures, `environment.json`, and model/training metadata where
applicable. Validate the bundle before interpreting it:

```bash
python3 experiments/validate_artifacts.py --results-root results/softwarex
```

The reproduction `dry-run` validates configuration and output planning and
demonstrates sample artifact shape. Latency, energy, fairness, availability, and
comparative performance tables use measured bundles accepted by the artifact
validator. `results/ARTIFACTS_POLICY.md` records this artifact workflow.
