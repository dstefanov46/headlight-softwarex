# Experiments

The evaluation includes Headlight, HPA, KEDA and Threshold across
gradual-ramp, flash-crowd and diurnal workloads and seeds 101–105.

See [evaluation documentation](../docs/experiments.md) and the
[60-run archive](../results/controller-archive/README.md).

```bash
python experiments/validate_softwarex_evaluation.py --output results/controller-archive/validation.json
```

This verifies the measured artifacts without Kubernetes. The configuration
dry-run is described in [reproduction](../docs/reproduction.md).
Synthetic generation and training entrypoints are `generate_headlight_v2_synthetic.py`
and `train_headlight_v2_synthetic.py`. Testbed and deployment helpers are in `scripts/`.
