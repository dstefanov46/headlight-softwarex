# Headlight

Headlight provides dependency-aware predictive autoscaling and availability-aware
replica placement across Kubernetes clusters managed by Karmada.

The evaluation contains **60 valid measured runs**: Headlight, HPA, KEDA and Threshold across
gradual-ramp, flash-crowd and diurnal workloads, with seeds 101–105.

The [evaluation index](results/controller-archive/README.md) identifies the exact
selected runs. Verify their artifacts and reproduce the reported mean and sample
standard deviation:

```bash
python experiments/validate_softwarex_evaluation.py --output results/controller-archive/validation.json
```

The validator returns `validation_status=validated` after checking the complete
60-cell matrix, selected summary status, artifact checksums and metric values.

## Components

- `ai-module/`: telemetry, frozen Headlight inference, synthetic training and Threshold.
- `actions-handler/`: Go REST service that batches placement actions.
- `controlled-replica-propagation/`: CRD and controller that guard replica removal.
- `experiments/`: workloads, configuration, testbed scripts and artifact validation.
- `results/`: the selected evaluation, frozen model and supporting provenance.
- `docs/`: [architecture](docs/architecture.md), [reproduction](docs/reproduction.md)
  and [evaluation](docs/experiments.md).

## Checks

```bash
make validate-config
make python-tests
make yaml-validate
make reproduction-dry-run
make docs-check
```

A dry-run checks configuration and orchestration without producing measurements.
See [evaluation scope](docs/evaluation.md) for assumptions and limits.
The software is licensed under Apache-2.0. Citation metadata is in `CITATION.cff`.
