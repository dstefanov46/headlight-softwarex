# Headlight

Headlight is an open-source experimental orchestration stack for predictive autoscaling and availability-aware replica placement across Kubernetes multi-cluster federations managed by Karmada.

This repository contains the **Headlight v1.0.0 immutable software release prepared for the SoftwareX submission**:

> **Headlight: dependency-aware predictive orchestration for Kubernetes multi-cluster federations**

## Release

**Version:** v1.0.0  
**License:** Apache License 2.0  
**Release repository:** https://github.com/dstefanov46/headlight-softwarex

This repository is an archival SoftwareX release snapshot rather than the development repository. The v1.0.0 source tree was derived from development commit:

```text
f390b7a9c5bc3ce6b867b1276e6600e3c8b7e2b4
```

The development repository and its commit history are maintained separately and are not part of this archival release.

The `v1.0.0` Git tag identifies the exact software version corresponding to the SoftwareX submission.

## Overview

Headlight combines a service-dependency-aware predictive model with a declarative execution layer for Kubernetes multi-cluster federations.

The active forecasting model:

- uses seven service-level telemetry features together with seven missing-value indicators;
- operates over a traffic-derived service dependency graph;
- applies temporal scoring and graph message passing;
- forecasts request demand, CPU demand, p95 latency and SLA-violation risk;
- converts forecasts into bounded replica and placement actions.

Model training and tuning are performed offline. Live inference uses a hash-verified frozen model bundle.

## Repository structure

The principal implementation directories are:

```text
ai-module/
actions-handler/
controlled-replica-propagation/
experiments/
docs/
```

## Experiment support

The repository includes five reference controllers in addition to Headlight:

- Kubernetes HPA;
- KEDA;
- LSTM-based predictive autoscaling;
- static allocation;
- threshold-based control.

Four parameterised workload profiles are provided:

- gradual ramp;
- flash crowd;
- diurnal;
- bursty.

The SoftwareX paper-facing comparison uses gradual ramp, flash crowd and diurnal workloads with paired seeds `101`–`105`.

The experiment framework also provides capacity qualification, offline model calibration, a fixed 48-candidate Headlight model grid, separate tuning seeds, frozen-model enforcement, quality gates, structured run validation, provenance recording and paper-artifact generation.

## Reproduction

Detailed instructions are provided in:

- [`docs/reproduction.md`](docs/reproduction.md)
- [`docs/experiments.md`](docs/experiments.md)
- [`docs/architecture.md`](docs/architecture.md)

The principal validation commands are:

```bash
make validate-config
make python-tests
make yaml-validate
make go-tests
```

A deterministic reproduction-contract check that does not execute the measured experiment campaign is available through:

```bash
make reproduction-dry-run
```

The dry run verifies the reproduction interface only and **must not be interpreted as comparative performance evidence**.

Only measured result bundles accepted by the repository validation procedure with:

```text
validation_status=validated
```

are eligible for quantitative performance tables or plots.

See [`docs/reproduction.md`](docs/reproduction.md) for the exact commands, configuration and output paths.

## Controlled replica propagation

Headlight uses the `ControlledReplicaPropagation` custom resource to express desired replica distributions across Karmada member clusters.

With `maxUnavailable: 0`, cross-cluster replica relocation is performed make-before-break: destination replicas must become Ready before the corresponding source replicas are removed.

## Software requirements

A complete experiment environment requires Kubernetes, Karmada, Istio, Prometheus, Kubernetes Metrics Server, cAdvisor, Docker, Helm, `kubectl`, Python 3 with PyTorch, Go and Locust.

## Citation

This repository contains **Headlight v1.0.0**, the software release associated with the SoftwareX submission.

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

The persistent Zenodo DOI for this exact release should be used when citing the software. Once Zenodo has archived the GitHub release, the DOI is linked from the GitHub Release page.

## License

Headlight is released under the **Apache License 2.0**.

See [`LICENSE`](LICENSE) for details.

## Support

For questions concerning the SoftwareX release:

**Dimitar Stefanov**  
Sunesis d.o.o., Ljubljana, Slovenia  
dimitar.stefanov@sunesis.si
