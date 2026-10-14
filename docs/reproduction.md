# Reproduction

## Verify the reported evaluation

From the repository root, run this Python standard-library command:

```bash
python experiments/validate_softwarex_evaluation.py --output results/controller-archive/validation.json
```

It verifies the exact 60-run matrix, each indexed artifact's SHA-256 and size,
selected summary status and identity, and metric values. Successful output has
`validation_status=validated`. Its twelve aggregates report the means and sample
standard deviations. The [archive](../results/controller-archive/README.md)
provides the complete run index.

## Configuration dry-run

```bash
make validate-config
make python-tests
make yaml-validate
make reproduction-dry-run
make docs-check
```

Dry-run records use `validation_status=dry-run`. They check configuration and
orchestration without starting Kubernetes workloads or producing measurements.

## Measured reproduction

Prepare the Kind/Karmada federation with Istio, Prometheus and Metrics Server.
Use the environment checklist in `experiments/config/softwarex_reproduction.yaml`,
the frozen HeadLight bundle and policy, Train Ticket advanced-ticket, the shared
gateway and recorded workload profiles. Run Headlight, HPA, KEDA and Threshold
for each workload and seed 101–105, retaining resolved configuration, images,
environment, request/telemetry traces, actions and measurement windows.

The configuration declares Ubuntu 24.04, 16 CPU cores and 48 GiB RAM as
prerequisites. These values are not an established minimum hardware requirement.
New measured runs require their own artifact validation and provenance.

The configuration dry-run is a staged demonstration, rather than a launcher for
the entire measured matrix. The release matrix is specified independently in
`experiments/config/softwarex_evaluation.json`. For prepared Linux testbeds,
the Headlight image helpers verify the frozen image inputs and prepare
and launch the seed-104/105 cells:

```bash
bash experiments/scripts/verify_and_build_headlight_v2_image.sh --verify-only
bash experiments/scripts/prepare_headlight_v2_seed104_105.sh results/new-v2-campaign
bash experiments/scripts/run_headlight_v2_seed104_105.sh results/new-v2-campaign
```

Run preparation only on the intended idle testbed. For other controller/seed
cells, use the selected index's recorded manifest, environment and image inputs,
and a fresh output directory. Preserve the emergency-disabled policy and
measurement timings recorded in the selected manifests.
