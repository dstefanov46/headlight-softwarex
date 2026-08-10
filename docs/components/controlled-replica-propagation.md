# Controlled Replica Propagation

![Controlled Replica Propagation context](../assets/controlled-replica-propagation.png)

CRP is the Kubernetes custom resource and controller that declares replica
counts per member cluster. `targetRef` identifies the workload, `placements`
sets desired counts, and `strategy.maxUnavailable` controls the availability
budget during availability-preserving replica relocation.

```yaml
apiVersion: policy.nephele-project.eu/v1alpha1
kind: ControlledReplicaPropagation
metadata:
  name: crp-example
spec:
  strategy: {maxUnavailable: 0}
  targetRef: {apiVersion: apps/v1, kind: Deployment, name: crp-example}
  placements:
    - {clusterName: member1, replicas: 5}
    - {clusterName: member2, replicas: 5}
```

For the SoftwareX example, the placement is changed to:

```yaml
spec:
  strategy: {maxUnavailable: 0}
  targetRef: {apiVersion: apps/v1, kind: Deployment, name: crp-example}
  placements:
    - {clusterName: member1, replicas: 2}
    - {clusterName: member2, replicas: 8}
```

The reconciler creates newly required replicas first and deletes old replicas
only after readiness satisfies the availability condition. This is
availability-preserving replica migration: make-before-break cross-cluster
movement, not movement of process memory or pod state. This makes CRP the
enforcement boundary for Actions Handler and AI Module decisions. Installers
are in `config/` and `dist/`; controller tests run with `make test`.

The illustrative 5/5 to 2/8 trace fixture can be validated without a Karmada
testbed:

```bash
python3 experiments/validate_crp_trace.py \
  --trace experiments/sample_data/softwarex/crp_5_5_to_2_8_trace.csv
```

The fixture demonstrates the CRP validation contract only. A real measured run
must validate the CRP trace captured from the Karmada testbed before it can be
used as availability evidence.

The dry-run and unit-test paths do not establish measured availability or
performance evidence.
