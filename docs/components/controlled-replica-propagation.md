# Controlled Replica Propagation

![Controlled Replica Propagation context](../assets/controlled-replica-propagation.png)

CRP is the Kubernetes custom resource and controller that declares replica
counts per member cluster. `targetRef` identifies the workload, `placements`
sets desired counts, and `strategy.maxUnavailable` controls the availability
budget during migration.

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

The reconciler creates newly required replicas first and deletes old replicas
only after readiness satisfies the availability condition. This makes CRP the
enforcement boundary for Actions Handler and AI Module decisions. Installers
are in `config/` and `dist/`; controller tests run with `make test`.

The dry-run and unit-test paths do not establish measured availability or
performance evidence.
