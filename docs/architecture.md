# Architecture and data flow

```text
Locust -> Train Ticket + Istio/cAdvisor -> Prometheus
       -> Headlight -> Actions Handler -> CRP -> Karmada -> member clusters
```

The AI Module aligns telemetry, checks the frozen model bundle and feature
contract, and proposes bounded replica and placement changes. Its Headlight
model uses causal temporal convolutions and directional graph propagation.
Nineteen node features describe demand, CPU, latency, errors, desired/ready/pending
replicas, pending age, action recency and missing/stale indicators. Seven edge
features describe directional call rate/ratio, latency, errors, cluster crossing
and missing/stale indicators.

The model predicts demand q50/q90 and candidate-dependent endpoint latency,
violation risk and success probability. The deterministic policy combines
resource cost, violation risk, churn and failure probability. The reported policy
limits total replica changes to one and submits at most one service change per
interval. Reactive emergency overrides are disabled in the measured evaluation.

The Actions Handler batches placement requests and updates CRP resources. CRP
creates destination replicas before allowing source removals, using federated
Deployment availability and `maxUnavailable`. The guard depends on fresh status,
destination capacity and a single controller owner.

Training and tuning run offline. Hash-verified frozen weights and policy inputs
are loaded before live inference. HPA and KEDA use their native controllers;
Threshold uses the Actions Handler/CRP path.

The [60-run evaluation](experiments.md) contains measured results. Configuration
dry-run records exercise planning and contracts without starting live workloads.
