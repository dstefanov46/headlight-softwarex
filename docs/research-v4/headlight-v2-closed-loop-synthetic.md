# Headlight closed-loop synthetic evaluation
Evidence note: Dry-run validation is not measured evidence; any measured results here apply only to the run and scope explicitly described.

This evaluation is a deterministic, offline engineering test of controller
behavior. It is not evidence from Kubernetes, Karmada, native HPA, or native
KEDA, and it must not be described as a production or federated edge-cloud
evaluation.

## Why closed loop matters

The training and holdout benchmark is **open loop**: every model sees fixed,
pre-generated histories and candidate labels. A prediction cannot change the
next sample. That setup measures forecast and candidate-ranking quality, but it
cannot show whether a sequence of decisions remains stable.

This benchmark is **closed loop**: at every 15-second step, a controller reads
the current observation, chooses replica counts and placements, and passes the
request through the shared bounded executor. The applied action changes desired
and ready replicas, startup queues, capacity pressure, latency, and therefore
the observation at the next step. Controllers share exogenous demand, noise,
missing-data events, capacity, initial conditions, and action limits for a
given case and seed; their endogenous trajectories can diverge because their
earlier actions differ.

## Matrix and controls

The frozen matrix contains 120 one-hour episodes: four cases, five paired seeds
(701--705), and six controllers. Metrics exclude the 24-step lookback warmup,
leaving 216 measured steps per episode.

The cases are gradual ramp, flash crowd, diurnal demand, and a constrained
capacity/delayed-readiness failure. The primary comparison contains:

- Headlight using the selected graph checkpoint;
- the selected graph-free ablation;
- an HPA-compatible CPU formula emulation (70% target, 10% tolerance,
  300-second downscale stabilization);
- a KEDA-compatible request-rate formula emulation (0.80 requests/second per
  replica, 0.10 activation threshold, 300-second cooldown); and
- the existing reactive CPU/latency/error threshold controller.

The look-ahead oracle is reported separately as a simulator-assisted sanity
bound. It is not deployable and is excluded from paired baseline comparisons.
The HPA and KEDA entries are explicit formula emulations running through the
same executor as HeadLight; they are not native controllers.

All controllers use 2--14 replicas per service, at most +2/-1 replicas per
decision, a four-step cooldown for replica or placement changes, the same
cluster-capacity checks, and the same startup/readiness model. Runtime
candidate actions are generated only from information available at the current
step. The oracle alone may preview future simulator state.

The v2 measurement contract materializes deterministic synthetic request
outcomes from each endpoint state. Requests are split between login, search,
and booking with fixed shares of 0.20/0.50/0.30. Successful request latencies
use a deterministic log-normal quantile grid calibrated from the synthetic
endpoint p50 and p95. Status 200 denotes success, status 500 an emulated
exception, and status 504 an emulated timeout. These values make controller
comparisons auditable; they are not observations of real HTTP traffic.

Run the matrix with the selected synthetic checkpoints:

```bash
python experiments/run_headlight_v2_closed_loop.py \
  --checkpoint-report results/synthetic-headlight-v2/training-v2-policy-accepted/training-report.json \
  --policy-config experiments/configs/models/headlight-v2-closed-loop-policy.json \
  --seeds 701,702,703,704,705 \
  --steps 240 \
  --output-dir results/synthetic-headlight-v2/closed-loop-v2-final
```

## Validated result

All 120 episodes completed, no episode failed, and the stored trace hashes
confirm that each controller received the same exogenous trace for each
case/seed pair.

| Controller | Success rate, mean +/- SD | Request SLA rate, mean +/- SD | Request p95, ms, mean +/- SD | Mean pods, mean +/- SD | Action churn, mean +/- SD | Migration proxy, mean +/- SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Headlight | 0.9819 +/- 0.0322 | 0.1651 +/- 0.1270 | 1891.8 +/- 1254.2 | 99.2 +/- 19.9 | 59.6 +/- 27.7 | 12.1 +/- 8.8 |
| Graph-free | 0.9818 +/- 0.0308 | 0.1684 +/- 0.1328 | 1889.6 +/- 1235.3 | 96.4 +/- 20.4 | 69.9 +/- 31.3 | 14.1 +/- 6.5 |
| HPA CPU emulation | 0.9854 +/- 0.0158 | 0.3055 +/- 0.1268 | 2901.6 +/- 1299.2 | 78.0 +/- 14.3 | 100.8 +/- 35.1 | 13.0 +/- 11.3 |
| KEDA request-rate emulation | 0.9566 +/- 0.0252 | 0.6148 +/- 0.1735 | 4522.6 +/- 1384.9 | 65.2 +/- 16.2 | 117.3 +/- 19.0 | 4.3 +/- 8.2 |
| Reactive threshold | 0.9880 +/- 0.0137 | 0.2176 +/- 0.1238 | 2119.4 +/- 996.9 | 79.9 +/- 16.0 | 67.5 +/- 25.3 | 5.4 +/- 10.1 |
| Oracle sanity bound | 0.9894 +/- 0.0134 | 0.1177 +/- 0.1142 | 1518.4 +/- 971.9 | 98.0 +/- 26.4 | 56.0 +/- 18.8 | 20.2 +/- 20.7 |

HeadLight's mean forecast q50 MAE was 0.7247 requests/second and mean end-to-end
inference/decision time was 9.92 ms. This timing includes the diagnostic shadow
and edge-ablation passes and is not a production-path latency estimate. For the
20 paired case/seed observations,
comparator-minus-HeadLight request-level SLA-violation differences were:

| Comparator | Mean difference | Approximate 95% CI |
| --- | ---: | ---: |
| Graph-free | +0.0032 | [-0.0098, +0.0163] |
| HPA CPU emulation | +0.1404 | [+0.1115, +0.1692] |
| KEDA request-rate emulation | +0.4497 | [+0.3554, +0.5441] |
| Reactive threshold | +0.0524 | [+0.0337, +0.0712] |

The credible conclusion is narrow: in this emulator, HeadLight lowers
request-level SLA violations relative to the three reactive baselines. It does
so with more pod-seconds than the reactive baselines, while its success rate is
slightly lower than HPA and Threshold. Stabilized placement generation reduced
HeadLight's migration proxy from 100.7 in the earlier engineering run to 12.1,
near HPA's 13.0 but above Threshold's 5.4. The graph-aware and graph-free
closed-loop results are not clearly separated at five paired seeds because the
confidence interval crosses zero. This result therefore establishes neither a
closed-loop graph benefit nor general controller superiority.

Decision attribution shows that the reactive guard modified 47.5% of
HeadLight decisions. Full and graph-free models selected different candidates
on 44.0% of identical observations, and zeroing valid edge features changed
the full model's SLA-risk output by a mean absolute 0.0114. Together with the
open-loop graph-vs-graph-free demand gate, this is evidence that the current
graph path is active; it does not justify a graph-architecture rewrite.

## Artifacts and limitations

The run writes machine-readable per-episode data, aggregate mean, median,
standard deviation and descriptive 95% intervals, overall and per-workload
paired differences, and a concise report under
`results/synthetic-headlight-v2/closed-loop-v2-final/`. Recorded measurements include:

- total, successful, and failed synthetic requests; success rate; HTTP status
  distribution; timeout and exception counts;
- offered and successful throughput; request p50, p95, and p99; request-level
  SLA violation rate; cumulative excess response time;
- the legacy endpoint-window p95 SLA-risk metric, severity, and longest
  violation duration;
- CPU-seconds, mean pod count, pod-seconds, replica changes, action churn,
  placement changes, and migration proxy;
- mean desired and ready replicas and placement counts per cluster, plus final
  per-service placement snapshots;
- capacity ceilings, rejected replicas, pending pod-seconds, readiness
  disruption, cooldown holds, decision/inference time, forecast accuracy, and
  failed episodes; and
- model-versus-guard attribution, selected objective components, candidate
  histograms, graph-free disagreement, and edge-ablation sensitivity.

These results remain synthetic and model-dependent. They do not validate
Kubernetes/Karmada actuation, true HPA/KEDA implementations, heterogeneous
hardware, WAN delay or bandwidth, control-plane latency, partitions, RBAC,
concurrent controller ownership, image pulls, API failures, or production
drift. The simulator equations and workload priors can favor controllers that
match them. The oracle uses privileged simulator access. Native local-cluster
and genuine multi-cluster experiments are still required before these numbers
can support the paper's operational claims.
