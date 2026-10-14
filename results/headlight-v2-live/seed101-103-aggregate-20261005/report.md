# Seed 101–103 controller metric aggregates

For every numeric outcome in the run summaries, this report gives the arithmetic mean, sample standard deviation, and standard error across seeds 101, 102, and 103. Standard error is sample SD / sqrt(n). The CSV includes each seed's raw value and the source summary path.

All 27 source summaries are marked `valid` and `measured`; load trace SHA-256 values match across the three controllers within every seed/workload pair. The seed 102/103 campaign status is complete (18/18 cells).

**Comparability note:** the seed 101 HPA/KEDA gradual-ramp summaries are from September 27, 2026. The seed 101 Headlight gradual-ramp summary is from October 4, and all seed 102/103 summaries are from October 5. The CSV therefore includes a separate n=2 sensitivity mean/SD/SE for the recent seed 102/103 HPA/KEDA gradual-ramp pair. Other seed 101 cells are from October 4.

The run metadata marks the seed 101 HeadLight gateway runs as evaluation and says prior comparator traces are not directly comparable. Load traces match byte-for-byte; treat cross-controller interpretation as descriptive and retain that provenance caveat.

The fixed `sla_latency_ms` value (1000 ms) is protocol configuration, not a measured outcome, and is recorded in the source manifest rather than averaged. The 200 ms and 300 ms violation rates are recalculated directly from raw request events within each recorded measurement window using the same successful-request population as the original metric. Categorical summary fields such as HTTP status distributions are expanded into per-code response-count metrics.

Detailed long-format values: `results/headlight-v2-live/seed101-103-aggregate-20261005/metric-aggregates.csv`.

## p95 latency (mean ± SE, ms)

| Workload | Headlight | HPA | KEDA |
|---|---:|---:|---:|
| gradual-ramp | 164.13 ± 8.65 | 164.16 ± 4.91 | 171.23 ± 5.28 |
| diurnal | 152.16 ± 1.74 | 161.81 ± 2.88 | 182.15 ± 1.68 |
| flash-crowd | 186.12 ± 6.64 | 195.84 ± 9.35 | 185.25 ± 5.20 |

## Metric names

- `total_requests` (requests)
- `successful_requests` (requests)
- `failed_requests` (requests)
- `success_rate` (fraction)
- `successful_throughput_rps` (requests/s)
- `offered_throughput_rps` (requests/s)
- `p50_latency_ms` (ms)
- `p95_latency_ms` (ms)
- `p99_latency_ms` (ms)
- `sla_violation_rate` (fraction)
- `sla_violation_rate_200ms` (fraction)
- `sla_violation_rate_300ms` (fraction)
- `timeout_count` (requests)
- `exception_count` (requests)
- `excess_response_time_seconds` (seconds)
- `cpu_seconds` (CPU-seconds)
- `mean_pod_count` (pods)
- `pod_seconds` (pod-seconds)
- `action_churn` (actions)
- `http_200_count` (HTTP responses)
- `http_500_count` (HTTP responses)
- `http_503_count` (HTTP responses)
