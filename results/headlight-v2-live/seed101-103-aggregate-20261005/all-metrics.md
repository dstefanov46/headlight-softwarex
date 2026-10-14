# Seed 101–103 metrics by workload and controller

Each cell is the arithmetic **mean ± standard error** across seeds 101, 102, and 103. Standard error is sample standard deviation divided by √3. Rates are fractions. The [CSV](metric-aggregates.csv) contains full precision, each seed's individual value, sample standard deviation, source summary path, and raw request event path.

All 27 source summaries are valid and measured. For each seed/workload pair, the three controller runs have matching load trace hashes.

**Comparability note:** the seed 101 HPA and KEDA gradual-ramp results are from September 27, 2026. The seed 101 Headlight gradual-ramp result is from October 4; seed 102/103 results are from October 5. The older comparator values are included and flagged in the CSV, which also reports n=2 sensitivity statistics for seeds 102/103. Seed 101 Headlight gateway metadata marks those runs evaluation and says prior comparator traces are not directly comparable; treat cross-controller comparisons as descriptive.

The original violation rate uses the 1000 ms threshold. The `sla_violation_rate_200ms` and `sla_violation_rate_300ms` metrics below are recomputed from each run's raw request events. For all three rates, a violation is a successful request whose recorded response time is strictly greater than the threshold; the denominator is successful requests with a recorded response time during that run's measurement window.

The fixed 1000 ms threshold is protocol configuration rather than an averaged outcome. HTTP status distributions are expanded into per-code response-count metrics.

## gradual-ramp

| Metric (unit) | Headlight | HPA | KEDA |
|---|---:|---:|---:|
| Total requests (requests) | 6379.33 ± 58.47 | 6401.00 ± 24.34 | 6361.33 ± 4.91 |
| Successful requests (requests) | 6379.00 ± 58.79 | 6401.00 ± 24.34 | 6361.33 ± 4.91 |
| Failed requests (requests) | 0.33 ± 0.33 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Success rate (fraction) | 0.999947 ± 0.000053 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Successful throughput (requests/s) | 5.32 ± 0.05 | 5.33 ± 0.02 | 5.30 ± 0.00 |
| Offered throughput (requests/s) | 5.32 ± 0.05 | 5.33 ± 0.02 | 5.30 ± 0.00 |
| p50 latency (ms) | 92.13 ± 3.57 | 89.00 ± 0.93 | 93.50 ± 2.70 |
| p95 latency (ms) | 164.13 ± 8.65 | 164.16 ± 4.91 | 171.23 ± 5.28 |
| p99 latency (ms) | 267.02 ± 18.14 | 260.00 ± 7.51 | 253.89 ± 7.63 |
| SLA violation rate >1000 ms (fraction) | 0.001361 ± 0.000118 | 0.001406 ± 0.000240 | 0.002410 ± 0.000139 |
| SLA violation rate >200 ms (fraction) | 0.030408 ± 0.004665 | 0.027775 ± 0.002451 | 0.029661 ± 0.001583 |
| SLA violation rate >300 ms (fraction) | 0.006279 ± 0.001588 | 0.006096 ± 0.000606 | 0.006446 ± 0.000400 |
| Timeouts (requests) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Exceptions (requests) | 0.33 ± 0.33 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Excess response time (seconds) | 7.83 ± 0.94 | 11.18 ± 1.97 | 16.18 ± 1.61 |
| CPU time (CPU-seconds) | 1183.03 ± 48.16 | 1230.71 ± 73.90 | 1563.87 ± 124.59 |
| Mean pod count (pods) | 39.52 ± 0.34 | 40.37 ± 0.55 | 43.42 ± 0.66 |
| Pod time (pod-seconds) | 47355.00 ± 392.40 | 48350.00 ± 651.79 | 51985.00 ± 785.76 |
| Action churn (actions) | 15.33 ± 1.20 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| HTTP 200 responses (HTTP responses) | 6379.00 ± 58.79 | 6401.00 ± 24.34 | 6361.33 ± 4.91 |
| HTTP 500 responses (HTTP responses) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| HTTP 503 responses (HTTP responses) | 0.33 ± 0.33 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## diurnal

| Metric (unit) | Headlight | HPA | KEDA |
|---|---:|---:|---:|
| Total requests (requests) | 11250.33 ± 20.54 | 11152.00 ± 47.43 | 10968.00 ± 12.86 |
| Successful requests (requests) | 11250.33 ± 20.54 | 11151.67 ± 47.76 | 10968.00 ± 12.86 |
| Failed requests (requests) | 0.00 ± 0.00 | 0.33 ± 0.33 | 0.00 ± 0.00 |
| Success rate (fraction) | 1.000000 ± 0.000000 | 0.999970 ± 0.000030 | 1.000000 ± 0.000000 |
| Successful throughput (requests/s) | 4.69 ± 0.01 | 4.65 ± 0.02 | 4.57 ± 0.01 |
| Offered throughput (requests/s) | 4.69 ± 0.01 | 4.65 ± 0.02 | 4.57 ± 0.01 |
| p50 latency (ms) | 88.82 ± 2.14 | 90.27 ± 1.07 | 95.73 ± 1.70 |
| p95 latency (ms) | 152.16 ± 1.74 | 161.81 ± 2.88 | 182.15 ± 1.68 |
| p99 latency (ms) | 252.30 ± 5.72 | 258.81 ± 1.85 | 285.20 ± 7.40 |
| SLA violation rate >1000 ms (fraction) | 0.001037 ± 0.000195 | 0.001435 ± 0.000053 | 0.003009 ± 0.000107 |
| SLA violation rate >200 ms (fraction) | 0.024538 ± 0.001553 | 0.026543 ± 0.001262 | 0.035196 ± 0.001566 |
| SLA violation rate >300 ms (fraction) | 0.004831 ± 0.000639 | 0.005951 ± 0.000341 | 0.008236 ± 0.000899 |
| Timeouts (requests) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Exceptions (requests) | 0.00 ± 0.00 | 0.33 ± 0.33 | 0.00 ± 0.00 |
| Excess response time (seconds) | 11.48 ± 4.32 | 18.24 ± 1.09 | 41.03 ± 2.72 |
| CPU time (CPU-seconds) | 2017.10 ± 124.98 | 2098.04 ± 87.47 | 3044.55 ± 7.72 |
| Mean pod count (pods) | 42.08 ± 0.41 | 42.90 ± 0.28 | 44.60 ± 0.39 |
| Pod time (pod-seconds) | 100945.00 ± 952.64 | 102950.00 ± 662.63 | 107025.00 ± 936.31 |
| Action churn (actions) | 35.33 ± 19.55 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| HTTP 200 responses (HTTP responses) | 11250.33 ± 20.54 | 11151.67 ± 47.76 | 10968.00 ± 12.86 |
| HTTP 500 responses (HTTP responses) | 0.00 ± 0.00 | 0.33 ± 0.33 | 0.00 ± 0.00 |
| HTTP 503 responses (HTTP responses) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## flash-crowd

| Metric (unit) | Headlight | HPA | KEDA |
|---|---:|---:|---:|
| Total requests (requests) | 4230.33 ± 11.26 | 4133.33 ± 69.75 | 4122.00 ± 22.03 |
| Successful requests (requests) | 4230.33 ± 11.26 | 4133.33 ± 69.75 | 4122.00 ± 22.03 |
| Failed requests (requests) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Success rate (fraction) | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Successful throughput (requests/s) | 3.53 ± 0.01 | 3.44 ± 0.06 | 3.44 ± 0.02 |
| Offered throughput (requests/s) | 3.53 ± 0.01 | 3.44 ± 0.06 | 3.44 ± 0.02 |
| p50 latency (ms) | 95.21 ± 4.26 | 96.51 ± 5.76 | 101.02 ± 4.19 |
| p95 latency (ms) | 186.12 ± 6.64 | 195.84 ± 9.35 | 185.25 ± 5.20 |
| p99 latency (ms) | 285.46 ± 13.03 | 300.80 ± 21.92 | 330.28 ± 19.95 |
| SLA violation rate >1000 ms (fraction) | 0.001418 ± 0.000135 | 0.002512 ± 0.000395 | 0.005418 ± 0.000322 |
| SLA violation rate >200 ms (fraction) | 0.039725 ± 0.004760 | 0.046647 ± 0.008364 | 0.040290 ± 0.002684 |
| SLA violation rate >300 ms (fraction) | 0.008674 ± 0.001181 | 0.010263 ± 0.003007 | 0.011887 ± 0.000966 |
| Timeouts (requests) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Exceptions (requests) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Excess response time (seconds) | 7.96 ± 0.21 | 15.01 ± 3.87 | 29.06 ± 1.84 |
| CPU time (CPU-seconds) | 1109.65 ± 32.77 | 1207.58 ± 144.42 | 1719.42 ± 93.13 |
| Mean pod count (pods) | 39.45 ± 0.73 | 41.42 ± 1.25 | 41.97 ± 0.19 |
| Pod time (pod-seconds) | 47285.00 ± 876.54 | 49640.00 ± 1504.35 | 50275.00 ± 229.84 |
| Action churn (actions) | 18.33 ± 1.33 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| HTTP 200 responses (HTTP responses) | 4230.33 ± 11.26 | 4133.33 ± 69.75 | 4122.00 ± 22.03 |
| HTTP 500 responses (HTTP responses) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| HTTP 503 responses (HTTP responses) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
