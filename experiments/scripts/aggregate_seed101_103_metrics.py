#!/usr/bin/env python3
"""Aggregate the three-seed Headlight, HPA, and KEDA qualification summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.lib.metrics import request_metrics  # noqa: E402

OUTPUT_DIR = ROOT / "results/headlight-v2-live/seed101-103-aggregate-20261005"
CAMPAIGN_STATUS = (
    ROOT / "results/headlight-v2-live/qualification-seed102-103-20261004/campaign-status.json"
)
WORKLOADS = ("gradual-ramp", "diurnal", "flash-crowd")
CONTROLLERS = ("headlight-v2", "hpa", "keda")
SEEDS = (101, 102, 103)

# These fields are numeric outcomes in summary.csv. sla_latency_ms is a fixed
# protocol threshold, while the remaining excluded numeric-looking fields are
# seed IDs or HeadLight tuning metadata rather than measured outcomes.
METRICS: dict[str, str] = {
    "total_requests": "requests",
    "successful_requests": "requests",
    "failed_requests": "requests",
    "success_rate": "fraction",
    "successful_throughput_rps": "requests/s",
    "offered_throughput_rps": "requests/s",
    "p50_latency_ms": "ms",
    "p95_latency_ms": "ms",
    "p99_latency_ms": "ms",
    "sla_violation_rate": "fraction",
    "sla_violation_rate_200ms": "fraction",
    "sla_violation_rate_300ms": "fraction",
    "timeout_count": "requests",
    "exception_count": "requests",
    "excess_response_time_seconds": "seconds",
    "cpu_seconds": "CPU-seconds",
    "mean_pod_count": "pods",
    "pod_seconds": "pod-seconds",
    "action_churn": "actions",
}

SEED101_PATHS: dict[tuple[str, str], Path] = {
    ("gradual-ramp", "headlight-v2"): ROOT
    / "results/headlight-v2-live/emergency-disabled-gradual-ramp-c4-seed-101-20261004",
    ("gradual-ramp", "hpa"): ROOT
    / "results/headlight-v2-live/paper-grid-candidate-gradual-ramp-seed-101-20260927/hpa",
    ("gradual-ramp", "keda"): ROOT
    / "results/headlight-v2-live/paper-grid-candidate-gradual-ramp-seed-101-20260927/keda",
    ("diurnal", "headlight-v2"): ROOT
    / "results/headlight-v2-live/qualification-diurnal-c4-seed-101-20261004",
    ("diurnal", "hpa"): ROOT / "results/socc26/hpa-diurnal-multi-cluster-headlight",
    ("diurnal", "keda"): ROOT / "results/socc26/keda-diurnal-multi-cluster-headlight",
    ("flash-crowd", "headlight-v2"): ROOT
    / "results/headlight-v2-live/qualification-flash-crowd-c4-seed-101-20261004",
    ("flash-crowd", "hpa"): ROOT / "results/socc26/hpa-flash-crowd-multi-cluster-headlight",
    ("flash-crowd", "keda"): ROOT / "results/socc26/keda-flash-crowd-multi-cluster-headlight",
}

EXPECTED_TRACE_SHA256 = {
    "gradual-ramp": "b1a3d85fce1cd8a4ce44bef382794c8c12fcb2647f389ae789a284cc19cb00ca",
    "diurnal": "ced8b35581298a0e88cdadc9939033d41372591ed75913aaea118fa99c057ce2",
    "flash-crowd": "015943997cc853524e85ae0aeb965bcdf92da58e4d1be2003a0ae6a485f73116",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def summary_path(output: Path, seed: int) -> Path:
    return output / f"seed-{seed}/summaries/summary.csv"


def read_one_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing summary: {relative(path)}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one summary row in {relative(path)}, found {len(rows)}")
    return rows[0]


def campaign_outputs() -> dict[tuple[int, str, str], Path]:
    status = json.loads(CAMPAIGN_STATUS.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("valid_cell_count") != 18:
        raise SystemExit("seed 102/103 campaign status is not complete with 18 valid cells")
    outputs: dict[tuple[int, str, str], Path] = {}
    for cell in status["cells"]:
        if cell.get("status") != "valid" or not cell.get("valid_output_dir"):
            raise SystemExit(f"campaign has an invalid or missing cell: {cell}")
        key = (int(cell["seed"]), str(cell["workload"]), str(cell["controller"]))
        outputs[key] = ROOT / cell["valid_output_dir"]
    if len(outputs) != 18:
        raise SystemExit(f"expected 18 seed-102/103 cells, found {len(outputs)}")
    return outputs


def get_output(
    seed: int,
    workload: str,
    controller: str,
    campaign: dict[tuple[int, str, str], Path],
) -> Path:
    if seed == 101:
        return SEED101_PATHS[(workload, controller)]
    return campaign[(seed, workload, controller)]


def load_validated_rows() -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, Any]]:
    campaign = campaign_outputs()
    rows: dict[tuple[int, str, str], dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for workload in WORKLOADS:
            for controller in CONTROLLERS:
                output = get_output(seed, workload, controller, campaign)
                summary = summary_path(output, seed)
                row = read_one_row(summary)
                expected_identity = (str(seed), controller, workload)
                actual_identity = (row.get("seed"), row.get("controller"), row.get("workload"))
                if actual_identity != expected_identity:
                    raise SystemExit(
                        f"identity mismatch in {relative(summary)}: "
                        f"expected={expected_identity}, actual={actual_identity}"
                    )
                if row.get("technical_validity") != "valid" or row.get("performance_outcome") != "measured":
                    raise SystemExit(
                        f"non-valid measurement in {relative(summary)}: "
                        f"{row.get('technical_validity')}/{row.get('performance_outcome')}"
                    )

                trace = output / f"seed-{seed}/raw/load_profile.csv"
                if not trace.is_file():
                    raise SystemExit(f"missing load trace: {relative(trace)}")
                trace_hash = sha256(trace)
                if trace_hash != EXPECTED_TRACE_SHA256[workload]:
                    raise SystemExit(
                        f"load trace hash mismatch in {relative(trace)}: {trace_hash}"
                    )

                metrics: dict[str, float] = {}
                for metric in METRICS:
                    if metric in {"sla_violation_rate_200ms", "sla_violation_rate_300ms"}:
                        continue
                    raw = row.get(metric, "")
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        raise SystemExit(f"missing/non-numeric {metric} in {relative(summary)}") from None
                    if not math.isfinite(value):
                        raise SystemExit(f"non-finite {metric} in {relative(summary)}")
                    metrics[metric] = value

                seed_dir = output / f"seed-{seed}"
                request_events = seed_dir / "raw/request_events.csv"
                measurement_window_path = seed_dir / "raw/measurement_window.json"
                if not request_events.is_file() or not measurement_window_path.is_file():
                    raise SystemExit(
                        f"missing raw request events or measurement window for {relative(output)}"
                    )
                measurement_window = json.loads(
                    measurement_window_path.read_text(encoding="utf-8")
                )
                started_at = float(measurement_window["started_at_unix"])
                finished_at = float(measurement_window["finished_at_unix"])
                original_recomputed = request_metrics(
                    request_events, 1000, started_at=started_at, finished_at=finished_at
                )
                if original_recomputed["successful_requests"] != int(row["successful_requests"]):
                    raise SystemExit(
                        f"successful request count does not match raw events in {relative(summary)}"
                    )
                if not math.isclose(
                    float(original_recomputed["sla_violation_rate"]),
                    float(row["sla_violation_rate"]),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise SystemExit(
                        f"recomputed 1000 ms violation rate disagrees with summary in {relative(summary)}"
                    )
                for threshold in (200, 300):
                    threshold_metrics = request_metrics(
                        request_events,
                        threshold,
                        started_at=started_at,
                        finished_at=finished_at,
                    )
                    metrics[f"sla_violation_rate_{threshold}ms"] = float(
                        threshold_metrics["sla_violation_rate"]
                    )

                try:
                    status_counts = json.loads(row.get("http_status_distribution", "{}"))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid HTTP status distribution in {relative(summary)}: {exc}") from exc
                if not isinstance(status_counts, dict):
                    raise SystemExit(f"HTTP status distribution is not an object in {relative(summary)}")
                for code, count in status_counts.items():
                    try:
                        metrics[f"http_{code}_count"] = float(count)
                    except (TypeError, ValueError):
                        raise SystemExit(f"invalid HTTP status count in {relative(summary)}") from None

                key = (seed, workload, controller)
                rows[key] = {
                    "metrics": metrics,
                    "summary": summary,
                    "request_events": request_events,
                    "output": output,
                    "trace_sha256": trace_hash,
                    "sla_latency_ms": float(row["sla_latency_ms"]),
                    "cpu_calculation_method": row.get("cpu_calculation_method", ""),
                    "pod_seconds_calculation_method": row.get("pod_seconds_calculation_method", ""),
                }
                source_rows.append(
                    {
                        "seed": seed,
                        "workload": workload,
                        "controller": controller,
                        "output_dir": relative(output),
                        "summary_csv": relative(summary),
                        "request_events_csv": relative(request_events),
                        "measurement_window_json": relative(measurement_window_path),
                        "load_trace_sha256": trace_hash,
                        "sla_latency_ms": float(row["sla_latency_ms"]),
                        "technical_validity": row["technical_validity"],
                        "performance_outcome": row["performance_outcome"],
                    }
                )

    for seed in SEEDS:
        for workload in WORKLOADS:
            traces = {
                rows[(seed, workload, controller)]["trace_sha256"]
                for controller in CONTROLLERS
            }
            if len(traces) != 1:
                raise SystemExit(f"controllers do not share a load trace for {seed}/{workload}")

    return rows, {"source_rows": source_rows, "campaign_status": json.loads(CAMPAIGN_STATUS.read_text())}


def all_metric_names(rows: dict[tuple[int, str, str], dict[str, Any]]) -> list[str]:
    return list(METRICS) + sorted(
        {
            metric
            for row in rows.values()
            for metric in row["metrics"]
            if metric.startswith("http_")
        }
    )


def aggregate_csv(rows: dict[tuple[int, str, str], dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for workload in WORKLOADS:
        for controller in CONTROLLERS:
            group = [rows[(seed, workload, controller)] for seed in SEEDS]
            legacy_pair = workload == "gradual-ramp" and controller in {"hpa", "keda"}
            for metric in metrics:
                values = [row["metrics"].get(metric, 0.0) for row in group]
                mean = statistics.fmean(values)
                sd = statistics.stdev(values)
                entry: dict[str, Any] = {
                    "workload": workload,
                    "controller": controller,
                    "metric": metric,
                    "unit": METRICS.get(metric, "HTTP responses"),
                    "n": len(values),
                    "mean": mean,
                    "sample_sd": sd,
                    "standard_error": sd / math.sqrt(len(values)),
                    "seed_101": values[0],
                    "seed_102": values[1],
                    "seed_103": values[2],
                    "seed_101_summary_csv": relative(group[0]["summary"]),
                    "seed_102_summary_csv": relative(group[1]["summary"]),
                    "seed_103_summary_csv": relative(group[2]["summary"]),
                    "comparability_note": (
                        "seed-101 comparator is from 2026-09-27; seeds 102/103 are from 2026-10-05"
                        if legacy_pair
                        else ""
                    ),
                    "n_102_103_mean": "",
                    "n_102_103_sample_sd": "",
                    "n_102_103_standard_error": "",
                }
                if legacy_pair:
                    recent_values = values[1:]
                    recent_sd = statistics.stdev(recent_values)
                    entry.update(
                        {
                            "n_102_103_mean": statistics.fmean(recent_values),
                            "n_102_103_sample_sd": recent_sd,
                            "n_102_103_standard_error": recent_sd / math.sqrt(2),
                        }
                    )
                output_rows.append(entry)
    return output_rows


def write_markdown(
    rows: dict[tuple[int, str, str], dict[str, Any]],
    metrics: list[str],
    csv_path: Path,
    aggregated: list[dict[str, Any]],
) -> None:
    lines = [
        "# Seed 101–103 controller metric aggregates",
        "",
        "For every numeric outcome in the run summaries, this report gives the arithmetic mean, sample standard deviation, and standard error across seeds 101, 102, and 103. Standard error is sample SD / sqrt(n). The CSV includes each seed's raw value and the source summary path.",
        "",
        "All 27 source summaries are marked `valid` and `measured`; load trace SHA-256 values match across the three controllers within every seed/workload pair. The seed 102/103 campaign status is complete (18/18 cells).",
        "",
        "**Comparability note:** the seed 101 HPA/KEDA gradual-ramp summaries are from September 27, 2026. The seed 101 Headlight gradual-ramp summary is from October 4, and all seed 102/103 summaries are from October 5. The CSV therefore includes a separate n=2 sensitivity mean/SD/SE for the recent seed 102/103 HPA/KEDA gradual-ramp pair. Other seed 101 cells are from October 4.",
        "",
        "The run metadata marks the seed 101 HeadLight gateway runs as evaluation and says prior comparator traces are not directly comparable. Load traces match byte-for-byte; treat cross-controller interpretation as descriptive and retain that provenance caveat.",
        "",
        "The fixed `sla_latency_ms` value (1000 ms) is protocol configuration, not a measured outcome, and is recorded in the source manifest rather than averaged. The 200 ms and 300 ms violation rates are recalculated directly from raw request events within each recorded measurement window using the same successful-request population as the original metric. Categorical summary fields such as HTTP status distributions are expanded into per-code response-count metrics.",
        "",
        f"Detailed long-format values: `{relative(csv_path)}`.",
        "",
        "## p95 latency (mean ± SE, ms)",
        "",
        "| Workload | Headlight | HPA | KEDA |",
        "|---|---:|---:|---:|",
    ]
    for workload in WORKLOADS:
        values = []
        for controller in CONTROLLERS:
            observations = [rows[(seed, workload, controller)]["metrics"]["p95_latency_ms"] for seed in SEEDS]
            mean = statistics.fmean(observations)
            se = statistics.stdev(observations) / math.sqrt(3)
            values.append(f"{mean:.2f} ± {se:.2f}")
        lines.append(f"| {workload} | " + " | ".join(values) + " |")
    lines.extend(["", "## Metric names", ""])
    for metric in metrics:
        lines.append(f"- `{metric}` ({metric_unit(metric)})")
    lines.append("")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    markdown = [
        "# Seed 101–103 metrics by workload and controller",
        "",
        "Each cell is the arithmetic **mean ± standard error** across seeds 101, 102, and 103. Standard error is sample standard deviation divided by √3. Rates are fractions. The [CSV](metric-aggregates.csv) contains full precision, each seed's individual value, sample standard deviation, source summary path, and raw request event path.",
        "",
        "All 27 source summaries are valid and measured. For each seed/workload pair, the three controller runs have matching load trace hashes.",
        "",
        "**Comparability note:** the seed 101 HPA and KEDA gradual-ramp results are from September 27, 2026. The seed 101 Headlight gradual-ramp result is from October 4; seed 102/103 results are from October 5. The older comparator values are included and flagged in the CSV, which also reports n=2 sensitivity statistics for seeds 102/103. Seed 101 HeadLight gateway metadata marks those runs evaluation and says prior comparator traces are not directly comparable; treat cross-controller comparisons as descriptive.",
        "",
        "The original violation rate uses the 1000 ms threshold. The `sla_violation_rate_200ms` and `sla_violation_rate_300ms` metrics below are recomputed from each run's raw request events. For all three rates, a violation is a successful request whose recorded response time is strictly greater than the threshold; the denominator is successful requests with a recorded response time during that run's measurement window.",
        "",
        "The fixed 1000 ms threshold is protocol configuration rather than an averaged outcome. HTTP status distributions are expanded into per-code response-count metrics.",
        "",
    ]
    for workload in WORKLOADS:
        markdown.extend(
            [
                f"## {workload}",
                "",
                "| Metric (unit) | Headlight | HPA | KEDA |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in metrics:
            by_controller = {
                row["controller"]: row
                for row in aggregated
                if row["workload"] == workload and row["metric"] == metric
            }
            values = [
                format_metric(by_controller[controller])
                for controller in CONTROLLERS
            ]
            markdown.append(
                f"| {metric_label(metric)} ({metric_unit(metric)}) | "
                + " | ".join(values)
                + " |"
            )
        markdown.append("")
    (OUTPUT_DIR / "all-metrics.md").write_text("\n".join(markdown), encoding="utf-8")


def metric_unit(metric: str) -> str:
    return METRICS.get(metric, "HTTP responses")


def metric_label(metric: str) -> str:
    labels = {
        "total_requests": "Total requests",
        "successful_requests": "Successful requests",
        "failed_requests": "Failed requests",
        "success_rate": "Success rate",
        "successful_throughput_rps": "Successful throughput",
        "offered_throughput_rps": "Offered throughput",
        "p50_latency_ms": "p50 latency",
        "p95_latency_ms": "p95 latency",
        "p99_latency_ms": "p99 latency",
        "sla_violation_rate": "SLA violation rate >1000 ms",
        "sla_violation_rate_200ms": "SLA violation rate >200 ms",
        "sla_violation_rate_300ms": "SLA violation rate >300 ms",
        "timeout_count": "Timeouts",
        "exception_count": "Exceptions",
        "excess_response_time_seconds": "Excess response time",
        "cpu_seconds": "CPU time",
        "mean_pod_count": "Mean pod count",
        "pod_seconds": "Pod time",
        "action_churn": "Action churn",
    }
    if metric.startswith("http_") and metric.endswith("_count"):
        return f"HTTP {metric.removeprefix('http_').removesuffix('_count')} responses"
    return labels.get(metric, metric)


def format_metric(row: dict[str, Any]) -> str:
    mean = float(row["mean"])
    standard_error = float(row["standard_error"])
    if row["metric"] in {
        "success_rate",
        "sla_violation_rate",
        "sla_violation_rate_200ms",
        "sla_violation_rate_300ms",
    }:
        return f"{mean:.6f} ± {standard_error:.6f}"
    return f"{mean:.2f} ± {standard_error:.2f}"


def main() -> int:
    rows, provenance = load_validated_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = all_metric_names(rows)
    aggregated = aggregate_csv(rows, metrics)
    csv_path = OUTPUT_DIR / "metric-aggregates.csv"
    fieldnames = list(aggregated[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)

    source_manifest = {
        "schema_version": "headlight.controller-metric-aggregate/v1",
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "method": {
            "mean": "arithmetic mean across seed-level summary metrics",
            "sample_sd": "sample standard deviation (n - 1 denominator)",
            "standard_error": "sample_sd / sqrt(n)",
            "n": 3,
        },
        "protocol_sla_latency_ms": 1000,
        "metric_units": {metric: METRICS.get(metric, "HTTP responses") for metric in metrics},
        "alternate_latency_thresholds": {
            "sla_violation_rate_200ms": "recomputed from raw requests as successful requests with response_time_ms > 200 divided by successful requests with recorded latency in measurement window",
            "sla_violation_rate_300ms": "recomputed from raw requests as successful requests with response_time_ms > 300 divided by successful requests with recorded latency in measurement window",
        },
        "comparability": {
            "seed_101_gradual_ramp_hpa_keda": "2026-09-27; older run/testbed source commit than the October runs",
            "seed_101_headlight_v2": "2026-10-04 evaluation gateway qualification; metadata says prior comparator traces are not directly comparable",
            "seed_101_diurnal_flash_crowd_hpa_keda": "2026-10-04",
            "seeds_102_103": "2026-10-05 campaign; 18/18 valid cells",
            "load_traces": "all controller traces matched the expected workload SHA-256 for every seed/workload pair",
        },
        "sources": provenance["source_rows"],
    }
    provenance_path = OUTPUT_DIR / "source-manifest.json"
    provenance_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(rows, metrics, csv_path, aggregated)
    print(f"AGGREGATE_OK groups={len(WORKLOADS) * len(CONTROLLERS)} metrics={len(metrics)} rows={len(aggregated)} n=3")
    print(f"CSV={relative(csv_path)}")
    print(f"REPORT={relative(OUTPUT_DIR / 'report.md')}")
    print(f"FULL_METRICS={relative(OUTPUT_DIR / 'all-metrics.md')}")
    print(f"PROVENANCE={relative(provenance_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
