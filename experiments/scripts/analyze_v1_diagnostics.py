#!/usr/bin/env python3
"""Join recorded v1 forecasts to exact future observations; never fit a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def identity(record: dict) -> tuple:
    return (record.get("run_id"), record.get("seed"), record["bundle_id"],
            record["policy_configuration_hash"], record["timestamp"])


def complete_members(record: dict, feature: str, service: str) -> bool:
    found = set()
    expected = set(record.get("expected_clusters", []))
    for endpoint in record.get("endpoint_observations", {}).get(feature, {}).get("endpoints", []):
        if endpoint.get("status") != "ok":
            continue
        for result in endpoint.get("result", []):
            labels = result.get("metric", {})
            node = next((labels[k].removesuffix("-service") for k in
                ["destination_workload", "deployment", "workload"] if labels.get(k)), None)
            if node is None:
                pod = labels.get("pod", "")
                node = next((n for n in sorted(record["node_order"], key=len, reverse=True)
                    if pod == n or pod.startswith(n+"-")), None)
            value = result.get("value", [None, None])[1]
            if node == service and value is not None and math.isfinite(float(value)):
                found.add(endpoint.get("cluster"))
    return bool(expected) and expected <= found


def scored_decisions(records: list[dict], window_end_unix: int | float | None) -> list[dict]:
    """Use predictions made before the measurement boundary; retain all observations as labels."""
    if window_end_unix is None:
        return [record for record in records if record.get("phase") == "decision"]
    return [record for record in records
            if record.get("phase") == "decision"
            and float(record["timestamp"]) < float(window_end_unix)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, action="append", required=True,
                        help="pod or host JSONL; repeat to merge retained copies")
    parser.add_argument("--measurement-window", type=Path,
                        help="measurement_window.json; scores decisions before finished_at_unix")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, sources, errors = {}, {}, []
    for path in args.diagnostics:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for number, line in enumerate(path.open(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("schema_version") != "headlight.v1-diagnostics/v1":
                errors.append({"source": str(path), "line": number, "record": value})
                continue
            if not value.get("run_id") or not value.get("seed"):
                raise ValueError(f"{path}:{number}: explicit run_id and seed required")
            key = (*identity(value), value["phase"])
            # Reconnected stdout/file copies may repeat the same original record.
            if key in records and records[key] != value:
                raise ValueError(f"conflicting diagnostic records for {key}")
            records[key] = value
    observations = {identity(v): v for v in records.values() if v["phase"] == "observation"}
    window_end = None
    if args.measurement_window:
        window = json.loads(args.measurement_window.read_text(encoding="utf-8"))
        window_end = window.get("finished_at_unix")
        if window_end is None:
            raise ValueError("measurement window has no finished_at_unix boundary")
    decisions = scored_decisions(list(records.values()), window_end)
    rows, services = [], {}
    for record in decisions:
        horizon = record["forecast_horizon_seconds"]
        key = identity(record)
        future_key = (*key[:-1], key[-1] + horizon)
        future = observations.get(future_key)
        current = observations.get(key)
        for service, decision in record["services"].items():
            prediction = decision.get("prediction")
            if not prediction:
                continue
            for forecast_name, feature_name, quantile in [
                ("request_q50", "request_rate", .5), ("request_q90", "request_rate", .9),
                ("cpu_q90", "cpu_usage_cores", .9), ("p95_latency_q90", "p95_latency_ms", .9),
                ("sla_violation_probability", "p95_latency_ms", None),
            ]:
                row = {"run_id": key[0], "seed": key[1], "bundle_id": key[2],
                    "service": service, "timestamp": key[-1], "target_timestamp": future_key[-1],
                    "forecast": forecast_name, "prediction": prediction[forecast_name],
                    "predictive_fresh": decision["predictive_fresh"],
                    "complete_member_coverage": False,
                    "actual": None, "persistence": None, "status": "missing_future_observation"}
                if row["prediction"] is None or not math.isfinite(row["prediction"]):
                    row["status"] = "invalid_prediction"
                elif future:
                    ni = future["node_order"].index(service)
                    fi = future["feature_order"].index(feature_name)
                    target = future["values"][ni][fi]
                    row["status"] = "future_feature_missing"
                    if target is not None and not future["missing"][ni][fi]:
                        if quantile is None:
                            target = int(target > record["sla_latency_ms"])
                        row["actual"] = target
                        row["status"] = "observed"
                        row["complete_member_coverage"] = complete_members(future, feature_name, service)
                        if current:
                            ci = current["node_order"].index(service)
                            cf = current["feature_order"].index(feature_name)
                            row["persistence"] = current["values"][ci][cf]
                            if quantile is None and row["persistence"] is not None:
                                row["persistence"] = int(row["persistence"] > record["sla_latency_ms"])
                        delta = target - row["prediction"]
                        row["error"] = -delta
                        row["absolute_error"] = abs(delta)
                        if quantile is None:
                            row["brier"] = delta**2
                            row["classification_correct"] = (row["prediction"] >= .5) == bool(target)
                        else:
                            row["pinball"] = max(quantile*delta, (quantile-1)*delta)
                            row["covered"] = target <= row["prediction"]
                rows.append(row)
                summary = services.setdefault(service, {}).setdefault(forecast_name,
                    {"predictions": 0, "observed": 0, "primary_observed": 0, "covered": 0,
                     "bias_sum": 0., "absolute_error_sum": 0., "pinball_sum": 0.,
                     "brier_sum": 0., "classification_correct": 0})
                summary["predictions"] += 1
                if row["status"] == "observed":
                    summary["observed"] += 1
                if row["status"] == "observed" and row["complete_member_coverage"] and row["predictive_fresh"]:
                    summary["primary_observed"] += 1
                    summary["covered"] += int(row.get("covered", False))
                    summary["bias_sum"] += row["error"]
                    summary["absolute_error_sum"] += row["absolute_error"]
                    summary["pinball_sum"] += row.get("pinball", 0.)
                    summary["brier_sum"] += row.get("brier", 0.)
                    summary["classification_correct"] += int(row.get("classification_correct", False))
    for service in services.values():
        for name, stats in service.items():
            n = stats["primary_observed"]
            stats.update(coverage=stats["covered"]/n if n and name != "sla_violation_probability" else None,
                         bias=stats["bias_sum"]/n if n else None,
                         mae=stats["absolute_error_sum"]/n if n else None,
                         pinball=stats["pinball_sum"]/n if n and name != "sla_violation_probability" else None,
                         brier=stats["brier_sum"]/n if n and name == "sla_violation_probability" else None,
                         classification_accuracy=stats["classification_correct"]/n if n and name == "sla_violation_probability" else None)
    if rows:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (args.output_dir / "forecast-outcomes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
    result = {"schema_version": "headlight.v1-outcome-analysis/v1", "sources": sources,
        "unique_records": len(records), "recording_errors": errors, "services": services,
        "scored_prediction_window_end_unix": window_end,
        "limitations": ["Exact timestamp joins only; missing outcomes are retained as statuses.",
            "Primary summaries require fresh predictions and complete labelled member endpoint coverage; partial observations remain in CSV.",
            "Overlapping samples are descriptive; this report does not establish independent-run calibration.",
            "Includes forecasts held by readiness or stale-input guards; predictive_fresh is recorded per row."]}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"unique_records": len(records), "outcome_rows": len(rows), "recording_errors": len(errors)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
