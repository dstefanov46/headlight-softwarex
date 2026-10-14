"""Dependency-light tensor contracts shared by Headlight data and models."""

from __future__ import annotations


REQUIRED_NODE_FEATURES = (
    "request_rate",
    "cpu_usage_cores",
    "cpu_utilization",
    "mean_latency_ms",
    "p95_latency_ms",
    "error_rate",
    "desired_replicas",
    "ready_replicas",
    "pending_replicas",
    "pending_age_seconds",
    "time_since_last_action_seconds",
    "request_rate_missing",
    "cpu_missing",
    "latency_missing",
    "error_missing",
    "request_rate_stale",
    "cpu_stale",
    "latency_stale",
    "error_stale",
)

DERIVED_REQUEST_FEATURES = (
    "request_rate_difference",
    "request_rate_short_slope",
    "request_rate_long_slope",
)

REQUIRED_EDGE_FEATURES = (
    "call_rate",
    "call_ratio",
    "latency_ms",
    "error_rate",
    "cross_cluster",
    "missing",
    "stale",
)
