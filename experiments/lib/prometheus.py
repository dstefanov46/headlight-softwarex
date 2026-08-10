"""Prometheus query builders for experiment exports."""

from __future__ import annotations


def _range_at_least(window: str, minimum_seconds: int) -> str:
    if window.endswith("s"):
        try:
            seconds = int(window[:-1])
        except ValueError:
            return window
        if seconds < minimum_seconds:
            return f"{minimum_seconds}s"
    return window


def cpu_time_query(namespace: str, window: str) -> str:
    scrape_safe_window = _range_at_least(window, 60)
    return (
        "sum by (pod) (increase(container_cpu_usage_seconds_total"
        f'{{namespace="{namespace}",container!="",pod!=""}}[{scrape_safe_window}]))'
    )


def memory_working_set_query(namespace: str) -> str:
    return (
        "sum by (pod) (container_memory_working_set_bytes"
        f'{{namespace="{namespace}",container!="",pod!=""}})'
    )


def pod_count_query(namespace: str) -> str:
    return f'count by (deployment) (kube_pod_status_ready{{namespace="{namespace}",condition="true"}} == 1)'


def request_rate_query(namespace: str, window: str) -> str:
    return f'sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}"}}[{window}]))'


def istio_latency_query(namespace: str, quantile: float, window: str) -> str:
    return (
        f"histogram_quantile({quantile}, sum by (le, destination_workload) "
        f'(rate(istio_request_duration_milliseconds_bucket{{destination_workload_namespace="{namespace}"}}[{window}])))'
    )


def failure_count_query(namespace: str, window: str) -> str:
    return (
        "sum(increase(istio_requests_total"
        f'{{destination_workload_namespace="{namespace}",response_code!~"2..|3.."}}[{window}])) or vector(0)'
    )


def export_queries(namespace: str, window: str) -> dict[str, str]:
    return {
        "cpu_seconds": cpu_time_query(namespace, window),
        "memory_working_set_bytes": memory_working_set_query(namespace),
        "pod_count": pod_count_query(namespace),
        "request_rate": request_rate_query(namespace, window),
        "istio_p95_latency_ms": istio_latency_query(namespace, 0.95, window),
        "failure_count": failure_count_query(namespace, window),
    }
