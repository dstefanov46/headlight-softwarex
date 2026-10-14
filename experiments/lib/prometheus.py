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
    return (
        "sum by (pod, container) (container_cpu_usage_seconds_total"
        f'{{namespace="{namespace}",container!="",pod!=""}})'
    )


def memory_working_set_query(namespace: str) -> str:
    return (
        "sum by (pod) (container_memory_working_set_bytes"
        f'{{namespace="{namespace}",container!="",pod!=""}})'
    )


def pod_count_query(namespace: str) -> str:
    # kube_pod_status_ready does not expose workload labels by default and
    # this testbed's kube-state-metrics does not allow-list Kubernetes labels.
    # Deployment pod names are stable in the form
    # <deployment>-<replicaset-hash>-<pod-suffix>, so derive a deployment
    # label before aggregating.  Non-Deployment pods do not match and remain
    # unlabeled; the ceiling diagnostic deliberately ignores those series.
    return (
        f'count by (deployment) (label_replace('
        f'kube_pod_status_ready{{namespace="{namespace}",condition="true"}} == 1, '
        f'"deployment", "$1", "pod", "(.+)-[a-z0-9]{{5,10}}-[a-z0-9]{{5}}"))'
    )


def request_rate_query(namespace: str, window: str) -> str:
    # Istio telemetry is scraped every 30s in the testbed. A 15s range can
    # contain no pair of samples and legitimately return an empty vector.
    window = _range_at_least(window, 60)
    return f'sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}"}}[{window}]))'


def istio_latency_query(namespace: str, quantile: float, window: str) -> str:
    window = _range_at_least(window, 60)
    return (
        f"histogram_quantile({quantile}, sum by (le, destination_workload) "
        f'(rate(istio_request_duration_milliseconds_bucket{{destination_workload_namespace="{namespace}"}}[{window}])))'
    )


def failure_count_query(namespace: str, window: str) -> str:
    window = _range_at_least(window, 60)
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
