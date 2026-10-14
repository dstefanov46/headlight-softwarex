"""Dependency-light readiness state helpers for the live controller."""

from __future__ import annotations

from typing import Mapping


class PendingAgeTracker:
    """Track the age of each service/cluster's oldest outstanding replica."""

    def __init__(self) -> None:
        self._first_seen: dict[tuple[str, str], float] = {}

    def ages(
        self,
        service: str,
        pending_by_cluster: Mapping[str, int],
        timestamp: float,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for cluster, count in pending_by_cluster.items():
            key = (service, cluster)
            if int(count) <= 0:
                self._first_seen.pop(key, None)
                result[cluster] = 0.0
                continue
            first_seen = self._first_seen.get(key)
            if first_seen is None or timestamp < first_seen:
                first_seen = timestamp
                self._first_seen[key] = first_seen
            result[cluster] = max(0.0, timestamp - first_seen)
        return result
