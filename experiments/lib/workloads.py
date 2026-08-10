"""Deterministic workload shape generation."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ShapePoint:
    elapsed_seconds: int
    users: int
    spawn_rate: int


def _linear(start: int, end: int, fraction: float) -> int:
    return int(round(start + (end - start) * max(0.0, min(1.0, fraction))))


def _gradual_ramp(locust: dict[str, Any], elapsed: int, duration: int) -> int:
    start = int(locust["start_users"])
    end = int(locust["end_users"])
    ramp = int(locust["ramp_seconds"])
    hold = int(locust.get("hold_seconds", max(0, duration - (2 * ramp))))
    if elapsed <= ramp:
        return _linear(start, end, elapsed / ramp)
    if elapsed <= ramp + hold:
        return end
    return _linear(end, start, (elapsed - ramp - hold) / max(1, duration - ramp - hold))


def _flash_crowd(locust: dict[str, Any], elapsed: int) -> int:
    base = int(locust["base_users"])
    spike = int(locust["spike_users"])
    spike_duration = int(locust["spike_duration_seconds"])
    starts = locust.get("spike_start_seconds", 0)
    if isinstance(starts, int):
        starts = [starts]
    return spike if any(start <= elapsed < start + spike_duration for start in starts) else base


def _diurnal(locust: dict[str, Any], elapsed: int) -> int:
    minimum = int(locust["min_users"])
    maximum = int(locust["max_users"])
    cycle = int(locust["cycle_seconds"])
    phase = int(locust.get("phase_offset_seconds", 0))
    amplitude = (maximum - minimum) / 2
    midpoint = minimum + amplitude
    return int(round(midpoint + amplitude * math.sin(2 * math.pi * (elapsed + phase) / cycle - math.pi / 2)))


def _bursty(locust: dict[str, Any], rng: random.Random, _elapsed: int = 0) -> int:
    base = int(locust["base_users"])
    burst = int(locust["burst_users"])
    probability = float(locust["burst_probability"])
    return burst if rng.random() < probability else base


def generate_shape(workload: dict[str, Any], seed: int, sampling_interval_seconds: int) -> list[ShapePoint]:
    pattern = workload["pattern"]
    duration = int(workload["duration_seconds"])
    locust = workload["locust"]
    spawn_rate = int(locust["spawn_rate"])
    rng = random.Random(seed)
    points: list[ShapePoint] = []
    handlers = {
        "gradual-ramp": partial(_gradual_ramp, locust, duration=duration),
        "flash-crowd": partial(_flash_crowd, locust),
        "diurnal": partial(_diurnal, locust),
        "bursty": partial(_bursty, locust, rng),
    }

    for elapsed in range(0, duration + 1, sampling_interval_seconds):
        try:
            users = handlers[pattern](elapsed)
        except KeyError as exc:
            raise ValueError(f"unsupported workload pattern `{pattern}`") from exc
        points.append(ShapePoint(elapsed, max(1, users), spawn_rate))
    return points


def write_shape_csv(path: Path, points: list[ShapePoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["elapsed_seconds", "users", "spawn_rate"])
        writer.writeheader()
        for point in points:
            writer.writerow(point.__dict__)
