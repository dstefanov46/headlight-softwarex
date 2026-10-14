"""Deterministic workload shape generation."""

from __future__ import annotations

import csv
import math
import os
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


def _capacity_users(workload: dict[str, Any]) -> int:
    locust = workload.get("locust", {})
    env_name = str(
        workload.get("capacity_users_env")
        or locust.get("capacity_users_env")
        or "HEADLIGHT_EVALUATION_CAPACITY"
    )
    for candidate in (os.environ.get(env_name), os.environ.get("HEADLIGHT_SELECTED_CAPACITY")):
        if candidate not in (None, ""):
            return max(1, int(candidate))
    return max(1, int(workload.get("capacity_users", locust.get("capacity_users", 10))))


def _resolve_users(value: Any, capacity: int) -> int:
    if isinstance(value, str):
        expression = value.strip().upper()
        if expression == "C":
            return capacity
        if expression.endswith("C"):
            return int(round(float(expression[:-1]) * capacity))
    return int(value)


def _resolved_locust(workload: dict[str, Any], capacity_users: int | None = None) -> dict[str, Any]:
    capacity = _capacity_users(workload) if capacity_users is None else int(capacity_users)
    if capacity <= 0:
        raise ValueError("workload capacity must be positive")
    locust = dict(workload["locust"])
    for key in (
        "start_users",
        "end_users",
        "base_users",
        "spike_users",
        "min_users",
        "max_users",
        "burst_users",
    ):
        if key in locust:
            locust[key] = _resolve_users(locust[key], capacity)
    locust["capacity_users"] = capacity
    return locust


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


def workload_phase_windows(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return contiguous measurement-relative boundaries for a phased workload."""
    windows: list[dict[str, Any]] = []
    offset = 0
    for phase in phases:
        pattern = str(phase.get("pattern", ""))
        duration = phase.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise ValueError("multi-phase phase durations must be positive integers")
        windows.append({
            "pattern": pattern,
            "start_offset_seconds": offset,
            "end_offset_seconds": offset + duration,
        })
        offset += duration
    return windows


def _multi_phase_shape(
    workload: dict[str, Any],
    seed: int,
    sampling_interval_seconds: int,
    warmup_seconds: int,
    duration_seconds: int,
    continuation_seconds: int,
    capacity_users: int | None,
) -> list[ShapePoint]:
    phases = workload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("multi-phase workload requires a non-empty phases list")
    parsed: list[tuple[str, int, dict[str, Any], Any]] = []
    rng = random.Random(seed)
    handlers = {
        "gradual-ramp": _gradual_ramp,
        "flash-crowd": _flash_crowd,
        "diurnal": _diurnal,
        "bursty": _bursty,
    }
    for phase in phases:
        phase_pattern = str(phase.get("pattern", ""))
        phase_duration = phase.get("duration_seconds")
        if (isinstance(phase_duration, bool) or not isinstance(phase_duration, int)
                or phase_duration <= 0):
            raise ValueError("multi-phase phase durations must be positive integers")
        if phase_pattern not in handlers:
            raise ValueError(f"unsupported workload pattern `{phase_pattern}` in multi-phase workload")
        phase_workload = {
            "locust": phase.get("locust", {}),
            "capacity_users": workload.get("capacity_users", 10),
            "capacity_users_env": workload.get("capacity_users_env"),
        }
        locust = _resolved_locust(phase_workload, capacity_users)
        if phase_pattern == "gradual-ramp":
            handler = partial(_gradual_ramp, locust, duration=phase_duration)
        elif phase_pattern == "flash-crowd":
            handler = partial(_flash_crowd, locust)
        elif phase_pattern == "diurnal":
            handler = partial(_diurnal, locust)
        else:
            handler = partial(_bursty, locust, rng)
        parsed.append((phase_pattern, phase_duration, locust, handler))
    windows = workload_phase_windows(phases)
    if windows[-1]["end_offset_seconds"] != duration_seconds:
        raise ValueError("phase durations must sum to workload duration")

    points: list[ShapePoint] = []
    total_seconds = warmup_seconds + duration_seconds + continuation_seconds
    for elapsed in range(0, total_seconds + 1, sampling_interval_seconds):
        profile_elapsed = max(0, elapsed - warmup_seconds)
        phase_start = 0
        selected = parsed[0]
        for candidate in parsed:
            selected = candidate
            if profile_elapsed < phase_start + candidate[1]:
                break
            phase_start += candidate[1]
        pattern, phase_duration, locust, handler = selected
        phase_elapsed = max(0, profile_elapsed - phase_start)
        if profile_elapsed >= duration_seconds and selected is parsed[-1]:
            if workload.get("continuation_mode", "natural") == "hold_endpoint":
                phase_elapsed = phase_duration
            elif continuation_seconds and pattern in {"gradual-ramp", "flash-crowd"}:
                phase_elapsed = phase_duration
        users = handler(phase_elapsed)
        points.append(ShapePoint(elapsed, max(1, users), int(locust["spawn_rate"])))
    return points


def generate_shape(
    workload: dict[str, Any],
    seed: int,
    sampling_interval_seconds: int,
    warmup_seconds: int = 0,
    duration_seconds: int | None = None,
    capacity_users: int | None = None,
    continuation_seconds: int = 0,
) -> list[ShapePoint]:
    pattern = workload["pattern"]
    duration = int(workload["duration_seconds"] if duration_seconds is None else duration_seconds)
    if duration <= 0:
        raise ValueError("workload duration must be positive")
    if isinstance(continuation_seconds, bool) or not isinstance(continuation_seconds, int) or continuation_seconds < 0:
        raise ValueError("continuation_seconds must be a non-negative integer")
    if pattern == "multi-phase":
        continuation_mode = workload.get("continuation_mode", "natural")
        if continuation_mode not in {"natural", "hold_endpoint"}:
            raise ValueError("multi-phase continuation_mode must be `natural` or `hold_endpoint`")
        return _multi_phase_shape(
            workload,
            seed,
            sampling_interval_seconds,
            warmup_seconds,
            duration,
            continuation_seconds,
            capacity_users,
        )
    locust = _resolved_locust(workload, capacity_users)
    spawn_rate = int(locust["spawn_rate"])
    rng = random.Random(seed)
    points: list[ShapePoint] = []
    handlers = {
        "gradual-ramp": partial(_gradual_ramp, locust, duration=duration),
        "flash-crowd": partial(_flash_crowd, locust),
        "diurnal": partial(_diurnal, locust),
        "bursty": partial(_bursty, locust, rng),
    }

    total_duration = warmup_seconds + duration + continuation_seconds
    for elapsed in range(0, total_duration + 1, sampling_interval_seconds):
        profile_elapsed = max(0, elapsed - warmup_seconds)
        # The forecast-outcome tail supplies future observations without
        # changing the registered measurement profile. Hold endpoint-defined
        # ramp/flash loads steady; periodic profiles keep cycling naturally.
        handler_elapsed = (
            min(profile_elapsed, duration)
            if continuation_seconds and pattern in {"gradual-ramp", "flash-crowd"}
            else profile_elapsed
        )
        try:
            users = handlers[pattern](handler_elapsed)
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
