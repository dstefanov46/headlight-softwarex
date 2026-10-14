#!/usr/bin/env python3
"""Build a hash-addressed Headlight profiling dataset from JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai-module" / "ai-module"))

from headlight.observability import (  # noqa: E402
    SchemaValidationError,
    build_profiling_dataset,
    write_profiling_dataset,
)


def _parse_seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed lists must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("seed lists must not be empty")
    return seeds


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaValidationError(f"missing input file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path}: expected a JSON object")
    return value


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise SchemaValidationError(f"missing profiling record file: {path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise SchemaValidationError(f"{path}:{line_number}: profiling record must be an object")
            records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True, help="v2 profiling JSONL input; repeatable")
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--adjacency", type=Path, required=True)
    parser.add_argument("--fitting-seeds", type=_parse_seed_list, required=True)
    parser.add_argument("--calibration-seeds", type=_parse_seed_list, required=True)
    parser.add_argument("--holdout-seeds", type=_parse_seed_list, required=True)
    parser.add_argument("--evaluation-seeds", type=_parse_seed_list, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = build_profiling_dataset(
            _load_jsonl(args.records),
            feature_schema=_load_json(args.feature_schema),
            adjacency=_load_json(args.adjacency),
            fitting_seeds=args.fitting_seeds,
            calibration_seeds=args.calibration_seeds,
            holdout_seeds=args.holdout_seeds,
            evaluation_seeds=args.evaluation_seeds,
        )
        write_profiling_dataset(args.output, dataset)
    except SchemaValidationError as exc:
        print(f"HEADLIGHT_V2_DATASET_ERROR {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "dataset": str(args.output),
                "dataset_hash": dataset["dataset_hash"],
                "feature_schema_hash": dataset["feature_schema_hash"],
                "adjacency_hash": dataset["adjacency_hash"],
                "samples": dataset["sample_count"],
                "excluded": dataset["excluded_record_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
