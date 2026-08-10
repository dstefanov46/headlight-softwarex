#!/usr/bin/env python3
"""Parse repository YAML files that are not Helm templates or generated results."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "results", ".venv", "venv", "env"}


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    text = path.read_text(encoding="utf-8")
    if "{{" in text and "}}" in text:
        return True
    return False


def main() -> int:
    failures: list[str] = []
    checked = 0
    for path in sorted(REPO_ROOT.rglob("*.yaml")):
        relative = path.relative_to(REPO_ROOT)
        if should_skip(relative):
            continue
        try:
            list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            checked += 1
        except yaml.YAMLError as exc:
            failures.append(f"{relative}: {exc}")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"validated YAML files: {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
