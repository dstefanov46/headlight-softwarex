#!/usr/bin/env python3
"""Train the graph-free LSTM with Headlight's identical frozen data contract."""

from __future__ import annotations

import sys
from pathlib import Path


AI_MODULE = Path(__file__).resolve().parents[1] / "ai-module"
sys.path.insert(0, str(AI_MODULE))

from headlight.training import main as training_main


if __name__ == "__main__":
    raise SystemExit(training_main(["--controller", "lstm", *sys.argv[1:]]))
