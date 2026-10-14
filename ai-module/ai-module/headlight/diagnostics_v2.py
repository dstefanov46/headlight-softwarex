"""Bounded, content-addressed diagnostic artifacts; no control or actuation."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class DiagnosticStore:
    def __init__(self, directory: str | Path, *, max_bytes: int = 128 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("diagnostic storage limit must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.used_bytes = sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())

    def _reserve(self, size: int) -> None:
        if self.used_bytes + size > self.max_bytes:
            raise OSError("diagnostic storage limit reached")

    def tensors(self, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
        canonical = hashlib.sha256()
        for name, value in sorted(arrays.items()):
            array = np.ascontiguousarray(value)
            canonical.update(json.dumps([name, str(array.dtype), array.shape]).encode())
            canonical.update(array.tobytes())
        path = self.directory / f"{canonical.hexdigest()}.npz"
        if not path.exists():
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            content = buffer.getvalue()
            self._reserve(len(content))
            temporary = path.with_suffix(".tmp")
            try:
                temporary.write_bytes(content)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            self.used_bytes += len(content)
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        }

    def record(self, value: Mapping[str, Any]) -> None:
        encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        self._reserve(len(encoded))
        with (self.directory / "decisions.jsonl").open("ab") as handle:
            handle.write(encoded)
        self.used_bytes += len(encoded)


def action_lifecycle(*, shadow: bool, changed: bool, accepted: Mapping[str, Any] | None) -> dict[str, Any]:
    requested = bool(changed and not shadow and accepted)
    return {
        "status": ("acknowledged" if accepted.get("acknowledged_at") is not None else "failed") if requested else "not_requested",
        "requested_at": accepted["requested_at"] if requested else None,
        "acknowledged_at": accepted["acknowledged_at"] if requested else None,
        "ready_at": None,
        "readiness_delay_seconds": None,
    }
