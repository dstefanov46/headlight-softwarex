"""Path validation helpers for command-line experiment tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from experiments.validate_configs import REPO_ROOT


PUBLIC_WRITABLE_ROOTS = {
    Path(tempfile.gettempdir()).resolve(strict=False),
    Path(os.sep, "var", "tmp").resolve(strict=False),
    Path(os.sep, "dev", "shm").resolve(strict=False),
}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_repo_path(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a CLI path that must stay inside the repository."""
    candidate = Path(path)
    resolved = candidate.resolve(strict=False) if candidate.is_absolute() else (REPO_ROOT / candidate).resolve(strict=False)
    repo_root = REPO_ROOT.resolve(strict=False)
    if not _is_relative_to(resolved, repo_root):
        raise ValueError(f"path must stay inside repository root {repo_root}: {path}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {resolved}")
    return resolved


def resolve_output_path(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    """Resolve an output path and reject direct writes into shared temp roots."""
    candidate = Path(path)
    resolved = candidate.resolve(strict=False) if candidate.is_absolute() else (base / candidate).resolve(strict=False)
    if resolved.parent in PUBLIC_WRITABLE_ROOTS:
        raise ValueError(f"refusing to write directly in publicly writable directory: {resolved}")
    return resolved
