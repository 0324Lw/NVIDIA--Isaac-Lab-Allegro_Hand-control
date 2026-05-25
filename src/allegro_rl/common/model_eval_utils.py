from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


def safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def resolve_first_existing(base: str | Path, candidates: Iterable[str]) -> Path:
    base = Path(base).expanduser().resolve()
    if base.is_file():
        return base
    for name in candidates:
        candidate = base / name
        if candidate.exists():
            return candidate
    return base
