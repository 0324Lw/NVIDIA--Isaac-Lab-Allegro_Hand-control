from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def flat_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in (d or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flat_dict(value, name))
        else:
            val = to_float(value)
            if val is not None:
                out[name] = val

    return out


def make_table(title: str, data: Dict[str, Any], width: int = 118) -> str:
    lines = ["-" * width, f"| {title:<{width - 4}} |", "-" * width]

    if not data:
        lines += [f"| {'<empty>':<{width - 4}} |", "-" * width]
        return "\n".join(lines)

    for key in sorted(data.keys()):
        value = data[key]
        key_s = (key[:72] + "...") if len(key) > 75 else key

        if isinstance(value, float):
            value_s = f"{value:.6e}" if abs(value) > 1e4 or 0 < abs(value) < 1e-3 else f"{value:.6f}"
        else:
            value_s = str(value)

        value_s = (value_s[:38] + "...") if len(value_s) > 41 else value_s
        lines.append(f"| {key_s:<76} | {value_s:>{width - 83}} |")

    lines.append("-" * width)
    return "\n".join(lines)
