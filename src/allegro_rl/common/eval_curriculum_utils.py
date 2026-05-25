from __future__ import annotations

from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def start_k_to_global_steps(start_k: float, curriculum_total_steps: int) -> int:
    return int(clamp01(start_k) * int(curriculum_total_steps))


def force_env_curriculum(env: Any, start_k: float, curriculum_total_steps: int | None = None) -> int:
    total = curriculum_total_steps
    if total is None:
        total = getattr(getattr(env, "cfg", None), "curriculum_total_steps", 1)
    steps = start_k_to_global_steps(start_k, int(total))
    if hasattr(env, "global_steps"):
        env.global_steps = steps
    return steps
