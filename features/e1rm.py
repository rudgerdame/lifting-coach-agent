"""Estimated 1RM and load calculations."""

from __future__ import annotations

MAX_REPS_FOR_E1RM = 15


def epley_e1rm(weight_kg: float, reps: int) -> float:
    """Epley formula: e1RM = weight * (1 + reps/30). Reps capped at 15."""
    if weight_kg <= 0 or reps <= 0:
        return 0.0
    capped = min(reps, MAX_REPS_FOR_E1RM)
    return weight_kg * (1.0 + capped / 30.0)


def volume_load_kg(reps: int, weight_kg: float) -> float:
    """Working-set volume: reps × weight."""
    return max(0, reps) * max(0.0, weight_kg)
