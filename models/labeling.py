"""Convert continuous ``performance_delta`` into ordinal readiness classes.

The session-readiness classifier predicts whether today's lift will land
**below**, **at**, or **above** your recent trend, instead of a precise kg delta.

Class boundaries are **adaptive per exercise**: a ``below_trend`` day means the
same thing (in standard-deviation terms) for a 200 kg deadlift as for a 12 kg
lateral raise. Boundaries are ``±k · std(performance_delta)`` for that exercise,
with a fallback to the global std when an exercise has too few logged sessions.

Thresholds are **fit on training data only** (per walk-forward fold) so the
label definition never leaks information from the held-out window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Ordinal order matters: index == integer class code.
CLASS_NAMES: tuple[str, str, str] = ("below_trend", "at_trend", "above_trend")
BELOW, AT, ABOVE = 0, 1, 2

DEFAULT_K = 0.5
DEFAULT_MIN_OBS = 6


@dataclass
class DeltaClassLabeler:
    """Fit adaptive per-exercise thresholds, then map deltas to ordinal classes."""

    k: float = DEFAULT_K
    min_obs: int = DEFAULT_MIN_OBS
    target_col: str = "performance_delta"
    exercise_col: str = "exercise"
    global_std_: float = field(default=1.0, init=False)
    exercise_std_: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, df: pd.DataFrame) -> "DeltaClassLabeler":
        deltas = df[self.target_col].astype(float)
        global_std = float(deltas.std(ddof=0))
        self.global_std_ = global_std if global_std > 0 else 1.0

        std_by_ex = df.groupby(self.exercise_col)[self.target_col].std(ddof=0)
        n_by_ex = df.groupby(self.exercise_col)[self.target_col].size()
        self.exercise_std_ = {}
        for ex, std in std_by_ex.items():
            enough = n_by_ex.get(ex, 0) >= self.min_obs
            usable = enough and pd.notna(std) and std > 0
            self.exercise_std_[str(ex)] = float(std) if usable else self.global_std_
        return self

    def threshold_for(self, exercise: str) -> float:
        """Half-width of the ``at_trend`` band (kg) for one exercise."""
        return self.k * self.exercise_std_.get(str(exercise), self.global_std_)

    def _thresholds(self, df: pd.DataFrame) -> np.ndarray:
        std = df[self.exercise_col].astype(str).map(self.exercise_std_)
        std = std.fillna(self.global_std_).to_numpy(dtype=float)
        return self.k * std

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return integer class codes (0=below, 1=at, 2=above) for each row."""
        thr = self._thresholds(df)
        delta = df[self.target_col].to_numpy(dtype=float)
        codes = np.full(len(df), AT, dtype=int)
        codes[delta <= -thr] = BELOW
        codes[delta >= thr] = ABOVE
        return codes

    def thresholds_summary(self) -> dict[str, float]:
        """Per-exercise band half-widths (kg), for reporting/transparency."""
        return {ex: round(self.k * std, 3) for ex, std in sorted(self.exercise_std_.items())}


def class_name(code: int) -> str:
    return CLASS_NAMES[int(code)]
