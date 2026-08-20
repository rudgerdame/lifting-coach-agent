"""Shared runtime context for agent tools — data, features, and model handle."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from features.pipeline import load_features
from models.predict import ReadinessPredictor


@dataclass
class CoachContext:
    """
    Lazy-loaded coaching runtime.

    One instance is passed to all agent tools so they share the same data view
    and model without reloading artifacts on every call.

    data_dir: path to a normalized data directory containing
        workout_sets.jsonl and recovery_daily.csv
        (default: data/synthetic, or DATA_DIR env var)
    """

    data_dir: Path = field(default_factory=lambda: Path("data/synthetic"))
    model_path: Path = field(default_factory=lambda: Path("models/artifacts/lgb_readiness.pkl"))
    meta_path: Path = field(default_factory=lambda: Path("models/artifacts/model_meta.pkl"))
    _features: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _workout_sets: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _recovery_daily: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _predictor: ReadinessPredictor | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls) -> CoachContext:
        data_dir = os.environ.get("DATA_DIR", "data/synthetic")
        return cls(data_dir=Path(data_dir))

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            self._features = load_features(self.data_dir)
        return self._features

    @property
    def workout_sets(self) -> pd.DataFrame:
        if self._workout_sets is None:
            path = self.data_dir / "workout_sets.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}")
            self._workout_sets = pd.read_json(path, lines=True)
        return self._workout_sets

    @property
    def recovery_daily(self) -> pd.DataFrame:
        if self._recovery_daily is None:
            path = self.data_dir / "recovery_daily.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}")
            self._recovery_daily = pd.read_csv(path)
        return self._recovery_daily

    @property
    def predictor(self) -> ReadinessPredictor:
        if self._predictor is None:
            self._predictor = ReadinessPredictor(
                model_path=self.model_path,
                meta_path=self.meta_path,
            )
        return self._predictor

    def predict_readiness(
        self,
        exercise: str,
        session_date: str | None = None,
        today: bool = False,
    ):
        """Predict readiness for an exercise.

        ``today=True``: score *today's planned session* using current recovery
        metrics — the primary production path.

        ``today=False`` + ``session_date``: score a specific past session date.
        ``today=False`` + no date: score the most recently logged session
        (useful for back-testing against known outcomes).
        """
        if today:
            return self.predictor.predict_today(
                exercise,
                self.workout_sets,
                self.recovery_daily,
            )
        return self.predictor.predict(self.features, exercise, session_date)
