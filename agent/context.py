"""Shared runtime context for agent tools — data, features, and model handle."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from features.pipeline import load_features, load_gravityos_features
from models.predict import ReadinessPredictor


@dataclass
class CoachContext:
    """
    Lazy-loaded coaching runtime.

    One instance is passed to all agent tools so they share the same data view
    and model without reloading artifacts on every call.
    """

    data_dir: Path | None = None
    gravityos_dir: Path | None = None
    model_path: Path = Path("models/artifacts/lgb_readiness.pkl")
    meta_path: Path = Path("models/artifacts/model_meta.pkl")
    _features: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _workout_sets: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _recovery_daily: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _predictor: ReadinessPredictor | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls) -> CoachContext:
        gravityos = os.environ.get("GRAVITYOS_DATA_DIR")
        if gravityos:
            return cls(gravityos_dir=Path(gravityos))
        return cls(data_dir=Path("data/synthetic"))

    def _resolve_data_dir(self) -> Path:
        if self.data_dir is not None:
            return self.data_dir
        if self.gravityos_dir is not None:
            raise ValueError("Raw sets/recovery CSV paths require data_dir, not gravityos_dir")
        raise ValueError("Set data_dir or gravityos_dir on CoachContext")

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            if self.data_dir is not None:
                self._features = load_features(self.data_dir)
            elif self.gravityos_dir is not None:
                self._features = load_gravityos_features(self.gravityos_dir)
            else:
                raise ValueError("Set data_dir or gravityos_dir on CoachContext")
        return self._features

    @property
    def workout_sets(self) -> pd.DataFrame:
        if self._workout_sets is None:
            if self.data_dir is not None:
                path = self.data_dir / "workout_sets.jsonl"
                if not path.exists():
                    raise FileNotFoundError(f"Missing {path}")
                self._workout_sets = pd.read_json(path, lines=True)
            elif self.gravityos_dir is not None:
                from ingestion.loaders import load_fitbod_csv, workout_sets_to_dataframe

                fitbod = self.gravityos_dir / "Fitbod" / "WorkoutExport.csv"
                if not fitbod.exists():
                    raise FileNotFoundError(f"Missing {fitbod}")
                self._workout_sets = workout_sets_to_dataframe(load_fitbod_csv(fitbod))
            else:
                raise ValueError("Set data_dir or gravityos_dir on CoachContext")
        return self._workout_sets

    @property
    def recovery_daily(self) -> pd.DataFrame:
        if self._recovery_daily is None:
            data_dir = self._resolve_data_dir()
            path = data_dir / "recovery_daily.csv"
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
    ):
        return self.predictor.predict(self.features, exercise, session_date)
