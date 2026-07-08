"""Baseline models for comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveAtTrend:
    """Predict performance_delta = 0 (session matches recent trend)."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))


class NaiveGlobalMean:
    """Predict constant target mean from the training fold."""

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, y_train: pd.Series) -> "NaiveGlobalMean":
        self.mean_ = float(y_train.mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_, dtype=float)


class NaivePerExerciseMean:
    """Predict per-exercise target mean from the training fold."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.exercise_mean_: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, *, exercise_col: str = "exercise", target_col: str = "performance_delta") -> "NaivePerExerciseMean":
        self.global_mean_ = float(train_df[target_col].mean())
        grouped = train_df.groupby(exercise_col)[target_col].mean()
        self.exercise_mean_ = {str(k): float(v) for k, v in grouped.items()}
        return self

    def predict(self, df: pd.DataFrame, *, exercise_col: str = "exercise") -> np.ndarray:
        if exercise_col not in df.columns:
            return np.full(len(df), self.global_mean_, dtype=float)
        mapped = df[exercise_col].astype(str).map(self.exercise_mean_)
        return mapped.fillna(self.global_mean_).to_numpy(dtype=float)


class MajorityClass:
    """Predict the most frequent readiness class from the training fold."""

    def __init__(self) -> None:
        self.cls_: int = 1  # default to at_trend

    def fit(self, y_codes: np.ndarray) -> "MajorityClass":
        if len(y_codes):
            self.cls_ = int(pd.Series(y_codes).mode().iloc[0])
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        return np.full(n_rows, self.cls_, dtype=int)


class PerExerciseMajorityClass:
    """Predict each exercise's most frequent class; fall back to global majority."""

    def __init__(self) -> None:
        self.global_cls_: int = 1
        self.exercise_cls_: dict[str, int] = {}

    def fit(
        self,
        df: pd.DataFrame,
        y_codes: np.ndarray,
        *,
        exercise_col: str = "exercise",
    ) -> "PerExerciseMajorityClass":
        tmp = pd.DataFrame({"_ex": df[exercise_col].astype(str).to_numpy(), "_y": y_codes})
        if len(tmp):
            self.global_cls_ = int(tmp["_y"].mode().iloc[0])
        modes = tmp.groupby("_ex")["_y"].agg(lambda s: int(s.mode().iloc[0]))
        self.exercise_cls_ = {str(k): int(v) for k, v in modes.items()}
        return self

    def predict(self, df: pd.DataFrame, *, exercise_col: str = "exercise") -> np.ndarray:
        if exercise_col not in df.columns:
            return np.full(len(df), self.global_cls_, dtype=int)
        mapped = df[exercise_col].astype(str).map(self.exercise_cls_)
        return mapped.fillna(self.global_cls_).to_numpy(dtype=int)
