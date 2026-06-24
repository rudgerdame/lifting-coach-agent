"""Load trained readiness model and predict performance_delta for an exercise-session.

``performance_delta`` (kg) = top_set_e1rm_kg − e1rm_trend, where ``e1rm_trend`` is the
mean top-set e1RM from the prior **3 sessions of the same exercise** (not the last
workout alone). Positive = likely above recent trend; negative = likely below.

See ``docs/feature-engineering.md`` (section ``performance_delta``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.pipeline import load_features, load_gravityos_features

ARTIFACTS_DIR = Path("models/artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lgb_readiness.pkl"
META_PATH = ARTIFACTS_DIR / "model_meta.pkl"
CATEGORICAL_COLS = ("exercise", "muscle_group", "split")

# Features most often linked to readiness in eval reports; surfaced in CLI output.
KEY_DRIVER_COLS = [
    "sleep_lag_1d",
    "sleep_deviation",
    "resting_hr_lag_1d",
    "acwr",
    "volume_trailing_7d",
    "days_since_last_session",
    "deload_flag",
    "training_days_trailing_7d",
]


@dataclass(frozen=True)
class ReadinessPrediction:
    """Predicted kg delta vs prior-3-session same-exercise e1RM trend (not vs last session)."""

    exercise: str
    session_date: str
    muscle_group: str
    split: str
    performance_delta_kg: float  # top_set e1RM minus 3-session rolling trend
    band: str
    key_drivers: dict[str, float | int | str | None]


def _as_category(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).astype("category")


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in feature_cols:
        if col not in CATEGORICAL_COLS and col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = _as_category(X[col])
    return X


def _band(delta_kg: float) -> str:
    if delta_kg < -1.5:
        return "below_trend"
    if delta_kg > 1.5:
        return "above_trend"
    return "at_trend"


def _resolve_exercise(features: pd.DataFrame, exercise: str) -> str:
    names = features["exercise"].astype(str).unique()
    if exercise in names:
        return exercise
    matches = [n for n in names if exercise.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous exercise {exercise!r}; matches: {matches[:5]}")
    raise ValueError(f"Exercise {exercise!r} not found in training history")


def _select_row(
    features: pd.DataFrame,
    exercise: str,
    session_date: str | None,
) -> pd.Series:
    resolved = _resolve_exercise(features, exercise)
    subset = features[features["exercise"] == resolved].copy()
    if subset.empty:
        raise ValueError(f"No sessions for exercise {resolved!r}")

    subset["_session_date"] = pd.to_datetime(subset["session_date"])
    if session_date:
        target = pd.to_datetime(session_date).date()
        row = subset[subset["session_date"] == target]
        if row.empty:
            raise ValueError(f"No session for {resolved!r} on {session_date}")
        return row.iloc[-1]

    return subset.sort_values("_session_date").iloc[-1]


def _key_drivers(row: pd.Series, feature_cols: list[str]) -> dict[str, float | int | str | None]:
    drivers: dict[str, float | int | str | None] = {}
    for col in KEY_DRIVER_COLS:
        if col in feature_cols and col in row.index:
            val = row[col]
            if pd.isna(val):
                drivers[col] = None
            elif isinstance(val, (np.floating, float)):
                drivers[col] = round(float(val), 3)
            elif isinstance(val, (np.integer, int)):
                drivers[col] = int(val)
            else:
                drivers[col] = val
    return drivers


class ReadinessPredictor:
    """Wraps saved LightGBM artifact + feature metadata for inference."""

    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Run: python -m models.train --data-dir data/synthetic"
            )
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        self.model = joblib.load(model_path)
        meta = joblib.load(meta_path)
        self.feature_cols: list[str] = meta["feature_cols"]
        self.target_col: str = meta["target_col"]

    def predict_row(self, row: pd.Series, features: pd.DataFrame | None = None) -> ReadinessPrediction:
        if features is not None:
            X = prepare_features(features.loc[[row.name]], self.feature_cols)
        else:
            X = prepare_features(row.to_frame().T, self.feature_cols)
        delta = float(self.model.predict(X)[0])
        return ReadinessPrediction(
            exercise=str(row["exercise"]),
            session_date=str(row["session_date"]),
            muscle_group=str(row.get("muscle_group", "unknown")),
            split=str(row.get("split", "unknown")),
            performance_delta_kg=round(delta, 2),
            band=_band(delta),
            key_drivers=_key_drivers(row, self.feature_cols),
        )

    def predict(
        self,
        features: pd.DataFrame,
        exercise: str,
        session_date: str | None = None,
    ) -> ReadinessPrediction:
        row = _select_row(features, exercise, session_date)
        return self.predict_row(row, features=features)


def load_feature_matrix(
    *,
    data_dir: Path | None = None,
    gravityos_dir: Path | None = None,
) -> pd.DataFrame:
    if data_dir is not None:
        return load_features(data_dir)
    if gravityos_dir is not None:
        return load_gravityos_features(gravityos_dir)
    raise ValueError("Provide data_dir or gravityos_dir")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict session readiness (performance_delta)")
    parser.add_argument("--data-dir", type=Path, help="Normalized data directory")
    parser.add_argument("--gravityos-dir", type=Path, help="Gravity OS data directory")
    parser.add_argument("--exercise", required=True, help="Exercise name (partial match OK)")
    parser.add_argument("--session-date", help="YYYY-MM-DD (default: most recent session)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    args = parser.parse_args()

    if not args.data_dir and not args.gravityos_dir:
        args.data_dir = Path("data/synthetic")

    features = load_feature_matrix(data_dir=args.data_dir, gravityos_dir=args.gravityos_dir)
    predictor = ReadinessPredictor(model_path=args.model, meta_path=args.meta)
    result = predictor.predict(features, args.exercise, args.session_date)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
