"""Load trained readiness models and predict session readiness for an exercise-session.

Primary output is a **readiness class** (``band``): ``below_trend`` / ``at_trend`` /
``above_trend`` vs the mean top-set e1RM of the prior **3 sessions of the same exercise**
(not the last workout alone), with per-class probabilities in ``class_probs``.

A secondary regression head still returns ``performance_delta_kg`` (kg) =
``top_set_e1rm_kg − e1rm_trend`` as a rough magnitude estimate. When the classifier
artifact is absent, ``band`` falls back to thresholding that delta at ±1.5 kg.

See ``docs/feature-engineering.md`` (sections ``Readiness class`` and ``performance_delta``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.pipeline import load_features
from models.labeling import CLASS_NAMES

ARTIFACTS_DIR = Path("models/artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lgb_readiness.pkl"
META_PATH = ARTIFACTS_DIR / "model_meta.pkl"
CLF_MODEL_PATH = ARTIFACTS_DIR / "lgb_readiness_clf.pkl"
CLF_META_PATH = ARTIFACTS_DIR / "clf_meta.pkl"
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
    """Predicted kg delta vs prior-3-session same-exercise e1RM trend (not vs last session).

    ``band`` is the readiness class. When the classifier artifact is present it is the
    classifier's argmax (``class_label``) and ``class_probs`` holds **calibrated**
    per-class probabilities (Platt/sigmoid); otherwise it falls back to thresholding the regression
    ``performance_delta_kg`` at ±1.5 kg and the class fields are ``None``.
    """

    exercise: str
    session_date: str
    muscle_group: str
    split: str
    performance_delta_kg: float  # top_set e1RM minus 3-session rolling trend
    band: str
    key_drivers: dict[str, float | int | str | None]
    class_label: str | None = None
    class_probs: dict[str, float] | None = None
    class_confidence: float | None = None
    prediction_source: str = "regression_band"


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
    """Wraps saved LightGBM artifacts + feature metadata for inference.

    Loads the regression model (kg delta) always, and the readiness classifier
    (below/at/above trend) when its artifact is present. When the classifier is
    available it becomes the primary ``band``; the regression delta is still
    returned for the magnitude estimate.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        meta_path: Path = META_PATH,
        clf_model_path: Path = CLF_MODEL_PATH,
        clf_meta_path: Path = CLF_META_PATH,
    ) -> None:
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

        # Classifier is optional for backward compatibility with older artifacts.
        self.clf = None
        self.clf_class_names: list[str] = list(CLASS_NAMES)
        if clf_model_path.exists() and clf_meta_path.exists():
            self.clf = joblib.load(clf_model_path)
            clf_meta = joblib.load(clf_meta_path)
            self.clf_class_names = clf_meta.get("class_names", list(CLASS_NAMES))

    def _class_proba(self, X: pd.DataFrame) -> dict[str, float]:
        """Per-class probabilities aligned to (below, at, above) order."""
        proba = self.clf.predict_proba(X)[0]
        aligned = {name: 0.0 for name in self.clf_class_names}
        for col_idx, cls in enumerate(self.clf.classes_):
            aligned[self.clf_class_names[int(cls)]] = round(float(proba[col_idx]), 4)
        return aligned

    def predict_row(self, row: pd.Series, features: pd.DataFrame | None = None) -> ReadinessPrediction:
        if features is not None:
            X = prepare_features(features.loc[[row.name]], self.feature_cols)
        else:
            X = prepare_features(row.to_frame().T, self.feature_cols)
        delta = float(self.model.predict(X)[0])

        class_label: str | None = None
        class_probs: dict[str, float] | None = None
        class_confidence: float | None = None
        if self.clf is not None:
            class_probs = self._class_proba(X)
            class_label = max(class_probs, key=class_probs.get)
            class_confidence = class_probs[class_label]
            band = class_label
            source = "classifier"
        else:
            band = _band(delta)
            source = "regression_band"

        return ReadinessPrediction(
            exercise=str(row["exercise"]),
            session_date=str(row["session_date"]),
            muscle_group=str(row.get("muscle_group", "unknown")),
            split=str(row.get("split", "unknown")),
            performance_delta_kg=round(delta, 2),
            band=band,
            key_drivers=_key_drivers(row, self.feature_cols),
            class_label=class_label,
            class_probs=class_probs,
            class_confidence=class_confidence,
            prediction_source=source,
        )

    def predict(
        self,
        features: pd.DataFrame,
        exercise: str,
        session_date: str | None = None,
    ) -> ReadinessPrediction:
        row = _select_row(features, exercise, session_date)
        return self.predict_row(row, features=features)

    def predict_today(
        self,
        exercise: str,
        workout_sets: pd.DataFrame,
        recovery_daily: pd.DataFrame,
        today: "pd.Timestamp | None" = None,
    ) -> ReadinessPrediction:
        """Predict readiness for *today's* planned session using current recovery metrics.

        Constructs a synthetic pre-workout row from today's recovery data and the
        load history up to now — no logged session for today is required.
        This is the primary production path: "how will my next workout go?"
        """
        from features.pipeline import build_today_row

        row = build_today_row(exercise, workout_sets, recovery_daily, today=today)
        return self.predict_row(row)


def load_feature_matrix(data_dir: Path) -> pd.DataFrame:
    return load_features(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict session readiness class (+ magnitude)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"), help="Normalized data directory")
    parser.add_argument("--exercise", required=True, help="Exercise name (partial match OK)")
    parser.add_argument("--session-date", help="YYYY-MM-DD (default: most recent session)")
    parser.add_argument(
        "--today",
        action="store_true",
        default=True,
        help="Score today's planned session using current recovery (default; primary production path)",
    )
    parser.add_argument(
        "--historical",
        dest="today",
        action="store_false",
        help="Score the most recently logged session instead of today",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    args = parser.parse_args()

    predictor = ReadinessPredictor(model_path=args.model, meta_path=args.meta)

    if args.today and not args.session_date:
        import pandas as pd
        sets = pd.read_json(args.data_dir / "workout_sets.jsonl", lines=True)
        recovery = pd.read_csv(args.data_dir / "recovery_daily.csv")
        result = predictor.predict_today(args.exercise, sets, recovery)
    else:
        features = load_feature_matrix(args.data_dir)
        result = predictor.predict(features, args.exercise, args.session_date)

    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
