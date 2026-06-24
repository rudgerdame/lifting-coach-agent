"""Deterministic workout planning helpers (no LLM)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.exercise_map import infer_split
from models.predict import ReadinessPrediction

PPL_CYCLE = ("push", "pull", "legs")
LOWER_MUSCLES = frozenset({"quads", "hamstrings", "glutes", "calves"})


@dataclass(frozen=True)
class ExercisePrescription:
    exercise: str
    muscle_group: str
    working_sets: int
    reps_per_set: int
    weight_kg: float
    load_note: str


@dataclass(frozen=True)
class WorkoutPlan:
    split: str
    inferred_from_days: int
    exercises: list[ExercisePrescription]
    readiness_summary: dict[str, object]
    deload_recommended: bool
    notes: list[str]


def _dominant_split_for_day(day_df: pd.DataFrame) -> str:
    splits = day_df["split"].dropna().astype(str)
    if splits.empty:
        return "unknown"
    return splits.mode().iloc[0]


def infer_next_split(features: pd.DataFrame, lookback_days: int = 5) -> tuple[str, list[str]]:
    """Infer next PPL split from recent gym-day rotation."""
    df = features.copy()
    df["_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
    gym_days = (
        df.groupby("_date", sort=True)
        .apply(_dominant_split_for_day, include_groups=False)
        .reset_index(name="split")
    )
    recent = gym_days.tail(lookback_days)
    day_splits = [s for s in recent["split"].tolist() if s in PPL_CYCLE]
    if not day_splits:
        return "push", []

    last = day_splits[-1]
    next_split = PPL_CYCLE[(PPL_CYCLE.index(last) + 1) % len(PPL_CYCLE)]
    return next_split, day_splits


def _recent_exercises_for_split(features: pd.DataFrame, split: str, max_exercises: int = 5) -> list[str]:
    df = features[features["split"] == split].copy()
    if df.empty:
        return []
    df["_date"] = pd.to_datetime(df["session_date"])
    last_seen = df.groupby("exercise")["_date"].max().sort_values(ascending=False)
    return last_seen.head(max_exercises).index.tolist()


def _median_prescription(sets_df: pd.DataFrame, exercise: str) -> tuple[int, int, float]:
    """Median working sets, reps per set, and weight from recent sessions."""
    df = sets_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["session_date"] = df["timestamp"].dt.date
    working = df[(df["exercise"] == exercise) & (~df["is_warmup"].fillna(False))]
    if working.empty:
        return 3, 10, 0.0

    per_session = (
        working.groupby("session_date")
        .agg(
            n_sets=("reps", "count"),
            median_reps=("reps", "median"),
            median_weight=("weight_kg", "median"),
        )
        .sort_index()
        .tail(3)
    )
    if per_session.empty:
        return 3, 10, 0.0

    return (
        int(round(per_session["n_sets"].median())),
        int(round(per_session["median_reps"].median())),
        round(float(per_session["median_weight"].median()), 1),
    )


def _is_lower_body(muscle_group: str, exercise: str) -> bool:
    if muscle_group.lower() in LOWER_MUSCLES:
        return True
    name = exercise.lower()
    return any(kw in name for kw in ("squat", "deadlift", "leg press", "lunge", "rdl"))


def _adjust_weight(
    weight_kg: float,
    band: str,
    *,
    deload: bool,
    acwr_high: bool,
    lower_body: bool,
) -> tuple[float, str]:
    if weight_kg <= 0:
        return weight_kg, "bodyweight or unknown — no load adjustment"

    if deload or acwr_high:
        w = round(weight_kg * 0.85, 1)
        return w, "deload/elevated ACWR — reduced ~15% [source: coaching_policy.md]"

    if band == "below_trend":
        w = round(weight_kg * 0.93, 1)
        return w, "below_trend — reduced ~7% [model]"

    if band == "above_trend":
        bump = 5.0 if lower_body else 2.5
        return round(weight_kg + bump, 1), f"above_trend — +{bump} kg [model]"

    return weight_kg, "at_trend — match recent working weight [model]"


def build_workout_plan(
    features: pd.DataFrame,
    sets_df: pd.DataFrame,
    *,
    split: str | None = None,
    lookback_days: int = 5,
    max_exercises: int = 5,
    readiness: ReadinessPrediction | None = None,
) -> WorkoutPlan:
    if split is None:
        split, rotation = infer_next_split(features, lookback_days)
    else:
        _, rotation = infer_next_split(features, lookback_days)
        split = split.lower().strip()

    exercises = _recent_exercises_for_split(features, split, max_exercises=max_exercises)
    notes: list[str] = []
    if not exercises:
        notes.append(f"No exercises found in history for split={split!r}.")

    band = readiness.band if readiness else "at_trend"
    drivers = readiness.key_drivers if readiness else {}
    acwr = drivers.get("acwr")
    acwr_high = acwr is not None and float(acwr) > 1.3
    deload_flag = drivers.get("deload_flag")
    deload = bool(deload_flag) or acwr_high

    prescriptions: list[ExercisePrescription] = []
    feat = features.copy()
    feat["_date"] = pd.to_datetime(feat["session_date"])

    for ex in exercises:
        n_sets, reps, weight = _median_prescription(sets_df, ex)
        row = feat[feat["exercise"] == ex].sort_values("_date").iloc[-1]
        muscle = str(row.get("muscle_group", "unknown"))
        if weight <= 0 and "max_working_weight_kg" in row.index:
            weight = float(row["max_working_weight_kg"] or 0)

        adj_weight, load_note = _adjust_weight(
            weight,
            band,
            deload=deload,
            acwr_high=acwr_high,
            lower_body=_is_lower_body(muscle, ex),
        )
        prescriptions.append(
            ExercisePrescription(
                exercise=ex,
                muscle_group=muscle,
                working_sets=max(1, n_sets),
                reps_per_set=max(1, reps),
                weight_kg=adj_weight,
                load_note=load_note,
            )
        )

    readiness_summary: dict[str, object] = {}
    if readiness:
        readiness_summary = {
            "anchor_exercise": readiness.exercise,
            "performance_delta_kg": readiness.performance_delta_kg,
            "band": readiness.band,
            "key_drivers": readiness.key_drivers,
        }

    return WorkoutPlan(
        split=split,
        inferred_from_days=lookback_days,
        exercises=prescriptions,
        readiness_summary=readiness_summary,
        deload_recommended=deload,
        notes=notes,
    )
