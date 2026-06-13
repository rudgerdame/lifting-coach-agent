"""Build session-level feature matrix from normalized workout + recovery data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from features.anomaly import (
    CALORIES_MIN_KCAL,
    CARBS_MIN_G,
    PROTEIN_MIN_G,
    SLEEP_MIN_HOURS,
    hard_floor_anomalies,
    impute_with_trailing_valid_mean,
    mad_lower_tail_anomalies,
)
from features.e1rm import epley_e1rm, volume_load_kg
from features.exercise_map import infer_muscle_group, infer_split

DELOAD_VOLUME_RATIO = 0.60
CONTINUITY_DROP_PCT = 0.20  # exclude exercise-session from train/eval

# Recovery columns: (value_col, imputed_flag_col, hard_floor)
RECOVERY_SIGNALS: list[tuple[str, str, float]] = [
    ("sleep_hours", "sleep_imputed", SLEEP_MIN_HOURS),
    ("calories_kcal", "calories_imputed", CALORIES_MIN_KCAL),
    ("protein_g", "protein_imputed", PROTEIN_MIN_G),
    ("carbs_g", "carbs_imputed", CARBS_MIN_G),
]


def impute_bad_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalous recovery readings and impute from prior valid history.

    Once flagged, an anomaly is:
      1. Recorded in an *_imputed flag (1 = was anomalous)
      2. Replaced with the trailing 7-day mean of valid prior days
      3. Never dropped — the calendar row is kept for lag/trailing features

    Detection: hard floor + personalized MAD lower-tail (see features/anomaly.py).
    """
    out = recovery.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    log_parts: list[str] = []
    for value_col, imputed_col, floor in RECOVERY_SIGNALS:
        if value_col not in out.columns:
            continue
        raw = out[value_col]
        hard = hard_floor_anomalies(raw, floor)
        mad = mad_lower_tail_anomalies(raw, exclude=hard)
        anomaly = hard | mad
        out[imputed_col] = anomaly.astype(int)
        # Past-only imputation: if no valid prior history exists, keep NaN rather
        # than backfilling with a global (future-informed) mean.
        out[value_col] = impute_with_trailing_valid_mean(raw, anomaly, fallback=None)
        if anomaly.any():
            log_parts.append(
                f"{int(anomaly.sum())} {value_col} "
                f"({int(hard.sum())} hard, {int(mad.sum())} MAD)"
            )

    if log_parts:
        print(f"[pipeline] imputed: {'; '.join(log_parts)}")

    out["date"] = out["date"].dt.date
    return out


def _assign_set_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        e1rm=df.apply(lambda r: epley_e1rm(r["weight_kg"], int(r["reps"])), axis=1),
        volume=df.apply(lambda r: volume_load_kg(int(r["reps"]), r["weight_kg"]), axis=1),
    )


def _add_load_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """ACWR, trailing volume, deload, calendar + week-over-week temporal features."""
    sessions_sorted = sessions.sort_values(["exercise", "session_date"]).copy()
    sessions_sorted["session_date_dt"] = pd.to_datetime(sessions_sorted["session_date"])

    trailing_rows = []
    for _, grp in sessions_sorted.groupby("exercise"):
        grp = grp.set_index("session_date_dt").sort_index()
        # Exclude the current session — knowable pre-workout from prior logs only.
        grp["volume_trailing_7d"] = grp["volume_load_kg"].rolling("7D", min_periods=1).sum().shift(1)
        grp["volume_trailing_28d"] = grp["volume_load_kg"].rolling("28D", min_periods=1).sum().shift(1)
        chronic = grp["volume_trailing_28d"] / 4.0
        grp["acwr"] = grp["volume_trailing_7d"] / chronic.replace(0, np.nan)
        grp["days_since_last_session"] = grp.index.to_series().diff().dt.days
        trailing_rows.append(grp.reset_index())

    out = pd.concat(trailing_rows, ignore_index=True)
    out["session_date"] = out["session_date_dt"].dt.date
    out = out.drop(columns=["session_date_dt"])

    out["day_of_week"] = pd.to_datetime(out["session_date"]).dt.dayofweek

    out["week"] = pd.to_datetime(out["session_date"]).dt.to_period("W")
    weekly = (
        out.groupby(["exercise", "week"], as_index=False)["volume_load_kg"]
        .sum()
        .rename(columns={"volume_load_kg": "weekly_volume"})
        .sort_values(["exercise", "week"])
    )
    # Completed weeks only — current week volume is unknown pre-workout.
    weekly["prev_week"] = weekly.groupby("exercise")["weekly_volume"].shift(1)
    weekly["prev_prev_week"] = weekly.groupby("exercise")["weekly_volume"].shift(2)
    weekly["volume_wow_pct"] = (
        (weekly["prev_week"] - weekly["prev_prev_week"])
        / weekly["prev_prev_week"].replace(0, np.nan)
    )
    weekly["weekly_volume_4w_mean"] = weekly.groupby("exercise")["prev_week"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    weekly["deload_flag"] = (
        weekly["prev_week"] < DELOAD_VOLUME_RATIO * weekly["weekly_volume_4w_mean"]
    ).astype(int)

    out = out.merge(
        weekly[["exercise", "week", "deload_flag", "volume_wow_pct"]],
        on=["exercise", "week"],
        how="left",
    )
    return out.drop(columns=["week"])


def _add_prior_session_frequency(
    sessions: pd.DataFrame,
    group_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Count prior exercise sessions in the last 10 calendar days for ``group_col``.

    Each exercise log row counts separately. Only strictly prior calendar dates
    count (no same-day leakage) — valid for pre-workout scoring.
    """
    out = sessions.copy()
    counts = pd.Series(0, index=out.index, dtype=int)

    for _, grp_idx in out.groupby(group_col, sort=False).groups.items():
        sub = out.loc[grp_idx].copy()
        sub["_session_date"] = pd.to_datetime(sub["session_date"]).dt.normalize()

        by_date = sub.groupby("_session_date", sort=True).size()
        date_index = by_date.index.to_numpy(dtype="datetime64[ns]")
        date_counts = by_date.to_numpy(dtype=int)

        prior_count_by_date: dict[pd.Timestamp, int] = {}
        left = 0
        running_total = 0
        for i, current_date in enumerate(date_index):
            window_start = current_date - np.timedelta64(10, "D")
            while left < i and date_index[left] < window_start:
                running_total -= int(date_counts[left])
                left += 1
            # Record count of strictly prior rows in [current_date - 10d, current_date).
            prior_count_by_date[pd.Timestamp(current_date)] = int(running_total)
            running_total += int(date_counts[i])

        counts.loc[sub.index] = sub["_session_date"].map(prior_count_by_date).to_numpy(dtype=int)

    out[out_col] = counts
    return out


def _add_global_schedule_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Global gym-calendar features (any exercise), pre-workout safe.

    training_days_trailing_7d: distinct workout dates in the prior 7 calendar days
    (session day excluded).
    days_since_last_workout: calendar days since the previous gym day (any lift).
    """
    out = sessions.copy()
    gym_dates = (
        pd.to_datetime(out["session_date"])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if gym_dates.empty:
        out["training_days_trailing_7d"] = np.nan
        out["days_since_last_workout"] = np.nan
        return out

    cal = pd.DataFrame({"date": pd.date_range(gym_dates.min(), gym_dates.max(), freq="D")})
    cal["trained"] = cal["date"].isin(gym_dates).astype(int)
    cal["training_days_trailing_7d"] = cal["trained"].rolling(7, min_periods=1).sum().shift(1)

    gym_only = cal.loc[cal["trained"] == 1, "date"].reset_index(drop=True)
    workout_gap = gym_only.diff().dt.days
    gap_by_date = dict(zip(gym_only, workout_gap))

    session_day = pd.to_datetime(out["session_date"]).dt.normalize()
    out["training_days_trailing_7d"] = session_day.map(
        cal.set_index("date")["training_days_trailing_7d"]
    )
    out["days_since_last_workout"] = session_day.map(gap_by_date)
    return out


def _add_continuity_flags(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Flag exercise-sessions where max working weight drops sharply vs the prior
    log for the same exercise name (likely gym/equipment change).

    Only that exercise-session is flagged — other exercises on the same day are kept.
    """
    out = sessions.sort_values(["exercise", "session_date"]).copy()
    if "max_working_weight_kg" not in out.columns:
        out["continuity_break"] = 0
        out["weight_drop_pct"] = np.nan
        return out

    out["prior_max_weight_kg"] = out.groupby("exercise")["max_working_weight_kg"].shift(1)
    out["weight_drop_pct"] = (
        (out["max_working_weight_kg"] - out["prior_max_weight_kg"])
        / out["prior_max_weight_kg"].replace(0, np.nan)
    )
    out["continuity_break"] = (
        out["prior_max_weight_kg"].notna() & (out["weight_drop_pct"] < -CONTINUITY_DROP_PCT)
    ).astype(int)
    return out


def _add_recovery_features(recovery_daily: pd.DataFrame) -> pd.DataFrame:
    recovery = impute_bad_recovery(recovery_daily)
    recovery = recovery.sort_values("date")

    def _rolling_block(col: str, prefix: str) -> None:
        if col not in recovery.columns:
            return
        recovery[f"{prefix}_trailing_7"] = recovery[col].rolling(7, min_periods=1).mean()
        recovery[f"{prefix}_trailing_28"] = recovery[col].rolling(28, min_periods=1).mean()
        acute = recovery[col].rolling(3, min_periods=1).mean()
        recovery[f"{prefix}_deviation"] = acute - recovery[f"{prefix}_trailing_28"]

    _rolling_block("sleep_hours", "sleep")
    if "sleep_trailing_7" in recovery.columns:
        # 7-day rolling mean; values are in hours (not a 7-hour window).
        recovery["sleep_trailing_7d"] = recovery["sleep_trailing_7"]
    # Session-date sleep bucket (Apple Health day total for workout date).
    recovery["sleep_lag_1d"] = recovery["sleep_hours"]
    recovery["sleep_lag_2d"] = recovery["sleep_hours"].shift(1)

    _rolling_block("calories_kcal", "calories")
    if "calories_trailing_7" in recovery.columns:
        recovery["calories_trailing_7"] = recovery["calories_trailing_7"]

    _rolling_block("protein_g", "protein")
    _rolling_block("carbs_g", "carbs")

    if "resting_hr_bpm" in recovery.columns:
        recovery["resting_hr_bpm"] = recovery["resting_hr_bpm"].ffill()
        _rolling_block("resting_hr_bpm", "resting_hr")
        recovery["resting_hr_lag_1d"] = recovery["resting_hr_bpm"]

    if "bodyweight_kg" in recovery.columns:
        recovery["bodyweight_kg"] = recovery["bodyweight_kg"].ffill()
        recovery["bodyweight_trailing_7"] = recovery["bodyweight_kg"].rolling(7, min_periods=1).mean()
        recovery["bodyweight_lag_1d"] = recovery["bodyweight_kg"].shift(1)

    return recovery


# Recovery fields that include same-day readings — shift to prior day for pre-workout use.
_RECOVERY_PREWORKOUT_SHIFT = (
    "sleep_trailing_7d",
    "calories_trailing_7",
    "protein_trailing_7",
    "carbs_trailing_7",
    "sleep_deviation",
    "calories_deviation",
    "protein_deviation",
    "carbs_deviation",
    "resting_hr_trailing_7",
    "resting_hr_deviation",
    "bodyweight_kg",
    "bodyweight_trailing_7",
    "sleep_imputed",
    "calories_imputed",
    "protein_imputed",
    "carbs_imputed",
)


def _recovery_for_preworkout(recovery_daily: pd.DataFrame) -> pd.DataFrame:
    """Recovery features as known before today's gym session."""
    recovery = _add_recovery_features(recovery_daily)
    for col in _RECOVERY_PREWORKOUT_SHIFT:
        if col in recovery.columns:
            recovery[col] = recovery[col].shift(1)
    return recovery


def build_session_features(
    workout_sets: pd.DataFrame,
    recovery_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate sets to session-level features.

    Expected workout_sets columns: timestamp, exercise, reps, weight_kg, is_warmup
    Expected recovery_daily columns: date, sleep_hours, calories_kcal, protein_g, carbs_g,
    optional resting_hr_bpm, bodyweight_kg
    """
    df = workout_sets.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["session_date"] = df["timestamp"].dt.date

    all_sets = _assign_set_metrics(df)
    working = all_sets[~all_sets["is_warmup"].fillna(False)]

    sessions = (
        working.groupby(["session_date", "exercise"], as_index=False)
        .agg(
            top_set_e1rm_kg=("e1rm", "max"),
            max_working_weight_kg=("weight_kg", "max"),
            volume_load_kg=("volume", "sum"),
            n_working_sets=("reps", "count"),
        )
    )

    sessions_all = (
        all_sets.groupby(["session_date", "exercise"], as_index=False)
        .agg(
            volume_load_all_kg=("volume", "sum"),
            n_sets_all=("reps", "count"),
        )
    )
    sessions = sessions.merge(sessions_all, on=["session_date", "exercise"], how="left")

    if "muscle_group" in df.columns and df["muscle_group"].notna().any():
        meta = df.groupby(["session_date", "exercise"], as_index=False).agg(
            muscle_group=("muscle_group", "first")
        )
        meta["muscle_group"] = meta["muscle_group"].fillna("unknown")
    else:
        meta = df.groupby(["session_date", "exercise"], as_index=False)["exercise"].first()
        meta["muscle_group"] = meta["exercise"].map(infer_muscle_group)
    sessions = sessions.merge(meta, on=["session_date", "exercise"], how="left")
    sessions["split"] = sessions["muscle_group"].map(infer_split)
    sessions = _add_continuity_flags(sessions)

    sessions = _add_load_features(sessions)
    sessions = _add_prior_session_frequency(
        sessions, "muscle_group", "muscle_group_sessions_trailing_10d"
    )
    sessions = _add_prior_session_frequency(
        sessions, "split", "split_sessions_trailing_10d"
    )
    sessions = _add_global_schedule_features(sessions)

    recovery = _recovery_for_preworkout(recovery_daily)
    sessions = sessions.merge(
        recovery,
        left_on="session_date",
        right_on="date",
        how="left",
    )
    return sessions


def load_features(data_dir: Path) -> pd.DataFrame:
    """Load workout_sets.jsonl + recovery_daily.csv from a normalized data directory."""
    sets_path = data_dir / "workout_sets.jsonl"
    recovery_path = data_dir / "recovery_daily.csv"
    if not sets_path.exists():
        raise FileNotFoundError(f"Missing {sets_path}")
    if not recovery_path.exists():
        raise FileNotFoundError(f"Missing {recovery_path}")
    sets = pd.read_json(sets_path, lines=True)
    recovery = pd.read_csv(recovery_path)
    return build_session_features(sets, recovery)


def load_gravityos_features(gravityos_data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load Fitbod + Apple Health from a Gravity OS data directory (local only).

    Expects:
      {dir}/Fitbod/WorkoutExport.csv
      {dir}/Apple Health Daily/*.csv
    """
    from ingestion.loaders import (
        aggregate_apple_health_dir,
        load_fitbod_csv,
        workout_sets_to_dataframe,
    )

    data_dir = Path(gravityos_data_dir or os.environ["GRAVITYOS_DATA_DIR"])
    fitbod_path = data_dir / "Fitbod" / "WorkoutExport.csv"
    health_dir = data_dir / "Apple Health Daily"

    if not fitbod_path.exists():
        raise FileNotFoundError(f"Missing Fitbod export: {fitbod_path}")
    if not health_dir.is_dir():
        raise FileNotFoundError(f"Missing Apple Health directory: {health_dir}")

    sets = workout_sets_to_dataframe(load_fitbod_csv(fitbod_path))
    recovery = aggregate_apple_health_dir(health_dir)
    return build_session_features(sets, recovery)
