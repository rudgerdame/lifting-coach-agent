"""Load workout and recovery data from CSV exports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ingestion.schema import RecoveryDaily, WorkoutSet

LBS_TO_KG = 0.453592
APPLE_SLEEP_COL = "Sleep Analysis [Total] (hr)"
APPLE_CALORIES_COL = "Dietary Energy (kcal)"
APPLE_PROTEIN_COL = "Protein (g)"
APPLE_CARBS_COL = "Carbohydrates (g)"
APPLE_WEIGHT_COL = "Weight (lbs)"
APPLE_RESTING_HR_COL = "Resting Heart Rate (bpm)"


def load_fitbod_csv(path: Path) -> list[WorkoutSet]:
    """Parse Fitbod WorkoutExport.csv into normalized WorkoutSet records."""
    df = pd.read_csv(path)
    required = {"Date", "Exercise", "Reps", "Weight(kg)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fitbod CSV missing columns: {missing}")

    sets: list[WorkoutSet] = []
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # One Fitbod session = one timestamp shared by all sets in that workout.
    for session_ts, session_df in df.groupby("Date"):
        session_date = session_ts.date()
        session_id = str(session_ts)
        for set_num, (_, row) in enumerate(session_df.iterrows(), start=1):
            is_warmup = bool(row.get("isWarmup", False))
            sets.append(
                WorkoutSet(
                    session_id=session_id,
                    timestamp=row["Date"].to_pydatetime(),
                    exercise=str(row["Exercise"]).strip(),
                    set_number=set_num,
                    reps=int(row["Reps"]),
                    weight_kg=float(row["Weight(kg)"]),
                    is_warmup=is_warmup,
                    notes=str(row["Note"]) if pd.notna(row.get("Note")) else None,
                )
            )
    return sets


def workout_sets_to_dataframe(sets: list[WorkoutSet]) -> pd.DataFrame:
    """Convert WorkoutSet records to a DataFrame for the feature pipeline."""
    rows = [s.model_dump(mode="json") for s in sets]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def aggregate_apple_health_file(path: Path) -> dict:
    """Aggregate one hourly Apple Health export file to daily recovery values."""
    df = pd.read_csv(path)
    if "Date/Time" not in df.columns:
        raise ValueError(f"{path.name}: missing Date/Time column")

    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    day = df["Date/Time"].dt.date.iloc[0]

    row: dict = {"date": day}
    if APPLE_SLEEP_COL in df.columns:
        sleep = df[APPLE_SLEEP_COL].dropna()
        row["sleep_hours"] = float(sleep.max()) if len(sleep) else None
    if APPLE_CALORIES_COL in df.columns:
        row["calories_kcal"] = float(df[APPLE_CALORIES_COL].fillna(0).sum())
    if APPLE_PROTEIN_COL in df.columns:
        row["protein_g"] = float(df[APPLE_PROTEIN_COL].fillna(0).sum())
    if APPLE_CARBS_COL in df.columns:
        row["carbs_g"] = float(df[APPLE_CARBS_COL].fillna(0).sum())
    if APPLE_WEIGHT_COL in df.columns:
        weight = df[APPLE_WEIGHT_COL].dropna()
        row["bodyweight_kg"] = float(weight.iloc[-1] * LBS_TO_KG) if len(weight) else None
    if APPLE_RESTING_HR_COL in df.columns:
        rhr = df[APPLE_RESTING_HR_COL].dropna()
        row["resting_hr_bpm"] = float(rhr.mean()) if len(rhr) else None
    return row


def aggregate_apple_health_dir(health_dir: Path) -> pd.DataFrame:
    """
    Aggregate all Apple Health daily CSV files under health_dir.

    Maps:
      Sleep Analysis [Total] (hr)  → sleep_hours (daily max)
      Dietary Energy (kcal)        → calories_kcal (daily sum)
      Protein (g)                  → protein_g (daily sum)
      Carbohydrates (g)            → carbs_g (daily sum)
      Weight (lbs)                 → bodyweight_kg (last reading, converted)
      Resting Heart Rate (bpm)     → resting_hr_bpm (daily mean of readings)
    """
    rows: list[dict] = []
    for path in sorted(health_dir.glob("*.csv")):
        try:
            rows.append(aggregate_apple_health_file(path))
        except (ValueError, pd.errors.EmptyDataError) as exc:
            print(f"[loaders] skipping {path.name}: {exc}")

    if not rows:
        raise FileNotFoundError(f"No usable Apple Health CSV files in {health_dir}")

    recovery = pd.DataFrame(rows).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return recovery.reset_index(drop=True)


def load_recovery_csv(path: Path) -> list[RecoveryDaily]:
    """Load daily recovery CSV with columns: date, sleep_hours, calories_kcal."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    records: list[RecoveryDaily] = []
    for _, row in df.iterrows():
        records.append(
            RecoveryDaily(
                date=row["date"],
                sleep_hours=_optional_float(row, "sleep_hours"),
                calories_kcal=_optional_float(row, "calories_kcal"),
                bodyweight_kg=_optional_float(row, "bodyweight_kg"),
            )
        )
    return records


def _optional_float(row: pd.Series, col: str) -> float | None:
    if col not in row.index or pd.isna(row[col]):
        return None
    return float(row[col])


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and preview workout CSV")
    parser.add_argument("--workouts", type=Path, help="Fitbod WorkoutExport.csv")
    parser.add_argument(
        "--gravityos-dir",
        type=Path,
        default=os.environ.get("GRAVITYOS_DATA_DIR"),
        help="Gravity OS Data directory (Fitbod + Apple Health)",
    )
    args = parser.parse_args()

    if args.workouts:
        sets = load_fitbod_csv(args.workouts)
        print(f"Loaded {len(sets)} sets from {args.workouts}")
        return

    if args.gravityos_dir:
        data_dir = Path(args.gravityos_dir)
        fitbod = data_dir / "Fitbod" / "WorkoutExport.csv"
        health = data_dir / "Apple Health Daily"
        sets = load_fitbod_csv(fitbod)
        recovery = aggregate_apple_health_dir(health)
        print(f"Loaded {len(sets)} sets from {fitbod}")
        print(f"Loaded {len(recovery)} recovery days from {health}")
        return

    parser.error("Provide --workouts or --gravityos-dir")


if __name__ == "__main__":
    main()
