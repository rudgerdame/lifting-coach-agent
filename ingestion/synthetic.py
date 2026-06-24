"""Generate synthetic workout + recovery data for demos and CI."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ingestion.schema import RecoveryDaily, WorkoutSet

# Exercise registry: PPL split, (name, muscle, base_kg, std_kg, is_bodyweight, prog_kg_per_week).
# Dumbbell bilateral loads are stored as effective total (per-DB × 2).

_Push = [
    # name,                              muscle,      base_kg  std   bw     prog/wk
    ("Dumbbell Incline Bench Press",    "chest",       54.0,  4.0, False,  0.40),
    ("Barbell Bench Press",             "chest",       70.0,  5.0, False,  0.50),
    ("Hammerstrength Incline Chest Press","chest",     54.0,  4.0, False,  0.30),
    ("Hammerstrength Chest Press",      "chest",       56.0,  4.0, False,  0.30),
    ("Dumbbell Shoulder Press",         "shoulders",   42.0,  3.0, False,  0.30),
    ("Hammerstrength Shoulder Press",   "shoulders",   60.0,  4.0, False,  0.30),
    ("Dumbbell Lateral Raise",          "shoulders",   22.0,  2.0, False,  0.10),
    ("Cable Tricep Pushdown",           "triceps",     40.0,  4.0, False,  0.30),
    ("Dumbbell Tricep Extension",       "triceps",     22.0,  2.0, False,  0.15),
    ("Low Cable Chest Fly",             "chest",       24.0,  2.0, False,  0.10),
    ("Dip",                             "chest",        0.0,  0.0, True,   0.00),
]

_Pull = [
    ("Pull Up",                         "back",         0.0,  0.0, True,   0.00),
    ("Cable Row",                       "back",        60.0,  5.0, False,  0.40),
    ("Barbell Curl",                    "biceps",      27.0,  2.0, False,  0.15),
    ("Incline Dumbbell Curl",           "biceps",      27.0,  2.0, False,  0.15),
    ("Concentration Curl",              "biceps",      22.0,  2.0, False,  0.10),
    ("Machine Bicep Curl",              "biceps",      22.0,  2.0, False,  0.15),
    ("Dumbbell Shrug",                  "back",        90.0,  5.0, False,  0.30),
]

_Legs = [
    ("Leg Extension",                   "quads",       65.0,  5.0, False,  0.30),
    ("Single Leg Leg Extension",        "quads",       45.0,  4.0, False,  0.20),
    ("Lying Hamstrings Curl",           "hamstrings",  34.0,  3.0, False,  0.20),
    ("Standing Leg Curl",               "hamstrings",  36.0,  3.0, False,  0.20),
    ("Barbell Hip Thrust",              "glutes",      80.0,  6.0, False,  0.50),
    ("Standing Barbell Calf Raise",     "calves",      61.0,  4.0, False,  0.15),
    ("Vertical Leg Raise",              "core",         0.0,  0.0, True,   0.00),
]

_DAY_TYPES = [
    ("push", _Push),
    ("pull", _Pull),
    ("legs", _Legs),
]

_DAYS_DEFAULT = 635
_SESSION_HOURS = [9, 10, 14, 15, 17, 18]
_N_EXERCISES_RANGE = (5, 7)
_SESSION_SKIP_PROB = 0.11
_RECOVERY_ROW_OMIT_PROB = 0.30
_RECOVERY_ANOMALY_PROB = 0.055
_RECOVERY_WARMUP_DAYS = 28
_SPARSE_FIELD_PROB = 0.18
_PROGRESSION_SCALE = 1.25
_PROGRESSION_PER_SESSION = 0.35
_PR_BUMP_PROB = 0.06
_PR_BUMP_KG_RANGE = (1.5, 3.0)
_DELOAD_WEEK_EVERY = 8
_READINESS_AR = 0.85
_READINESS_SHOCK_STD = 0.35
_SLEEP_READINESS_COEF = 0.45
_RHR_READINESS_COEF = -1.0
_CALORIES_READINESS_COEF = 65.0
_SLEEP_LAG1_COEF = 1.4
_SLEEP_LAG2_COEF = 0.8
_RHR_TRAIL7_COEF = -0.06
_VOLUME_TRAIL7_COEF = -0.00035
_TRAINING_DAYS_COEF = -0.35
_DAYS_SINCE_WORKOUT_COEF = 0.08
_RESIDUAL_NOISE_STD = 0.38
_WORKING_SETS = 3
_WARMUP_REPS = 8
_WORKING_REP_MIN = 10
_WORKING_REP_MAX = 17
_WARMUP_WEIGHT_FRAC = 0.65


def generate_synthetic(
    days: int = _DAYS_DEFAULT,
    seed: int = 42,
) -> tuple[list[WorkoutSet], list[RecoveryDaily]]:
    """Generate synthetic PPL training logs."""
    rng = random.Random(seed)
    sets: list[WorkoutSet] = []
    recovery: list[RecoveryDaily] = []

    start = date.today() - timedelta(days=days)
    day_type_idx = 0
    exercise_loads: dict[str, float] = {}
    exercise_volume_by_date: dict[str, list[tuple[date, float]]] = {}
    latent_readiness = 0.0
    sleep_history: list[tuple[date, float]] = []
    rhr_history: list[tuple[date, float]] = []
    workout_dates: list[date] = []

    for offset in range(days):
        current = start + timedelta(days=offset)
        weeks_elapsed = offset / 7.0

        latent_readiness = _READINESS_AR * latent_readiness + rng.gauss(0, _READINESS_SHOCK_STD)

        sleep = max(5.0, min(9.5, 7.4 + _SLEEP_READINESS_COEF * latent_readiness + rng.gauss(0, 0.55)))
        calories = max(1800.0, 2600.0 + _CALORIES_READINESS_COEF * latent_readiness + rng.gauss(0, 220.0))

        # Simulate missed Apple Health sync / partial logs.
        if rng.random() < _RECOVERY_ANOMALY_PROB:
            sleep = round(rng.uniform(0.5, 3.5), 2)
        if rng.random() < _RECOVERY_ANOMALY_PROB:
            calories = round(rng.uniform(0.0, 800.0), 0)

        protein_g = max(50.0, calories * 0.28 / 4.0 + rng.gauss(0, 15))
        carbs_g = max(80.0, calories * 0.42 / 4.0 + rng.gauss(0, 25))
        if rng.random() < _RECOVERY_ANOMALY_PROB:
            protein_g = round(rng.uniform(0.0, 20.0), 1)
        if rng.random() < _RECOVERY_ANOMALY_PROB:
            carbs_g = round(rng.uniform(0.0, 30.0), 1)

        resting_hr = max(
            48.0,
            min(72.0, 58.0 + _RHR_READINESS_COEF * latent_readiness + (7.4 - sleep) * 1.2 + rng.gauss(0, 2.5)),
        )
        bodyweight = round(rng.gauss(78.0, 1.2), 1)
        if rng.random() < _SPARSE_FIELD_PROB:
            resting_hr = None
        if rng.random() < _SPARSE_FIELD_PROB:
            bodyweight = None

        # Omit early timeline + random recovery gaps.
        sleep_history.append((current, sleep))
        if resting_hr is not None:
            rhr_history.append((current, resting_hr))

        if offset >= _RECOVERY_WARMUP_DAYS and rng.random() >= _RECOVERY_ROW_OMIT_PROB:
            recovery.append(
                RecoveryDaily(
                    date=current,
                    sleep_hours=round(sleep, 2),
                    calories_kcal=round(calories, 0),
                    protein_g=round(protein_g, 1),
                    carbs_g=round(carbs_g, 1),
                    bodyweight_kg=bodyweight,
                    resting_hr_bpm=round(resting_hr, 1) if resting_hr is not None else None,
                )
            )

        # Mon/Tue/Thu/Sat (~4x/week) with occasional spillover and skipped sessions.
        if current.weekday() not in (0, 1, 3, 5) and rng.random() > 0.10:
            continue
        if rng.random() < _SESSION_SKIP_PROB:
            continue

        # --- session ---
        day_label, exercise_pool = _DAY_TYPES[day_type_idx % 3]
        day_type_idx += 1

        n_exercises = rng.randint(*_N_EXERCISES_RANGE)
        chosen = rng.sample(exercise_pool, min(n_exercises, len(exercise_pool)))

        session_id = f"{current}:{day_label}"
        ts = datetime.combine(current, datetime.min.time(), tzinfo=timezone.utc).replace(
            hour=rng.choice(_SESSION_HOURS)
        )

        prev_sleep = sleep_history[-2][1] if len(sleep_history) >= 2 else 7.4
        rhr_cutoff = current - timedelta(days=1)
        rhr_window = [v for d, v in rhr_history if d <= rhr_cutoff and d > rhr_cutoff - timedelta(days=7)]
        rhr_trail7 = sum(rhr_window) / len(rhr_window) if rhr_window else 58.0
        trained_cutoff = current - timedelta(days=1)
        training_days_trailing_7d = sum(
            1 for d in workout_dates if d <= trained_cutoff and d > trained_cutoff - timedelta(days=7)
        )
        days_since_last_workout = (
            (current - workout_dates[-1]).days if workout_dates else 7
        )

        week_num = int(weeks_elapsed)
        deload_multiplier = 0.90 if week_num > 0 and week_num % _DELOAD_WEEK_EVERY == 0 else 1.0

        for ex_name, muscle, base_kg, std_kg, is_bw, prog_rate in chosen:
            vol_cutoff = current - timedelta(days=1)
            ex_volumes = [
                v
                for d, v in exercise_volume_by_date.get(ex_name, [])
                if d <= vol_cutoff and d > vol_cutoff - timedelta(days=7)
            ]
            volume_trailing_7d = sum(ex_volumes)
            performance_boost = (
                _SLEEP_LAG1_COEF * (sleep - 7.4)
                + _SLEEP_LAG2_COEF * (prev_sleep - 7.4)
                + _RHR_TRAIL7_COEF * (rhr_trail7 - 58.0)
                + _VOLUME_TRAIL7_COEF * volume_trailing_7d
                + _TRAINING_DAYS_COEF * training_days_trailing_7d
                + _DAYS_SINCE_WORKOUT_COEF * min(days_since_last_workout, 7)
                + rng.gauss(0, _RESIDUAL_NOISE_STD)
            )
            if is_bw:
                weight_now = 0.0
            else:
                if ex_name not in exercise_loads:
                    exercise_loads[ex_name] = base_kg + rng.gauss(0, std_kg * 0.25)
                exercise_loads[ex_name] += prog_rate * _PROGRESSION_SCALE * _PROGRESSION_PER_SESSION
                if rng.random() < _PR_BUMP_PROB:
                    exercise_loads[ex_name] += rng.uniform(*_PR_BUMP_KG_RANGE)
                weight_now = exercise_loads[ex_name] * deload_multiplier

            ex_volume_kg = 0.0
            for set_num in range(1, _WORKING_SETS + 2):  # +1 warmup set
                is_warmup = set_num == 1
                reps = _WARMUP_REPS if is_warmup else rng.randint(_WORKING_REP_MIN, _WORKING_REP_MAX)
                weight = (
                    weight_now * _WARMUP_WEIGHT_FRAC
                    if is_warmup
                    else weight_now + performance_boost
                )
                weight = max(0.0, round(weight, 1))
                if not is_warmup:
                    ex_volume_kg += reps * weight
                sets.append(
                    WorkoutSet(
                        session_id=session_id,
                        timestamp=ts,
                        exercise=ex_name,
                        muscle_group=muscle,
                        set_number=set_num,
                        reps=reps,
                        weight_kg=weight,
                        is_warmup=is_warmup,
                    )
                )
            if not is_bw:
                exercise_volume_by_date.setdefault(ex_name, []).append((current, ex_volume_kg))

        workout_dates.append(current)

    return sets, recovery


def write_synthetic(out_dir: Path, days: int = _DAYS_DEFAULT, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sets, recovery = generate_synthetic(days=days, seed=seed)

    sets_path = out_dir / "workout_sets.jsonl"
    with sets_path.open("w", encoding="utf-8") as f:
        for s in sets:
            f.write(s.model_dump_json() + "\n")

    recovery_rows = [r.model_dump() for r in recovery]
    pd.DataFrame(recovery_rows).to_csv(out_dir / "recovery_daily.csv", index=False)

    meta = {
        "days": days,
        "seed": seed,
        "n_sets": len(sets),
        "n_recovery_days": len(recovery),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(sets)} sets and {len(recovery)} recovery days to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--days", type=int, default=_DAYS_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_synthetic(args.out, days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
