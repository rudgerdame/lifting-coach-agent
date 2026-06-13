"""Generate synthetic workout + recovery data for reproducible demos.

Calibrated from a real ~18-month Fitbod PPL export to match realistic
exercise selection, rep ranges, weight progressions, and session frequency.
No personal data is used or committed.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ingestion.schema import RecoveryDaily, WorkoutSet

# ---------------------------------------------------------------------------
# Exercise registry — calibrated from a real Fitbod PPL export.
# Tuple: (exercise_name, muscle_group, base_kg, weight_std_kg, is_bodyweight, prog_kg_per_week)
#
# Dumbbell bilateral movements store the effective total load (per-DB × 2)
# so the feature pipeline's volume and e1RM calculations are consistent.
# Bodyweight exercises (Pull Up, Dip, Vertical Leg Raise) use weight=0.
# ---------------------------------------------------------------------------

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

_SESSION_HOURS = [9, 10, 14, 15, 17, 18]
_N_EXERCISES_RANGE = (4, 6)
_WORKING_SETS = 3
_WARMUP_REPS = 8
_WORKING_REP_MIN = 10
_WORKING_REP_MAX = 17
_WARMUP_WEIGHT_FRAC = 0.65


def generate_synthetic(
    days: int = 540,
    seed: int = 42,
) -> tuple[list[WorkoutSet], list[RecoveryDaily]]:
    """Generate ~18 months of synthetic PPL training logs without personal data."""
    rng = random.Random(seed)
    sets: list[WorkoutSet] = []
    recovery: list[RecoveryDaily] = []

    start = date.today() - timedelta(days=days)
    day_type_idx = 0

    for offset in range(days):
        current = start + timedelta(days=offset)
        weeks_elapsed = offset / 7.0

        # --- recovery data ---
        sleep = max(5.0, min(9.5, rng.gauss(7.4, 0.8)))
        calories = max(1800.0, rng.gauss(2600.0, 280.0))

        # ~3 % of days simulate a missed Apple Health sync or logging gap;
        # the feature pipeline imputes these from the prior 7-day valid average.
        if rng.random() < 0.03:
            sleep = round(rng.uniform(0.5, 3.5), 2)
        if rng.random() < 0.03:
            calories = round(rng.uniform(0.0, 800.0), 0)

        protein_g = max(50.0, calories * 0.28 / 4.0 + rng.gauss(0, 15))
        carbs_g = max(80.0, calories * 0.42 / 4.0 + rng.gauss(0, 25))
        if rng.random() < 0.03:
            protein_g = round(rng.uniform(0.0, 20.0), 1)
        if rng.random() < 0.03:
            carbs_g = round(rng.uniform(0.0, 30.0), 1)

        resting_hr = max(48.0, min(72.0, rng.gauss(58.0, 3.5) + (7.4 - sleep) * 1.5))

        recovery.append(
            RecoveryDaily(
                date=current,
                sleep_hours=round(sleep, 2),
                calories_kcal=round(calories, 0),
                protein_g=round(protein_g, 1),
                carbs_g=round(carbs_g, 1),
                bodyweight_kg=round(rng.gauss(78.0, 1.2), 1),
                resting_hr_bpm=round(resting_hr, 1),
            )
        )

        # --- session scheduling ---
        # Target Mon/Tue/Thu/Sat (~4x/week) with 10 % spillover to other days,
        # then a 15 % chance of skipping a planned session (rest, travel, etc.).
        if current.weekday() not in (0, 1, 3, 5) and rng.random() > 0.10:
            continue
        if rng.random() < 0.15:
            continue

        # --- session setup ---
        day_label, exercise_pool = _DAY_TYPES[day_type_idx % 3]
        day_type_idx += 1

        n_exercises = rng.randint(*_N_EXERCISES_RANGE)
        chosen = rng.sample(exercise_pool, min(n_exercises, len(exercise_pool)))

        session_id = f"{current}:{day_label}"
        ts = datetime.combine(current, datetime.min.time(), tzinfo=timezone.utc).replace(
            hour=rng.choice(_SESSION_HOURS)
        )

        # sleep last night nudges performance up/down slightly
        sleep_bonus = (sleep - 7.4) * 1.2

        for ex_name, muscle, base_kg, std_kg, is_bw, prog_rate in chosen:
            weight_now = base_kg + (weeks_elapsed * prog_rate) + rng.gauss(0, std_kg)
            if is_bw:
                weight_now = 0.0

            for set_num in range(1, _WORKING_SETS + 2):  # +1 warmup set
                is_warmup = set_num == 1
                reps = _WARMUP_REPS if is_warmup else rng.randint(_WORKING_REP_MIN, _WORKING_REP_MAX)
                weight = (
                    weight_now * _WARMUP_WEIGHT_FRAC
                    if is_warmup
                    else weight_now + sleep_bonus + rng.gauss(0, 0.8)
                )
                sets.append(
                    WorkoutSet(
                        session_id=session_id,
                        timestamp=ts,
                        exercise=ex_name,
                        muscle_group=muscle,
                        set_number=set_num,
                        reps=reps,
                        weight_kg=max(0.0, round(weight, 1)),
                        is_warmup=is_warmup,
                    )
                )

    return sets, recovery


def write_synthetic(out_dir: Path, days: int = 540, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sets, recovery = generate_synthetic(days=days, seed=seed)

    sets_path = out_dir / "workout_sets.jsonl"
    with sets_path.open("w", encoding="utf-8") as f:
        for s in sets:
            f.write(s.model_dump_json() + "\n")

    recovery_rows = [r.model_dump() for r in recovery]
    pd.DataFrame(recovery_rows).to_csv(out_dir / "recovery_daily.csv", index=False)

    meta = {"days": days, "seed": seed, "n_sets": len(sets), "n_recovery_days": len(recovery)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(sets)} sets and {len(recovery)} recovery days to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--days", type=int, default=540)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_synthetic(args.out, days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
