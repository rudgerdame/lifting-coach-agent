# Data schema

Versioned export format shared across lifting-coach-agent, Gravity OS, and (future) iOS app.

## Workout sets (`workout_sets.jsonl`)

One JSON object per set. `schema_version` must be present on every record.

```json
{
  "schema_version": 1,
  "session_id": "2026-06-03-push",
  "timestamp": "2026-06-03T14:38:14Z",
  "exercise": "Barbell Back Squat",
  "muscle_group": "legs",
  "set_number": 2,
  "reps": 5,
  "weight_kg": 100.0,
  "bodyweight_kg": 82.5,
  "is_warmup": false,
  "notes": ""
}
```

### Fitbod CSV mapping

| Fitbod column | Schema field |
|---------------|--------------|
| `Date` | `timestamp` |
| `Exercise` | `exercise` |
| `Reps` | `reps` |
| `Weight(kg)` | `weight_kg` |
| `isWarmup` | `is_warmup` |
| `Note` | `notes` |

Session ID is derived: `{date}:{primary_muscle_group_or_split}`.

## Daily recovery (`recovery_daily.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `date` | YYYY-MM-DD | Calendar day |
| `sleep_hours` | float | Total sleep (Apple Health: `Sleep Analysis [Total] (hr)` daily aggregate) |
| `calories_kcal` | float | Dietary energy (Apple Health: `Dietary Energy (kcal)` daily aggregate) |
| `bodyweight_kg` | float | Optional |
| `resting_hr_bpm` | float | Resting heart rate (Apple Health: `Resting Heart Rate (bpm)` daily mean) |

### Apple Health mapping

Aggregate hourly export to daily totals for:

- `Sleep Analysis [Total] (hr)` → `sleep_hours`
- `Dietary Energy (kcal)` → `calories_kcal`
- `Weight (lbs)` → `bodyweight_kg` (convert to kg)
- `Resting Heart Rate (bpm)` → `resting_hr_bpm` (daily mean of readings)

## Session features (internal, `features/` output)

One row per training session, used for ML:

| Column | Description |
|--------|-------------|
| `session_date` | Date of session |
| `exercise` | Primary lift or session label |
| `top_set_e1rm_kg` | Best estimated 1RM that session |
| `volume_load_kg` | Sum(reps × weight) for working sets |
| `acwr` | Acute (7d) / chronic (28d) workload ratio |
| `training_days_trailing_7d` | Distinct gym days in prior 7 calendar days (session day excluded) |
| `days_since_last_workout` | Calendar days since previous gym day (any exercise) |
| `sleep_trailing_7d` | Mean sleep hours over prior 7 calendar days (pre-workout shifted) |
| `calories_trailing_7` | Mean daily calories over prior 7 days |
| `protein_trailing_7` | Mean daily protein (g) over prior 7 days |
| `carbs_trailing_7` | Mean daily carbs (g) over prior 7 days |
| `sleep_lag_1d` | Sleep hours on session date (Apple Health day bucket) |
| `sleep_lag_2d` | Sleep hours on prior calendar day |
| `resting_hr_lag_1d` | Resting HR (bpm) on session date |
| `resting_hr_trailing_7` | Mean resting HR over prior 7 days (pre-workout shifted) |
| `performance_delta` | Target: top_set_e1rm vs recent trend |

## Versioning

Increment `schema_version` on breaking changes. Loaders should accept v1 and log warnings on unknown versions.
