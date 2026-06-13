


# Feature engineering & target rationale

This document defines the **target variable**, every **model feature**, and the **diligence defense** for why each is computed this way. It is the reference for model review, portfolio interviews, and future changes.

---

## Pre-workout scoring

The readiness model is trained for **prediction before the gym** — not post-hoc explanation.

At scoring time you know:

- **Today's split** — push, pull, or legs (user-provided or inferred from planned exercises)
- **Which exercise** you're about to perform
- **All history** through yesterday's logs (Fitbod + Apple Health)

Same-day features (`intensity_pct`, session volume, set counts) are **not** model inputs. Trailing load, ACWR, WoW volume, deload, and recovery trailing/deviation values exclude today's session or use data through **yesterday** only.

---

## Target variable

### `performance_delta` (kg)

**Definition**

```
performance_delta = top_set_e1rm_kg − e1rm_trend
```

Where:

- `top_set_e1rm_kg` = max estimated 1RM (Epley) among working sets in that session for a given exercise
- `e1rm_trend` = rolling mean of the prior **3 sessions of the same exercise within the same continuity segment** (shifted so the current session is excluded)

**Important:** The trend is **session-ordered per exercise**, not calendar-ordered. If your last bench press was 10 days ago and you trained legs/back in between, the trend uses your last 3 bench sessions only — not whatever you did on prior calendar days. `days_since_last_session` captures the calendar gap explicitly as a separate feature.

**Continuity segments:** After a `continuity_break` flag (see below), the trend resets — prior sessions on old equipment do not anchor the trend for the new segment.

**Why this target**

| Choice | Rationale |
|--------|-----------|
| Session-level, per exercise | Matches how a lifter experiences a workout ("squat felt strong today") and avoids pooling incompatible lifts |
| Delta vs trend, not raw e1RM | Raw e1RM drifts with training age and program phase. The model should predict **deviation from your recent baseline**, not absolute strength |
| 3-session trend window | Short enough to track mesocycle progress; long enough to smooth one bad/good set. Standard "recent form" horizon for autoregulation |
| Regression on kg delta | Continuous and interpretable ("+4 kg vs trend"). Can be bucketed later into over / at / under for classification |

**Limitations (disclose in review)**

- Correlation ≠ causation for all recovery features
- Exercise swaps and technique changes break trend continuity
- Accessory lifts have noisier e1RM estimates than compounds
- No RPE — objective proxies only

---

### `continuity_break` (excluded from train/eval, not a model feature)

**Formula:** Flag = 1 when max working-set weight for this **exercise name** drops more than **20%** vs the prior logged session of that same exercise:

```
weight_drop_pct = (max_weight_today − max_weight_prior) / max_weight_prior
continuity_break = weight_drop_pct < −0.20
```

**Scope:** Per exercise-session only — other exercises on the same gym day are unchanged.

**Why exclude:** Sharp drops with the same Fitbod name usually mean **different equipment, gym, or stack** — not a real −30 kg performance change vs trend. Those rows are dropped from training/eval; `e1rm_trend` resets for subsequent sessions of that exercise.

---

## Recovery data quality: anomaly detection

Before recovery features are computed, sleep, calories, protein, and carbs pass through **hybrid anomaly detection** (`features/anomaly.py`).

### What happens once an anomaly is detected

Anomalies are **not dropped**. Each flagged day goes through three steps:

1. **Flag** — `{signal}_imputed = 1` (e.g. `sleep_imputed`, `protein_imputed`)
2. **Replace** — the raw value is overwritten with the **mean of valid readings from the prior 7 days** (excluding the current day)
3. **Continue** — the calendar row stays in the dataset so lag and trailing features remain continuous

If there is no prior valid history, the value remains missing for that day
(past-only policy; no future-informed fallback).

The imputation flags enter the model so LightGBM can treat estimated days differently from measured ones.

### Layer 1 — Hard floor (instrumentation failure)

| Signal | Floor | Defense |
|--------|-------|---------|
| `sleep_hours` | < 4 h | Almost always a missed wearable sync, not a real night |
| `calories_kcal` | < 1000 kcal | App not opened / partial day log, not a real intake day |
| `protein_g` | < 30 g | Partial log / missed nutrition sync |
| `carbs_g` | < 50 g | Partial log / missed nutrition sync |

### Layer 2 — Personalized lower-tail (MAD)

For each day, compare the value to the **prior 28 days of valid readings**:

```
flag if value < median − 3 × 1.4826 × MAD
```

| Parameter | Value | Defense |
|-----------|-------|---------|
| Lower tail only | yes | We detect missing data, not "great sleep" outliers |
| MAD not std | robust | Personal baselines are skewed; MAD resists a few extreme days |
| 3× scaled MAD | ~3σ equivalent | Standard conservative threshold for univariate outliers |
| Min 7 days history | required | Avoid flagging early sparse logs before a baseline exists |

### Imputation

Anomalous values are replaced with the **mean of valid readings from the prior 7 days** (excluding the current day). If there is no prior valid history, the value remains missing.

**Why impute instead of drop**

- Dropping removes the recovery **row**, breaking the date index and leaving sessions with missing lag/trailing features
- Imputation preserves continuity while down-weighting bad days via `sleep_imputed` / `calories_imputed` flags

---

## Training-load features

### `volume_load_kg`

**Formula:** `Σ (reps × weight_kg)` for working sets in the session (excludes warmups).

**Model status:** Computed in the pipeline for trailing-load math only — **not a model feature** (same-day, unknown pre-workout).

---

### `volume_trailing_7d`

**Formula:** Rolling 7-day **sum** of prior `volume_load_kg` for the same exercise (time-aware `"7D"` window, **excluding today's session** via `.shift(1)`).

**Defense**

- Captures **recent training density** knowable before you walk in
- Per-exercise (not global) because push/pull/legs rotate independently in PPL
- 7 days aligns with the "acute" window in sports-science load monitoring

---

### `acwr` (acute:chronic workload ratio)

**Formula:**

```
acute   = volume_trailing_7d
chronic = volume_trailing_28d / 4    # average weekly load over 28 days
acwr    = acute / chronic
```

**Defense**

- ACWR is the standard load-management metric in sports science (Gabbett 2016)
- Values > ~1.3–1.5 suggest ramping load faster than the body has adapted to → higher injury/fatigue risk and potential underperformance
- Values < ~0.8 suggest detraining → may over- or under-perform depending on context
- Uses volume (objective) because RPE is unavailable — volume load is the best proxy

**Caveat:** ACWR validity is debated for resistance training vs team sports; included because it's interpretable and portfolio-defensible, not because it's proven for hypertrophy blocks.

---

### `intensity_pct` (not in model)

**Formula:** `weight_kg / e1rm` for the working set that produced the session's max e1RM.

**Model status:** **Removed** — requires today's logged sets; unknown pre-workout. With Epley, this collapses to `1 / (1 + reps/30)` on the top set anyway.

---

### `n_working_sets` / `n_sets_all` (not in model)

**Model status:** **Removed** — same-day set counts; unknown pre-workout.

---

### `split`

**Formula:** PPL bucket from muscle group — `push` (chest/shoulders/triceps), `pull` (back/biceps), `legs` (quads/hamstrings/glutes/calves).

**Defense**

- Known before the gym ("today is push day")
- Categorical feature alongside `exercise` and `muscle_group`

---

### `split_sessions_trailing_10d`

**Formula:** Count of prior exercise sessions for the same **split** in the last 10 calendar days (same logic as muscle-group count).

**Defense**

- Whole-split fatigue on PPL — many push exercises in 10 days even if each muscle group count is moderate

---

### `deload_flag`

**Formula:** Binary, from **last completed week** per exercise:

```
prev_week_volume < 0.60 × rolling_4week_mean(prev_week_volume)
```

**Defense**

- Knowable pre-workout — does not use current week's in-progress volume
- Flags intentional or accidental low-load weeks

---

### `day_of_week`

**Formula:** `session_date.dayofweek` (0 = Monday, 6 = Sunday).

**Defense**

- Captures **circadian / schedule effects** (e.g., early Monday sessions after weekend sleep shift)
- Weak prior alone, but tree models can interact it with sleep lags
- No one-hot encoding — LightGBM handles ordinal day reasonably for small N

---

### `days_since_last_session`

**Formula:** Calendar days since the previous session **of the same exercise** (`session_date.diff()` within exercise group).

**Defense**

- Complements `day_of_week`: captures **training gap** even when weekday repeats (e.g. bench every 7 days always on Monday still varies if you miss a week)
- Long gaps may mean detraining or freshness — both affect performance vs trend
- Explicitly separates "calendar day" from "time since last time you did this lift"

---

### `training_days_trailing_7d`

**Formula:** Count of **distinct calendar days with any logged workout** in the prior 7 calendar days (session day excluded). Equivalent to `7 − rest_days_trailing_7d`.

**Defense**

- Captures **global weekly schedule density** — how many days you were in the gym recently, regardless of split or exercise
- Complements per-exercise `days_since_last_session`: you may bench after only 2 days, but have trained legs/back on 5 of the last 7 days
- Useful when training frequency is irregular (travel, missed weeks, extra sessions)

---

### `days_since_last_workout`

**Formula:** Calendar days since the previous **gym day** (any exercise), on the current session date.

**Defense**

- Systemic freshness signal — "how long since I last trained at all?"
- Differs from `days_since_last_session` when rotating PPL: legs today may follow a pull day 1 calendar day ago, but bench was 5 days ago
- Pairs with `training_days_trailing_7d` for irregular schedules

---

### `volume_wow_pct`

**Formula:** Week-over-week volume change using **last two completed weeks** (excludes current week):

```
(prev_week_volume − prev_prev_week_volume) / prev_prev_week_volume
```

**Defense**

- Mesocycle ramping/deloading signal knowable before today's session
- Positive = last week built load vs the week before

---

### `muscle_group_sessions_trailing_10d`

**Formula:** Count of **exercise sessions** (one row per exercise in a workout) for the same muscle group in the prior 10 calendar days. Each exercise counts separately — two chest exercises on one push day add 2, not 1.

**Important pre-workout rule:** same-day exercises are excluded. The count is based on strictly prior calendar dates only.

**Example:** Push day with Incline Bench + Cable Fly (both chest) after 3 prior chest exercises in the last 10 days → both rows see 3.

**Defense**

- Proxy for **muscle-group fatigue / overwork** at exercise granularity, not day granularity
- Captures PPL patterns where multiple exercises hit the same group in one workout
- Complements exercise-level ACWR and `days_since_last_session` with group-level volume frequency

---

### `trailing_days` (removed)

Removed after walk-forward diagnosis: it correlated almost perfectly with CV fold index
(older test folds = higher `trailing_days`), so the model learned log recency instead of
physiology. Use `days_since_last_session` (per exercise) for gap effects at inference.

---

All recovery features are merged onto session rows by `session_date`. They describe the athlete's state **entering** the session.

### `sleep_trailing_7d`

**Formula:** Rolling **7-day mean** of `sleep_hours` (after anomaly imputation), shifted one day for pre-workout use so the session day is excluded.

**Naming:** The `d` is the window length in days (like `volume_trailing_7d`). Values are in **hours**, not a 7-hour window.

**Defense**

- Mean daily sleep over the week is stable when imputed or missing days reduce the effective window count (`min_periods=1`)
- Chronic sleep debt affects strength and RPE tolerance over days, not just one night
- 7-day window matches acute recovery monitoring practice

---

### `calories_trailing_7`

**Formula:** Rolling 7-day **mean** of `calories_kcal` (after imputation).

**Defense**

- Mean daily intake normalizes for sparse logging — a 5-day window with one imputed day does not deflate the total like a sum would
- Energy availability affects recovery and training capacity over a week
- Defensible as a proxy for fuelling; not a substitute for macro tracking

---

### `sleep_deviation`

**Formula:**

```
sleep_deviation = mean(sleep last 3 days) − mean(sleep last 28 days)
```

Both windows are shifted one day for pre-workout use (session day excluded).

**Defense**

- Captures **acute vs chronic** sleep state — recent nights vs your normal month
- Positive deviation = recent sleep better than baseline → hypothesized positive effect on performance_delta
- 3-day acute / 28-day chronic mirrors ACWR's acute:chronic structure for recovery

---

### `calories_deviation`

**Formula:**

```
calories_deviation = mean(calories last 3 days) − mean(calories last 28 days)
```

**Defense**

- Same acute:chronic logic as sleep_deviation
- Short-term under-fuelling may correlate with underperformance vs trend

---

### `sleep_lag_1d` / `sleep_lag_2d`

**Formula:**

- `sleep_lag_1d` = `sleep_hours` on the **session date** (Apple Health daily total for the workout day — typically last night’s sleep attributed to the morning you wake)
- `sleep_lag_2d` = `sleep_hours` on the **prior calendar day**

**Defense**

- Aligns with how Apple Health buckets sleep onto calendar days in the Gravity OS export
- `sleep_lag_1d` is the night-before signal for typical evening sessions; for early-morning sessions the same-day bucket may still be filling in (pre-workout limitation)
- `sleep_lag_2d` captures delayed fatigue from two calendar days back (travel/weekend patterns)

---

### `protein_trailing_7` / `protein_deviation`

**Formula:** Same trailing-7 **mean** / acute-vs-28-day deviation pattern as calories, applied to `protein_g` from Apple Health (`Protein (g)` daily sum).

**Defense**

- Protein supports recovery and muscle repair; trailing intake may correlate with performance vs trend on high-volume blocks
- Deviation captures short-term under-eating protein relative to your baseline

---

### `carbs_trailing_7` / `carbs_deviation`

**Formula:** Same trailing-7 **mean** / acute-vs-28-day deviation pattern applied to `carbs_g` from Apple Health (`Carbohydrates (g)` daily sum).

**Defense**

- Carb availability affects same-day glycogen and session energy, especially on leg/high-volume days
- More granular than total calories alone for fuelling hypothesis

---

### `resting_hr_lag_1d` / `resting_hr_trailing_7` / `resting_hr_deviation`

**Formula:**

- `resting_hr_lag_1d` = `resting_hr_bpm` on the **session date** (daily mean of Apple Health `Resting Heart Rate (bpm)` readings)
- `resting_hr_trailing_7` = rolling 7-day **mean** of `resting_hr_bpm`, shifted one day for pre-workout use
- `resting_hr_deviation` = mean(RHR last 3 days) − mean(RHR last 28 days), pre-workout shifted

**Defense**

- Elevated resting HR vs personal baseline is a common autonomic stress / under-recovery signal
- Trailing mean smooths day-to-day wearable noise; deviation captures acute vs chronic autonomic state
- Lower RHR vs baseline may correlate with fresher performance — model learns direction from data
- Omitted automatically when resting HR is unavailable in the Apple Health export

---

### `bodyweight_kg` / `bodyweight_trailing_7` / `bodyweight_lag_1d`

**Formula:** Daily bodyweight from Apple Health (`Weight` → kg). Trailing 7-day mean and 1-day lag (same pattern as sleep/calories).

**Defense**

- Body mass shifts affect absolute load capacity and e1RM on barbell lifts
- Trailing mean smooths daily scale noise; lag captures recent weight change before the session
- Omitted automatically when bodyweight is unavailable in the recovery feed

---

### `protein_imputed` / `carbs_imputed`

**Formula:** Binary flags (1 = anomalous macro day imputed via same pipeline as sleep/calories).

**Defense**

- Same rationale as `sleep_imputed` — transparency when macro data was estimated

---

### `sleep_imputed` / `calories_imputed`

**Formula:** Binary flags (1 = value was anomalous and imputed for that day).

**Defense**

- Tells the model when recovery inputs are **estimated, not measured**
- Prevents imputed "normal-looking" values from being treated with full confidence
- Transparency for diligence review and SHAP interpretation

---

## Leakage safeguards

| Rule | Implementation |
|------|----------------|
| Time-ordered validation | 3-fold expanding walk-forward CV in `models/train.py` |
| Trend excludes current session | `e1rm_trend` uses `.shift(1)` on rolling mean |
| Lag features only look backward | `.shift(1)`, `.shift(2)` on sorted daily recovery |
| Trailing windows exclude future | Rolling computed on past dates only; ACWR uses `"7D"`/`"28D"` backward windows |
| Anomaly baseline uses prior days only | MAD window `[i−28, i)` excludes day `i` |

---

## Features computed but not in model

| Column | Purpose |
|--------|---------|
| `top_set_e1rm_kg` | Used to derive target; not a feature (would leak outcome) |
| `e1rm_trend` | Used to derive target |
| `volume_trailing_28d` | Intermediate for ACWR |

---

## References (for diligence packet)

- Gabbett TJ (2016) — ACWR and training-load monitoring
- Fullagar et al. (2015) — Sleep and athletic performance review
- Epley (1985) — e1RM estimation formula used in `features/e1rm.py`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-13 | Continuity break filter (>20% weight drop per exercise) excludes bad trend rows from train/eval |
| 2026-06-13 | Pre-workout feature set: drop same-day features, add split, shift trailing load/recovery |
| 2026-06-13 | Added temporal features, protein/carbs, volume ablation, anomaly imputation doc |
| 2026-06-13 | Initial doc: target, 15 features, MAD anomaly detection, imputation flags |
