# Coaching policy — load recommendations

source: coaching_policy.md

## Philosophy

Conservative by default: protect long-term progression over single-session PRs. The readiness model is a **classifier** that predicts a readiness **band** — `below_trend`, `at_trend`, or `above_trend` — relative to the **mean of your prior 3 same-exercise sessions'** top-set e1RM (**not** vs your last workout alone), with a probability for each class. Use the class and its confidence as a signal, not a guarantee.

## Readiness bands (model output)

The class boundary is **adaptive per exercise** (±0.5 standard deviations of that lift's own session-to-session variation), so a band means the same thing for a heavy deadlift as for a light isolation lift.

| Band | Meaning | Load guidance |
|------|---------|---------------|
| below_trend | likely under your recent trend | Reduce working weight ~5–10% or cut 1–2 working sets; avoid max attempts |
| at_trend | likely on your recent trend | Match recent working weights; standard progression rules apply |
| above_trend | likely above your recent trend | Small load increase OK (+2.5 kg upper / +5 kg lower) if sleep and ACWR are stable |

## Combine model with rules

Always check in order:

1. **Deload flag** or ACWR > 1.3 → deload prescription overrides aggressive progression.
2. **Low confidence** (top class probability < ~0.5) → treat as `at_trend` and hold load; do not chase an uncertain up/down call.
3. **Sleep deviation** sharply negative → hold load even if model is optimistic.
4. **Continuity break** on an exercise (equipment change) → do not compare loads across the break; re-establish baseline.

## Known limitation: below_trend recall

The classifier has **low recall for `below_trend`** (~8–42% depending on calibration). Most genuinely below-trend sessions get predicted as `at_trend`. This is accepted behavior: planning a normal session on a day that turns out worse than expected is a minor outcome. When the model *does* call `below_trend` it is high-precision (>60%) — treat it as a strong signal and reduce load. Do not interpret a `at_trend` prediction as a guarantee the session will go normally; it means the features available pre-workout didn't clearly indicate an off day.

## Workout planning

- Infer next PPL split from last 3–5 logged gym days (see `personal_preferences.md`).
- Exercises must come from user history unless they ask for alternatives.
- Sets × reps default to **median of recent sessions** for that exercise; adjust load per bands above.

## Citation requirement

Every coaching recommendation must cite:

- `[history]` — logged sessions
- `[model]` — readiness class, probabilities, and key drivers
- `[source: filename]` — retrieved research snippet

Do not state hypertrophy volume targets or deload rules without a corpus citation.
