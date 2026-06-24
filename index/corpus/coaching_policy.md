# Coaching policy — load recommendations

source: coaching_policy.md

## Philosophy

Conservative by default: protect long-term progression over single-session PRs. The readiness model predicts **performance_delta** (kg): top-set e1RM minus the **mean of your prior 3 same-exercise sessions'** top-set e1RM — **not** vs your last workout alone. Use it as a signal, not a guarantee.

## Readiness bands (model output)

| Band | performance_delta (kg) | Load guidance |
|------|------------------------|---------------|
| below_trend | < −1.5 | Reduce working weight ~5–10% or cut 1–2 working sets; avoid max attempts |
| at_trend | −1.5 to +1.5 | Match recent working weights; standard progression rules apply |
| above_trend | > +1.5 | Small load increase OK (+2.5 kg upper / +5 kg lower) if sleep and ACWR are stable |

## Combine model with rules

Always check in order:

1. **Deload flag** or ACWR > 1.3 → deload prescription overrides aggressive progression.
2. **Sleep deviation** sharply negative → hold load even if model is optimistic.
3. **Continuity break** on an exercise (equipment change) → do not compare loads across the break; re-establish baseline.

## Workout planning

- Infer next PPL split from last 3–5 logged gym days (see `personal_preferences.md`).
- Exercises must come from user history unless they ask for alternatives.
- Sets × reps default to **median of recent sessions** for that exercise; adjust load per bands above.

## Citation requirement

Every coaching recommendation must cite:

- `[history]` — logged sessions
- `[model]` — readiness prediction and key drivers
- `[source: filename]` — retrieved research snippet

Do not state hypertrophy volume targets or deload rules without a corpus citation.
