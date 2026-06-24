# Personal workout preferences

> Agent context — not generic LLM knowledge. Update when your training preferences change.

## Split

- **Primary split:** Push / Pull / Legs (PPL)
- **Next-session inference:** Look at the last **3–5 logged workout days** and infer the next split from recent rotation (e.g. push → pull → legs → push).
- Do not assume a fixed calendar; use actual session history from Fitbod logs.

## Preferred exercises

- **Rule:** Any exercise that appears in the user's training history is treated as a **preferred / acceptable** exercise.
- When planning a session, prioritize exercises the user has logged recently for that split or muscle group.
- Avoid suggesting novel movements not present in history unless the user explicitly asks for alternatives.

## Rep and set patterns

- Derive typical **reps per working set** and **working set count per exercise** from the user's logs (trailing history), not from generic templates.
- Use recent session aggregates as defaults when proposing today's plan (e.g. median working sets and rep range per exercise over last N sessions).
- Progression should respect what the user actually runs (e.g. ~3 working sets, ~10–17 reps if that matches history).

## Data sources

- Workout logs: Fitbod export via `GRAVITYOS_DATA_DIR` or normalized `workout_sets.jsonl`
- Recovery: Apple Health daily aggregates (sleep, calories, macros, bodyweight)
- Readiness signal: `performance_delta` model + feature pipeline (see `docs/feature-engineering.md`)

## Open decisions

- **Coaching policy** (conservative vs aggressive loading thresholds): TBD — will be defined in `coaching_policy.md` once agent eval begins.
