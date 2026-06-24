"""LangChain tools wrapping CoachContext, predictor, and corpus retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from agent.context import CoachContext
from agent.planner import build_workout_plan, infer_next_split
from index.retrieve import CorpusRetriever


def _json(data: object) -> str:
    return json.dumps(data, indent=2, default=str)


def make_tools(ctx: CoachContext, retriever: CorpusRetriever | None = None):
    """Build tool callables with shared context injected via closure."""

    @tool
    def query_history(
        exercise: str | None = None,
        split: str | None = None,
        last_n_sessions: int = 5,
    ) -> str:
        """Return recent logged exercise-sessions from user history. Filter by exercise or split (push/pull/legs)."""
        feat = ctx.features.copy()
        feat["_date"] = pd.to_datetime(feat["session_date"])

        if exercise:
            names = feat["exercise"].astype(str).unique()
            if exercise not in names:
                matches = [n for n in names if exercise.lower() in n.lower()]
                if len(matches) == 1:
                    exercise = matches[0]
                elif matches:
                    return _json({"error": f"Ambiguous exercise {exercise!r}", "matches": matches[:8]})
                else:
                    return _json({"error": f"Exercise {exercise!r} not found in history"})

        if split:
            feat = feat[feat["split"].str.lower() == split.lower().strip()]

        if exercise:
            feat = feat[feat["exercise"] == exercise]

        gym_days = sorted(feat["_date"].dt.date.unique(), reverse=True)[:last_n_sessions]
        if not gym_days:
            return _json({"sessions": [], "message": "No matching sessions in history."})

        rows = feat[feat["_date"].dt.date.isin(gym_days)].sort_values("_date", ascending=False)
        sessions = []
        for (day, ex), grp in rows.groupby(["_date", "exercise"]):
            row = grp.iloc[-1]
            sessions.append(
                {
                    "session_date": str(day.date() if hasattr(day, "date") else day),
                    "exercise": str(row["exercise"]),
                    "split": str(row.get("split", "")),
                    "muscle_group": str(row.get("muscle_group", "")),
                    "top_set_e1rm_kg": round(float(row.get("top_set_e1rm_kg", 0)), 1),
                    "max_working_weight_kg": round(float(row.get("max_working_weight_kg", 0)), 1),
                    "volume_load_kg": round(float(row.get("volume_load_kg", 0)), 1),
                    "n_working_sets": int(row.get("n_working_sets", 0)),
                }
            )

        return _json({"sessions": sessions[:20], "citation": "[history]"})

    @tool
    def predict_readiness(exercise: str, session_date: str | None = None) -> str:
        """
        Predict performance_delta (kg) vs prior-3-session same-exercise e1RM trend.
        session_date: YYYY-MM-DD or omit for latest logged session.
        """
        try:
            result = ctx.predict_readiness(exercise, session_date)
        except (FileNotFoundError, ValueError) as exc:
            return _json({"error": str(exc)})

        payload = asdict(result)
        payload["definition"] = (
            "performance_delta = top_set_e1rm_kg - mean(prior 3 same-exercise top-set e1RMs); "
            "not vs last session alone"
        )
        payload["citation"] = "[model]"
        return _json(payload)

    @tool
    def search_corpus(query: str, k: int = 3) -> str:
        """Search hypertrophy/recovery/coaching research snippets. Returns citable sources."""
        if retriever is None:
            return _json({"error": "Corpus index missing. Run: python -m index.build"})
        hits = retriever.search(query, k=k)
        return _json(
            {
                "hits": [
                    {
                        "source": h.source,
                        "title": h.title,
                        "text": h.text,
                        "score": round(h.score, 3),
                        "citation": f"[source: {h.source}]",
                    }
                    for h in hits
                ]
            }
        )

    @tool
    def plan_workout(split: str | None = None) -> str:
        """
        Propose next workout: infer PPL split from recent rotation unless split is given.
        Exercises, sets, reps, and loads come from user history + readiness model.
        """
        features = ctx.features
        sets_df = ctx.workout_sets

        target_split = split.lower().strip() if split else None
        if target_split is None:
            target_split, rotation = infer_next_split(features)
        else:
            _, rotation = infer_next_split(features)

        anchor_exercises = [
            ex
            for ex in features[features["split"] == target_split]["exercise"].unique()
        ][:1]
        readiness = None
        if anchor_exercises:
            try:
                readiness = ctx.predict_readiness(str(anchor_exercises[0]))
            except (FileNotFoundError, ValueError):
                readiness = None

        plan = build_workout_plan(
            features,
            sets_df,
            split=target_split,
            readiness=readiness,
        )
        return _json(
            {
                "split": plan.split,
                "recent_split_rotation": rotation,
                "deload_recommended": plan.deload_recommended,
                "readiness_summary": plan.readiness_summary,
                "exercises": [asdict(e) for e in plan.exercises],
                "notes": plan.notes,
                "citations": ["[history]", "[model]", "[source: coaching_policy.md]"],
            }
        )

    @tool
    def plan_block(weeks: int = 4) -> str:
        """Stub: high-level training block outline (not a full periodization engine)."""
        return _json(
            {
                "weeks": weeks,
                "structure": [
                    {"week": 1, "focus": "accumulation", "volume": "MEV→MAV", "intensity": "moderate"},
                    {"week": 2, "focus": "progression", "volume": "MAV", "intensity": "moderate+"},
                    {"week": 3, "focus": "overreach", "volume": "near MRV", "intensity": "moderate-high"},
                    {"week": 4, "focus": "deload", "volume": "40-60% reduction", "intensity": "moderate"},
                ][:weeks],
                "note": "Template only — call search_corpus for volume landmarks and deload rules.",
                "citation": "[source: volume_landmarks.md] [source: deload_fatigue.md]",
            }
        )

    @tool
    def explain(topic: str = "readiness model") -> str:
        """Explain how the readiness model or a feature topic works (SHAP themes + corpus)."""
        report_path = Path("eval/model_report.md")
        shap_summary = ""
        if report_path.exists():
            text = report_path.read_text(encoding="utf-8")
            if "## SHAP" in text:
                shap_summary = text.split("## SHAP")[1].split("##")[0].strip()[:800]

        corpus_hits = []
        if retriever is not None:
            corpus_hits = [
                {"source": h.source, "title": h.title, "text": h.text[:400]}
                for h in retriever.search(topic, k=2)
            ]

        return _json(
            {
                "topic": topic,
                "performance_delta_definition": (
                    "top_set_e1rm_kg minus mean of prior 3 same-exercise top-set e1RMs "
                    "(not vs last session)"
                ),
                "model_report_excerpt": shap_summary or "Run models.train to generate eval/model_report.md",
                "corpus_hits": corpus_hits,
                "citations": ["[model]", "[source: feature-engineering.md]"],
            }
        )

    return [
        query_history,
        predict_readiness,
        search_corpus,
        plan_workout,
        plan_block,
        explain,
    ]
