"""Keyword router for tool-only / eval runs (no LLM)."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent.context import CoachContext
from agent.tools import make_tools
from index.retrieve import CorpusRetriever


def _exercise_from_question(q: str) -> str:
    """Map NL exercise hints to a feature-table name (synthetic data defaults)."""
    if "squat" in q:
        return "squat"
    if "bench" in q or "incline" in q:
        return "Incline Bench"
    return "Incline Bench"


_EXERCISE_KEYWORDS = (
    "bench", "incline", "squat", "deadlift", "row", "curl", "press",
    "pulldown", "pullup", "dip", "lunge", "thrust", "extension", "fly",
)


def _mentions_exercise(q: str) -> bool:
    return any(w in q for w in _EXERCISE_KEYWORDS)


def route_question(question: str, tools: dict[str, BaseTool]) -> tuple[str, str]:
    """Return (tool_name, json_output) for a natural-language question."""
    q = question.lower()

    # History questions beat readiness even when they mention an exercise
    # e.g. "What did I bench recently?" → query_history, not predict_readiness
    _HISTORY_WORDS = ("history", "last", "logged", "when did", "recently", "recent", "what did", "what have i")
    if any(w in q for w in _HISTORY_WORDS):
        exercise = _exercise_from_question(q) if _mentions_exercise(q) else None
        return "query_history", tools["query_history"].invoke(
            {"exercise": exercise, "split": None, "last_n_sessions": 5}
        )

    # Exercise-specific readiness: "What should I try benching today?" → predict_readiness
    if _mentions_exercise(q) and any(
        w in q for w in ("try", "do", "ready", "readiness", "heavy", "performance", "today", "attempt", "should")
    ):
        exercise = _exercise_from_question(q)
        return "predict_readiness", tools["predict_readiness"].invoke(
            {"exercise": exercise, "session_date": None}
        )

    if any(w in q for w in ("plan", "workout", "train next", "what should i")):
        return "plan_workout", tools["plan_workout"].invoke({"split": None})
    if "explain" in q or q.startswith("why") or " why " in f" {q} ":
        return "explain", tools["explain"].invoke({"topic": question})
    if any(w in q for w in ("ready", "readiness", "heavy", "performance")) or _mentions_exercise(q):
        exercise = _exercise_from_question(q)
        return "predict_readiness", tools["predict_readiness"].invoke(
            {"exercise": exercise, "session_date": None}
        )
    if any(w in q for w in ("volume", "deload", "sleep", "protein", "hypertrophy")):
        return "search_corpus", tools["search_corpus"].invoke({"query": question, "k": 3})
    if "block" in q or "week" in q:
        return "plan_block", tools["plan_block"].invoke({"weeks": 4})
    return "plan_workout", tools["plan_workout"].invoke({"split": None})


def tools_only_answer(
    ctx: CoachContext,
    retriever: CorpusRetriever | None,
    question: str,
) -> str:
    tools = {t.name: t for t in make_tools(ctx, retriever)}
    _, output = route_question(question, tools)
    return output
