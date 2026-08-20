"""Lifting Coach Agent — FastAPI server.

Endpoints
---------
GET  /health                      liveness + data-source info
GET  /readiness/{exercise}        predict today's session readiness
GET  /plan                        next workout plan
GET  /history                     recent session history
GET  /search                      corpus search
POST /ask                         natural-language question → tool output
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.deps import get_ctx, get_retriever, get_tools, init_context
from agent.context import CoachContext
from agent.router import route_question
from index.retrieve import CorpusRetriever


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_context()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lifting Coach Agent",
    description="ML readiness model + RAG coaching agent.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(raw: str) -> Any:
    """Parse JSON tool output, returning dict/list or wrapping plain strings."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw}


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    tool_used: str
    result: Any


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health(ctx: CoachContext = Depends(get_ctx)):
    """Liveness check — returns data source and corpus status."""
    retriever = get_retriever()
    return {
        "status": "ok",
        "data_dir": str(ctx.data_dir),
        "corpus_index": retriever is not None,
    }


@app.get("/readiness/{exercise}", tags=["coach"])
def readiness(
    exercise: str,
    today: bool = Query(True, description="Score today's planned session (default). Pass false to score most recent logged session."),
    session_date: str | None = Query(None, description="YYYY-MM-DD — score a specific past session (implies today=false)."),
    ctx: CoachContext = Depends(get_ctx),
):
    """Predict readiness for an exercise session.

    Default scores *today's planned session* using last night's recovery metrics
    and load history up to now. This is the primary production path.
    """
    use_today = today and session_date is None
    try:
        result = ctx.predict_readiness(exercise, session_date=session_date, today=use_today)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "exercise": result.exercise,
        "session_date": str(result.session_date),
        "band": result.band,
        "performance_delta_kg": result.performance_delta_kg,
        "class_label": result.class_label,
        "class_probs": result.class_probs,
        "class_confidence": result.class_confidence,
        "key_drivers": result.key_drivers,
        "prediction_source": result.prediction_source,
        "citation": "[model]",
    }


@app.get("/plan", tags=["coach"])
def plan(
    split: str | None = Query(None, description="Override split: push / pull / legs"),
    tools: dict = Depends(get_tools),
):
    """Return the next workout plan (exercises, sets, load) based on training history."""
    raw = tools["plan_workout"].invoke({"split": split})
    return _parse(raw)


@app.get("/history", tags=["data"])
def history(
    exercise: str | None = Query(None, description="Filter by exercise name (partial match OK)"),
    split: str | None = Query(None, description="Filter by split: push / pull / legs"),
    last_n: int = Query(5, ge=1, le=50, description="Number of sessions to return"),
    tools: dict = Depends(get_tools),
):
    """Return recent logged exercise sessions."""
    raw = tools["query_history"].invoke(
        {"exercise": exercise, "split": split, "last_n_sessions": last_n}
    )
    return _parse(raw)


@app.get("/search", tags=["corpus"])
def search(
    q: str = Query(..., description="Natural-language query"),
    k: int = Query(3, ge=1, le=10, description="Number of results"),
    retriever: CorpusRetriever | None = Depends(get_retriever),
    tools: dict = Depends(get_tools),
):
    """Search the hypertrophy/recovery/coaching research corpus."""
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Corpus index not available. Run: python -m index.build",
        )
    raw = tools["search_corpus"].invoke({"query": q, "k": k})
    return _parse(raw)


@app.post("/ask", response_model=AskResponse, tags=["coach"])
def ask(
    body: AskRequest,
    tools: dict = Depends(get_tools),
):
    """Natural-language coaching question — routes to the best tool automatically.

    Returns the raw tool JSON output. For an LLM-synthesised answer use the CLI
    with ``LLM_PROVIDER=cursor`` or ``LLM_PROVIDER=ollama``.
    """
    try:
        tool_used, raw = route_question(body.question, tools)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(tool_used=tool_used, result=_parse(raw))


@app.get("/block", tags=["coach"])
def block(
    weeks: int = Query(4, ge=2, le=16, description="Number of training weeks"),
    tools: dict = Depends(get_tools),
):
    """Return a multi-week mesocycle plan."""
    raw = tools["plan_block"].invoke({"weeks": weeks})
    return _parse(raw)


@app.get("/progress/{exercise}", tags=["coach"])
def progress(
    exercise: str,
    last_n: int = Query(30, ge=5, le=100, description="Number of sessions to return"),
    ctx: CoachContext = Depends(get_ctx),
):
    """Return e1RM history + readiness band for each past session of an exercise.

    Used for progress charts in the UI. Runs the classifier on each historical
    session row (not today's recovery — these are retrospective scores).
    """
    import pandas as pd

    features = ctx.features
    # Resolve exercise name (partial match)
    names = features["exercise"].astype(str).unique()
    if exercise not in names:
        matches = [n for n in names if exercise.lower() in n.lower()]
        if len(matches) == 1:
            exercise = matches[0]
        elif len(matches) > 1:
            raise HTTPException(status_code=400, detail=f"Ambiguous exercise {exercise!r}; matches: {matches[:5]}")
        else:
            raise HTTPException(status_code=404, detail=f"Exercise {exercise!r} not found in history")

    ex_rows = (
        features[features["exercise"] == exercise]
        .copy()
        .sort_values("session_date")
        .tail(last_n)
    )

    predictor = ctx.predictor
    sessions = []
    for _, row in ex_rows.iterrows():
        try:
            result = predictor.predict_row(row, features=features)
            sessions.append({
                "session_date": str(row["session_date"]),
                "top_set_e1rm_kg": round(float(row["top_set_e1rm_kg"]), 2) if pd.notna(row.get("top_set_e1rm_kg")) else None,
                "volume_load_kg": round(float(row["volume_load_kg"]), 1) if pd.notna(row.get("volume_load_kg")) else None,
                "acwr": round(float(row["acwr"]), 3) if pd.notna(row.get("acwr")) else None,
                "deload_flag": int(row.get("deload_flag", 0)),
                "band": result.band,
                "performance_delta_kg": result.performance_delta_kg,
                "class_confidence": result.class_confidence,
            })
        except Exception:
            # Skip rows where prediction fails (e.g. missing features)
            sessions.append({
                "session_date": str(row["session_date"]),
                "top_set_e1rm_kg": round(float(row["top_set_e1rm_kg"]), 2) if pd.notna(row.get("top_set_e1rm_kg")) else None,
                "volume_load_kg": round(float(row["volume_load_kg"]), 1) if pd.notna(row.get("volume_load_kg")) else None,
                "band": None,
            })

    return {"exercise": exercise, "sessions": sessions, "citation": "[history] [model]"}


@app.get("/explain", tags=["corpus"])
def explain(
    topic: str = Query("readiness model", description="Topic to explain"),
    tools: dict = Depends(get_tools),
):
    """Explain how the readiness model or a feature topic works."""
    raw = tools["explain"].invoke({"topic": topic})
    return _parse(raw)
