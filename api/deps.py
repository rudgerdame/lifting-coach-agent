"""Shared FastAPI dependencies — CoachContext + CorpusRetriever, loaded once at startup."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from agent.context import CoachContext
from agent.tools import make_tools
from index.retrieve import CorpusRetriever

load_dotenv()

_ctx: CoachContext | None = None
_retriever: CorpusRetriever | None = None
_tools: dict | None = None


def init_context() -> None:
    """Called once at server startup via lifespan."""
    global _ctx, _retriever, _tools

    gravityos = os.environ.get("GRAVITYOS_DATA_DIR")
    data_dir = os.environ.get("DATA_DIR", "data/synthetic")

    if gravityos:
        _ctx = CoachContext(gravityos_dir=Path(gravityos))
    else:
        _ctx = CoachContext(data_dir=Path(data_dir))

    try:
        _retriever = CorpusRetriever()
    except FileNotFoundError:
        _retriever = None

    _tools = {t.name: t for t in make_tools(_ctx, _retriever)}


def get_ctx() -> CoachContext:
    if _ctx is None:
        raise RuntimeError("Context not initialised — call init_context() at startup")
    return _ctx


def get_retriever() -> CorpusRetriever | None:
    return _retriever


def get_tools() -> dict:
    if _tools is None:
        raise RuntimeError("Tools not initialised — call init_context() at startup")
    return _tools
