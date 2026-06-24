"""LangGraph ReAct agent wiring context, tools, and LLM."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from agent.context import CoachContext
from agent.llm import get_chat_model
from agent.prompts import SYSTEM_PROMPT
from agent.tools import make_tools
from index.retrieve import CorpusRetriever


def build_agent(
    ctx: CoachContext | None = None,
    retriever: CorpusRetriever | None = None,
):
    """
    Compile a ReAct agent: LLM chooses tools → tools return JSON → LLM synthesizes cited answer.
    """
    ctx = ctx or CoachContext.from_env()
    if retriever is None:
        try:
            retriever = CorpusRetriever()
        except FileNotFoundError:
            retriever = None

    tools = make_tools(ctx, retriever)
    model = get_chat_model()
    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
