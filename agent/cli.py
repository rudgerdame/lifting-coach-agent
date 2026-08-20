"""CLI for the lifting coach LangGraph agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.context import CoachContext
from agent.cursor_client import ask_cursor, cursor_endpoint_summary
from agent.graph import build_agent
from agent.llm import check_llm_connection, llm_endpoint_summary
from agent.prompts import SYSTEM_PROMPT
from agent.router import tools_only_answer
from index.retrieve import CorpusRetriever


def _llm_provider() -> str:
    load_dotenv()
    return os.environ.get("LLM_PROVIDER", "ollama").strip().lower()


def _cursor_answer(ctx: CoachContext, retriever: CorpusRetriever | None, question: str) -> str:
    """Tools first, then Cursor SDK synthesizes a cited coaching answer."""
    tool_output = tools_only_answer(ctx, retriever, question)
    prompt = f"""{SYSTEM_PROMPT}

User question: {question}

Tool results (JSON — ground your answer in this data only):
{tool_output}

Write a concise coaching answer. Include [history], [model], and [source: filename] citations."""
    return ask_cursor(prompt, cwd=Path.cwd())


def _print_response(result: dict) -> None:
    messages = result.get("messages", [])
    if not messages:
        print("(no response)")
        return
    last = messages[-1]
    content = getattr(last, "content", str(last))
    print(content)


def main() -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Lifting coach agent (LangGraph + tools)")
    parser.add_argument("question", nargs="?", help="Coaching question (omit for interactive mode)")
    parser.add_argument("--data-dir", type=Path, help="Normalized data directory (default: data/synthetic)")
    parser.add_argument("--check-llm", action="store_true", help="Test LLM endpoint and exit")
    parser.add_argument("--tools-only", action="store_true", help="Run tools without LLM (no Ollama required)")
    args = parser.parse_args()

    if args.check_llm:
        provider = _llm_provider()
        if provider == "cursor":
            print(f"LLM target: {cursor_endpoint_summary()}")
            print("Cursor SDK — run a question to verify auth (no separate ping).")
            raise SystemExit(0)
        ok, msg = check_llm_connection()
        print(f"LLM target: {llm_endpoint_summary()}")
        print(msg)
        raise SystemExit(0 if ok else 1)

    if args.data_dir:
        ctx = CoachContext(data_dir=args.data_dir)
    else:
        ctx = CoachContext.from_env()

    try:
        retriever = CorpusRetriever()
    except FileNotFoundError:
        print("Warning: FAISS index not found — run: python -m index.build", file=sys.stderr)
        retriever = None

    if args.tools_only:
        if not args.question:
            print("--tools-only requires a question argument", file=sys.stderr)
            raise SystemExit(1)
        print(tools_only_answer(ctx, retriever, args.question))
        return

    if _llm_provider() == "cursor":
        if not args.question:
            print("Cursor mode requires a question argument (interactive mode not yet supported).", file=sys.stderr)
            raise SystemExit(1)
        try:
            print(_cursor_answer(ctx, retriever, args.question))
        except Exception as exc:
            print(f"Cursor agent failed: {exc}", file=sys.stderr)
            print(f"Target: {cursor_endpoint_summary()}", file=sys.stderr)
            print("Fallback: --tools-only", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    try:
        agent = build_agent(ctx, retriever)
    except Exception as exc:
        print(f"Failed to start agent: {exc}", file=sys.stderr)
        print(f"LLM target: {llm_endpoint_summary()}", file=sys.stderr)
        print(
            "Ollama: start server and pull a model. Use --tools-only to skip LLM.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    ok, ping_msg = check_llm_connection()
    if not ok:
        print(f"LLM unreachable: {ping_msg}", file=sys.stderr)
        print(f"LLM target: {llm_endpoint_summary()}", file=sys.stderr)
        print(
            "Fix: start Ollama locally or set OLLAMA_HOST in .env. Or run with --tools-only.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        if args.question:
            result = agent.invoke({"messages": [HumanMessage(content=args.question)]})
            _print_response(result)
            return

        print("Lifting coach agent (type 'quit' to exit)")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in {"quit", "exit", "q"}:
                break
            result = agent.invoke({"messages": [HumanMessage(content=question)]})
            print("\nCoach:", end=" ")
            _print_response(result)
    except Exception as exc:
        if exc.__class__.__name__ == "APIConnectionError" or "ConnectError" in exc.__class__.__name__:
            print(f"LLM connection failed: {exc}", file=sys.stderr)
            print(f"LLM target: {llm_endpoint_summary()}", file=sys.stderr)
            print("Try: --tools-only  or  fix Ollama on Brethren", file=sys.stderr)
            raise SystemExit(1) from exc
        raise


if __name__ == "__main__":
    main()
