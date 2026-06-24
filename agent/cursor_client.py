"""Cursor SDK client — same pattern as graVityOS telegram weekly review."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_CURSOR_MODEL = "composer-2"


def _api_key() -> str:
    load_dotenv()
    key = (os.environ.get("CURSOR_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise ValueError(
            "CURSOR_API_KEY missing. Add it to .env (from https://cursor.com/dashboard/cloud-agents)."
        )
    return key


def _model_id() -> str:
    load_dotenv()
    return (
        os.environ.get("CURSOR_MODEL_ID")
        or os.environ.get("LLM_MODEL")
        or DEFAULT_CURSOR_MODEL
    ).strip()


def _extract_result(result: object) -> str:
    if getattr(result, "status", None) == "error":
        raise RuntimeError(f"Cursor agent error: {result}")

    text = getattr(result, "result", None) or getattr(result, "text", None)
    if text:
        return str(text).strip()
    return str(result).strip()


async def _ask_cursor_async(prompt: str, cwd: Path) -> str:
    """Async bridge — required on Windows (sync bridge uses select() on pipes)."""
    from cursor_sdk import AgentOptions, AsyncAgent, AsyncClient, LocalAgentOptions

    workdir = str(cwd)
    async with await AsyncClient.launch_bridge(workspace=workdir) as client:
        result = await AsyncAgent.prompt(
            prompt,
            AgentOptions(
                api_key=_api_key(),
                model=_model_id(),
                local=LocalAgentOptions(cwd=workdir),
            ),
            client=client,
        )
    return _extract_result(result)


def _ask_cursor_sync(prompt: str, cwd: Path) -> str:
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

    workdir = str(cwd)
    result = Agent.prompt(
        prompt,
        AgentOptions(
            api_key=_api_key(),
            model=_model_id(),
            local=LocalAgentOptions(cwd=workdir),
        ),
    )
    return _extract_result(result)


def ask_cursor(prompt: str, *, cwd: Path | None = None) -> str:
    """
    One-shot local Cursor agent run (Gravity OS polish pattern).

    Requires: pip install cursor-sdk, CURSOR_API_KEY in .env, internet.
    On Windows uses the async bridge (sync path fails with WinError 10038).
    """
    try:
        import cursor_sdk  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install cursor-sdk: pip install cursor-sdk") from exc

    root = cwd or Path.cwd()
    if sys.platform == "win32":
        return asyncio.run(_ask_cursor_async(prompt, root))
    return _ask_cursor_sync(prompt, root)


def cursor_endpoint_summary() -> str:
    load_dotenv()
    bridge = "async bridge" if sys.platform == "win32" else "sync bridge"
    return f"cursor-sdk local ({bridge}) cwd={Path.cwd()} model={_model_id()}"
