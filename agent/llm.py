"""LLM provider abstraction — Brethren Ollama (homelab) or hosted OpenAI-compatible API."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Gravity OS homelab: Brethren = local model server (16 GB VRAM).
# Access from Tzeentch/dev machine via Tailscale MagicDNS — same pattern as rasbora:3000.
# See graVityOS/AI and Automation/Multi-Computer AI Homelab.md
DEFAULT_OLLAMA_HOST = "brethren"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"  # Copilot default on vault; fits Brethren VRAM budget


def normalize_ollama_base_url(raw: str) -> str:
    """
    Accept hostname (brethren), host:port (brethren:11434), or full URL.
    Always return OpenAI-compatible base ending in /v1.
    """
    value = raw.strip().rstrip("/")
    if not value:
        raise ValueError("empty Ollama host/URL")

    if "://" not in value:
        if ":" in value:
            value = f"http://{value}"
        else:
            value = f"http://{value}:11434"

    parsed = urlparse(value)
    if not parsed.hostname:
        raise ValueError(f"invalid Ollama URL: {raw!r}")

    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1"):
        return value.rstrip("/")
    if path in ("", "/"):
        return f"{value}/v1"
    return f"{value}/v1"


def _ollama_base_url() -> str:
    explicit = os.environ.get("OPENAI_BASE_URL", "").strip()
    if explicit:
        return normalize_ollama_base_url(explicit)

    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip()
    if host.lower() in {"local", "localhost", "127.0.0.1"}:
        return "http://localhost:11434/v1"
    return normalize_ollama_base_url(host)


def _resolve_model() -> str:
    return (
        os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_OLLAMA_MODEL
    )


def get_chat_model() -> ChatOpenAI:
    """
    Provider swap via env (or .env in repo root):

      LLM_PROVIDER=ollama (default) | openai
      OLLAMA_HOST=brethren (hostname only) | http://brethren:11434 | localhost
      OPENAI_BASE_URL — overrides OLLAMA_HOST
      OPENAI_API_KEY — default 'ollama' for local Ollama
      LLM_MODEL / OPENAI_MODEL — default qwen3:8b
    """
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        base_url = _ollama_base_url()
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        model = _resolve_model()
    else:
        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER is not ollama")
        model = _resolve_model() if _resolve_model() != DEFAULT_OLLAMA_MODEL else "gpt-4o-mini"

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.2,
    )


def llm_endpoint_summary() -> str:
    """Human-readable endpoint for CLI error messages."""
    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider != "ollama":
        return f"provider={provider} model={_resolve_model()}"
    return f"{_ollama_base_url()} model={_resolve_model()}"


def check_llm_connection(timeout_s: float = 5.0) -> tuple[bool, str]:
    """Ping Ollama /api/tags before starting the agent."""
    load_dotenv()
    if os.environ.get("LLM_PROVIDER", "ollama").lower() != "ollama":
        return True, "skipped (non-ollama provider)"

    base = _ollama_base_url()
    root = base.removesuffix("/v1").rstrip("/")
    url = f"{root}/api/tags"
    try:
        resp = httpx.get(url, timeout=timeout_s)
        resp.raise_for_status()
        return True, f"ok {url}"
    except httpx.HTTPError as exc:
        return False, f"cannot reach {url}: {exc}"
