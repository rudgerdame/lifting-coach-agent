"""Run gold Q&A eval on agent tools and write eval/results.md.

Modes
-----
Default (no --llm):   tool + router cases only — fast, no LLM required.
--llm cursor:         also runs llm_synthesis cases through Cursor SDK.
--llm ollama:         also runs llm_synthesis cases through Ollama.

LLM synthesis cases check:
  - required citation tags appear in the generated answer ([model], [history], etc.)
  - must_contain substrings appear (grounding check)
  - must_not_contain substrings are absent (hallucination check)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.context import CoachContext
from agent.router import route_question
from agent.tools import make_tools
from index.retrieve import CorpusRetriever

GOLD_PATH = Path("eval/gold_qa.jsonl")
RESULTS_PATH = Path("eval/results.md")


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    detail: str
    tool_used: str | None = None
    failures: list[str] = field(default_factory=list)


def _load_gold(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _check_output(output: str, case: dict) -> list[str]:
    failures: list[str] = []
    for needle in case.get("must_contain", []):
        if needle not in output:
            failures.append(f"missing substring: {needle!r}")
    for citation in case.get("required_citations", []):
        if citation not in output:
            failures.append(f"missing citation: {citation!r}")
    return failures


def _llm_answer(question: str, tool_output: str, provider: str) -> str:
    """Call LLM to synthesise a cited coaching answer from tool output."""
    from agent.prompts import SYSTEM_PROMPT

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User question: {question}\n\n"
        f"Tool results (JSON — ground your answer in this data only):\n{tool_output}\n\n"
        "Write a concise coaching answer. Include [history], [model], and "
        "[source: filename] citations where relevant. Do not invent facts."
    )

    if provider == "cursor":
        from agent.cursor_client import ask_cursor
        return ask_cursor(prompt, cwd=Path.cwd())

    if provider == "ollama":
        from agent.llm import build_llm
        llm = build_llm()
        return llm.invoke(prompt).content

    raise ValueError(f"Unknown LLM provider {provider!r}")


def _run_case(case: dict, tools: dict, llm_provider: str | None = None) -> CaseResult:
    case_id = case["id"]
    mode = case.get("mode", "tool")

    # Skip LLM synthesis cases when no provider is configured
    if mode == "llm_synthesis" and not llm_provider:
        return CaseResult(case_id, True, "skipped (no --llm provider)", None)

    try:
        if mode == "router":
            tool_used, output = route_question(case["question"], tools)
            failures = _check_output(output, case)
            expected = case.get("expected_tool")
            if expected and tool_used != expected:
                failures.append(f"expected tool {expected!r}, got {tool_used!r}")

        elif mode == "llm_synthesis":
            # Run the router to get tool output, then synthesise through LLM
            tool_used, tool_output = route_question(case["question"], tools)
            llm_answer = _llm_answer(case["question"], tool_output, llm_provider)
            output = llm_answer
            failures = _check_output(output, case)
            # Hallucination check
            for bad in case.get("must_not_contain", []):
                if bad.lower() in output.lower():
                    failures.append(f"hallucination: found {bad!r} not in tool output")

        else:
            tool_name = case["tool"]
            tool_args = case.get("tool_args") or {}
            if tool_name not in tools:
                return CaseResult(case_id, False, f"unknown tool {tool_name!r}", tool_name, ["tool not found"])
            output = tools[tool_name].invoke(tool_args)
            tool_used = tool_name
            failures = _check_output(output, case)

        passed = not failures
        detail = "ok" if passed else "; ".join(failures)
        return CaseResult(case_id, passed, detail, tool_used, failures)
    except Exception as exc:
        return CaseResult(case_id, False, str(exc), case.get("tool"), [str(exc)])


def _build_report(
    results: list[CaseResult],
    data_dir: Path,
    llm_provider: str | None = None,
) -> str:
    # Skipped LLM cases count as passed for the score denominator only when provider is absent
    countable = [r for r in results if r.detail != "skipped (no --llm provider)"]
    passed = sum(1 for r in countable if r.passed)
    total = len(countable)
    skipped = len(results) - total
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    llm_note = f"`--llm {llm_provider}`" if llm_provider else "tool + router only (no `--llm`)"

    lines = [
        "# Agent eval results",
        "",
        f"> Generated by `python -m eval.run_all`. {ts}",
        "",
        f"- **Data:** `{data_dir}`",
        f"- **Gold set:** `{GOLD_PATH}`",
        f"- **LLM eval:** {llm_note}",
        f"- **Score:** {passed}/{total} passed" + (f" ({skipped} skipped)" if skipped else ""),
        "",
        "## Summary",
        "",
        "| Case | Mode | Tool | Pass | Detail |",
        "|------|------|------|------|--------|",
    ]
    for r in results:
        mark = "yes" if r.passed else ("skip" if r.detail.startswith("skipped") else "no")
        lines.append(f"| `{r.case_id}` | — | `{r.tool_used or '-'}` | {mark} | {r.detail} |")

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- **Tool cases:** direct tool invoke + JSON structure",
            "- **Router cases:** keyword router picks expected tool",
            "- **Faithfulness:** required citation tags and source filenames in output",
            "- **LLM synthesis:** citation + grounding check on Cursor/Ollama answers (opt-in via `--llm`)",
            "",
            "## Limitations",
            "",
            "- Synthetic data only; regenerate gold set if schema changes materially.",
            "- LLM synthesis cases skipped unless `--llm cursor` or `--llm ollama` is passed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run agent gold Q&A eval")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--llm",
        choices=["cursor", "ollama"],
        default=None,
        help="Also run llm_synthesis cases through this provider (requires LLM_PROVIDER env or --llm flag)",
    )
    args = parser.parse_args()

    # Allow LLM_PROVIDER env var as fallback for --llm
    llm_provider = args.llm or (
        os.environ.get("LLM_PROVIDER", "").strip().lower() or None
    )
    # Only activate LLM eval when a valid provider is set
    if llm_provider not in (None, "cursor", "ollama"):
        llm_provider = None

    if not args.gold.exists():
        print(f"Missing {args.gold}", file=sys.stderr)
        raise SystemExit(1)

    if not (args.data_dir / "workout_sets.jsonl").exists():
        print(f"Missing data in {args.data_dir}", file=sys.stderr)
        raise SystemExit(1)

    try:
        retriever = CorpusRetriever()
    except FileNotFoundError:
        print("Warning: FAISS index missing — corpus cases may fail. Run: python -m index.build", file=sys.stderr)
        retriever = None

    ctx = CoachContext(data_dir=args.data_dir)
    tools = {t.name: t for t in make_tools(ctx, retriever)}

    cases = _load_gold(args.gold)
    results = [_run_case(case, tools, llm_provider=llm_provider) for case in cases]

    report = _build_report(results, args.data_dir, llm_provider=llm_provider)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    countable = [r for r in results if r.detail != "skipped (no --llm provider)"]
    passed = sum(1 for r in countable if r.passed)
    llm_note = f" [llm={llm_provider}]" if llm_provider else ""
    print(f"Wrote {args.out} ({passed}/{len(countable)} passed){llm_note}")
    raise SystemExit(0 if passed == len(countable) else 1)


if __name__ == "__main__":
    main()
