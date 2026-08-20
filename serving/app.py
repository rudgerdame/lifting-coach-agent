"""Lifting Coach Agent — Streamlit UI.

Connects to the FastAPI backend (default http://localhost:8000).
Set API_BASE_URL env var or use the sidebar to override.

Run:
    streamlit run serving/app.py
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lifting Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏋️ Lifting Coach")
    api_url = st.text_input("API URL", value=API_BASE)

    st.divider()

    # Health check
    try:
        resp = httpx.get(f"{api_url}/health", timeout=3)
        info = resp.json()
        st.success(f"API online — `{info.get('data_dir', '?')}`")
        corpus_ok = info.get("corpus_index", False)
        if corpus_ok:
            st.caption("✅ Corpus index loaded")
        else:
            st.caption("⚠️ Corpus index missing — run `python -m index.build`")
    except Exception:
        st.error("API offline — start the server first:\n`uvicorn api.main:app --reload`")
        st.stop()

    st.divider()
    st.caption("**Quick actions**")
    quick = st.radio(
        "Jump to",
        ["Chat", "Readiness", "Progress", "Workout plan", "History", "Search corpus"],
        label_visibility="collapsed",
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(path: str, **params) -> dict:
    r = httpx.get(f"{api_url}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    r = httpx.post(f"{api_url}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _band_badge(band: str) -> str:
    colours = {"above_trend": "🟢", "at_trend": "🟡", "below_trend": "🔴"}
    return colours.get(band, "⚪") + f" **{band.replace('_', ' ').title()}**"


def _render_readiness(data: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Exercise", data.get("exercise", "—"))
    col2.metric("Session date", str(data.get("session_date", "—")))
    col3.metric("Δ e1RM (kg)", f"{data.get('performance_delta_kg', 0):+.2f}")

    st.markdown(f"### Readiness: {_band_badge(data.get('band', ''))}")

    probs = data.get("class_probs")
    if probs:
        st.progress(
            probs.get("above_trend", 0),
            text=f"Above trend: {probs.get('above_trend', 0):.0%}",
        )
        st.progress(
            probs.get("at_trend", 0),
            text=f"At trend: {probs.get('at_trend', 0):.0%}",
        )
        st.progress(
            probs.get("below_trend", 0),
            text=f"Below trend: {probs.get('below_trend', 0):.0%}",
        )

    drivers = data.get("key_drivers", {})
    if drivers:
        with st.expander("Key drivers"):
            for k, v in drivers.items():
                if v is not None:
                    st.markdown(f"- **{k.replace('_', ' ')}**: `{v}`")


# ── Chat tab ──────────────────────────────────────────────────────────────────

def _chat_tab() -> None:
    st.header("Ask the coach")
    st.caption("Natural language questions → tool output. Try: *Am I ready for bench today?*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a coaching question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = _post("/ask", {"question": prompt})
                    tool = resp.get("tool_used", "?")
                    result = resp.get("result", {})

                    # If it's a readiness result render it nicely
                    if tool == "predict_readiness" and "band" in result:
                        st.caption(f"Tool: `{tool}`")
                        _render_readiness(result)
                        content = f"*Tool: {tool}* — {result.get('band', '')} ({result.get('performance_delta_kg', 0):+.2f} kg)"
                    else:
                        pretty = json.dumps(result, indent=2, default=str)
                        st.caption(f"Tool: `{tool}`")
                        st.code(pretty, language="json")
                        content = f"*Tool: {tool}*\n```json\n{pretty}\n```"
                except Exception as exc:
                    content = f"⚠️ Error: {exc}"
                    st.error(content)

        st.session_state.messages.append({"role": "assistant", "content": content})


# ── Readiness tab ─────────────────────────────────────────────────────────────

def _readiness_tab() -> None:
    st.header("Session readiness")
    st.caption("Predicts how today's planned session will go based on current recovery + training load.")

    exercise = st.text_input("Exercise name", placeholder="Dumbbell Incline Bench Press")
    col1, col2 = st.columns(2)
    today_mode = col1.toggle("Score today (default)", value=True)
    session_date = col2.text_input(
        "Specific date (YYYY-MM-DD)",
        disabled=today_mode,
        placeholder="2025-06-15",
    )

    if st.button("Predict", type="primary"):
        with st.spinner("Running model…"):
            try:
                params = {"today": today_mode}
                if not today_mode and session_date:
                    params["session_date"] = session_date
                data = _get(f"/readiness/{exercise}", **params)
                _render_readiness(data)
            except httpx.HTTPStatusError as exc:
                st.error(f"Error {exc.response.status_code}: {exc.response.json().get('detail', exc)}")
            except Exception as exc:
                st.error(str(exc))


# ── Workout plan tab ──────────────────────────────────────────────────────────

def _plan_tab() -> None:
    st.header("Next workout plan")

    split_override = st.selectbox(
        "Split override (optional)",
        ["Auto", "push", "pull", "legs"],
        index=0,
    )

    if st.button("Generate plan", type="primary"):
        with st.spinner("Building plan…"):
            try:
                params = {} if split_override == "Auto" else {"split": split_override}
                data = _get("/plan", **params)

                col1, col2 = st.columns(2)
                col1.metric("Split", data.get("split", "?").upper())
                deload = data.get("deload_recommended", False)
                col2.metric("Deload recommended", "Yes ⚠️" if deload else "No ✅")

                if deload:
                    st.warning("Deload recommended — reduce volume/intensity this week.")

                exercises = data.get("exercises", [])
                if exercises:
                    st.subheader("Exercises")
                    for ex in exercises:
                        with st.expander(f"**{ex.get('exercise', '?')}** — {ex.get('muscle_group', '')}"):
                            for key in ("sets", "reps", "load_kg", "notes"):
                                if key in ex:
                                    st.markdown(f"- **{key}**: {ex[key]}")
                            band = ex.get("readiness_band")
                            if band:
                                st.markdown(f"- **readiness**: {_band_badge(band)}")
                else:
                    st.code(json.dumps(data, indent=2), language="json")

            except Exception as exc:
                st.error(str(exc))


# ── History tab ───────────────────────────────────────────────────────────────

def _history_tab() -> None:
    st.header("Session history")

    col1, col2, col3 = st.columns(3)
    exercise = col1.text_input("Exercise (optional)", placeholder="Incline Bench")
    split = col2.selectbox("Split (optional)", ["All", "push", "pull", "legs"])
    last_n = col3.slider("Sessions", 1, 20, 5)

    if st.button("Load history", type="primary"):
        with st.spinner("Fetching…"):
            try:
                params = {"last_n": last_n}
                if exercise:
                    params["exercise"] = exercise
                if split != "All":
                    params["split"] = split
                data = _get("/history", **params)

                sessions = data.get("sessions", [])
                if sessions:
                    import pandas as pd
                    df = pd.DataFrame(sessions)
                    display_cols = [c for c in
                        ["session_date", "exercise", "split", "top_set_e1rm_kg",
                         "volume_load_kg", "n_working_sets", "performance_delta_kg"]
                        if c in df.columns]
                    st.dataframe(df[display_cols], use_container_width=True)
                else:
                    st.info("No sessions found.")

            except Exception as exc:
                st.error(str(exc))


# ── Corpus search tab ─────────────────────────────────────────────────────────

def _search_tab() -> None:
    st.header("Research corpus")
    st.caption("Search hypertrophy, recovery, and coaching research snippets.")

    query = st.text_input("Search query", placeholder="How much volume for hypertrophy?")
    k = st.slider("Results", 1, 8, 3)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching…"):
            try:
                data = _get("/search", q=query, k=k)
                hits = data.get("hits", [])
                if hits:
                    for hit in hits:
                        with st.expander(
                            f"**{hit.get('title', '?')}** — `{hit.get('source', '?')}` (score: {hit.get('score', 0):.3f})"
                        ):
                            st.markdown(hit.get("text", ""))
                            st.caption(hit.get("citation", ""))
                else:
                    st.info("No results.")
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", str(exc))
                st.error(detail)
            except Exception as exc:
                st.error(str(exc))


# ── Progress tab ──────────────────────────────────────────────────────────────

def _progress_tab() -> None:
    st.header("Exercise progress")
    st.caption("e1RM trend and readiness band history for any exercise.")

    import pandas as pd

    exercise = st.text_input("Exercise name", placeholder="Dumbbell Incline Bench Press", key="progress_ex")
    last_n = st.slider("Sessions to show", 10, 100, 30, key="progress_n")

    if st.button("Load progress", type="primary") and exercise:
        with st.spinner("Loading…"):
            try:
                data = _get(f"/progress/{exercise}", last_n=last_n)
                sessions = data.get("sessions", [])
                if not sessions:
                    st.info("No sessions found.")
                    return

                df = pd.DataFrame(sessions)
                df["session_date"] = pd.to_datetime(df["session_date"])

                # ── e1RM over time ────────────────────────────────────────────
                st.subheader(f"e1RM — {data.get('exercise', exercise)}")

                import altair as alt

                band_colours = {
                    "above_trend": "#22c55e",
                    "at_trend":    "#eab308",
                    "below_trend": "#ef4444",
                    None:          "#94a3b8",
                }
                df["colour"] = df["band"].map(band_colours).fillna("#94a3b8")
                df["band_label"] = df["band"].fillna("unknown").str.replace("_", " ").str.title()

                line = (
                    alt.Chart(df)
                    .mark_line(color="#6366f1", strokeWidth=2)
                    .encode(
                        x=alt.X("session_date:T", title="Date"),
                        y=alt.Y("top_set_e1rm_kg:Q", title="Top-set e1RM (kg)", scale=alt.Scale(zero=False)),
                        tooltip=["session_date:T", "top_set_e1rm_kg:Q", "band_label:N"],
                    )
                )
                points = (
                    alt.Chart(df)
                    .mark_circle(size=80)
                    .encode(
                        x="session_date:T",
                        y=alt.Y("top_set_e1rm_kg:Q", scale=alt.Scale(zero=False)),
                        color=alt.Color(
                            "band_label:N",
                            scale=alt.Scale(
                                domain=["Above Trend", "At Trend", "Below Trend", "Unknown"],
                                range=["#22c55e", "#eab308", "#ef4444", "#94a3b8"],
                            ),
                            legend=alt.Legend(title="Readiness"),
                        ),
                        tooltip=["session_date:T", "top_set_e1rm_kg:Q", "band_label:N",
                                 alt.Tooltip("performance_delta_kg:Q", format="+.2f", title="Delta (kg)"),
                                 alt.Tooltip("class_confidence:Q", format=".0%", title="Confidence")],
                    )
                )
                st.altair_chart((line + points).interactive(), use_container_width=True)

                # ── Volume + ACWR ─────────────────────────────────────────────
                with st.expander("Volume & ACWR"):
                    vol_df = df[df["volume_load_kg"].notna()].copy()
                    if not vol_df.empty:
                        vol_chart = (
                            alt.Chart(vol_df)
                            .mark_bar(color="#818cf8", opacity=0.7)
                            .encode(
                                x=alt.X("session_date:T", title="Date"),
                                y=alt.Y("volume_load_kg:Q", title="Volume load (kg)"),
                                tooltip=["session_date:T", "volume_load_kg:Q",
                                         alt.Tooltip("acwr:Q", format=".2f", title="ACWR")],
                            )
                        )
                        acwr_line = (
                            alt.Chart(vol_df[vol_df["acwr"].notna()])
                            .mark_line(color="#f59e0b", strokeWidth=2, strokeDash=[4, 2])
                            .encode(
                                x="session_date:T",
                                y=alt.Y("acwr:Q", title="ACWR", scale=alt.Scale(zero=False)),
                            )
                        )
                        st.altair_chart(
                            alt.layer(vol_chart).resolve_scale(y="independent").interactive(),
                            use_container_width=True,
                        )
                        st.altair_chart(acwr_line.interactive(), use_container_width=True)

                # ── Summary stats ─────────────────────────────────────────────
                with st.expander("Summary"):
                    valid = df[df["top_set_e1rm_kg"].notna()]
                    if not valid.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Sessions", len(valid))
                        col2.metric("Best e1RM", f"{valid['top_set_e1rm_kg'].max():.1f} kg")
                        col3.metric("Latest e1RM", f"{valid.iloc[-1]['top_set_e1rm_kg']:.1f} kg")
                        delta_pct = (
                            (valid.iloc[-1]["top_set_e1rm_kg"] - valid.iloc[0]["top_set_e1rm_kg"])
                            / valid.iloc[0]["top_set_e1rm_kg"] * 100
                        ) if len(valid) > 1 else 0
                        col4.metric("Progress", f"{delta_pct:+.1f}%")

                        band_counts = df["band_label"].value_counts()
                        st.caption("Readiness distribution: " + " · ".join(
                            f"{k}: {v}" for k, v in band_counts.items()
                        ))

            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", str(exc))
                st.error(detail)
            except Exception as exc:
                st.error(str(exc))


# ── Route to active tab ───────────────────────────────────────────────────────

tabs = {
    "Chat": _chat_tab,
    "Readiness": _readiness_tab,
    "Progress": _progress_tab,
    "Workout plan": _plan_tab,
    "History": _history_tab,
    "Search corpus": _search_tab,
}
tabs[quick]()
