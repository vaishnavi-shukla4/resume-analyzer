"""Sidebar navigation using native Streamlit widgets."""

from __future__ import annotations

import streamlit as st

MODES = {
    "Single Analysis": "single",
    "Demo / Sample": "demo",
    "About": "about",
}


def render_sidebar() -> str:
    st.sidebar.header("Navigation")
    choice = st.sidebar.radio("Mode", list(MODES.keys()), index=0)
    st.sidebar.divider()
    st.sidebar.markdown("**Signals**")
    st.sidebar.markdown(
        "- Traditional NLP: TF-IDF, skills, keywords\n"
        "- LLM: contextual match via Groq\n"
        "- Final score: configurable weighted blend"
    )
    return MODES[choice]
