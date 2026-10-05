"""Page header using native Streamlit components."""

from __future__ import annotations

import streamlit as st


def render_header() -> None:
    st.title("AI Resume Analyzer")
    st.caption("NLP + LLM powered resume-to-job matching")
    st.markdown(
        "Upload a resume and a job description. Traditional NLP measures overlap; "
        "the LLM adds contextual reasoning. The final score blends both signals."
    )
