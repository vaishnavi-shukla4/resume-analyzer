"""Render analysis results with native Streamlit components."""

from __future__ import annotations

import streamlit as st

from src.components.charts import render_all_charts
from src.models.schemas import MatchResult
from src.utils.export import to_csv, to_json


def _pct(value: float | None) -> str:
    if value is None:
        return "LLM unavailable"
    return f"{value * 100:.1f}%"


def render_results(result: MatchResult) -> None:
    st.subheader("Match overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Match Score", _pct(result.final_match_score))
    col2.metric("Traditional NLP Score", _pct(result.traditional_nlp_score))
    if result.llm.available and result.llm_assessment_score is not None:
        col3.metric("LLM Assessment", _pct(result.llm_assessment_score))
    else:
        col3.metric("LLM Assessment", "LLM unavailable")

    st.info(result.score_explanation)

    render_all_charts(result)

    st.subheader("Skills")
    skill_cols = st.columns(3)
    skill_cols[0].write("**Matched skills**")
    matched = result.traditional.matched_skills
    skill_cols[0].write(", ".join(matched) if matched else "None detected")
    skill_cols[1].write("**Missing skills**")
    missing = result.traditional.missing_skills
    skill_cols[1].write(", ".join(missing) if missing else "None detected")
    skill_cols[2].write("**Partial skills (LLM)**")
    if result.llm.available:
        partial = result.llm.partial_skills
        skill_cols[2].write(", ".join(partial) if partial else "None reported")
    else:
        skill_cols[2].write("LLM unavailable")

    st.subheader("Alignment")
    align_cols = st.columns(3)
    align_cols[0].metric("Keyword coverage", _pct(result.traditional.keyword_coverage))
    if result.llm.available:
        align_cols[1].metric("Experience alignment", _pct(result.llm.experience_alignment))
        align_cols[2].metric("Education alignment", _pct(result.llm.education_alignment))
    else:
        align_cols[1].metric("Experience alignment", "LLM unavailable")
        align_cols[2].metric("Education alignment", "LLM unavailable")

    st.subheader("Resume sections")
    if result.traditional.resume_sections:
        for name, text in result.traditional.resume_sections.items():
            with st.expander(name.title(), expanded=False):
                st.write(text if text else "Section not found in resume.")
    else:
        st.write("No labeled sections were detected.")

    st.subheader("LLM contextual analysis")
    if result.llm.available:
        st.write("**Candidate summary**")
        st.write(result.llm.candidate_summary or "No summary returned.")
        st.write("**Role alignment**")
        st.write(result.llm.role_alignment or "No role alignment returned.")
        rec_cols = st.columns(3)
        rec_cols[0].write("**Strengths**")
        rec_cols[0].write("\n".join(f"- {item}" for item in result.llm.strengths) or "None")
        rec_cols[1].write("**Weaknesses**")
        rec_cols[1].write("\n".join(f"- {item}" for item in result.llm.weaknesses) or "None")
        rec_cols[2].write("**Recommendations**")
        rec_cols[2].write("\n".join(f"- {item}" for item in result.llm.recommendations) or "None")
        st.write("**Explanation / reasoning**")
        st.write(result.llm.reasoning)
        if result.llm.model_used:
            st.caption(f"Model used: {result.llm.model_used}")
    else:
        error = result.llm.error
        st.warning("LLM unavailable")
        if error:
            st.write(f"Error category: `{error.category}`")
            st.write(error.message)

    st.subheader("Traditional NLP details")
    st.write("**TF-IDF similarity:**", _pct(result.traditional.tfidf_similarity))
    st.write("**Job keywords:**", ", ".join(result.traditional.job_keywords[:15]) or "None")
    st.write("**Resume keywords:**", ", ".join(result.traditional.resume_keywords[:15]) or "None")

    json_payload = to_json(result)
    csv_payload = to_csv(result)
    downloads = st.columns(2)
    downloads[0].download_button(
        "Export JSON",
        data=json_payload,
        file_name="resume_match_analysis.json",
        mime="application/json",
    )
    downloads[1].download_button(
        "Export CSV",
        data=csv_payload,
        file_name="resume_match_analysis.csv",
        mime="text/csv",
    )
