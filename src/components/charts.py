"""Plotly charts for match scores, skills, and keyword coverage."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.models.schemas import MatchResult


def _gauge(title: str, value: float) -> go.Figure:
    percent = max(0.0, min(1.0, value)) * 100
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4F8BF9"},
                "steps": [
                    {"range": [0, 40], "color": "#3a2a2a"},
                    {"range": [40, 70], "color": "#3a3a2a"},
                    {"range": [70, 100], "color": "#2a3a2a"},
                ],
            },
        )
    )
    figure.update_layout(height=260, margin=dict(t=40, b=20, l=20, r=20), template="plotly_dark")
    return figure


def render_match_gauge(final_score: float) -> None:
    st.plotly_chart(_gauge("Final Match Score", final_score), use_container_width=True)


def render_score_comparison(
    traditional: float,
    llm_score: float | None,
    final_score: float,
) -> None:
    labels = ["Traditional NLP", "LLM Assessment", "Final Match"]
    values = [
        traditional * 100,
        (llm_score * 100) if llm_score is not None else 0,
        final_score * 100,
    ]
    colors = ["#6C9BD2", "#C084FC" if llm_score is not None else "#555555", "#34D399"]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[
                f"{traditional * 100:.1f}%",
                "LLM unavailable" if llm_score is None else f"{llm_score * 100:.1f}%",
                f"{final_score * 100:.1f}%",
            ],
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Resume / JD similarity signals",
        yaxis_title="Score (%)",
        yaxis_range=[0, 115],
        template="plotly_dark",
        height=360,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_skill_bar(matched: list[str], missing: list[str], partial: list[str]) -> None:
    figure = go.Figure(
        go.Bar(
            x=["Matched", "Partial", "Missing"],
            y=[len(matched), len(partial), len(missing)],
            marker_color=["#34D399", "#FBBF24", "#F87171"],
            text=[len(matched), len(partial), len(missing)],
            textposition="auto",
        )
    )
    figure.update_layout(
        title="Skill match counts",
        template="plotly_dark",
        height=320,
        yaxis_title="Count",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_keyword_coverage(coverage: float) -> None:
    figure = go.Figure(
        go.Bar(
            x=["Keyword coverage"],
            y=[coverage * 100],
            marker_color="#60A5FA",
            text=[f"{coverage * 100:.1f}%"],
            textposition="auto",
        )
    )
    figure.update_layout(
        title="Job-description keyword coverage",
        yaxis_range=[0, 100],
        template="plotly_dark",
        height=280,
    )
    st.plotly_chart(figure, use_container_width=True)


def render_all_charts(result: MatchResult) -> None:
    top = st.columns(2)
    with top[0]:
        render_match_gauge(result.final_match_score)
    with top[1]:
        render_keyword_coverage(result.traditional.keyword_coverage)
    render_score_comparison(
        result.traditional_nlp_score,
        result.llm_assessment_score if result.llm.available else None,
        result.final_match_score,
    )
    partial = result.llm.partial_skills if result.llm.available else []
    render_skill_bar(
        result.traditional.matched_skills,
        result.traditional.missing_skills,
        partial,
    )
