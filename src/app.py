"""Streamlit UI for the AI Resume Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.match_engine import MatchEngine, run_full_analysis
from src.analysis.traditional_nlp import TraditionalNLPAnalyzer
from src.components.header import render_header
from src.components.results import render_results
from src.components.sidebar import render_sidebar
from src.extraction.pdf_extractor import extract_text
from src.models.schemas import AnalysisError
from src.utils.cache import LRUCache
from src.utils.config_loader import (
    get_analysis_weights,
    get_groq_api_key,
    get_project_root,
    load_config,
    load_dotenv_file,
)


def _load_runtime():
    load_dotenv_file()
    try:
        config = load_config()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return None, AnalysisError(
            category="missing_config",
            message="Could not load config.yaml.",
            details=str(exc),
        )
    analysis_cfg = config.get("analysis", {})
    cache_cfg = config.get("cache", {})
    nlp = TraditionalNLPAnalyzer(keyword_top_k=int(analysis_cfg.get("keyword_top_k", 25)))
    llm = LLMAnalyzer(config=config, api_key=get_groq_api_key())
    engine = MatchEngine(get_analysis_weights(config))
    cache = LRUCache(
        max_size=int(cache_cfg.get("max_size", 100)),
        enabled=bool(cache_cfg.get("enabled", True)),
    )
    return {
        "config": config,
        "nlp": nlp,
        "llm": llm,
        "engine": engine,
        "cache": cache,
        "minimum_text_length": int(analysis_cfg.get("minimum_text_length", 20)),
        "max_text_length": int(analysis_cfg.get("max_text_length", 50000)),
    }, None


def _read_sample(name: str) -> str:
    path = get_project_root() / "sample_data" / name
    return path.read_text(encoding="utf-8")


def _analyze(
    resume_text: str,
    job_text: str,
    source_name: str,
    source_type: str,
    runtime: dict,
) -> None:
    outcome = run_full_analysis(
        resume_text,
        job_text,
        source_name=source_name,
        source_type=source_type,
        nlp=runtime["nlp"],
        llm=runtime["llm"],
        engine=runtime["engine"],
        cache=runtime["cache"],
        minimum_text_length=runtime["minimum_text_length"],
        max_text_length=runtime["max_text_length"],
    )
    if not outcome.ok or outcome.result is None:
        error = outcome.error
        st.error(error.message if error else "Analysis failed.")
        if error:
            st.caption(f"Error category: {error.category}")
        return
    render_results(outcome.result)


def _single_analysis(runtime: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Resume")
        uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        pasted = st.text_area("Or paste resume text", height=220)
    with right:
        st.subheader("Job description")
        job_text = st.text_area("Paste the job description", height=320)

    if st.button("Analyze Resume", type="primary"):
        resume_text = ""
        source_name = "pasted"
        source_type = "pasted"
        extract_error = None
        if uploaded is not None:
            data = uploaded.getvalue()
            source_name = uploaded.name
            resume_text, extract_error = extract_text(data, uploaded.name)
            source_type = "pdf" if uploaded.name.lower().endswith(".pdf") else "txt"
        elif pasted.strip():
            resume_text = pasted
        if extract_error is not None:
            st.error(extract_error.message)
            st.caption(f"Error category: {extract_error.category}")
            return
        _analyze(resume_text, job_text, source_name, source_type, runtime)


def _demo_analysis(runtime: dict) -> None:
    resume_text = _read_sample("sample_resume.txt")
    job_text = _read_sample("sample_job_description.txt")
    st.subheader("Demo / Sample")
    st.markdown(
        "Fictional candidate vs a junior data scientist / data analyst posting. "
        "Expected: Python, SQL, Pandas, and Machine Learning match; Power BI, AWS, and Docker are missing."
    )
    cols = st.columns(2)
    cols[0].text_area("Sample resume", resume_text, height=320, disabled=True)
    cols[1].text_area("Sample job description", job_text, height=320, disabled=True)
    if st.button("Analyze Resume", type="primary"):
        _analyze(resume_text, job_text, "sample_resume.txt", "sample", runtime)


def _about() -> None:
    st.subheader("About")
    st.markdown(
        """
**Problem.** Keyword filters miss context. Pure chatbots miss measurable overlap.

**This project** combines:

1. Traditional NLP — TF-IDF cosine similarity, skill dictionary matching, keyword coverage, section extraction.
2. Groq LLM — structured JSON for experience/education alignment, partial skills, strengths, weaknesses, and recommendations.
3. Match engine — configurable weights in `config.yaml`. The LLM score is shown separately and is never copied as the final score.

**If Groq is unavailable**, traditional NLP still runs. The UI labels LLM results as **LLM unavailable** with an error category. It does not invent a Neutral or zero prediction.
        """
    )
    st.markdown(
        """
Resume + Job Description  
→ Traditional NLP (TF-IDF, skills, keywords)  
→ LLM (contextual JSON via Groq)  
→ Match engine  
→ Final score and recommendations
        """
    )


def main() -> None:
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
    render_header()
    mode = render_sidebar()
    runtime, config_error = _load_runtime()
    if config_error is not None:
        st.error(config_error.message)
        st.caption(config_error.details or "")
        return
    assert runtime is not None
    llm: LLMAnalyzer = runtime["llm"]
    if not llm.is_available():
        error = llm.availability_error()
        st.warning(
            error.message if error else "LLM unavailable. Traditional NLP will still run."
        )
    if mode == "single":
        _single_analysis(runtime)
    elif mode == "demo":
        _demo_analysis(runtime)
    else:
        _about()


if __name__ == "__main__":
    main()
