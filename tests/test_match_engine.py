from __future__ import annotations

from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.match_engine import MatchEngine, run_full_analysis, validate_texts
from src.analysis.traditional_nlp import TraditionalNLPAnalyzer
from src.models.schemas import AnalysisError, LLMMatchResult, LLMStructuredOutput, TraditionalNLPResult
from src.utils.cache import LRUCache
from src.utils.config_loader import get_analysis_weights, load_config


def _traditional() -> TraditionalNLPResult:
    return TraditionalNLPResult(
        tfidf_similarity=0.50,
        keyword_coverage=0.40,
        matched_skills=["Python"],
        missing_skills=["Docker"],
        explicit_skill_ratio=0.50,
    )


def test_final_score_is_weighted_not_copied_from_llm() -> None:
    config = load_config()
    engine = MatchEngine(get_analysis_weights(config))
    structured = LLMStructuredOutput(
            overall_match_score=0.99,
            candidate_summary="Strong python intern",
            role_alignment="Partial junior match",
            matched_skills=["Python"],
            missing_skills=["Docker", "AWS"],
            partial_skills=["pandas"],
            experience_alignment=0.40,
            education_alignment=0.80,
            strengths=["Python"],
            weaknesses=["No Docker"],
            recommendations=["Learn Docker"],
            reasoning="Evidence-based",
        )
    llm = LLMMatchResult.from_structured(
        structured,
        model_used="openai/gpt-oss-120b",
    )
    result = engine.combine(_traditional(), llm)
    assert result.llm_assessment_score == 0.99
    assert result.final_match_score != 0.99
    assert 0.0 <= result.final_match_score <= 1.0
    assert result.llm_included_in_final is True


def test_llm_unavailable_does_not_use_zero_as_prediction() -> None:
    config = load_config()
    engine = MatchEngine(get_analysis_weights(config))
    llm = LLMMatchResult.unavailable(
        AnalysisError(category="missing_api_key", message="GROQ_API_KEY is not set.")
    )
    result = engine.combine(_traditional(), llm)
    assert result.llm_assessment_score is None
    assert result.llm_included_in_final is False
    assert "LLM unavailable" in result.score_explanation
    expected = engine.traditional_score(_traditional())
    assert abs(result.final_match_score - expected) < 1e-9


def test_empty_inputs_are_rejected() -> None:
    error = validate_texts("", "job description text here", 20, 50000)
    assert error is not None
    assert error.category == "empty_resume"
    error_job = validate_texts("resume text that is long enough", "", 20, 50000)
    assert error_job is not None
    assert error_job.category == "empty_job"


def test_run_full_analysis_empty_and_sample(tmp_path) -> None:
    config = load_config()
    nlp = TraditionalNLPAnalyzer()
    llm = LLMAnalyzer(config=config, api_key="")
    engine = MatchEngine(get_analysis_weights(config))
    empty = run_full_analysis(
        "",
        "job",
        source_name="x",
        source_type="txt",
        nlp=nlp,
        llm=llm,
        engine=engine,
        minimum_text_length=20,
        max_text_length=50000,
    )
    assert empty.ok is False
    assert empty.error is not None
    assert empty.error.category == "empty_resume"

    resume = (tmp_path / "r.txt").read_text() if False else (
        "EDUCATION\nBachelor of Science 2024\nSKILLS\nPython SQL Pandas Machine Learning Git FastAPI REST APIs\n"
        "EXPERIENCE\nIntern built pandas and scikit-learn models.\n"
    )
    job = "Requires Python SQL Pandas Machine Learning Power BI AWS Docker Statistics"
    outcome = run_full_analysis(
        resume,
        job,
        source_name="sample",
        source_type="sample",
        nlp=nlp,
        llm=llm,
        engine=engine,
        cache=LRUCache(max_size=8, enabled=True),
        minimum_text_length=20,
        max_text_length=50000,
    )
    assert outcome.ok is True
    assert outcome.result is not None
    assert outcome.result.llm.available is False
    assert "Python" in outcome.result.traditional.matched_skills
    assert "Docker" in outcome.result.traditional.missing_skills
