"""Programmatic weighted match engine. LLM score is never used as the final score by itself."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.extraction.text_cleaner import clean_text
from src.models.schemas import (
    AnalysisError,
    AnalysisOutcome,
    JobDescription,
    LLMMatchResult,
    MatchResult,
    TraditionalNLPResult,
)
from src.utils.cache import LRUCache

if TYPE_CHECKING:
    from src.analysis.llm_analyzer import LLMAnalyzer
    from src.analysis.traditional_nlp import TraditionalNLPAnalyzer


class MatchEngine:
    def __init__(self, weights: dict[str, float]) -> None:
        required = {
            "tfidf",
            "keyword_coverage",
            "skill_alignment",
            "experience_alignment",
            "education_alignment",
            "explicit_skill_ratio",
        }
        missing = required - set(weights)
        if missing:
            raise KeyError(f"Missing match weights: {sorted(missing)}")
        self.weights = {key: float(weights[key]) for key in required}

    def traditional_score(self, traditional: TraditionalNLPResult) -> float:
        parts = {
            "tfidf": traditional.tfidf_similarity,
            "keyword_coverage": traditional.keyword_coverage,
            "explicit_skill_ratio": traditional.explicit_skill_ratio,
        }
        return self._weighted_average(parts)

    def llm_skill_alignment(
        self, llm: LLMMatchResult, traditional: TraditionalNLPResult
    ) -> float:
        if not llm.available:
            return traditional.explicit_skill_ratio
        matched = len(llm.matched_skills)
        missing = len(llm.missing_skills)
        partial = len(llm.partial_skills)
        total = matched + missing + partial
        if total == 0:
            return traditional.explicit_skill_ratio
        return (matched + 0.5 * partial) / total

    def combine(self, traditional: TraditionalNLPResult, llm: LLMMatchResult) -> MatchResult:
        traditional_nlp_score = self.traditional_score(traditional)
        if llm.available and llm.overall_match_score is not None:
            components = {
                "tfidf": traditional.tfidf_similarity,
                "keyword_coverage": traditional.keyword_coverage,
                "skill_alignment": self.llm_skill_alignment(llm, traditional),
                "experience_alignment": float(llm.experience_alignment or 0.0),
                "education_alignment": float(llm.education_alignment or 0.0),
                "explicit_skill_ratio": traditional.explicit_skill_ratio,
            }
            final_match_score = self._weighted_average(components)
            llm_assessment_score = llm.overall_match_score
            included = True
            explanation = (
                "Final Match Score is a weighted blend of traditional NLP features "
                "(TF-IDF similarity, keyword coverage, explicit skill ratio) and LLM "
                "contextual features (skill alignment, experience alignment, education alignment). "
                "The LLM Assessment is a separate model judgment and is not copied as the final score."
            )
        else:
            components = {
                "tfidf": traditional.tfidf_similarity,
                "keyword_coverage": traditional.keyword_coverage,
                "explicit_skill_ratio": traditional.explicit_skill_ratio,
            }
            final_match_score = self._weighted_average(components)
            llm_assessment_score = None
            included = False
            category = llm.error.category if llm.error else "unknown"
            explanation = (
                "LLM unavailable"
                f" ({category}). Final Match Score uses only traditional NLP signals: "
                "TF-IDF similarity, keyword coverage, and explicit matched-skill ratio."
            )
        return MatchResult(
            traditional=traditional,
            llm=llm,
            traditional_nlp_score=traditional_nlp_score,
            llm_assessment_score=llm_assessment_score,
            final_match_score=final_match_score,
            skill_matches=traditional.skill_matches,
            weights_used=dict(self.weights),
            llm_included_in_final=included,
            score_explanation=explanation,
        )

    def _weighted_average(self, components: dict[str, float]) -> float:
        total_weight = 0.0
        total_value = 0.0
        for name, value in components.items():
            weight = self.weights[name]
            total_weight += weight
            total_value += weight * float(value)
        if total_weight <= 0:
            return 0.0
        return total_value / total_weight


def validate_texts(
    resume_text: str,
    job_text: str,
    minimum_text_length: int,
    max_text_length: int,
) -> Optional[AnalysisError]:
    resume = (resume_text or "").strip()
    job = (job_text or "").strip()
    if not resume:
        return AnalysisError(category="empty_resume", message="Resume text is empty.")
    if not job:
        return AnalysisError(category="empty_job", message="Job description text is empty.")
    if len(resume) < minimum_text_length:
        return AnalysisError(
            category="empty_resume",
            message=f"Resume is too short (minimum {minimum_text_length} characters).",
        )
    if len(job) < minimum_text_length:
        return AnalysisError(
            category="empty_job",
            message=f"Job description is too short (minimum {minimum_text_length} characters).",
        )
    if len(resume) > max_text_length:
        return AnalysisError(
            category="huge_resume",
            message=f"Resume exceeds the maximum length of {max_text_length} characters.",
        )
    if len(job) > max_text_length:
        return AnalysisError(
            category="malformed_input",
            message=f"Job description exceeds the maximum length of {max_text_length} characters.",
        )
    return None


def run_full_analysis(
    resume_text: str,
    job_text: str,
    *,
    source_name: str,
    source_type: str,
    nlp: "TraditionalNLPAnalyzer",
    llm: "LLMAnalyzer",
    engine: "MatchEngine",
    cache: Optional[LRUCache] = None,
    minimum_text_length: int = 20,
    max_text_length: int = 50000,
) -> AnalysisOutcome:
    error = validate_texts(resume_text, job_text, minimum_text_length, max_text_length)
    if error is not None:
        return AnalysisOutcome(ok=False, error=error)

    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_text)
    cache_key = None
    if cache is not None:
        cache_key = LRUCache.make_key(
            source_name,
            cleaned_resume,
            cleaned_job,
            "llm" if llm.is_available() else "nlp-only",
        )
        cached = cache.get(cache_key)
        if isinstance(cached, MatchResult):
            return AnalysisOutcome(ok=True, result=cached)

    resume = nlp.build_resume_document(cleaned_resume, source_name, source_type)
    job = JobDescription(raw_text=job_text, cleaned_text=cleaned_job)
    traditional = nlp.analyze(resume, job)
    llm_result = llm.analyze(resume, job)
    result = engine.combine(traditional, llm_result)
    if cache is not None and cache_key is not None:
        cache.put(cache_key, result)
    return AnalysisOutcome(ok=True, result=result)
