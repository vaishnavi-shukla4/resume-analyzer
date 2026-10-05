"""Central data models. Every module must import from here."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ErrorCategory = Literal[
    "missing_api_key",
    "invalid_api_key",
    "model_unavailable",
    "rate_limit",
    "network_failure",
    "malformed_response",
    "invalid_pdf",
    "empty_resume",
    "empty_job",
    "huge_resume",
    "malformed_input",
    "missing_config",
    "extraction_failed",
]


class AnalysisError(BaseModel):
    category: ErrorCategory
    message: str
    details: Optional[str] = None


class ResumeSection(BaseModel):
    name: str
    text: str = ""
    entities: list[str] = Field(default_factory=list)


class SkillMatch(BaseModel):
    canonical: str
    variants_found: list[str] = Field(default_factory=list)
    source: Literal["resume", "job", "both"] = "both"
    match_type: Literal["matched", "missing", "partial"] = "matched"


class ResumeDocument(BaseModel):
    source_name: str
    source_type: Literal["pdf", "txt", "pasted", "sample"]
    raw_text: str
    cleaned_text: str
    sections: dict[str, ResumeSection] = Field(default_factory=dict)


class JobDescription(BaseModel):
    raw_text: str
    cleaned_text: str


class TraditionalNLPResult(BaseModel):
    tfidf_similarity: float
    keyword_coverage: float
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    resume_keywords: list[str] = Field(default_factory=list)
    job_keywords: list[str] = Field(default_factory=list)
    resume_sections: dict[str, str] = Field(default_factory=dict)
    resume_skills: list[str] = Field(default_factory=list)
    job_skills: list[str] = Field(default_factory=list)
    explicit_skill_ratio: float = 0.0
    skill_matches: list[SkillMatch] = Field(default_factory=list)

    @field_validator("tfidf_similarity", "keyword_coverage", "explicit_skill_ratio")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        return float(min(1.0, max(0.0, value)))


class LLMStructuredOutput(BaseModel):
    """Exact JSON the LLM must return. Used for Groq schema + Pydantic validation."""

    overall_match_score: float
    candidate_summary: str
    role_alignment: str
    matched_skills: list[str]
    missing_skills: list[str]
    partial_skills: list[str]
    experience_alignment: float
    education_alignment: float
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    reasoning: str

    model_config = {"extra": "forbid"}

    @field_validator(
        "overall_match_score",
        "experience_alignment",
        "education_alignment",
    )
    @classmethod
    def _normalize_score(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be numeric")
        number = float(value)
        if 1.0 < number <= 100.0:
            number = number / 100.0
        if number < 0.0 or number > 1.0:
            raise ValueError("score must be between 0 and 1 (or 0-100 percent)")
        return number

    @field_validator(
        "matched_skills",
        "missing_skills",
        "partial_skills",
        "strengths",
        "weaknesses",
        "recommendations",
    )
    @classmethod
    def _string_lists(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("expected a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("list items must be strings")
            text = item.strip()
            if text:
                cleaned.append(text)
        return cleaned

    @field_validator("candidate_summary", "role_alignment", "reasoning")
    @classmethod
    def _required_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("expected a string")
        return value.strip()


class LLMMatchResult(BaseModel):
    available: bool
    overall_match_score: Optional[float] = None
    candidate_summary: str = ""
    role_alignment: str = ""
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    partial_skills: list[str] = Field(default_factory=list)
    experience_alignment: Optional[float] = None
    education_alignment: Optional[float] = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning: str = ""
    error: Optional[AnalysisError] = None
    model_used: Optional[str] = None

    @classmethod
    def from_structured(cls, payload: LLMStructuredOutput, model_used: str) -> "LLMMatchResult":
        return cls(
            available=True,
            overall_match_score=payload.overall_match_score,
            candidate_summary=payload.candidate_summary,
            role_alignment=payload.role_alignment,
            matched_skills=payload.matched_skills,
            missing_skills=payload.missing_skills,
            partial_skills=payload.partial_skills,
            experience_alignment=payload.experience_alignment,
            education_alignment=payload.education_alignment,
            strengths=payload.strengths,
            weaknesses=payload.weaknesses,
            recommendations=payload.recommendations,
            reasoning=payload.reasoning,
            error=None,
            model_used=model_used,
        )

    @classmethod
    def unavailable(cls, error: AnalysisError) -> "LLMMatchResult":
        return cls(available=False, error=error)


class MatchResult(BaseModel):
    traditional: TraditionalNLPResult
    llm: LLMMatchResult
    traditional_nlp_score: float
    llm_assessment_score: Optional[float] = None
    final_match_score: float
    skill_matches: list[SkillMatch] = Field(default_factory=list)
    weights_used: dict[str, float] = Field(default_factory=dict)
    llm_included_in_final: bool = False
    score_explanation: str = ""

    @model_validator(mode="after")
    def _scores_bounded(self) -> "MatchResult":
        self.traditional_nlp_score = min(1.0, max(0.0, self.traditional_nlp_score))
        self.final_match_score = min(1.0, max(0.0, self.final_match_score))
        if self.llm_assessment_score is not None:
            self.llm_assessment_score = min(1.0, max(0.0, self.llm_assessment_score))
        return self


class AnalysisOutcome(BaseModel):
    ok: bool
    result: Optional[MatchResult] = None
    error: Optional[AnalysisError] = None
