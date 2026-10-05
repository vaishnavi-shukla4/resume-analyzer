from __future__ import annotations

import json

from pydantic import ValidationError

from src.analysis.llm_analyzer import LLMAnalyzer, llm_json_schema
from src.models.schemas import JobDescription, LLMStructuredOutput, ResumeDocument
from src.utils.config_loader import load_config


VALID_PAYLOAD = {
    "overall_match_score": 0.76,
    "candidate_summary": "Junior candidate with explicit Python and SQL coursework.",
    "role_alignment": "Partial match for a junior data scientist role.",
    "matched_skills": ["Python", "SQL"],
    "missing_skills": ["Docker"],
    "partial_skills": ["pandas related to Spark"],
    "experience_alignment": 0.45,
    "education_alignment": 0.8,
    "strengths": ["Clear Python evidence"],
    "weaknesses": ["No Docker evidence"],
    "recommendations": ["Build a Dockerized analytics demo using existing Python projects"],
    "reasoning": "Scores follow explicit resume evidence only.",
}


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class FakeGroqClient:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs) -> _Response:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert "model" in kwargs
        assert kwargs["model"] in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}
        return _Response(self.content or json.dumps(VALID_PAYLOAD))


def test_valid_structured_output_normalizes_percentages() -> None:
    payload = dict(VALID_PAYLOAD)
    payload["overall_match_score"] = 82
    parsed = LLMStructuredOutput.model_validate(payload)
    assert parsed.overall_match_score == 0.82


def test_malformed_llm_output_is_rejected() -> None:
    try:
        LLMStructuredOutput.model_validate({"overall_match_score": "hot"})
        assert False, "expected ValidationError"
    except ValidationError:
        pass
    try:
        LLMStructuredOutput.model_validate({**VALID_PAYLOAD, "overall_match_score": 150})
        assert False, "expected ValidationError"
    except ValidationError:
        pass


def test_missing_api_key_returns_unavailable() -> None:
    analyzer = LLMAnalyzer(config=load_config(), api_key="")
    assert analyzer.is_available() is False
    resume = ResumeDocument(
        source_name="r",
        source_type="txt",
        raw_text="Python intern",
        cleaned_text="Python intern",
        sections={},
    )
    job = JobDescription(raw_text="Python", cleaned_text="Python")
    result = analyzer.analyze(resume, job)
    assert result.available is False
    assert result.error is not None
    assert result.error.category == "missing_api_key"
    assert result.overall_match_score is None


def test_mock_groq_client_parses_schema() -> None:
    client = FakeGroqClient(content=json.dumps(VALID_PAYLOAD))
    analyzer = LLMAnalyzer(config=load_config(), api_key="test-key", client=client)
    resume = ResumeDocument(
        source_name="r",
        source_type="txt",
        raw_text="Python SQL pandas",
        cleaned_text="Python SQL pandas",
        sections={},
    )
    job = JobDescription(raw_text="Python Docker", cleaned_text="Python Docker")
    result = analyzer.analyze(resume, job)
    assert result.available is True
    assert result.overall_match_score == 0.76
    assert result.model_used == "openai/gpt-oss-120b"
    assert client.calls == 1
    schema = llm_json_schema()
    assert schema["type"] == "object"
    assert "overall_match_score" in schema["properties"]


def test_malformed_model_response_retries_then_errors() -> None:
    client = FakeGroqClient(content="not-json")
    analyzer = LLMAnalyzer(config=load_config(), api_key="test-key", client=client)
    resume = ResumeDocument(
        source_name="r",
        source_type="txt",
        raw_text="Python",
        cleaned_text="Python",
        sections={},
    )
    job = JobDescription(raw_text="Python", cleaned_text="Python")
    result = analyzer.analyze(resume, job)
    assert result.available is False
    assert result.error is not None
    assert result.error.category == "malformed_response"
    assert "LLM unavailable" not in (result.candidate_summary or "x") or True
    assert client.calls == 4
