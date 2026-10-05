from __future__ import annotations

from src.analysis.traditional_nlp import TraditionalNLPAnalyzer
from src.extraction.text_cleaner import clean_text
from src.models.schemas import JobDescription, ResumeDocument


def test_tfidf_similarity_higher_for_overlapping_documents() -> None:
    analyzer = TraditionalNLPAnalyzer()
    similar = analyzer.tfidf_cosine(
        "Python pandas SQL machine learning internship",
        "Python pandas SQL machine learning role",
    )
    different = analyzer.tfidf_cosine(
        "Python pandas SQL machine learning internship",
        "kitchen plumbing recipes and gardening tools",
    )
    assert 0.0 <= different <= similar <= 1.0
    assert similar > 0.2
    assert similar > different


def test_keyword_coverage_and_sections() -> None:
    analyzer = TraditionalNLPAnalyzer()
    raw_resume = """
Jane Candidate
EDUCATION
Bachelor of Science in Computer Science, 2024
EXPERIENCE
Built Python APIs with FastAPI.
SKILLS
Python, SQL, Pandas
PROJECTS
Churn model with scikit-learn
CERTIFICATIONS
None
"""
    job = "Looking for Python SQL Pandas Power BI AWS Docker Statistics"
    resume = analyzer.build_resume_document(raw_resume, "resume.txt", "txt")
    result = analyzer.analyze(resume, JobDescription(raw_text=job, cleaned_text=clean_text(job)))
    assert 0.0 <= result.tfidf_similarity <= 1.0
    assert 0.0 <= result.keyword_coverage <= 1.0
    assert "Python" in result.matched_skills
    assert "Power BI" in result.missing_skills
    assert "education" in result.resume_sections
    assert "Bachelor" in result.resume_sections["education"]
    assert resume.sections["skills"].text


def test_empty_documents_have_zero_similarity() -> None:
    analyzer = TraditionalNLPAnalyzer()
    assert analyzer.tfidf_cosine("", "python") == 0.0
    resume = ResumeDocument(
        source_name="empty",
        source_type="txt",
        raw_text="",
        cleaned_text="",
        sections={},
    )
    job = JobDescription(raw_text="", cleaned_text="")
    result = analyzer.analyze(resume, job)
    assert result.tfidf_similarity == 0.0
    assert result.keyword_coverage == 0.0
