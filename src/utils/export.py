"""Export MatchResult to JSON and CSV."""

from __future__ import annotations

import csv
import io
import json

from src.models.schemas import MatchResult


def to_json(result: MatchResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2)


def to_csv(result: MatchResult) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["field", "value"])
    writer.writerow(["final_match_score", f"{result.final_match_score:.4f}"])
    writer.writerow(["traditional_nlp_score", f"{result.traditional_nlp_score:.4f}"])
    llm_score = (
        f"{result.llm_assessment_score:.4f}"
        if result.llm_assessment_score is not None and result.llm.available
        else "LLM unavailable"
    )
    writer.writerow(["llm_assessment_score", llm_score])
    writer.writerow(["llm_included_in_final", str(result.llm_included_in_final)])
    writer.writerow(["tfidf_similarity", f"{result.traditional.tfidf_similarity:.4f}"])
    writer.writerow(["keyword_coverage", f"{result.traditional.keyword_coverage:.4f}"])
    writer.writerow(["explicit_skill_ratio", f"{result.traditional.explicit_skill_ratio:.4f}"])
    writer.writerow(["matched_skills_nlp", "; ".join(result.traditional.matched_skills)])
    writer.writerow(["missing_skills_nlp", "; ".join(result.traditional.missing_skills)])
    if result.llm.available:
        writer.writerow(["matched_skills_llm", "; ".join(result.llm.matched_skills)])
        writer.writerow(["missing_skills_llm", "; ".join(result.llm.missing_skills)])
        writer.writerow(["partial_skills_llm", "; ".join(result.llm.partial_skills)])
        writer.writerow(["experience_alignment", f"{result.llm.experience_alignment:.4f}"])
        writer.writerow(["education_alignment", f"{result.llm.education_alignment:.4f}"])
        writer.writerow(["candidate_summary", result.llm.candidate_summary])
        writer.writerow(["role_alignment", result.llm.role_alignment])
        writer.writerow(["strengths", "; ".join(result.llm.strengths)])
        writer.writerow(["weaknesses", "; ".join(result.llm.weaknesses)])
        writer.writerow(["recommendations", "; ".join(result.llm.recommendations)])
        writer.writerow(["reasoning", result.llm.reasoning])
        writer.writerow(["model_used", result.llm.model_used or ""])
    else:
        category = result.llm.error.category if result.llm.error else "unknown"
        message = result.llm.error.message if result.llm.error else "LLM unavailable"
        writer.writerow(["llm_status", "LLM unavailable"])
        writer.writerow(["llm_error_category", category])
        writer.writerow(["llm_error_message", message])
    return buffer.getvalue()
