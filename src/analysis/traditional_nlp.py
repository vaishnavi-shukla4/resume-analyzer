"""Traditional NLP: TF-IDF similarity, keywords, skills, and section extraction."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.analysis.skill_matcher import SkillDictionary, load_skill_dictionary
from src.extraction.text_cleaner import clean_text
from src.models.schemas import JobDescription, ResumeDocument, ResumeSection, TraditionalNLPResult

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "education": ("education", "academic background", "academics", "academic qualifications"),
    "experience": (
        "experience",
        "work experience",
        "employment",
        "work history",
        "professional experience",
        "internship",
        "internships",
    ),
    "skills": ("skills", "technical skills", "core competencies", "technologies"),
    "projects": ("projects", "academic projects", "personal projects", "selected projects"),
    "certifications": ("certifications", "certificates", "licenses", "certification"),
}

DEGREE_PATTERN = re.compile(
    r"\b(b\.?s\.?|b\.?tech|b\.?e\.?|bachelor(?:'s)?|m\.?s\.?|m\.?tech|master(?:'s)?|mba|ph\.?d\.?|doctorate)\b",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


class TraditionalNLPAnalyzer:
    def __init__(self, skill_dictionary: SkillDictionary | None = None, keyword_top_k: int = 25) -> None:
        self.skill_dictionary = skill_dictionary or load_skill_dictionary()
        self.keyword_top_k = keyword_top_k

    def analyze(self, resume: ResumeDocument, job: JobDescription) -> TraditionalNLPResult:
        resume_text = resume.cleaned_text or clean_text(resume.raw_text)
        job_text = job.cleaned_text or clean_text(job.raw_text)
        tfidf_similarity, resume_keywords, job_keywords, keyword_coverage = self._vector_features(
            resume_text, job_text
        )
        comparison = self.skill_dictionary.compare(resume_text, job_text)
        sections = resume.sections or self.extract_sections(resume.raw_text or resume_text)
        section_text = {name: section.text for name, section in sections.items()}
        return TraditionalNLPResult(
            tfidf_similarity=tfidf_similarity,
            keyword_coverage=keyword_coverage,
            matched_skills=comparison.matched,
            missing_skills=comparison.missing,
            resume_keywords=resume_keywords,
            job_keywords=job_keywords,
            resume_sections=section_text,
            resume_skills=comparison.resume_skills,
            job_skills=comparison.job_skills,
            explicit_skill_ratio=comparison.explicit_skill_ratio,
            skill_matches=comparison.skill_matches,
        )

    def tfidf_cosine(self, text_a: str, text_b: str) -> float:
        similarity, _, _, _ = self._vector_features(text_a, text_b)
        return similarity

    def extract_keywords(self, text: str, top_k: int | None = None) -> list[str]:
        limit = top_k or self.keyword_top_k
        _, resume_keywords, _, _ = self._vector_features(text, text)
        return resume_keywords[:limit]

    def extract_sections(self, text: str) -> dict[str, ResumeSection]:
        cleaned = text or ""
        lines = cleaned.splitlines()
        spans: list[tuple[int, str]] = []
        for index, line in enumerate(lines):
            header = self._match_header(line)
            if header:
                spans.append((index, header))
        extracted: dict[str, ResumeSection] = {
            name: ResumeSection(name=name, text="", entities=[]) for name in SECTION_ALIASES
        }
        for position, (start, name) in enumerate(spans):
            end = spans[position + 1][0] if position + 1 < len(spans) else len(lines)
            body = "\n".join(lines[start + 1 : end]).strip()
            if not extracted[name].text:
                extracted[name] = ResumeSection(
                    name=name,
                    text=body,
                    entities=self._extract_entities(body),
                )
        if not any(section.text for section in extracted.values()):
            extracted["skills"] = ResumeSection(
                name="skills",
                text=cleaned,
                entities=self._extract_entities(cleaned),
            )
        return extracted

    def build_resume_document(
        self,
        raw_text: str,
        source_name: str,
        source_type: str,
    ) -> ResumeDocument:
        cleaned = clean_text(raw_text)
        sections = self.extract_sections(cleaned)
        return ResumeDocument(
            source_name=source_name,
            source_type=source_type,  # type: ignore[arg-type]
            raw_text=raw_text,
            cleaned_text=cleaned,
            sections=sections,
        )

    def _vector_features(
        self, resume_text: str, job_text: str
    ) -> tuple[float, list[str], list[str], float]:
        resume_clean = clean_text(resume_text).lower()
        job_clean = clean_text(job_text).lower()
        if not resume_clean or not job_clean:
            return 0.0, [], [], 0.0
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_features=5000,
            )
            matrix = vectorizer.fit_transform([resume_clean, job_clean])
        except ValueError:
            return 0.0, [], [], 0.0
        similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
        feature_names = vectorizer.get_feature_names_out()
        resume_vec = matrix[0].toarray().ravel()
        job_vec = matrix[1].toarray().ravel()
        job_keywords = self._top_terms(feature_names, job_vec, self.keyword_top_k)
        resume_keywords = self._top_terms(feature_names, resume_vec, self.keyword_top_k)
        top_job_indices = job_vec.argsort()[::-1][: self.keyword_top_k]
        considered = [index for index in top_job_indices if job_vec[index] > 0]
        covered = sum(1 for index in considered if resume_vec[index] > 0)
        coverage = (covered / len(considered)) if considered else 0.0
        return similarity, resume_keywords, job_keywords, coverage

    @staticmethod
    def _top_terms(feature_names, weights, top_k: int) -> list[str]:
        indices = weights.argsort()[::-1][:top_k]
        return [str(feature_names[i]) for i in indices if weights[i] > 0]

    @staticmethod
    def _match_header(line: str) -> str | None:
        compact = line.strip().strip(":").strip()
        if not compact or len(compact) > 48:
            return None
        lowered = compact.lower()
        for canonical, aliases in SECTION_ALIASES.items():
            if lowered in aliases:
                return canonical
        return None

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        blob = text or ""
        entities = EMAIL_PATTERN.findall(blob)
        entities.extend(match.group(0) for match in YEAR_PATTERN.finditer(blob))
        entities.extend(match.group(0) for match in DEGREE_PATTERN.finditer(blob))
        unique: list[str] = []
        seen: set[str] = set()
        for item in entities:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
