"""Configurable skill dictionary matching with alias normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.models.schemas import SkillMatch
from src.utils.config_loader import get_skills_path


class SkillComparison:
    def __init__(
        self,
        matched: list[str],
        missing: list[str],
        resume_only: list[str],
        resume_skills: list[str],
        job_skills: list[str],
        skill_matches: list[SkillMatch],
        explicit_skill_ratio: float,
    ) -> None:
        self.matched = matched
        self.missing = missing
        self.resume_only = resume_only
        self.resume_skills = resume_skills
        self.job_skills = job_skills
        self.skill_matches = skill_matches
        self.explicit_skill_ratio = explicit_skill_ratio


class SkillDictionary:
    def __init__(self, skills_path: Path | None = None) -> None:
        path = skills_path or get_skills_path()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("skills", [])
        self._canonical_to_aliases: dict[str, list[str]] = {}
        self._alias_to_canonical: list[tuple[str, str]] = []
        for entry in entries:
            canonical = str(entry["canonical"]).strip()
            aliases = [canonical.lower()] + [str(a).strip().lower() for a in entry.get("aliases", [])]
            unique_aliases = []
            seen = set()
            for alias in aliases:
                if alias and alias not in seen:
                    seen.add(alias)
                    unique_aliases.append(alias)
            self._canonical_to_aliases[canonical] = unique_aliases
            for alias in unique_aliases:
                self._alias_to_canonical.append((alias, canonical))
        self._alias_to_canonical.sort(key=lambda item: len(item[0]), reverse=True)
        self._patterns: list[tuple[re.Pattern[str], str, str]] = []
        for alias, canonical in self._alias_to_canonical:
            escaped = re.escape(alias)
            pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
            self._patterns.append((pattern, canonical, alias))

    def canonical_skills(self) -> list[str]:
        return list(self._canonical_to_aliases.keys())

    def normalize(self, token: str) -> str | None:
        lowered = token.strip().lower()
        if not lowered:
            return None
        for alias, canonical in self._alias_to_canonical:
            if alias == lowered:
                return canonical
        return None

    def extract_skills(self, text: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        haystack = text or ""
        for pattern, canonical, _alias in self._patterns:
            if canonical in seen:
                continue
            if pattern.search(haystack):
                seen.add(canonical)
                found.append(canonical)
        return found

    def extract_skill_hits(self, text: str) -> dict[str, list[str]]:
        hits: dict[str, list[str]] = {}
        haystack = text or ""
        for pattern, canonical, alias in self._patterns:
            if pattern.search(haystack):
                hits.setdefault(canonical, [])
                if alias not in hits[canonical]:
                    hits[canonical].append(alias)
        return hits

    def compare(self, resume_text: str, job_text: str) -> SkillComparison:
        resume_hits = self.extract_skill_hits(resume_text)
        job_hits = self.extract_skill_hits(job_text)
        resume_skills = list(resume_hits.keys())
        job_skills = list(job_hits.keys())
        job_set = set(job_skills)
        resume_set = set(resume_skills)
        matched = sorted(job_set & resume_set)
        missing = sorted(job_set - resume_set)
        resume_only = sorted(resume_set - job_set)
        skill_matches: list[SkillMatch] = []
        for name in matched:
            skill_matches.append(
                SkillMatch(
                    canonical=name,
                    variants_found=sorted(set(resume_hits.get(name, []) + job_hits.get(name, []))),
                    source="both",
                    match_type="matched",
                )
            )
        for name in missing:
            skill_matches.append(
                SkillMatch(
                    canonical=name,
                    variants_found=job_hits.get(name, []),
                    source="job",
                    match_type="missing",
                )
            )
        ratio = (len(matched) / len(job_set)) if job_set else 0.0
        return SkillComparison(
            matched=matched,
            missing=missing,
            resume_only=resume_only,
            resume_skills=resume_skills,
            job_skills=job_skills,
            skill_matches=skill_matches,
            explicit_skill_ratio=ratio,
        )


def load_skill_dictionary(path: Path | None = None) -> SkillDictionary:
    return SkillDictionary(skills_path=path)
