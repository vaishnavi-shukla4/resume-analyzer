from __future__ import annotations

from src.analysis.skill_matcher import load_skill_dictionary


def test_normalizes_skill_variants() -> None:
    dictionary = load_skill_dictionary()
    assert dictionary.normalize("python programming") == "Python"
    assert dictionary.normalize("ML") == "Machine Learning"
    assert dictionary.normalize("natural language processing") == "Natural Language Processing"
    assert dictionary.normalize("structured query language") == "SQL"


def test_extracts_and_compares_skills() -> None:
    dictionary = load_skill_dictionary()
    resume = "Used python programming, pandas, and scikit-learn. Experience with REST APIs and git."
    job = "Requires Python, SQL, Pandas, Machine Learning, Power BI, AWS, Docker, Statistics"
    comparison = dictionary.compare(resume, job)
    assert "Python" in comparison.matched
    assert "Pandas" in comparison.matched
    assert "SQL" in comparison.missing
    assert "Docker" in comparison.missing
    assert "Power BI" in comparison.missing
    assert comparison.explicit_skill_ratio > 0
    assert all(item.match_type in {"matched", "missing"} for item in comparison.skill_matches)


def test_does_not_invent_unrelated_skills() -> None:
    dictionary = load_skill_dictionary()
    skills = dictionary.extract_skills("Used pandas for data analysis.")
    assert "Spark" not in skills
    assert "Pandas" in skills
