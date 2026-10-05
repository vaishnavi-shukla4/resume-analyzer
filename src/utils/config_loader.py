"""Load config.yaml and project-root/.env. Never load src/.env or tests/.env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config.yaml").is_file() and (parent / "src").is_dir():
            return parent
    return Path(__file__).resolve().parents[2]


def load_dotenv_file() -> Path:
    env_path = get_project_root() / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    return env_path


def get_groq_api_key() -> Optional[str]:
    load_dotenv_file()
    key = os.getenv("GROQ_API_KEY")
    if key is None:
        return None
    stripped = key.strip()
    if not stripped or stripped == "your_groq_api_key_here":
        return None
    return stripped


def get_skills_path() -> Path:
    return get_project_root() / "src" / "data" / "skills.json"


def get_config_path() -> Path:
    return get_project_root() / "config.yaml"


def _require(mapping: dict[str, Any], *keys: str) -> Any:
    cursor: Any = mapping
    path = []
    for key in keys:
        path.append(key)
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError(".".join(path))
        cursor = cursor[key]
    return cursor


@lru_cache(maxsize=1)
def load_config(path: Optional[str] = None) -> dict[str, Any]:
    config_file = Path(path) if path else get_config_path()
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing config file: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping")
    _require(data, "llm", "model")
    _require(data, "llm", "fallback_model")
    _require(data, "analysis", "weights")
    _require(data, "analysis", "minimum_text_length")
    _require(data, "cache", "enabled")
    _require(data, "cache", "max_size")
    return data


def get_analysis_weights(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("analysis", {}).get("weights", {})
    required = [
        "tfidf",
        "keyword_coverage",
        "skill_alignment",
        "experience_alignment",
        "education_alignment",
        "explicit_skill_ratio",
    ]
    missing = [name for name in required if name not in raw]
    if missing:
        raise KeyError(f"Missing analysis.weights fields: {missing}")
    return {name: float(raw[name]) for name in required}
