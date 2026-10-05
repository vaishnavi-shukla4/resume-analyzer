"""Text normalization for resume and job-description documents."""

from __future__ import annotations

import re
import unicodedata

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_BULLETS = re.compile(r"[•●○■▪‣◦]+")


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _BULLETS.sub("-", normalized)
    normalized = _MULTI_SPACE.sub(" ", normalized)
    normalized = _MULTI_NEWLINE.sub("\n\n", normalized)
    return "\n".join(line.strip() for line in normalized.split("\n")).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\x00", "")
    normalized = normalize_whitespace(normalized)
    return normalized
