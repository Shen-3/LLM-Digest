"""Simple heuristic quality scoring for representative article selection."""
from __future__ import annotations

from typing import Set

from . import utils

_PREFERRED_SOURCES: Set[str] = {
    "bbc",
    "reuters",
    "ap",
    "associated press",
    "science daily",
    "national geographic",
    "nhk",
}


def quality_score(title: str, content: str, source: str) -> float:
    """Compute a lightweight quality score for ranking representative articles."""
    title_len = len(title or "")
    content_len = len(content or "")
    source_normalized = (source or "").strip().lower()

    title_score = min(title_len / 120.0, 1.0)
    content_score = utils.logarithmic_bonus(content_len, base=50.0)
    source_bonus = 0.2 if source_normalized in _PREFERRED_SOURCES else 0.0

    return round(title_score + content_score + source_bonus, 4)


__all__ = ["quality_score"]
