"""Data ingestion and preprocessing utilities."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List

from langdetect import DetectorFactory, LangDetectException, detect

from . import utils

DetectorFactory.seed = 42  # deterministic language detection


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing operations."""

    language_threshold: float = 0.6
    title_similarity_threshold: float = 0.88


_STOPWORDS: Dict[str, set[str]] = {
    "en": {
        "the",
        "a",
        "an",
        "of",
        "and",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "from",
        "la",
        "el",
        "los",
        "las",
        "de",
        "der",
        "die",
        "das",
        "und",
        "ein",
        "eine",
    },
}


def load_news(path: str) -> List[dict]:
    """Load news articles from JSON."""
    data = utils.load_json(path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of article objects")
    return [dict(item) for item in data]


def normalize_text(raw: str | None) -> str:
    """Normalize text by stripping HTML, whitespace, and NFC normalization."""
    if not raw:
        return ""
    text = utils.strip_html(raw)
    text = unicodedata.normalize("NFKC", text)
    return utils.normalize_whitespace(text)


def detect_language(text: str) -> str:
    """Detect the language code for a piece of text."""
    cleaned = normalize_text(text)
    if not cleaned:
        return "unknown"
    try:
        return detect(cleaned)
    except LangDetectException:
        return "unknown"


def _canonical_title(title: str, language: str | None = None) -> str:
    tokens = utils.tokenize(title)
    if not tokens:
        return ""
    language = language or "en"
    stopwords = _STOPWORDS.get(language, _STOPWORDS["en"])
    filtered = [t for t in tokens if t not in stopwords]
    if not filtered:
        filtered = tokens
    return " ".join(filtered)


def _is_duplicate(title: str, existing: Iterable[str], *, threshold: float) -> bool:
    for other in existing:
        if not other:
            continue
        if SequenceMatcher(a=title, b=other).ratio() >= threshold:
            return True
    return False


def deduplicate(items: List[dict], *, config: PreprocessConfig | None = None) -> List[dict]:
    """Remove near-duplicate articles based on URL and fuzzy title matching."""
    config = config or PreprocessConfig()
    seen_urls: set[str] = set()
    canonical_titles: Dict[str, str] = {}
    deduped: List[dict] = []

    for item in items:
        url = item.get("url", "").strip().lower()
        if url and url in seen_urls:
            continue

        title = normalize_text(item.get("title") or "")
        language = item.get("language") or detect_language(item.get("content", "") or title)
        languaged_title = _canonical_title(title, language)

        if _is_duplicate(languaged_title, canonical_titles.values(), threshold=config.title_similarity_threshold):
            continue

        item_copy = {**item}
        item_copy["title"] = title
        item_copy["description"] = normalize_text(item.get("description"))
        item_copy["content"] = normalize_text(item.get("content"))
        item_copy["language"] = language

        seen_urls.add(url)
        canonical_titles[item_copy.get("id", f"idx-{len(deduped)}")] = languaged_title
        deduped.append(item_copy)

    return deduped


__all__ = [
    "PreprocessConfig",
    "load_news",
    "normalize_text",
    "detect_language",
    "deduplicate",
]
