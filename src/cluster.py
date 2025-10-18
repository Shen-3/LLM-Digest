"""Clustering utilities for grouping news articles into topics."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import numpy as np
from sklearn.cluster import DBSCAN

from . import utils

_STOPWORDS = {
    "en": {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "at", "from"},
    "es": {"el", "la", "los", "las", "de", "y", "para", "con", "en"},
    "de": {"der", "die", "das", "und", "mit", "für", "den", "im"},
    "fr": {"le", "la", "les", "des", "et", "pour", "dans", "avec"},
    "pt": {"o", "a", "os", "as", "de", "e", "para", "com"},
    "ru": {"и", "в", "на", "с", "по", "за", "для"},
    "ja": {"これ", "それ", "そして", "また", "ため"},
}


def dbscan_clusters(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Perform DBSCAN clustering with cosine distance."""
    if embeddings.size == 0:
        return np.array([])
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = model.fit_predict(embeddings)
    return labels


def _token_frequencies(texts: Iterable[str], language: str | None) -> Counter:
    lang = (language or "en").split("-")[0]
    stopwords = _STOPWORDS.get(lang, _STOPWORDS["en"])
    counter: Counter = Counter()
    for text in texts:
        for token in utils.tokenize(text):
            if len(token) <= 2:
                continue
            if token in stopwords:
                continue
            counter[token] += 1
    return counter


def extract_keywords(texts: List[str], top_k: int, lang: str | None = None) -> List[str]:
    """Extract keywords based on token frequency."""
    if not texts:
        return []
    frequencies = _token_frequencies(texts, lang)
    most_common = [token for token, _ in frequencies.most_common(top_k)]
    return most_common


def topic_label(keywords: List[str], max_tokens: int = 5) -> str:
    """Create a short topic name based on top keywords."""
    if not keywords:
        return "Miscellaneous"
    trimmed = keywords[:max_tokens]
    return " ".join(trimmed).title()


__all__ = [
    "dbscan_clusters",
    "extract_keywords",
    "topic_label",
]
