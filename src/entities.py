"""Lightweight entity extraction helpers using spaCy when available."""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Dict, Iterable, List
from functools import lru_cache

# Expose the spaCy Language type only for static type checkers. At runtime
# spaCy may be missing; we import it in a try/except below.
if TYPE_CHECKING:
    from spacy.language import Language
import logging

logger = logging.getLogger(__name__)

try:  # optional dependency ---
    import spacy
except ImportError:  # pragma: no cover - spaCy not installed
    spacy = None  # type: ignore


Entity = Dict[str, str]


def _best_text_sample(article: dict) -> str:
    for field in ("content", "description", "title"):
        value = article.get(field)
        if value:
            return value
    return ""


@lru_cache(maxsize=2)
def _load_pipeline(model_name: str) -> Optional["Language"]:
    """Load a spaCy model if available."""
    if spacy is None:
        logger.info("spaCy is not installed; entity extraction is disabled")
        return None
    try:
        return spacy.load(model_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load spaCy model %s: %s", model_name, exc)
        return None


def extract_entities(text: str, model_name: str) -> List[Entity]:
    """Extract named entities from text using the configured spaCy model."""
    nlp = _load_pipeline(model_name)
    if not text or nlp is None:
        return []
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]


def annotate_entities(articles: Iterable[dict], model_name: str) -> List[dict]:
    """Attach entity lists to each article."""
    annotated: List[dict] = []
    for article in articles:
        text = _best_text_sample(article)
        entities = extract_entities(text, model_name)
        enriched = {**article, "entities": entities}
        annotated.append(enriched)
    return annotated


__all__ = ["annotate_entities", "extract_entities"]
