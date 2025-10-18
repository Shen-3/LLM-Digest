"""Multilingual embedding utilities using sentence-transformers."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache a sentence-transformers model."""
    return SentenceTransformer(model_name)


def embed_texts(
    texts: Iterable[str],
    model_name: str,
    batch_size: int,
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts into embeddings with optional L2 normalization."""
    model = _load_model(model_name)
    texts_list: List[str] = [text or "" for text in texts]
    embeddings = model.encode(texts_list, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize)
    if not normalize:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


__all__ = ["embed_texts"]
