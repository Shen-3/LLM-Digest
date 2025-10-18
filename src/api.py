"""FastAPI application exposing the news digest and search endpoints."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from . import embeddings, metrics, utils
import .search as search_module

CONFIG_PATH = os.environ.get("NEWS_TOPICS_CONFIG", "configs/config.yaml")
CONFIG = utils.load_config(CONFIG_PATH)
DIGEST_PATH = CONFIG["paths"]["output"]
VECTORS_PATH = CONFIG["paths"].get("vectors", "artifacts/article_vectors.npy")
MODEL_CFG = CONFIG["model"]
SEARCH_CFG = CONFIG.get("search", {})
ANN_INDEX_PATH = SEARCH_CFG.get("ann_index", "artifacts/ann_index.bin")

app = FastAPI(title="Multilingual News Digest")


@lru_cache(maxsize=1)
def load_digest_cached(path: str) -> Dict:
    """Load the digest JSON with simple caching."""
    resolved = str(Path(path).resolve())
    return utils.load_json(resolved)


def _load_embeddings() -> np.ndarray:
    if not Path(VECTORS_PATH).exists():
        raise HTTPException(status_code=404, detail="Embeddings file not found. Run the pipeline first.")
    return np.load(VECTORS_PATH)


def _all_articles() -> List[Dict]:
    digest = load_digest_cached(DIGEST_PATH)
    return digest.get("articles", [])


def search(query: str, k: int = 10) -> List[Dict]:
    """Perform cosine similarity search over article embeddings."""
    if not query:
        return []
    query_vec = embeddings.embed_texts([query], MODEL_CFG["name"], MODEL_CFG.get("batch_size", 64), MODEL_CFG.get("normalize", True))[0]
    use_ann = SEARCH_CFG.get("use_ann", False) and search.ann_index_exists(ANN_INDEX_PATH)
    articles = _all_articles()

    if use_ann:
        try:
            ann_results = search.ann_search(
                ANN_INDEX_PATH,
                query_vec,
                k,
                ef_search=int(SEARCH_CFG.get("ann_ef_search", 100)),
            )
            indices_scores = ann_results
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f"ANN search failed: {exc}") from exc
    else:
        matrix = _load_embeddings()
        scores = utils.cosine_similarity(matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:k]
        indices_scores = [(int(idx), float(scores[idx])) for idx in top_indices]

    articles = _all_articles()
    results = []
    for idx, score in indices_scores:
        if idx >= len(articles):
            continue
        article = dict(articles[idx])
        article["score"] = float(score)
        results.append(article)
    return results


@app.get("/digest")
def get_digest() -> Dict:
    return load_digest_cached(DIGEST_PATH)


@app.get("/topics")
def list_topics() -> List[Dict]:
    digest = load_digest_cached(DIGEST_PATH)
    topics = digest.get("digest", [])
    return [
        {
            "topic_id": topic["topic_id"],
            "topic_name": topic.get("topic_name"),
            "articles": topic.get("articles_count"),
        }
        for topic in topics
    ]


@app.get("/topic/{topic_id}")
def get_topic(topic_id: int) -> Dict:
    digest = load_digest_cached(DIGEST_PATH)
    for topic in digest.get("digest", []):
        if topic.get("topic_id") == topic_id:
            return topic
    raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")


@app.get("/search")
def search_endpoint(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=50)) -> List[Dict]:
    try:
        return search(q, k=k)
    except HTTPException as exc:
        raise exc


@app.get("/metrics")
def metrics_endpoint() -> Dict:
    vectors_path = CONFIG["paths"].get("vectors")
    try:
        return metrics.compute_kpis(DIGEST_PATH, vectors_path=vectors_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


__all__ = ["app", "load_digest_cached", "search"]
