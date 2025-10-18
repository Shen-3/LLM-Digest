from __future__ import annotations

import numpy as np

from src import digest, quality


def _sample_article(idx: int, topic: str) -> dict:
    return {
        "id": f"art-{idx}",
        "title": f"Sample title {idx} about {topic}",
        "description": f"Description {idx}",
        "content": f"Content body {idx} focusing on {topic}",
        "url": f"https://example.com/{idx}",
        "source": "BBC",
        "language": "en",
        "published_at": "2024-03-01T00:00:00Z",
    }


def test_digest_structure_respects_schema() -> None:
    items = [_sample_article(0, "climate"), _sample_article(1, "climate"), _sample_article(2, "energy")]
    clusters = np.array([0, 0, 1])
    embeddings_matrix = np.zeros((3, 4))

    result = digest.build_digest(items, clusters, embeddings_matrix, k_keywords=5, quality_fn=quality.quality_score)

    assert {"generated_at", "total_articles", "topics_found", "digest", "articles"} <= set(result)
    assert result["total_articles"] == 3
    assert result["topics_found"] >= 2

    for topic in result["digest"]:
        assert {"topic_id", "topic_name", "articles_count", "keywords", "representative_articles"} <= set(topic)
        assert "top_entities" in topic
        representatives = topic["representative_articles"]
        assert topic["articles_count"] >= len(representatives)
        for article in representatives:
            assert {"title", "url", "source", "language", "quality_score", "published_at"} <= set(article)

    assert len(result["articles"]) == 3
    assert all(article["embedding_index"] == idx for idx, article in enumerate(result["articles"]))
    assert all("entities" in article for article in result["articles"])
