from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import metrics


def test_metrics_compute_and_chart(tmp_path) -> None:
    digest_data = {
        "generated_at": "2024-03-01T00:00:00Z",
        "total_articles": 4,
        "topics_found": 2,
        "digest": [
            {
                "topic_id": 0,
                "topic_name": "Climate Action",
                "articles_count": 2,
                "keywords": ["climate", "summit"],
                "representative_articles": [],
            },
            {
                "topic_id": 1,
                "topic_name": "Energy",
                "articles_count": 2,
                "keywords": ["energy", "solar"],
                "representative_articles": [],
            },
        ],
    }

    digest_path = tmp_path / "digest.json"
    digest_path.write_text(json.dumps(digest_data), encoding="utf-8")

    vectors_path = tmp_path / "vectors.npy"
    np.save(vectors_path, np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]))

    kpis = metrics.compute_kpis(str(digest_path), vectors_path=str(vectors_path))
    assert set(kpis) == {
        "topics_found",
        "total_articles",
        "avg_articles_per_topic",
        "max_topic_size",
        "top_keywords",
        "keyword_diversity",
        "avg_intra_topic_similarity",
        "entity_coverage",
        "top_entities",
    }
    assert kpis["topics_found"] == 2
    assert isinstance(kpis["avg_articles_per_topic"], float)

    out_path = tmp_path / "topics_bar.png"
    metrics.save_bar_chart(str(digest_path), str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
