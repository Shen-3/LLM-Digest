"""Metrics computation and visualization for the news digest."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

import matplotlib.pyplot as plt

from . import utils


def _average_pairwise_similarity(vectors: np.ndarray) -> float:
    if vectors.shape[0] <= 1:
        return 1.0
    similarity_matrix = vectors @ vectors.T
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    upper = similarity_matrix[triu_indices]
    if upper.size == 0:
        return 1.0
    return float(np.mean(upper))


def compute_kpis(digest_path: str, *, vectors_path: str | None = None) -> Dict[str, float | int | list]:
    """Compute KPI metrics from the digest JSON."""
    digest = utils.load_json(digest_path)
    topics = digest.get("digest", [])
    articles = digest.get("articles", [])

    topics_found = digest.get("topics_found", len(topics))
    total_articles = digest.get("total_articles", 0)
    articles_per_topic = [topic.get("articles_count", 0) for topic in topics]
    avg_articles_per_topic = utils.safe_average(articles_per_topic)
    max_topic_size = max(articles_per_topic, default=0)

    keyword_counter: Counter[str] = Counter()
    keyword_total = 0
    for topic in topics:
        kws = topic.get("keywords", [])
        keyword_total += len(kws)
        keyword_counter.update(kws)
    top_keywords = [keyword for keyword, _ in keyword_counter.most_common(10)]
    keyword_diversity = round(len(keyword_counter) / keyword_total, 3) if keyword_total else 0.0

    entity_counter: Counter[str] = Counter()
    for topic in topics:
        for entity in topic.get("top_entities", []):
            text = entity.get("text")
            if text:
                entity_counter[text] += entity.get("count", 1)
    top_entities = [
        {"text": text, "count": count}
        for text, count in entity_counter.most_common(10)
    ]

    vectors = None
    if vectors_path and Path(vectors_path).exists():
        vectors = np.load(vectors_path)

    topic_embeddings: Dict[int, List[int]] = defaultdict(list)
    for article in articles:
        topic_id = article.get("topic_id")
        embedding_index = article.get("embedding_index")
        if topic_id is None or embedding_index is None:
            continue
        topic_embeddings[int(topic_id)].append(int(embedding_index))

    intra_similarities: List[float] = []
    if vectors is not None:
        for indices in topic_embeddings.values():
            if not indices:
                continue
            subset = vectors[indices]
            intra_similarities.append(_average_pairwise_similarity(subset))
    avg_intra_similarity = round(utils.safe_average(intra_similarities), 3)

    entity_coverage = round(
        sum(1 for article in articles if article.get("entities")) / len(articles), 3
    ) if articles else 0.0

    return {
        "topics_found": topics_found,
        "total_articles": total_articles,
        "avg_articles_per_topic": round(avg_articles_per_topic, 2),
        "max_topic_size": max_topic_size,
        "top_keywords": top_keywords,
        "keyword_diversity": keyword_diversity,
        "avg_intra_topic_similarity": avg_intra_similarity,
        "entity_coverage": entity_coverage,
        "top_entities": top_entities,
    }


def save_bar_chart(digest_path: str, out_path: str) -> None:
    """Create and persist a bar chart of article counts per topic."""
    digest = utils.load_json(digest_path)
    topics = digest.get("digest", [])

    labels = [f"#{topic['topic_id']}" for topic in topics]
    counts = [topic.get("articles_count", 0) for topic in topics]

    if not labels:
        labels = ["No Topics"]
        counts = [0]

    utils.ensure_dir(out_path)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts, color="#4C72B0")
    plt.title("Articles per Topic")
    plt.xlabel("Topic")
    plt.ylabel("Article Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute KPIs for the news digest")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()
    config = utils.load_config(args.config)
    digest_path = config["paths"]["output"]
    plot_dir = Path(config["paths"].get("plot_dir", "plots"))
    out_path = str(plot_dir / "topics_bar.png")

    vectors_path = config["paths"].get("vectors")
    kpis = compute_kpis(digest_path, vectors_path=vectors_path)
    print("KPI Summary")
    for key, value in kpis.items():
        print(f"- {key}: {value}")

    save_bar_chart(digest_path, out_path)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
