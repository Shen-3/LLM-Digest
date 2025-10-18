"""Pipeline orchestration for building the news topic digest."""
from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from . import cluster, embeddings, entities, preprocess, quality, search, utils

logger = logging.getLogger(__name__)


Article = dict


def _article_text(article: Article) -> str:
    parts = [article.get("title", ""), article.get("description", ""), article.get("content", "")]
    return " \n".join(part for part in parts if part)


def _prepare_embeddings_payload(items: List[Article], model_cfg: dict) -> Tuple[np.ndarray, List[str]]:
    texts = [_article_text(item) for item in items]
    emb = embeddings.embed_texts(
        texts,
        model_name=model_cfg["name"],
        batch_size=model_cfg.get("batch_size", 64),
        normalize=model_cfg.get("normalize", True),
    )
    ids = [item.get("id", f"article-{idx}") for idx, item in enumerate(items)]
    return emb, ids


def build_digest(
    items: List[Article],
    clusters: np.ndarray,
    embeddings_matrix: np.ndarray,
    k_keywords: int,
    quality_fn: Callable[[str, str, str], float],
) -> Dict:
    """Build the digest dictionary from clustered items."""
    _ = embeddings_matrix  # retained for signature parity and future enhancements
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(clusters):
        label_to_indices[int(label)].append(idx)

    base_labels = sorted(label_to_indices)
    sorted_labels = [label for label in base_labels if label != -1]
    if -1 in label_to_indices:
        sorted_labels.append(-1)
    digest_topics = []
    topic_id_mapping = {label: topic_id for topic_id, label in enumerate(sorted_labels)}

    for label in sorted_labels:
        indices = label_to_indices[label]
        topic_articles = [items[i] for i in indices]
        language = topic_articles[0].get("language") if topic_articles else None
        texts = [_article_text(article) for article in topic_articles]
        keywords = cluster.extract_keywords(texts, k_keywords, language)
        topic_name = cluster.topic_label(keywords)
        if label == -1:
            topic_name = "Miscellaneous"

        entities_counter: Counter[tuple[str, str]] = Counter()
        for article in topic_articles:
            for entity in article.get("entities", []):
                key = (entity.get("text", ""), entity.get("label", ""))
                if key[0]:
                    entities_counter[key] += 1
        top_entities = [
            {"text": text, "label": label_name, "count": count}
            for (text, label_name), count in entities_counter.most_common(5)
        ]

        ranked = sorted(
            (
                {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "language": article.get("language", "unknown"),
                    "quality_score": quality_fn(
                        article.get("title", ""),
                        article.get("content", ""),
                        article.get("source", ""),
                    ),
                    "published_at": article.get("published_at"),
                    "index": idx,
                }
                for idx, article in zip(indices, topic_articles)
            ),
            key=lambda a: (a["quality_score"], a.get("published_at", "")),
            reverse=True,
        )

        representative = [
            {
                key: value
                for key, value in article.items()
                if key in {"title", "url", "source", "language", "quality_score", "published_at"}
            }
            for article in ranked[:3]
        ]

        topic_id = topic_id_mapping[label]
        digest_topics.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "articles_count": len(indices),
                "keywords": keywords,
                "representative_articles": representative,
                "top_entities": top_entities,
            }
        )

    article_records = []
    for idx, article in enumerate(items):
        label = int(clusters[idx]) if idx < len(clusters) else -1
        topic_id = topic_id_mapping.get(label)
        article_records.append(
            {
                "id": article.get("id", f"article-{idx}"),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "language": article.get("language", "unknown"),
                "published_at": article.get("published_at"),
                "topic_id": topic_id,
                "embedding_index": idx,
                "entities": article.get("entities", []),
            }
        )

    digest_payload = {
        "generated_at": utils.utcnow_iso(),
        "total_articles": len(items),
        "topics_found": len(digest_topics),
        "digest": digest_topics,
        "articles": article_records,
    }
    return digest_payload


def save_digest(digest_data: Dict, path: str | Path) -> None:
    """Persist the digest as JSON."""
    utils.save_json(digest_data, path)


def save_embeddings(embeddings_matrix: np.ndarray, path: str | Path) -> None:
    """Persist article embeddings to disk."""
    utils.ensure_dir(path)
    np.save(path, embeddings_matrix)


def print_console_report(digest_data: Dict) -> None:
    """Print a human-readable digest summary to the console."""
    header = f"News Digest | Generated: {digest_data['generated_at']}\n"
    header += f"Total articles: {digest_data['total_articles']} | Topics: {digest_data['topics_found']}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for topic in digest_data.get("digest", []):
        topic_line = f"[#{topic['topic_id']}] {topic['topic_name']} (articles: {topic['articles_count']})"
        print(topic_line)
        keywords = ", ".join(topic.get("keywords", [])[:5])
        if keywords:
            print(f"  keywords: {keywords}")
        for article in topic.get("representative_articles", []):
            print(f"  - {article['title']} ({article['source']})")
        print()


def run_pipeline(config_path: str) -> Dict:
    config = utils.load_config(config_path)
    input_path = config["paths"]["input"]
    output_path = config["paths"]["output"]
    vectors_path = config["paths"].get("vectors", "artifacts/article_vectors.npy")
    history_dir = config["paths"].get("history_dir")
    run_log_path = config["paths"].get("run_log")

    raw_items = preprocess.load_news(input_path)
    cleaned_items = preprocess.deduplicate(raw_items)
    enrichment_cfg = config.get("enrichment", {})
    if enrichment_cfg.get("enable_entities", False):
        model_name = enrichment_cfg.get("spacy_model", "xx_ent_wiki_sm")
        cleaned_items = entities.annotate_entities(cleaned_items, model_name)

    embeddings_matrix, _ = _prepare_embeddings_payload(cleaned_items, config["model"])
    clusters = cluster.dbscan_clusters(
        embeddings_matrix,
        eps=config["cluster"].get("eps", 0.7),
        min_samples=config["cluster"].get("min_samples", 3),
    )

    digest_data = build_digest(
        cleaned_items,
        clusters,
        embeddings_matrix,
        k_keywords=config["keywords"].get("per_topic", 8),
        quality_fn=quality.quality_score,
    )

    save_digest(digest_data, output_path)
    save_embeddings(embeddings_matrix, vectors_path)
    search_cfg = config.get("search", {})
    if search_cfg.get("use_ann", False):
        try:
            search.build_ann_index(
                embeddings_matrix,
                search_cfg.get("ann_index", "artifacts/ann_index.bin"),
                m=int(search_cfg.get("ann_m", 16)),
                ef_construction=int(search_cfg.get("ann_ef_construction", 200)),
                ef_search=int(search_cfg.get("ann_ef_search", 100)),
            )
        except RuntimeError as exc:
            logger.warning("ANN index build failed: %s", exc)
    if history_dir:
        history_dir_path = Path(history_dir)
        history_dir_path.mkdir(parents=True, exist_ok=True)
        history_path = history_dir_path / f"digest_{digest_data['generated_at'].replace(':', '-')}.json"
        utils.save_json(digest_data, history_path)
    if run_log_path:
        run_record = {
            "generated_at": digest_data["generated_at"],
            "topics_found": digest_data["topics_found"],
            "total_articles": digest_data["total_articles"],
            "config": config_path,
            "output": output_path,
        }
        utils.append_run_record(run_log_path, run_record)

    print_console_report(digest_data)
    return digest_data


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the multilingual news digest")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration file")
    return parser.parse_args(args=args)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parsed = parse_args()
    run_pipeline(parsed.config)


if __name__ == "__main__":  # pragma: no cover
    main()
