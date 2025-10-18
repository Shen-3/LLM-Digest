"""Streamlit UI for exploring the news digest."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import embeddings, search, utils  # noqa: E402

CONFIG = utils.load_config("configs/config.yaml")
DIGEST_PATH = Path(CONFIG["paths"]["output"]).resolve()
HISTORY_DIR = CONFIG["paths"].get("history_dir")
VECTORS_PATH = CONFIG["paths"].get("vectors", "artifacts/article_vectors.npy")
MODEL_CFG = CONFIG["model"]
SEARCH_CFG = CONFIG.get("search", {})
ANN_INDEX_PATH = SEARCH_CFG.get("ann_index", "artifacts/ann_index.bin")


@st.cache_data(show_spinner=False)
def list_digests(default_path: str, history_dir: str | None) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    default = Path(default_path)
    if default.exists():
        entries.append(("Latest run", str(default)))
    if history_dir:
        for path in sorted(Path(history_dir).glob("digest_*.json"), reverse=True):
            label = path.name.replace("digest_", "").replace(".json", "")
            entries.append((label, str(path.resolve())))
    seen = set()
    unique_entries: List[Tuple[str, str]] = []
    for label, path in entries:
        if path in seen:
            continue
        seen.add(path)
        unique_entries.append((label, path))
    return unique_entries


@st.cache_data(show_spinner=False)
def load_digest(path: str) -> Dict:
    return utils.load_json(path)


@st.cache_resource(show_spinner=False)
def load_embeddings() -> np.ndarray:
    path = Path(VECTORS_PATH)
    if not path.exists():
        return np.empty((0, 0))
    return np.load(path)


def _supports_vector_search(selected_digest: Path) -> bool:
    return selected_digest.resolve() == DIGEST_PATH


def search_articles(query: str, digest: Dict, k: int = 10) -> List[Dict]:
    matrix = load_embeddings()
    if matrix.size == 0 or not query:
        return []
    query_vec = embeddings.embed_texts(
        [query],
        model_name=MODEL_CFG["name"],
        batch_size=MODEL_CFG.get("batch_size", 64),
        normalize=MODEL_CFG.get("normalize", True),
    )[0]
    use_ann = SEARCH_CFG.get("use_ann", False) and search.ann_index_exists(ANN_INDEX_PATH)
    if use_ann:
        try:
            ann_results = search.ann_search(
                ANN_INDEX_PATH,
                query_vec,
                k,
                ef_search=int(SEARCH_CFG.get("ann_ef_search", 100)),
            )
            indices_scores = ann_results
        except RuntimeError:
            indices_scores = []
    else:
        scores = utils.cosine_similarity(matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:k]
        indices_scores = [(int(i), float(scores[i])) for i in top_indices]

    articles = digest.get("articles", [])
    results = []
    for idx, score in indices_scores:
        if idx >= len(articles):
            continue
        article = dict(articles[idx])
        article["score"] = float(score)
        results.append(article)
    return results


def similar_articles(embedding_index: int, digest: Dict, k: int = 5) -> List[Dict]:
    matrix = load_embeddings()
    if matrix.size == 0 or embedding_index >= matrix.shape[0]:
        return []
    article_vector = matrix[embedding_index]
    use_ann = SEARCH_CFG.get("use_ann", False) and search.ann_index_exists(ANN_INDEX_PATH)
    if use_ann:
        try:
            ann_results = search.ann_search(
                ANN_INDEX_PATH,
                article_vector,
                k + 1,
                ef_search=int(SEARCH_CFG.get("ann_ef_search", 100)),
            )
            indices_scores = [(idx, score) for idx, score in ann_results if idx != embedding_index]
        except RuntimeError:
            indices_scores = []
    else:
        scores = utils.cosine_similarity(matrix, article_vector)
        sorted_idx = np.argsort(scores)[::-1]
        indices_scores = [
            (int(idx), float(scores[idx]))
            for idx in sorted_idx
            if idx != embedding_index
        ][:k]

    articles = digest.get("articles", [])
    results = []
    for idx, score in indices_scores[:k]:
        if idx >= len(articles):
            continue
        article = dict(articles[idx])
        article["score"] = float(score)
        results.append(article)
    return results


def main() -> None:
    st.set_page_config(page_title="Multilingual News Digest", layout="wide")
    digest_entries = list_digests(str(DIGEST_PATH), HISTORY_DIR)
    labels = [entry[0] for entry in digest_entries]
    selected_label = st.sidebar.selectbox("Digest run", labels, index=0 if labels else None)
    selected_path = next((path for label, path in digest_entries if label == selected_label), str(DIGEST_PATH))
    digest = load_digest(selected_path)
    supports_vectors = _supports_vector_search(Path(selected_path))

    st.title("Multilingual News Topics")
    st.caption(f"Digest generated at {digest.get('generated_at', 'N/A')}")

    articles = digest.get("articles", [])
    languages = sorted({article.get("language", "unknown") for article in articles if article.get("language")})
    selected_languages = st.sidebar.multiselect("Languages", languages, default=languages)

    topics = digest.get("digest", [])
    language_filtered_topics = []
    for topic in topics:
        topic_id = topic.get("topic_id")
        topic_articles = [article for article in articles if article.get("topic_id") == topic_id]
        if selected_languages and not any(article.get("language") in selected_languages for article in topic_articles):
            continue
        language_filtered_topics.append((topic, topic_articles))

    for topic, topic_articles in language_filtered_topics:
        st.subheader(f"Topic #{topic['topic_id']}: {topic['topic_name']}")
        st.write(
            f"Articles: {topic.get('articles_count', 0)} | "
            f"Keywords: {', '.join(topic.get('keywords', [])[:8])}"
        )
        if topic.get("top_entities"):
            entities = ", ".join(f"{e['text']} ({e['count']})" for e in topic.get("top_entities", []) if e.get("text"))
            if entities:
                st.caption(f"Entities: {entities}")
        with st.expander("Representative Articles", expanded=True):
            for article in topic.get("representative_articles", []):
                st.markdown(
                    f"- [{article.get('title', 'Untitled')}]({article.get('url', '#')}) — {article.get('source', 'Unknown')}"
                )
        if topic_articles:
            st.markdown("**All articles in topic:**")
            for article in topic_articles:
                st.markdown(
                    f"• {article.get('title', 'Untitled')} "
                    f"({article.get('source', 'Unknown')}, {article.get('language', 'unknown')})"
                )
        st.divider()

    st.sidebar.header("Search Articles")
    if supports_vectors:
        query = st.sidebar.text_input("Enter keywords", "climate")
        k = st.sidebar.slider("Top results", min_value=5, max_value=20, value=10)
        if query:
            results = search_articles(query, digest, k=k)
            st.sidebar.write("### Search Results")
            for result in results:
                st.sidebar.markdown(
                    f"[{result.get('title', 'Untitled')}]({result.get('url', '#')})\nScore: {result.get('score', 0.0):.3f}"
                )
        st.sidebar.header("Similar Articles")
        article_choices = {
            f"{article.get('title', 'Untitled')} — {article.get('source', 'Unknown')}": article
            for article in articles
            if article.get("embedding_index") is not None
        }
        article_labels = list(article_choices.keys())
        if article_labels:
            selected_article_label = st.sidebar.selectbox("Select article", article_labels, index=0)
            similar_k = st.sidebar.slider("Similar results", min_value=3, max_value=15, value=5)
            selected_article = article_choices[selected_article_label]
            similar = similar_articles(int(selected_article["embedding_index"]), digest, k=similar_k)
            st.sidebar.write("### Similar Articles")
            for item in similar:
                st.sidebar.markdown(
                    f"[{item.get('title', 'Untitled')}]({item.get('url', '#')})\nScore: {item.get('score', 0.0):.3f}"
                )
        else:
            st.sidebar.info("No articles with embeddings available for similarity search.")
    else:
        st.sidebar.info("Vector search and similarity require the latest digest run.")


if __name__ == "__main__":
    main()
