"""Search utilities supporting optional ANN backends."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from . import utils

logger = logging.getLogger(__name__)

try:  # optional dependency
    import hnswlib
except ImportError:  # pragma: no cover - optional
    hnswlib = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - hints only
    from hnswlib import Index as HnswIndex
else:
    HnswIndex = Any


_META_SUFFIX = ".meta.json"


def _meta_path(index_path: str | Path) -> Path:
    path = Path(index_path)
    return path.with_suffix(path.suffix + _META_SUFFIX)


def build_ann_index(
    embeddings_matrix: np.ndarray,
    index_path: str | Path,
    *,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> None:
    """Build and persist an HNSW index for cosine similarity."""
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed. Install it or disable ANN search in the config.")
    if embeddings_matrix.size == 0:
        logger.info("Skipping ANN index build because no embeddings were provided")
        return
    matrix = embeddings_matrix.astype(np.float32)
    dim = matrix.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=matrix.shape[0], ef_construction=ef_construction, M=m)
    ids = np.arange(matrix.shape[0])
    index.add_items(matrix, ids)
    index.set_ef(ef_search)

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(index_path))
    utils.save_json({"dim": dim, "count": int(matrix.shape[0]), "ef_search": ef_search}, _meta_path(index_path))
    logger.info("Saved ANN index with %s vectors to %s", matrix.shape[0], index_path)


def ann_index_exists(index_path: str | Path) -> bool:
    path = Path(index_path)
    return path.exists() and _meta_path(path).exists()


@lru_cache(maxsize=1)
def _load_ann_index(index_path: str, ef_search: int) -> Tuple[HnswIndex, Dict[str, int]]:
    if hnswlib is None:
        raise RuntimeError("hnswlib is not available")
    path = Path(index_path)
    meta_path = _meta_path(path)
    if not meta_path.exists():
        raise RuntimeError(f"Missing ANN metadata at {meta_path}")
    metadata = utils.load_json(meta_path)
    dim = int(metadata.get("dim", 0))
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(path))
    index.set_ef(ef_search)
    return index, metadata


def ann_search(
    index_path: str,
    query_vector: np.ndarray,
    k: int,
    *,
    ef_search: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Search ANN index and return (idx, similarity) pairs."""
    query_vector = query_vector.astype(np.float32)
    if ef_search is None:
        ef_search = int(utils.load_json(_meta_path(index_path)).get("ef_search", 100))
    index, _ = _load_ann_index(index_path, ef_search)
    labels, distances = index.knn_query(query_vector, k=k)
    results: List[Tuple[int, float]] = []
    for idx, dist in zip(labels[0], distances[0]):
        if idx == -1:
            continue
        similarity = float(1.0 - dist)
        results.append((int(idx), similarity))
    return results


__all__ = ["build_ann_index", "ann_search", "ann_index_exists"]
