"""Utility helpers shared across the news topic modeling pipeline."""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    """Create the directory for a file path if it does not already exist."""
    directory = Path(path).expanduser().resolve().parent
    directory.mkdir(parents=True, exist_ok=True)


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load JSON data from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    """Serialize JSON payloads with UTF-8 encoding."""
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)


def utcnow_iso() -> str:
    """Return the current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace characters to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def strip_html(text: str) -> str:
    """Remove rudimentary HTML tags using a lightweight regex."""
    return re.sub(r"<[^>]+>", " ", text)


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a matrix and a single vector."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if vector.ndim != 1:
        raise ValueError("vector must be 1-dimensional")
    dot = matrix @ vector
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    denom = np.clip(matrix_norms * vector_norm, a_min=1e-12, a_max=None)
    return dot / denom


def safe_average(values: Iterable[float]) -> float:
    """Return the arithmetic mean or 0.0 if the iterable is empty."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def logarithmic_bonus(length: int, *, base: float = 10.0) -> float:
    """Compute a small logarithmic bonus for positive lengths."""
    if length <= 0:
        return 0.0
    return math.log(length + 1, base)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"[\\w]+", text.lower(), flags=re.UNICODE)


def append_run_record(path: str | os.PathLike[str], record: Dict[str, Any]) -> None:
    """Append a run record to a JSON log, storing a list structure."""
    ensure_dir(path)
    log_path = Path(path)
    if log_path.exists():
        try:
            data = load_json(log_path)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:  # pragma: no cover - invalid file
            data = []
    else:
        data = []
    data.append(record)
    save_json(data, log_path)
