"""Toolkit for multilingual news topic modeling and digest generation."""
from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - best effort metadata fetch
    __version__ = version("news-topics")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
