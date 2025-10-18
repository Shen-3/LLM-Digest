"""Optional RSS ingestion to refresh the sample dataset."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import feedparser

from . import preprocess, utils


def _hash_entry(url: str, title: str) -> str:
    payload = (url or "") + "::" + (title or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _canonical_datetime(entry: feedparser.FeedParserDict) -> str:
    published = entry.get("published") or entry.get("updated")
    if not published:
        return utils.utcnow_iso()
    try:
        parsed = datetime(*entry.published_parsed[:6])  # type: ignore[attr-defined]
        return parsed.strftime(utils.ISO_FORMAT)
    except Exception:  # pragma: no cover - defensive fallback
        return utils.utcnow_iso()


def _fp_value_to_str(raw: object) -> str | None:
    """Safely coerce common feedparser values to a plain str or None.

    feedparser sometimes returns strings, None, lists of FeedParserDict or
    FeedParserDict-like objects. This helper normalizes those common cases to
    a simple str or None so downstream callers (like preprocess.normalize_text)
    can accept a well-typed input.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    # common case: list of items
    if isinstance(raw, list):
        if not raw:
            return None
        first = raw[0]
        if isinstance(first, str):
            return first
        # try FeedParserDict-like access
        try:
            return first.get("value") or first.get("title") or first.get("href")  # type: ignore[attr-defined]
        except Exception:
            return str(first)
    # FeedParserDict-like single object
    try:
        return raw.get("value") or raw.get("title")  # type: ignore[arg-type]
    except Exception:
        return str(raw)


def fetch_rss_items(feeds: Iterable[str], limit_per_feed: int = 20) -> List[dict]:
    """Download RSS entries and normalize them to the project schema."""
    articles: List[dict] = []
    for feed_url in feeds:
        parsed = feedparser.parse(feed_url)
        feed_obj = parsed.get("feed")
        for entry in parsed.entries[:limit_per_feed]:
            # safely extract feed-level fields (parsed.get("feed") can be various types)
            feed_link_raw = None
            feed_title_raw = None
            feed_language_raw = None
            if isinstance(feed_obj, dict):
                feed_link_raw = feed_obj.get("link")
                feed_title_raw = feed_obj.get("title", "")
                feed_language_raw = feed_obj.get("language")

            url = _fp_value_to_str(entry.get("link") or feed_link_raw or "") or ""
            title = preprocess.normalize_text(_fp_value_to_str(entry.get("title")))
            description = preprocess.normalize_text(_fp_value_to_str(entry.get("summary")))
            content = preprocess.normalize_text(_fp_value_to_str(entry.get("description")) or description)
            source = preprocess.normalize_text(_fp_value_to_str(feed_title_raw)) or "Unknown"
            published_at = _canonical_datetime(entry)
            language = _fp_value_to_str(entry.get("language") or feed_language_raw)
            articles.append(
                {
                    "id": _hash_entry(url, title),
                    "title": title,
                    "description": description,
                    "content": content,
                    "url": url,
                    "source": source,
                    "language": language,
                    "published_at": published_at,
                }
            )
    return articles


def save_items_to_json(items: List[dict], path: str) -> None:
    """Persist RSS articles to JSON."""
    utils.save_json(items, path)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch RSS feeds and build a news dataset")
    parser.add_argument("--feeds", default="data/feeds.txt", help="Path to list of RSS feed URLs")
    parser.add_argument("--output", default="data/sample_news.json", help="Where to store the aggregated JSON")
    parser.add_argument("--limit", type=int, default=20, help="Maximum entries per feed")
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()
    feeds_text = Path(args.feeds).read_text(encoding="utf-8")
    feeds = [line.strip() for line in feeds_text.splitlines() if line.strip()]
    items = fetch_rss_items(feeds, limit_per_feed=args.limit)
    save_items_to_json(items, args.output)
    print(f"Fetched {len(items)} articles")


if __name__ == "__main__":  # pragma: no cover
    main()
