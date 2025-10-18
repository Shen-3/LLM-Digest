"""Optional RSS ingestion to refresh the sample dataset."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

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


def fetch_rss_items(feeds: Iterable[str], limit_per_feed: int = 20) -> List[dict]:
    """Download RSS entries and normalize them to the project schema."""
    articles: List[dict] = []
    for feed_url in feeds:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:limit_per_feed]:
            url = entry.get("link") or ""
            title = preprocess.normalize_text(entry.get("title"))
            description = preprocess.normalize_text(entry.get("summary"))
            content = preprocess.normalize_text(entry.get("description") or description)
            source = preprocess.normalize_text(parsed.feed.get("title", "")) or "Unknown"
            published_at = _canonical_datetime(entry)
            language = entry.get("language") or parsed.feed.get("language")
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


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
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
