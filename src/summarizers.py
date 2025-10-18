"""Summarization backends supporting stub and optional LLM calls."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List

from . import utils

try:  # optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional
    OpenAI = None  # type: ignore


class Summarizer(ABC):
    """Abstract summarizer interface."""

    def __init__(self, language: str = "English") -> None:
        self.language = language

    @abstractmethod
    def summarize_topic(self, topic: Dict) -> str:
        raise NotImplementedError


class StubSummarizer(Summarizer):
    """Deterministic template-based summarizer."""

    def summarize_topic(self, topic: Dict) -> str:
        title = topic.get("topic_name", "Topic")
        count = topic.get("articles_count", 0)
        keywords = topic.get("keywords", [])
        keywords_text = ", ".join(keywords[:5]) if keywords else "a mix of themes"
        top_entities = topic.get("top_entities", [])
        entity_text = ", ".join(e.get("text", "") for e in top_entities[:3] if e.get("text"))
        if not entity_text:
            entity_text = "varied actors"

        sentences = [
            f"**{title}** appears in {count} article(s) today.",
            f"Key phrases include {keywords_text}.",
            f"Primary entities mentioned: {entity_text}.",
        ]

        if count > 1:
            sentences.append("Multiple outlets covered this story, indicating broad interest across regions.")
        else:
            sentences.append("Currently highlighted by a single outlet, providing an initial perspective on the topic.")

        sentences.append("Refer to the curated articles below for source details.")
        return " ".join(sentences)


class LLMSummarizer(Summarizer):
    """LLM-backed summarizer using the OpenAI client when configured."""

    def __init__(self, model: str, language: str = "English", api_key: str | None = None, api_base: str | None = None) -> None:
        super().__init__(language=language)
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Install it or use the stub backend.")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Provide it to enable the LLM backend.")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def _build_prompt(self, topic: Dict) -> str:
        lines = [
            "Summarize the topic below in 4-6 factual sentences.",
            "Include a brief bullet list of key takeaways and cite sources using markdown links.",
            f"Write in {self.language}.",
            "",
            f"Topic label: {topic.get('topic_name', 'Topic')}",
            f"Article count: {topic.get('articles_count', 0)}",
            f"Keywords: {', '.join(topic.get('keywords', []))}",
            "Representative Articles:",
        ]
        for article in topic.get("representative_articles", []):
            lines.append(
                f"- Title: {article.get('title', 'Untitled')}\n  URL: {article.get('url', '#')}\n  Source: {article.get('source', 'Unknown')}\n"
            )
        return "\n".join(lines)

    def summarize_topic(self, topic: Dict) -> str:
        prompt = self._build_prompt(topic)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise news summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()


def build_markdown(
    digest_path: str,
    out_md: str = "llm_digest_stub.md",
    *,
    language: str = "English",
    backend: str = "stub",
    llm_model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_topics: int | None = None,
) -> None:
    """Create a Markdown digest using the configured summarizer backend."""
    digest = utils.load_json(digest_path)
    topics = digest.get("digest", [])
    if max_topics is not None:
        topics = topics[:max_topics]

    if backend == "llm":
        if llm_model is None:
            raise RuntimeError("llm backend selected but no model provided")
        summarizer: Summarizer = LLMSummarizer(llm_model, language=language, api_key=api_key, api_base=api_base)
    else:
        summarizer = StubSummarizer(language=language)

    lines: List[str] = [f"# Daily Digest ({language})", "", f"Generated at: {digest.get('generated_at', 'N/A')}", ""]
    for topic in topics:
        lines.append(summarizer.summarize_topic(topic))
        lines.append("")
        lines.append("Representative Articles:")
        for article in topic.get("representative_articles", []):
            lines.append(
                f"- [{article.get('title', 'Untitled')}]({article.get('url', '#')}) — {article.get('source', 'Unknown')}"
            )
        lines.append("")

    utils.ensure_dir(out_md)
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


__all__ = ["build_markdown", "Summarizer", "StubSummarizer", "LLMSummarizer"]
