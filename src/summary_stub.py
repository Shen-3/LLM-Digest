"""CLI entry for generating Markdown digests via stub or LLM summarizers."""
from __future__ import annotations

import argparse
from typing import Sequence

from . import summarizers, utils


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown digest summary")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--language", default="English", help="Output language label")
    parser.add_argument("--output", default="llm_digest_stub.md", help="Destination Markdown file")
    parser.add_argument("--backend", choices={"stub", "llm"}, default="stub", help="Summarization backend")
    parser.add_argument("--llm-model", default=None, help="LLM model name (required when backend=llm)")
    parser.add_argument("--api-key", default=None, help="API key for LLM backend (falls back to env variable)")
    parser.add_argument("--api-base", default=None, help="Custom API base URL for LLM providers")
    parser.add_argument("--max-topics", type=int, default=None, help="Limit number of topics in summary")
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()
    config = utils.load_config(args.config)
    summary_cfg = config.get("summary", {})
    digest_path = config["paths"]["output"]

    backend = args.backend or summary_cfg.get("backend", "stub")
    llm_model = args.llm_model or summary_cfg.get("llm_model")
    api_key = args.api_key or summary_cfg.get("api_key")
    api_base = args.api_base or summary_cfg.get("api_base")
    max_topics = args.max_topics or summary_cfg.get("max_topics")

    summarizers.build_markdown(
        digest_path,
        out_md=args.output,
        language=args.language,
        backend=backend,
        llm_model=llm_model,
        api_key=api_key,
        api_base=api_base,
        max_topics=max_topics,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
