# Multilingual News Topic Modeling & Daily Digest

A student-scale project for clustering multilingual news articles, generating a daily digest, and exposing the results through a FastAPI service and optional Streamlit UI. The entire pipeline runs locally on CPU hardware and uses open-source models.

## Quickstart

1. **Create & activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline**
   ```bash
   python -m src.digest --config configs/config.yaml
   ```
   This produces `news_digest_output.json` and prints a console summary.
4. **Compute KPIs & chart**
   ```bash
   python -m src.metrics --config configs/config.yaml
   ```
   Find the bar chart at `plots/topics_bar.png`.
5. **Generate stub Markdown digest**
   ```bash
   python -m src.summary_stub --config configs/config.yaml
   ```
   The file `llm_digest_stub.md` is created alongside the JSON digest. Switch to an LLM-backed summary with `--backend llm --llm-model <model>` and set `OPENAI_API_KEY`.
6. **Serve the API**
   ```bash
   uvicorn src.api:app --reload --port 8000
   ```
   Visit `http://127.0.0.1:8000/docs` for the interactive docs.
7. **(Optional) Launch the Streamlit UI**
   ```bash
   streamlit run ui/streamlit_app.py
   ```
8. **Run tests**
   ```bash
   pytest
   ```

## Configuration

The default configuration lives in `configs/config.yaml`. Adjust cluster parameters, keyword counts, and filesystem paths as needed:

```yaml
paths:
  input: "data/sample_news.json"
  output: "news_digest_output.json"
  plot_dir: "plots"
  vectors: "artifacts/article_vectors.npy"
  history_dir: "artifacts/digests"
  run_log: "artifacts/run_history.json"

model:
  name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  batch_size: 64
  normalize: true

cluster:
  eps: 0.7
  min_samples: 3

keywords:
  per_topic: 8

enrichment:
  enable_entities: true
  spacy_model: "xx_ent_wiki_sm"

search:
  use_ann: false
  ann_index: "artifacts/ann_index.bin"
  ann_m: 16
  ann_ef_construction: 200
  ann_ef_search: 100

summary:
  backend: "stub"
  llm_model: null
  max_topics: null
```

## Project Structure

```
news-topics/
├── configs/
├── data/
├── scripts/
├── src/
├── tests/
└── ui/
```

Key modules:
- `src.preprocess`: Cleaning, language detection, and deduplication.
- `src.entities`: Optional spaCy-based entity extraction.
- `src.embeddings`: Multilingual MiniLM sentence-transformer embeddings.
- `src.cluster`: DBSCAN topic clustering, keyword extraction, labeling.
- `src.digest`: Pipeline orchestration, digest history logging, and ANN index generation.
- `src.search`: Utilities for building/loading HNSW indices and cosine search.
- `src.metrics`: KPI computation (including coherence proxies) and matplotlib charting.
- `src.summarizers`: Shared summarization backends (stub + optional LLM).
- `src.summary_stub`: CLI entry for rendering Markdown digests.
- `src.api`: FastAPI app exposing digest, metrics, and search endpoints.

## Data Inputs

Sample input data is located at `data/sample_news.json` and follows the schema defined in the project prompt. The optional `data/feeds.txt` lists RSS feeds that can be ingested via `src.ingest_rss` to refresh the dataset.

## Make Targets

The provided `Makefile` wraps common tasks:
- `make install`
- `make run`
- `make api`
- `make ui`
- `make test`

## Optional Enhancements

- **ANN search:** Install `hnswlib` (see `requirements_midterm.txt`) and set `search.use_ann: true` to build an HNSW index reused by the API/UI.
- **Entity extraction:** Install a multilingual spaCy model such as `xx_ent_wiki_sm`; entities are added to articles and per-topic summaries.
- **LLM summaries:** Install the `openai` package, set `OPENAI_API_KEY`, and configure `summary.backend: llm` with your preferred model.
- **Digest history:** Each run appends metadata to `artifacts/run_history.json` and stores a timestamped digest in `artifacts/digests/`; the Streamlit UI offers a date selector.
- **Similarity explorer:** The UI sidebar exposes both keyword search and article-to-article similarity (uses ANN if available).

## Future Work

- Swap DBSCAN for BERTopic or Hierarchical HDBSCAN for richer topic discovery.
- Extend evaluations with coherence metrics and trend tracking.
