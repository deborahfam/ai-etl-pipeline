# FlowAI ETL (`ai-etl-pipeline`)

Portfolio project: an **AI-augmented ETL toolkit** with a small pipeline engine, multi-provider LLM gateway, and demos that mix tabular data, NLP, and document (image/PDF) workflows—similar in spirit to tools like Unstract (document extraction), Pathway (data movement), and Mage.ai (pipeline ergonomics).

## Alignment with a typical “ETL + LLM” interview brief

| Ask | Delivered here |
|-----|----------------|
| Design & implement an ETL process manager in Python | `Pipeline` DAG, `@extract` / `@transform` / `@load` / `@ai_transform`, `PipelineContext`, retries & snapshots (see `config/settings.yaml`) |
| Show **runnable** ETL examples | `pipelines/demo_*.py`, `python -m pipelines.run_all` |
| Integrate an **LLM via API** (or similar) | `LLMGateway` + Anthropic / OpenAI / OpenRouter / LM Studio adapters |
| Use the LLM for **text and/or images** | Text: profiling, anomalies, enrichment, advisor. Images: invoice vision extraction (`demo_invoice_processing`) |
| **Ingenious** use: auto dataset descriptions | `auto_profile` → semantic column + dataset narrative |
| **Ingenious** use: anomalies + **natural language** explanations | `detect_anomalies` → statistical pass + LLM classification & explanation |
| Good engineering practices | Typing, logging, Pydantic models, tests (`pytest`), config YAML, `.env` |

**Showcase command for “real data + brief checklist”:** after fetching open data, run `python3 -m pipelines.demo_open_data_etl` — it writes `output/RECRUITER_BRIEF_ALIGNMENT.md` mapping requirements to code paths.

## What it demonstrates

- **Architecture**: DAG-based `Pipeline` with `@extract` / `@transform` / `@load` / `@ai_transform`, shared `PipelineContext`, event bus, and optional lineage-style state.
- **LLM integration**: Single `LLMGateway` with **Anthropic**, **OpenAI**, **OpenRouter**, and **LM Studio** adapters; automatic provider detection, fallback chain, token/cost tracking, and SQLite-backed response cache.
- **Multimodal**: CSV/JSON/API extractors, **invoice images** via vision-capable models, and **PDF** intake (bytes → optional structured vision extraction).
- **“Intelligence” layer**: Profiling, anomaly detection (stats + explanations), quality scoring, pipeline advisor, PII redaction, semantic schema mapping.

## Requirements

- **Python 3.11+** (tested install path on 3.11–3.13)
- At least one LLM provider configured (see `.env.example`). Demos that call the LLM need keys; the test suite uses mocks.

## Setup

```bash
cd ai-etl-pipeline
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install -e ".[dev]"
cp .env.example .env        # then add your API keys
```

Generate **invoice images** and a **sample PDF** (idempotent):

```bash
python3 data/bootstrap_demo_data.py
```

**Real open datasets** (Titanic + Palmer Penguins) into `data/real/` — not committed; run once:

```bash
python3 data/fetch_open_datasets.py
```

Sources and licenses: `data/real/README.md`.

## Run the demos

```bash
python3 -m pipelines.run_all                      # four demos
python3 -m pipelines.run_all --quick              # skip vision invoice demo
python3 -m pipelines.run_all --with-open-data     # include Titanic open-data pipeline (needs fetch above)

python3 -m pipelines.demo_open_data_etl           # ETL + LLM on real Titanic CSV + alignment report
python3 -m pipelines.demo_sales_analytics
python3 -m pipelines.demo_customer_reviews
python3 -m pipelines.demo_invoice_processing
python3 -m pipelines.demo_multiformat_ingestion
```

Artifacts are written under `output/` (gitignored): DuckDB, Parquet, CSV, and Markdown-style reports depending on the demo.

## Tests

```bash
python3 -m pytest tests/ -v
```

## Project layout

| Path | Role |
|------|------|
| `src/engine/` | Pipeline runtime, steps, decorators, context, events, Pydantic models |
| `src/llm/` | Gateway, adapters, cache, cost tracker |
| `src/extractors/` | CSV, API, web, image, PDF |
| `src/transformers/` | Clean, validate, enrich, PII, semantic mapping |
| `src/loaders/` | DuckDB, SQLite, file exports |
| `src/intelligence/` | Profiler, anomalies, documents, quality, advisor |
| `pipelines/` | End-to-end demos |
| `data/` | Sample CSV/JSON, `data/real/` for fetched open data, invoices, PDF |
| `config/` | YAML settings for pipeline defaults and provider hints |

### Cloud note (AWS and similar)

This repo is **deliberately local-first** (DuckDB, files, your API keys). In an interview you can describe how the same boundaries map to AWS: **S3** for landing/staging files, **Glue** or **Lambda** for extract/transform orchestration, **RDS / Redshift / Athena** for load/query, and **Bedrock** (or a Lambda calling OpenAI/Anthropic) for LLM steps—without coupling the codebase to one vendor. That is separate from proving Python ETL + LLM design skills in this project.

## License

MIT
