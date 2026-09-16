# Ferrite — Rust-Native Embedding + Retrieval Service — Design

**Date:** 2026-09-16

## Goal

A Rust-native embedding + retrieval module (Candle inference + LanceDB vector store + Tokio async runtime) that is a drop-in replacement for a Python/LangChain embedding+retrieval pipeline. Shipped with a benchmark suite that measures P99 latency, RAM/instance, and RPS/core against an identical Python/LangChain baseline on the identical dataset, Docker Compose orchestration, and tests covering embedding, retrieval, and edge cases.

## Success Criteria

1. A publishable Rust crate `ferrite` exposing a small, deep public interface: `embed`, `ingest`, `search`, `stats`.
2. An HTTP service (`/v1/embed`, `/v1/ingest`, `/v1/search`, `/v1/health`, `/v1/stats`) that mirrors the Python baseline's routes exactly, so a single harness benchmarks both.
3. A benchmark suite producing an apples-to-apples comparison (P99 query latency, RPS/core, peak RSS MB/instance) between Ferrite and the Python/LangChain baseline on the identical Quora-derived dataset.
4. `docker compose up --build` + a single bench command reproduces all reported numbers.
5. Tests pass for: embedding numerics, retrieval round-trip, empty index, duplicates, `top_k > size`, long-text truncation, empty-string embed, Unicode.

## Architecture

Single Cargo crate with multiple binary entry points. The library (`src/lib.rs`) is the deep module: all pipeline behavior sits behind a tiny interface. The CLI service, the HTTP router, and the benchmark harness all cross the same seam.

```
ferrite/
├── Cargo.toml            # lib + 2 bins in one crate
├── src/
│   ├── lib.rs            # pub exports; Ferrite facade, config types, error type
│   ├── pipeline.rs       # Ferrite facade: embed / ingest / search / stats
│   ├── embedding.rs      # Embedder: model load, tokenize, encode, mask-mean-pool, L2 norm
│   ├── store.rs          # VectorStore: lance table open/create, add, query, index
│   ├── http.rs           # axum router + handlers returning types from the facade
│   └── bin/
│       ├── ferrite.rs    # HTTP service + `ingest`/`search` CLI subcommands
│       └── bench.rs      # benchmark harness (HTTP-target and lib-target modes)
├── tests/                # integration tests cross the public seam
├── benchmark/
│   ├── python-baseline/  # FastAPI + LangChain + sentence-transformers + Chroma
│   └── orchestrate.py    # run both targets, collect JSON, emit comparison
├── docker/
│   ├── ferrite.Dockerfile
│   └── baseline.Dockerfile
├── docker-compose.yml    # services: ferrite, baseline, bench
├── data/                 # gitignored: dataset, lancedb stores, bench reports
└── README.md             # usage + generated benchmark numbers
```

## Components

### 1. Embedding pipeline (`src/embedding.rs`)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `refs/pr/21` (the revision the candle bert example pins), 384 dims, safetensors weights (~90 MB). Downloaded via `hf-hub` once at image build / first run into `HF_HOME`, then `HF_HUB_OFFLINE=1` at runtime.
- **Tokenization:** `tokenizers` crate (HuggingFace bindings). `encode_batch` with `PaddingStrategy::BatchLongest`, truncation to 256 tokens (sentence-transformers default) and special tokens enabled. Padding configured on the tokenizer; `encode_batch` yields `ids` + `attention_mask` for the whole batch.
- **Tensors:** `input_ids`, `token_type_ids` (zeros), `attention_mask` -> `candle` tensors on CPU `Device`.
- **Inference:** `candle_transformers::models::bert::BertModel::load(VarBuilder, &Config)` (the `model_type` fallback in `load` transparently handles the `bert.` weight prefix in all-MiniLM-L6-v2's safetensors). `forward(&input_ids, &token_type_ids, Some(&attention_mask))` returns last-hidden-state.
- **Pooling:** mask-mean-pool, exactly matching sentence-transformers (*default* `include_padding_embeddings=false`): `sum(embeddings * mask.unsqueeze(2)) / sum(mask)`, then L2-normalize (`broadcast_div(broadcast_mul(emb, mask), sqrt(sum(sqr)))`). This is the candle bert example's default path, which the upstream README asserts produces the same numeric result as `sentence_transformers`.
- **Threading:** CPU-bound inference runs on a `std::thread` pool (e.g. `std::thread::scope` / dedicated worker threads sized to available cores); the async service calls it via `tokio::task::spawn_blocking` so the tokio runtime stays responsive.
- **Batching:** configurable `batch_size` (default 32). Deterministic: ensemble of one-by-one encodes equals batch encode.

### 2. Retrieval (`src/store.rs`)

- **Backend:** embedded LanceDB (no server), URI `data/lance` (configurable).
- **Schema** (arrow 58, matching `lancedb 0.38.0`'s arrow dependency): `id: Utf8` (or `Uuid`), `text: Utf8`, `vector: FixedSizeList(Float32, 384)`, `metadata: Utf8` (JSON string) or `Map`.
- **Ingest:** build a `RecordBatch` from already-computed embeddings; `table.add` (or `create_table` on first use). Idempotent re-ingest allowed (duplicates test relies on app-level ids; Lance row ids remain distinct).
- **Search:** encode+normalize the query inline; `table.query().limit(k).nearest_to(vec).distance_type(L2).execute()`; decode results. L2 over unit-normalized vectors ranks identically to cosine. `top_k`, `nprobes`, `refine_factor` are pass-through configuration.
- **Index:** configurable `IndexMode` — `Flat` (default; exact, good to ~100K rows) or `IvfPq { num_partitions }`. `create_index` idempotent (skips if already built).
- **Empty index:** searching an empty table returns an empty result set (no panic).

### 3. HTTP service (`src/http.rs`, bin `ferrite.rs`)

axum 0.8 router on `:8080` (port configurable), routes shared with the Python baseline:

| Route | Request | Response |
|---|---|---|
| `POST /v1/embed` | `{"texts": ["..."]}` | `{"embeddings":[[f32;384]], "model": "...", "dim": 384, "latency_ms": 1.2}` |
| `POST /v1/ingest` | `{"items":[{"id":"...","text":"...","metadata":"..."}]}` | `{"ingested": N}` |
| `POST /v1/search` | `{"query":"...","top_k":5}` | `{"results":[{"id","text","score","distance"}]}` |
| `GET /v1/health` | — | `{"status":"ok"}` |
| `GET /v1/stats` | — | `{"rows": N, "index": "...", "model": "...", "dim":384}` |

Validation: `top_k` clamped to `[1, 1000]`; `texts` batch capped (e.g. 256) and non-empty; error responses are JSON `{"error": "..."}` with proper status codes.

### 4. Python baseline (`benchmark/python-baseline/`)

FastAPI app with the exact same five routes, backed by the LangChain `SentenceTransformerEmbeddings` wrapper -> Chroma (in-process, persistent) vector store. Uses the same `all-MiniLM-L6-v2` model path, same dataset file, same top-k semantics. Its only job is to be the reference for the benchmark. Dependencies pinned in `requirements.txt`.

### 5. Benchmark suite (bin `src/bin/bench.rs`)

- **Dataset:** the harness itself is dataset-self-contained. `ferrite bench dataset` fetches `sentence-transformers/quora-duplicates` (HF) via the `hf-hub` crate, reads the parquet with `parquet`+`arrow-rs`, takes a deterministic seeded 25,000-question sample, and writes `data/quora_questions.jsonl` (one question per line). The same file feeds Rust and Python — byte-identical input.
- **Harness:** `ferrite bench run --target <http|lib>` drives the target over HTTP (or in-process facade), with a lib-target mode for CI histogram sanity.
  - *Latency:* warm-up, then a fixed-window run at a target concurrency; per-request latency recorded into `hdrhistogram`; report P50/P99/P99.9.
  - *Throughput:* concurrency ramp (1, 2, 4, ... cores) until P99 breaches a saturation threshold; peak sustained success-rate = max RPS; **RPS/core** = max_RPS / logical cores.
  - *Memory:* sample target service peak RSS (`/proc/<pid>/status` `VmHWM` on Linux, else `getrusage.ru_maxrss`).
  - Output: `data/reports/ferrite.json` / `baseline.json`, identical `{metrics, env, params}` schema.
- **Comparison:** `ferrite bench compare ferrite.json baseline.json` writes `data/reports/comparison.json` (per-metric ratios); `ferrite bench table` renders the markdown table for the README from that file, so README numbers are exactly the measured ones.

### 6. Docker Compose

- `ferrite` service: build `docker/ferrite.Dockerfile` (multi-stage: bare-builder -> runtime `debian-slim`), model baked in, volume `./data` mounted, port 8080.
- `baseline` service: build `docker/baseline.Dockerfile`, volume `./data`, port 8081.
- `bench` service: same image as `ferrite`; command runs `ferrite bench dataset` + `ferrite bench run` against both targets + `compare` + `table`; exit code non-zero on regression/failure.
- One command: `docker compose up --build` then `docker compose run --rm bench`.

## Datasets / determinism

- Source: `sentence-transformers/quora-duplicates` parquet, fetched via `hf-hub` crate.
- Sample: 25,000 rows, seeded RNG (fixed seed constant), `text_1` column only, one question per line in `data/quora_questions.jsonl`.
- Re-use: `ferrite bench dataset` skips download when the file already exists (idempotent).

## Config (env vars, all with defaults)

`FERRITE_PORT` (8080), `FERRITE_MODEL_REPO`, `FERRITE_MODEL_REVISION` (`refs/pr/21`), `FERRITE_HF_HOME`, `FERRITE_INDEX` (`flat`|`ivf_pq`), `FERRITE_IVF_PARTITIONS`, `FERRITE_TOP_K` (10), `FERRITE_BATCH_SIZE` (32), `FERRITE_MAX_TEXT_BATCH` (256), `FERRITE_DATA_DIR` (`data`), `LANCE_URI` (`data/lance`).

## Edge Cases (covered by tests, per requirement 5)

| Case | Expected |
|---|---|
| Empty index search | `[]`, no panic |
| Duplicate texts ingested | distinct Lance row ids; search returns `top_k` results; no crash |
| `top_k` > row count | clamped to row count / returns all |
| `top_k` = 0 or negative | rejected (400) / clamped to 1 |
| Long text > 256 tokens | truncated to 256 |
| Empty string embed | valid 384-dim vector (matches sentence-transformers value); document behavior |
| Unicode / emoji text | tokenized correctly, no panic |
| Batch vs single encode | identical embeddings (determinism) |
| `texts` empty array | 400 error |
| Re-ingest same id | appended (app-level id preserved), search still correct |

## Testing Strategy (TDD)

- **Unit (`src/embedding.rs`, `src/store.rs`):** pooling math vs hand-computed values; truncation; empty string; store round-trip (small in-memory temp lance dir); empty-index query; duplicate ingest; `top_k` clamp.
- **Integration (`tests/`):** full facade `Ferrite` vs small in-process temp dirs — embed -> ingest -> search -> verify top hit contains exact text; parallel searches; stats.
- **HTTP (`src/http.rs`):** `tower::ServiceExt::oneshot` against the router — route validation, error JSON, e2e search.
- **Bench sanity (`bench.rs`):** harness runs against the in-process target, asserts histograms non-empty and metrics in plausible ranges (no network needed).

## Benchmark Target Environment

The dataset-preparation and orchestration steps are Rust-side (`ferrite bench dataset | run | compare | table`), so the bench container needs only the Ferrite image; the Python baseline is a reference service only.

Baseline numbers in the README are produced on this machine (macOS, Apple Silicon, 14 logical cores / 64 GB) and reproduced inside Docker Linux containers. Both sides run in the same container payload (identical CPU quotas when compared).

## Deliverables

- Publishable `ferrite` crate + HTTP service.
- `benchmark/python-baseline/` reference pipeline.
- Benchmark harness + orchestration + generated comparison.
- Docker Compose one-command run.
- Tests green; clippy + `cargo fmt` clean.
- `README.md` with exact measured numbers (from `comparison.json`), usage, and GitHub-ready licensing note.

## Out of Scope (YAGNI)

- GPU / Metal acceleration (CPU-only for parity with baseline).
- Multiple embedding models at runtime (config swap possible, but only MiniLM ships).
- Streaming responses, authN/Z, multi-user isolation.
- Distributed LanceDB (embedded only).