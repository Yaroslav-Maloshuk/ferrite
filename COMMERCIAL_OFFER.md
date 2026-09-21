# Ferrite — Commercial Offer (Fixed-Scope PoC / Migration)

**What this is:** a fixed-price engagement to replace an existing Python/LangChain/Chroma
embedding-retrieval pipeline with the Ferrite Rust stack, deployed **inside your own
infrastructure** — same API surface, measured against your own data.

**TL;DR:** You get a working, benchmarked, air-gapped retrieval service and a runbook to
own it — for a few hundred dollars, flat.

---

## Scope (all included)

1. **Dataset migration** — you provide your corpus (JSONL: `id` + `text`).
   We ingest it into LanceDB on your box, index it, and expose it over
   `POST /v1/search|ingest|embed`.
2. **Head-to-head benchmark** against your current pipeline on your data:
   throughput (RPS), P99 latency, RSS footprint. Report delivered as
   `comparison.json` + a rendered table.
3. **Air-gapped delivery** — offline Docker image with the embedding model baked in,
   pinned SHA-256 integrity, zero network calls at runtime (verified).
4. **Runbook** — end-to-end instructions to rebuild, re-ingest, and operate the service
   after handover, including `docker save` transfer for closed networks.
5. **Support window** — 30 days of email/Slack fixes for anything in the delivered scope.

## Pricing

| Item | Price (USD, one-off) |
| --- | --- |
| PoC / migration (sprints 1–4 above) | **$500 – $1,500** (fixed after intake call) |
| Deploy into your VPC / air-gapped LAN | +$300 – $800 |
| Plug in **your own** fine-tuned `model.safetensors` | +$300 – $500 |
| Ongoing maintenance (optional) | from $250 / month |

Fixed prices are locked after a 20-minute intake call (dataset size, index type, node count).
No hours metering, no surprise invoices.

## Why it works (the evidence you can verify)

- Rendered benchmark (native, 14-core macOS): **196 RPS vs 138 RPS**, **275 MB vs
  696 MB RSS** against the reference Chroma pipeline — full methodology in the repo.
- **Cosine parity** with Chroma/HNSW semantics (`score = 1 − distance`, IVF-PQ supported).
- **Enterprise surface**: bearer auth (`/v1/health` public), per-request audit trail,
  native HTTPS (rustls), constant-time key comparison.
- MIT core — you are not buying a black box; you are buying a running, owned, documented
  deployment plus the engineering to reach it.

## Process

1. **Intake call (20 min)** — scope, dataset, success criteria → fixed price.
2. **You send corpus** (JSONL) — no proprietary code required on our side.
3. **We build** the store + benchmark in a staging VM of your spec.
4. **Report + handover** — `comparison.json`, rendered table, Docker bundle, runbook.
5. **30-day support** starts.

## Terms (brief)

- Ferrite source stays MIT; this agreement covers our engineering, delivery, and support.
- Deliverables: Docker image + runbook + benchmark report + 30-day support.
- Typical delivery: 1–2 weeks from corpus receipt.