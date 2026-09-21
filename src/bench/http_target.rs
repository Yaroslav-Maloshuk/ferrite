//! HTTP benchmark target.
//!
//! Measures latency/RPS against a remote `ferrite serve` (or byte-compatible
//! baseline) service over `reqwest`. Before the measured window the harness
//! ingests the first `n_docs` dataset lines through `POST /v1/ingest` (chunked
//! at [`INGEST_BATCH`] to stay inside a 256-item embed batch cap), so
//! service-mode benchmarks exercise the service's real ingest + search path.
//! Search probes are timed `POST /v1/search` requests.

use std::time::Duration;

use hdrhistogram::Histogram;
use reqwest::Client;
use serde_json::json;

use super::{BenchConfig, INGEST_BATCH, load_questions, measure_window};
use crate::pipeline::IngestItem;

/// Shared `reqwest` client for HTTP-mode benchmarks. If `FERRITE_API_KEY` is set
/// (e.g. the target service runs with authentication), every request carries
/// `Authorization: Bearer <key>`.
pub(crate) fn http_client() -> anyhow::Result<Client> {
    let mut builder = Client::builder().user_agent("ferrite-bench");
    if let Ok(key) = std::env::var("FERRITE_API_KEY")
        && !key.is_empty()
    {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))?,
        );
        builder = builder.default_headers(headers);
    }
    Ok(builder.build()?)
}

/// POST the first `cfg.n_docs` dataset lines to `/v1/ingest`, requiring 2xx on
/// every batch. Duplicate ids across runs are acceptable for a benchmark.
pub async fn ingest_over_http(
    client: &Client,
    base_url: &str,
    cfg: &BenchConfig,
) -> anyhow::Result<()> {
    let texts = load_questions(&cfg.dataset, cfg.n_docs);
    anyhow::ensure!(
        !texts.is_empty(),
        "dataset contains no ingest documents: {}",
        cfg.dataset.display()
    );
    let items: Vec<IngestItem> = texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| IngestItem {
            id: format!("doc-{i}"),
            text,
            metadata: None,
        })
        .collect();
    let mut sent = 0usize;
    for chunk in items.chunks(INGEST_BATCH) {
        let res = client
            .post(format!("{base_url}/v1/ingest"))
            .json(&json!({ "items": chunk }))
            .send()
            .await?;
        anyhow::ensure!(
            res.status().is_success(),
            "POST /v1/ingest failed with {}: {}",
            res.status(),
            res.text().await.unwrap_or_default()
        );
        sent += chunk.len();
    }
    anyhow::ensure!(
        sent == items.len(),
        "ingest shipped {sent} of {} documents",
        items.len()
    );
    Ok(())
}

/// One timed `POST /v1/search {query, top_k}`.
pub async fn search_over_http(
    client: &Client,
    base_url: &str,
    query: &str,
    top_k: usize,
) -> anyhow::Result<Duration> {
    let start = std::time::Instant::now();
    let res = client
        .post(format!("{base_url}/v1/search"))
        .json(&json!({ "query": query, "top_k": top_k }))
        .send()
        .await?;
    anyhow::ensure!(
        res.status().is_success(),
        "POST /v1/search failed with {}: {}",
        res.status(),
        res.text().await.unwrap_or_default()
    );
    Ok(start.elapsed())
}

/// One fixed-window measurement against the remote service, probing with the
/// first `cfg.n_queries` questions (cycled).
pub async fn http_run_once(
    client: &Client,
    base_url: &str,
    cfg: &BenchConfig,
) -> anyhow::Result<(Histogram<u64>, u64)> {
    let probes = load_questions(&cfg.dataset, cfg.n_queries);
    anyhow::ensure!(
        !probes.is_empty(),
        "dataset contains no probe questions: {}",
        cfg.dataset.display()
    );
    let top_k = cfg.top_k;
    let mut i = 0usize;
    let (hist, count) = measure_window(
        cfg.concurrency,
        Duration::from_secs(cfg.window_secs),
        move || {
            let q = probes[i % probes.len()].clone();
            i += 1;
            let client = client.clone();
            let base_url = base_url.to_string();
            async move { search_over_http(&client, &base_url, &q, top_k).await }
        },
    )
    .await;
    Ok((hist, count))
}

/// Run one fixed-window measurement against the HTTP service at `base_url`.
/// Ingests the first `cfg.n_docs` dataset lines through `/v1/ingest` before
/// measuring the search window.
pub async fn http_bench(
    base_url: &str,
    cfg: &BenchConfig,
) -> anyhow::Result<(Histogram<u64>, u64)> {
    let client = http_client()?;
    ingest_over_http(&client, base_url, cfg).await?;
    http_run_once(&client, base_url, cfg).await
}

#[cfg(test)]
mod tests {
    use super::BenchConfig;
    use crate::bench::{Target, ramp_and_report, run_bench};
    use crate::config::FerriteConfig;
    use crate::http::router;
    use crate::pipeline::Ferrite;
    use std::sync::Arc;

    fn write_dataset(tmp: &std::path::Path) -> std::path::PathBuf {
        let dataset = tmp.join("quora_questions.jsonl");
        std::fs::write(&dataset, "cat on a couch\nguitar player\npasta recipe\n").unwrap();
        dataset
    }

    fn cfg(
        tmp: &std::path::Path,
        base: String,
        out: std::path::PathBuf,
    ) -> (BenchConfig, FerriteConfig) {
        let ferrite_config = FerriteConfig {
            lance_uri: tmp.join("lance").display().to_string(),
            ..Default::default()
        };
        let cfg = BenchConfig {
            label: "http-test".into(),
            target: Target::Http(base),
            concurrency: 2,
            window_secs: 1,
            n_docs: 3,
            n_queries: 3,
            top_k: 2,
            dataset: write_dataset(tmp),
            out,
            target_pid: None,
            p99_saturation_factor: 2.0,
            ferrite_config: ferrite_config.clone(),
        };
        (cfg, ferrite_config)
    }

    async fn server_for(
        tmp: &std::path::Path,
    ) -> (
        String,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let ferrite_config = FerriteConfig {
            lance_uri: tmp.join("server-lance").display().to_string(),
            ..Default::default()
        };
        let ferrite = Arc::new(Ferrite::init(&ferrite_config).await.unwrap());
        let app = router(ferrite, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            tokio::select! {
                _ = axum::serve(listener, app) => {}
                _ = rx => {}
            };
        });
        (format!("http://{addr}"), tx, handle)
    }

    #[tokio::test]
    async fn http_target_ingests_and_searches() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, tx, handle) = server_for(tmp.path()).await;
        let out = tmp.path().join("report.json");
        let (cfg, _) = cfg(tmp.path(), base, out);
        let report = run_bench(&cfg).await.unwrap();
        assert!(report.metrics.rps > 0.0);
        assert_eq!(report.params.n_docs, 3);
        let _ = tx.send(());
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn http_ramp_reports_peak_concurrency() {
        let tmp = tempfile::tempdir().unwrap();
        let (base, tx, handle) = server_for(tmp.path()).await;
        let out = tmp.path().join("report.json");
        let (cfg, _) = cfg(tmp.path(), base, out);
        let report = ramp_and_report(&cfg).await.unwrap();
        assert!(report.metrics.rps > 0.0);
        assert!(report.params.concurrency >= 1);
        let _ = tx.send(());
        handle.await.unwrap();
    }
}
