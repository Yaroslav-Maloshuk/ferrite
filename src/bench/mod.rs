use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};

use crate::config::FerriteConfig;
use crate::pipeline::{Ferrite, IngestItem};

pub mod compare;
pub mod dataset;
pub mod http_target;

pub const DATASET_JSONL: &str = "quora_questions.jsonl";

/// Documents ingested (or sent per `/v1/ingest` POST) between embed calls.
/// `Ferrite::embed` caps batches at `max_text_batch` (256), so anything larger
/// must be chunked.
pub const INGEST_BATCH: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvInfo {
    pub cores: usize,
    pub os: String,
    pub rust: String,
    pub container: bool,
}

impl EnvInfo {
    pub fn current() -> Self {
        let container = std::path::Path::new("/.dockerenv").exists()
            || std::path::Path::new("/run/.containerenv").exists();
        Self {
            cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            os: std::env::consts::OS.to_string(),
            rust: env!("CARGO_PKG_VERSION").to_string(),
            container,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Params {
    pub n_docs: usize,
    pub n_queries: usize,
    pub top_k: usize,
    pub concurrency: usize,
    pub window_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub p999_ms: f64,
    pub rps: f64,
    pub rps_per_core: f64,
    pub peak_rss_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub label: String,
    pub env: EnvInfo,
    pub params: Params,
    pub metrics: Metrics,
}

/// VmHWM (Linux) or getrusage ru_maxrss (macOS), in KiB.
pub fn peak_rss_kb() -> u64 {
    if let Ok(stat) = std::fs::read_to_string("/proc/self/status") {
        for line in stat.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest
                    .split_whitespace()
                    .next()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                return kb;
            }
        }
    }
    #[cfg(unix)]
    {
        let mut rusage: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut rusage) };
        if rc == 0 {
            #[cfg(target_os = "macos")]
            {
                return (rusage.ru_maxrss / 1024) as u64; // bytes -> KiB
            }
            #[cfg(not(target_os = "macos"))]
            {
                return rusage.ru_maxrss as u64; // Linux: already KiB
            }
        }
    }
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Target {
    Lib,
    Http(String),
}

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub label: String,
    pub target: Target,
    pub concurrency: usize,
    pub window_secs: u64,
    pub n_docs: usize,
    pub n_queries: usize,
    pub top_k: usize,
    pub dataset: PathBuf,
    pub out: PathBuf,
    pub ferrite_config: FerriteConfig,
}

impl BenchConfig {
    fn with_concurrency(&self, concurrency: usize) -> Self {
        let mut cfg = self.clone();
        cfg.concurrency = concurrency;
        cfg
    }
}

/// Read at most `n` non-empty lines from `path`.
pub fn load_questions(path: &Path, n: usize) -> Vec<String> {
    fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .take(n)
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn ms(h: &Histogram<u64>) -> (f64, f64, f64) {
    (
        h.value_at_quantile(0.50) as f64 / 1000.0,
        h.value_at_quantile(0.99) as f64 / 1000.0,
        h.value_at_quantile(0.999) as f64 / 1000.0,
    )
}

/// Run one fixed-window measurement at a given concurrency, returning
/// (histogram, completed_count).
pub async fn measure_window<F, Fut>(
    concurrency: usize,
    window: Duration,
    mut f: F,
) -> (Histogram<u64>, u64)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<Duration>> + Send + 'static,
{
    let mut hist = Histogram::<u64>::new(3).unwrap();
    let mut count: u64 = 0;
    let deadline = std::time::Instant::now() + window;
    while std::time::Instant::now() < deadline {
        let tasks: Vec<_> = (0..concurrency)
            .map(|_| {
                let fut = f();
                tokio::spawn(fut)
            })
            .collect();
        for t in tasks {
            if let Ok(Ok(lat)) = t.await {
                // `record` (not `saturating_record`): the histogram auto-resizes
                // beyond its initial +-2 unit range, so real latencies are kept.
                let _ = hist.record(lat.as_micros() as u64);
                count += 1;
            }
        }
    }
    (hist, count)
}

/// Initialize an in-process `Ferrite`, ingest the first `cfg.n_docs` lines of
/// the dataset (batched at [`INGEST_BATCH`] to stay inside the embed cap), and
/// return it wrapped for reuse across ramp points.
async fn prepare_lib(cfg: &BenchConfig) -> anyhow::Result<Arc<Ferrite>> {
    let ferrite = Ferrite::init(&cfg.ferrite_config).await?;
    let items: Vec<IngestItem> = load_questions(&cfg.dataset, cfg.n_docs)
        .into_iter()
        .enumerate()
        .map(|(i, text)| IngestItem {
            id: format!("doc-{i}"),
            text,
            metadata: None,
        })
        .collect();
    anyhow::ensure!(
        !items.is_empty(),
        "dataset contains no ingest documents: {}",
        cfg.dataset.display()
    );
    let mut ingested = 0usize;
    for chunk in items.chunks(INGEST_BATCH) {
        ingested += ferrite.ingest(chunk).await?;
    }
    anyhow::ensure!(
        ingested == items.len(),
        "ingest stored {ingested} of {} documents",
        items.len()
    );
    Ok(Arc::new(ferrite))
}

/// One fixed-window measurement through an already-initialized `Ferrite`,
/// probing with the first `cfg.n_queries` questions (cycled).
async fn run_once(
    ferrite: &Arc<Ferrite>,
    cfg: &BenchConfig,
) -> anyhow::Result<(Histogram<u64>, u64)> {
    let probes = load_questions(&cfg.dataset, cfg.n_queries);
    anyhow::ensure!(
        !probes.is_empty(),
        "dataset contains no probe questions: {}",
        cfg.dataset.display()
    );
    let top_k = cfg.top_k;
    let ferrite = ferrite.clone();
    let mut i = 0usize;
    let (hist, count) = measure_window(
        cfg.concurrency,
        Duration::from_secs(cfg.window_secs),
        move || {
            let q = probes[i % probes.len()].clone();
            i += 1;
            let ferrite = ferrite.clone();
            async move {
                let start = std::time::Instant::now();
                ferrite.search(&q, top_k).await?;
                Ok(start.elapsed())
            }
        },
    )
    .await;
    Ok((hist, count))
}

fn write_report(cfg: &BenchConfig, report: &BenchReport) -> anyhow::Result<()> {
    if !cfg.out.as_os_str().is_empty() {
        if let Some(parent) = cfg.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cfg.out, serde_json::to_string_pretty(report)?)?;
    }
    Ok(())
}

fn build_report(
    cfg: &BenchConfig,
    hist: Histogram<u64>,
    count: u64,
    concurrency: usize,
) -> anyhow::Result<BenchReport> {
    let (p50, p99, p999) = ms(&hist);
    let rps = count as f64 / cfg.window_secs.max(1) as f64;
    let env = EnvInfo::current();
    let metrics = Metrics {
        p50_ms: p50,
        p99_ms: p99,
        p999_ms: p999,
        rps,
        rps_per_core: rps / env.cores.max(1) as f64,
        peak_rss_mb: peak_rss_kb() / 1024,
    };
    let report = BenchReport {
        label: cfg.label.clone(),
        env,
        params: Params {
            n_docs: cfg.n_docs,
            n_queries: cfg.n_queries,
            top_k: cfg.top_k,
            concurrency,
            window_secs: cfg.window_secs,
        },
        metrics,
    };
    write_report(cfg, &report)?;
    Ok(report)
}

/// Run a single benchmark: ingest `n_docs`, then a measured window of
/// `n_queries` probes at the configured concurrency.
pub async fn run_bench(cfg: &BenchConfig) -> anyhow::Result<BenchReport> {
    let (hist, count) = match &cfg.target {
        Target::Lib => {
            let ferrite = prepare_lib(cfg).await?;
            run_once(&ferrite, cfg).await
        }
        Target::Http(base) => http_target::http_bench(base, cfg).await,
    }?;
    build_report(cfg, hist, count, cfg.concurrency)
}

/// Ramp `run_bench` over concurrency 1,2,4,8,cores,cores*2 and keep the
/// report with the peak throughput. Lib targets init `Ferrite` once and reuse
/// it across ramp points; HTTP targets ingest `n_docs` through `/v1/ingest`
/// once, then measure a fresh search window per ramp point against the
/// remote service.
pub async fn ramp_and_report(cfg: &BenchConfig) -> anyhow::Result<BenchReport> {
    let (http, ferrite) = match &cfg.target {
        Target::Http(base) => {
            let client = http_target::http_client()?;
            http_target::ingest_over_http(&client, base, cfg).await?;
            (Some((base.clone(), client)), None)
        }
        Target::Lib => (None, Some(prepare_lib(cfg).await?)),
    };
    let cores = EnvInfo::current().cores;
    let mut best: Option<(f64, Histogram<u64>, u64, usize)> = None;
    for c in [1usize, 2, 4, 8, cores, cores * 2] {
        let c = c.max(1);
        let run_cfg = cfg.with_concurrency(c);
        let (hist, count) = match (&http, &ferrite) {
            (Some((base, client)), _) => http_target::http_run_once(client, base, &run_cfg).await?,
            (None, Some(f)) => run_once(f, &run_cfg).await?,
            (None, None) => unreachable!("either an HTTP base URL or a lib Ferrite is prepared"),
        };
        let rps = count as f64 / run_cfg.window_secs.max(1) as f64;
        if best.as_ref().is_none_or(|(best_rps, ..)| rps > *best_rps) {
            best = Some((rps, hist, count, c));
        }
    }
    let (_, hist, count, best_concurrency) = best.expect("at least one ramp point ran");
    build_report(cfg, hist, count, best_concurrency)
}

#[cfg(test)]
mod bench_runner_test {
    use super::*;

    #[tokio::test]
    async fn lib_target_produces_report() {
        let tmp = tempfile::tempdir().unwrap();
        let config = FerriteConfig {
            lance_uri: tmp.path().join("lance").display().to_string(),
            data_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let dataset = tmp.path().join("quora_questions.jsonl");
        std::fs::write(
            &dataset,
            "the cat sits outside\na man is playing guitar\npasta is delicious\n",
        )
        .unwrap();
        let out = tmp.path().join("report.json");
        let cfg = BenchConfig {
            label: "test".into(),
            target: Target::Lib,
            concurrency: 2,
            window_secs: 1,
            n_docs: 3,
            n_queries: 3,
            top_k: 2,
            dataset: dataset.clone(),
            out: out.clone(),
            ferrite_config: config,
        };
        let report = run_bench(&cfg).await.unwrap();
        assert!(report.metrics.rps > 0.0);
        assert!(report.metrics.p99_ms >= 0.0);
        assert!(report.metrics.peak_rss_mb > 0);
        assert!(
            out.is_file(),
            "report file must be written when --out is set"
        );
    }

    #[tokio::test]
    async fn lib_target_ingests_large_batches() {
        let tmp = tempfile::tempdir().unwrap();
        let config = FerriteConfig {
            lance_uri: tmp.path().join("lance").display().to_string(),
            data_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let dataset = tmp.path().join("quora_questions.jsonl");
        let lines: String = (0..INGEST_BATCH + 50)
            .map(|i| format!("question number {i}\n"))
            .collect();
        std::fs::write(&dataset, lines).unwrap();
        let cfg = BenchConfig {
            label: "test-large".into(),
            target: Target::Lib,
            concurrency: 2,
            window_secs: 1,
            n_docs: INGEST_BATCH + 50,
            n_queries: 3,
            top_k: 2,
            dataset: dataset.clone(),
            out: tmp.path().join("report.json"),
            ferrite_config: config,
        };
        let report = run_bench(&cfg).await.unwrap();
        assert_eq!(report.params.n_docs, INGEST_BATCH + 50);
        assert!(report.metrics.rps > 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdrhistogram::Histogram;

    #[test]
    fn histogram_percentiles() {
        let mut h = Histogram::<u64>::new(3).unwrap();
        for i in 0..1000 {
            h.record(i as u64).unwrap();
        }
        assert_eq!(h.value_at_quantile(0.50), 499);
        assert_eq!(h.value_at_quantile(0.99), 989);
        assert_eq!(h.value_at_quantile(0.999), 998);
    }

    #[test]
    fn peak_rss_is_positive() {
        assert!(peak_rss_kb() > 0);
    }

    #[test]
    fn records_real_microsecond_latencies() {
        let mut h = Histogram::<u64>::new(3).unwrap();
        for v in 7000u64..8500 {
            let _ = h.record(v);
        }
        assert!(h.value_at_quantile(0.50) > 7000);
        assert!(h.value_at_quantile(0.50) < 8500);
        assert!(h.min() > 6000, "no values clamped to the initial range");
    }
}
