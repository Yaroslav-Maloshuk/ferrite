use std::time::Duration;

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};

pub mod dataset;

pub const DATASET_JSONL: &str = "quora_questions.jsonl";

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
                return rusage.ru_maxrss; // Linux: already KiB
            }
        }
    }
    0
}

#[expect(dead_code)]
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
                hist.saturating_record(lat.as_micros() as u64);
                count += 1;
            }
        }
    }
    (hist, count)
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
}
