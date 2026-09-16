//! HTTP benchmark target.
//!
//! Measures latency/RPS against a remote `ferrite serve` service over
//! `reqwest`. Fully implemented in Task 15; until then every HTTP-mode
//! benchmark fails fast with a clear error.

use hdrhistogram::Histogram;

use super::BenchConfig;

/// Run one fixed-window measurement against the HTTP service at `base_url`.
pub async fn http_bench(
    _base_url: &str,
    _cfg: &BenchConfig,
) -> anyhow::Result<(Histogram<u64>, u64)> {
    anyhow::bail!("HTTP target not implemented yet (Task 15)")
}
