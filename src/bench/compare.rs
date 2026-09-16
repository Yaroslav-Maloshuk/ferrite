use serde::{Deserialize, Serialize};

use super::{BenchReport, EnvInfo};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricPair {
    pub ferrite: f64,
    pub baseline: f64,
    pub speedup: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Comparison {
    pub p50_ms: MetricPair,
    pub p99_ms: MetricPair,
    pub p999_ms: MetricPair,
    pub rps: MetricPair,
    pub rps_per_core: MetricPair,
    pub peak_rss_mb: MetricPair,
}

/// Speedup for a lower-is-better metric (latency, RSS): baseline/ferrite.
/// >1 means ferrite is faster / uses less memory.
fn latency_speedup(ferrite: f64, baseline: f64) -> f64 {
    if ferrite == 0.0 {
        f64::INFINITY
    } else {
        baseline / ferrite
    }
}

/// Speedup for a higher-is-better metric (throughput): ferrite/baseline.
/// >1 means ferrite is faster.
fn throughput_speedup(ferrite: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        f64::INFINITY
    } else {
        ferrite / baseline
    }
}

fn pair(ferrite: f64, baseline: f64, spi: &impl Fn(f64, f64) -> f64) -> MetricPair {
    MetricPair {
        ferrite,
        baseline,
        speedup: spi(ferrite, baseline),
    }
}

pub fn compare(ours: &BenchReport, baseline: &BenchReport) -> Comparison {
    let o = &ours.metrics;
    let b = &baseline.metrics;
    Comparison {
        p50_ms: pair(o.p50_ms, b.p50_ms, &latency_speedup),
        p99_ms: pair(o.p99_ms, b.p99_ms, &latency_speedup),
        p999_ms: pair(o.p999_ms, b.p999_ms, &latency_speedup),
        rps: pair(o.rps, b.rps, &throughput_speedup),
        rps_per_core: pair(o.rps_per_core, b.rps_per_core, &throughput_speedup),
        peak_rss_mb: pair(o.peak_rss_mb as f64, b.peak_rss_mb as f64, &latency_speedup),
    }
}

pub fn render_table(c: &Comparison, baseline_env: &EnvInfo, ferrite_env: &EnvInfo) -> String {
    let row = |name: &str, m: &MetricPair| {
        format!(
            "| {name} | {:.1} | {:.1} | {:.1}x |",
            m.ferrite, m.baseline, m.speedup
        )
    };
    let mut s = String::new();
    s.push_str("| Metric | Ferrite | Python/LangChain | Speedup |\n");
    s.push_str("| --- | --- | --- | --- |\n");
    s.push_str(&row("P50 (ms)", &c.p50_ms));
    s.push('\n');
    s.push_str(&row("P99 (ms)", &c.p99_ms));
    s.push('\n');
    s.push_str(&row("P99.9 (ms)", &c.p999_ms));
    s.push('\n');
    s.push_str(&row("Throughput RPS", &c.rps));
    s.push('\n');
    s.push_str(&row("RPS/core", &c.rps_per_core));
    s.push('\n');
    s.push_str(&row("Peak RSS (MB)", &c.peak_rss_mb));
    s.push_str(&format!(
        "\nFERRITE_ENV: os={} cores={} rust={} container={}\n",
        ferrite_env.os, ferrite_env.cores, ferrite_env.rust, ferrite_env.container
    ));
    s.push_str(&format!(
        "BASELINE_ENV: os={} cores={} rust={} container={}\n",
        baseline_env.os, baseline_env.cores, baseline_env.rust, baseline_env.container
    ));
    s
}

#[cfg(test)]
mod compare_tests {
    use super::*;
    use crate::bench::{Metrics, Params};

    fn report(p99: f64, rps: f64, rss: u64) -> BenchReport {
        BenchReport {
            label: String::new(),
            env: EnvInfo::current(),
            params: Params {
                n_docs: 0,
                n_queries: 0,
                top_k: 0,
                concurrency: 0,
                window_secs: 0,
            },
            metrics: Metrics {
                p50_ms: p99,
                p99_ms: p99,
                p999_ms: p99,
                rps,
                rps_per_core: rps,
                peak_rss_mb: rss,
            },
        }
    }

    #[test]
    fn compare_reports_speedups() {
        let ours = report(1000.0, 100.0, 60);
        let baseline = report(2000.0, 50.0, 300);
        let c = compare(&ours, &baseline);
        assert_eq!(c.p99_ms.speedup, 2.0);
        assert_eq!(c.rps.speedup, 2.0);
        assert_eq!(c.peak_rss_mb.speedup, 5.0);
    }

    #[test]
    fn render_table_contains_header_and_values() {
        let ours = report(1000.0, 100.0, 60);
        let baseline = report(2000.0, 50.0, 300);
        let c = compare(&ours, &baseline);
        let t = render_table(&c, &ours.env, &baseline.env);
        assert!(t.contains("Metric"));
        assert!(t.contains("2.0x"));
        assert!(t.contains("Ferrite"));
    }
}
