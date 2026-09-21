use std::path::PathBuf;

use clap::{Parser, Subcommand};

use ferrite::bench::compare::{Comparison, compare, render_table};
use ferrite::bench::dataset::{TARGET_ROWS, fetch_and_sample};
use ferrite::bench::{
    BenchConfig, BenchReport, DATASET_JSONL, EnvInfo, Target, ramp_and_report, run_bench,
};
use ferrite::config::FerriteConfig;

#[derive(Parser)]
#[command(name = "ferrite-bench", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate the deterministic Quora dataset (idempotent).
    Dataset {
        #[arg(long, default_value = DATASET_JSONL)]
        out: PathBuf,
        #[arg(long, default_value_t = TARGET_ROWS)]
        rows: usize,
    },
    /// Run a benchmark and emit a JSON report.
    Run {
        #[arg(long, default_value = "ferrite")]
        label: String,
        #[arg(long)]
        target_http: Option<String>,
        /// PID of the HTTP target service; reports its `peak_rss_mb` instead of
        /// the harness process (ignored for the lib target).
        #[arg(long)]
        target_pid: Option<u32>,
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        #[arg(long, default_value_t = 10)]
        window_secs: u64,
        #[arg(long, default_value_t = 5_000)]
        n_docs: usize,
        #[arg(long, default_value_t = 1_000)]
        n_queries: usize,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value = DATASET_JSONL)]
        dataset: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        ramp: bool,
        /// Stop the ramp once P99 exceeds `P99@concurrency=1 * FACTOR`;
        /// `1` disables early stopping (run every ramp point).
        #[arg(long, default_value_t = 2.0)]
        p99_saturation_factor: f64,
    },
    /// Compare two benchmark reports and emit a comparison JSON.
    Compare {
        #[arg(long)]
        ours: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Render a comparison JSON as a markdown table.
    Table {
        #[arg(long)]
        comparison: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Dataset { out, rows } => {
            let n = fetch_and_sample(&out, rows)?;
            println!("dataset ready: {n} questions -> {}", out.display());
        }
        Command::Run {
            label,
            target_http,
            target_pid,
            concurrency,
            window_secs,
            n_docs,
            n_queries,
            top_k,
            dataset,
            out,
            ramp,
            p99_saturation_factor,
        } => {
            let target = match target_http {
                Some(base) => Target::Http(base),
                None => Target::Lib,
            };
            let cfg = BenchConfig {
                label,
                target,
                concurrency,
                window_secs,
                n_docs,
                n_queries,
                top_k,
                dataset,
                out,
                target_pid,
                p99_saturation_factor,
                ferrite_config: FerriteConfig::from_env(),
            };
            let report = if ramp {
                ramp_and_report(&cfg).await?
            } else {
                run_bench(&cfg).await?
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Compare {
            ours,
            baseline,
            out,
        } => {
            let read_json = |path: &PathBuf| -> anyhow::Result<BenchReport> {
                Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
            };
            let c = compare(&read_json(&ours)?, &read_json(&baseline)?);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&c)?)?;
            println!("comparison written -> {}", out.display());
        }
        Command::Table { comparison } => {
            let c: Comparison = serde_json::from_str(&std::fs::read_to_string(&comparison)?)?;
            let env = EnvInfo::current();
            print!("{}", render_table(&c, &env, &env));
        }
    }
    Ok(())
}
