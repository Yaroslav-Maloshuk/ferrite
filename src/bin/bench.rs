use std::path::PathBuf;

use clap::{Parser, Subcommand};

use ferrite::bench::dataset::{TARGET_ROWS, fetch_and_sample};
use ferrite::bench::{BenchConfig, DATASET_JSONL, Target, ramp_and_report, run_bench};
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
            concurrency,
            window_secs,
            n_docs,
            n_queries,
            top_k,
            dataset,
            out,
            ramp,
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
                ferrite_config: FerriteConfig::from_env(),
            };
            let report = if ramp {
                ramp_and_report(&cfg).await?
            } else {
                run_bench(&cfg).await?
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
