use std::path::PathBuf;

use clap::{Parser, Subcommand};

use ferrite::bench::DATASET_JSONL;
use ferrite::bench::dataset::{TARGET_ROWS, fetch_and_sample};

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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Dataset { out, rows } => {
            let n = fetch_and_sample(&out, rows)?;
            println!("dataset ready: {n} questions -> {}", out.display());
        }
    }
    Ok(())
}
