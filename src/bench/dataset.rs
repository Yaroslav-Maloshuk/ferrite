//! Deterministic Quora questions dataset generation (identical input for
//! both Ferrite and the Python baseline).
use std::path::{Path, PathBuf};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub const SEED: u64 = 42;
pub const TARGET_ROWS: usize = 25_000;
const REPO: &str = "sentence-transformers/quora-duplicates";
const PARQUET: &str = "pair-class/train-00000-of-00001.parquet";
const TEXT_COL: &str = "sentence1";

fn cache_path() -> PathBuf {
    let base = std::env::temp_dir().join("ferrite-dataset");
    std::fs::create_dir_all(&base).expect("create dataset cache");
    base.join(PARQUET.replace('/', "_"))
}

fn download_parquet() -> anyhow::Result<PathBuf> {
    let cached = cache_path();
    if cached.is_file() {
        return Ok(cached);
    }
    let client = hf_hub::HFClientSync::new()?;
    let (owner, name) = hf_hub::split_id(REPO);
    let repo = client.dataset(owner, name);
    let file = repo.download_file().filename(PARQUET).send()?;
    std::fs::copy(&file, &cached)?;
    Ok(cached)
}

fn read_text_column(path: &Path) -> anyhow::Result<Vec<String>> {
    use arrow_array::Array;
    use arrow_array::StringArray;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        if let Some(col) = batch.column_by_name(TEXT_COL) {
            let arr = col.as_any().downcast_ref::<StringArray>();
            if let Some(arr) = arr {
                for i in 0..arr.len() {
                    out.push(arr.value(i).to_string());
                }
            }
        }
    }
    Ok(out)
}

/// Write `rows` deterministically sampled questions to `out` (one per line).
/// Idempotent: if `out` already exists it is left untouched.
pub fn fetch_and_sample(out: &Path, rows: usize) -> anyhow::Result<usize> {
    if out.is_file() {
        return Ok(std::fs::read_to_string(out)?.lines().count());
    }
    let parquet = download_parquet()?;
    let mut texts = read_text_column(&parquet)?;
    let mut rng = StdRng::seed_from_u64(SEED);
    texts.shuffle(&mut rng);
    texts.truncate(rows);
    texts.retain(|t| !t.trim().is_empty());
    let mut content = String::new();
    for t in &texts {
        content.push_str(t.trim());
        content.push('\n');
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, content)?;
    Ok(texts.len())
}

#[cfg(test)]
mod dataset_test {
    use super::*;

    #[test]
    fn generate_is_idempotent_and_writes_lines() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("q.jsonl");
        let n1 = fetch_and_sample(&out, 500).unwrap();
        assert_eq!(n1, 500);
        let lines = std::fs::read_to_string(&out).unwrap();
        assert_eq!(lines.lines().count(), 500);
        let n2 = fetch_and_sample(&out, 500).unwrap();
        assert_eq!(n2, 500, "idempotent re-run does not duplicate");
        let lines2 = std::fs::read_to_string(&out).unwrap();
        assert_eq!(lines, lines2, "byte-identical on re-run");
    }

    #[test]
    fn same_seed_same_sample() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.jsonl");
        let b = dir.path().join("b.jsonl");
        fetch_and_sample(&a, 100).unwrap();
        fetch_and_sample(&b, 100).unwrap();
        assert_eq!(
            std::fs::read_to_string(&a).unwrap(),
            std::fs::read_to_string(&b).unwrap()
        );
    }
}
