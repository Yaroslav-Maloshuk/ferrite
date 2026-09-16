use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IndexMode {
    Flat,
    IvfPq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FerriteConfig {
    pub model_repo: String,
    pub model_revision: String,
    pub model_dir: PathBuf,
    pub index: IndexMode,
    pub top_k: usize,
    pub batch_size: usize,
    pub max_text_batch: usize,
    pub data_dir: PathBuf,
    pub lance_uri: String,
    pub port: u16,
}

impl Default for FerriteConfig {
    fn default() -> Self {
        Self {
            model_repo: "sentence-transformers/all-MiniLM-L6-v2".into(),
            model_revision: "refs/pr/21".into(),
            model_dir: std::env::var_os("FERRITE_MODEL_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/models")),
            index: IndexMode::Flat,
            top_k: 10,
            batch_size: 32,
            max_text_batch: 256,
            data_dir: std::env::var_os("FERRITE_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data")),
            lance_uri: "data/lance".into(),
            port: 8080,
        }
    }
}

impl FerriteConfig {
    pub fn from_env() -> Self {
        #[allow(clippy::redundant_closure)]
        let mut c = Self::default();
        if let Ok(v) = std::env::var("FERRITE_INDEX") {
            c.index = match v.to_ascii_lowercase().as_str() {
                "flat" => IndexMode::Flat,
                "ivf_pq" => IndexMode::IvfPq,
                other => panic!("FERRITE_INDEX must be flat|ivf_pq, got {other}"),
            };
        }
        if let Ok(v) = std::env::var("FERRITE_TOP_K") {
            c.top_k = v.parse().expect("FERRITE_TOP_K must be a usize");
        }
        if let Ok(v) = std::env::var("FERRITE_BATCH_SIZE") {
            c.batch_size = v.parse().expect("FERRITE_BATCH_SIZE must be a usize");
        }
        if let Ok(v) = std::env::var("FERRITE_LANCE_URI") {
            c.lance_uri = v;
        }
        if let Ok(v) = std::env::var("FERRITE_PORT") {
            c.port = v.parse().expect("FERRITE_PORT must be a u16");
        }
        c
    }
}
