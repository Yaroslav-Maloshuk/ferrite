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
    /// Pin model.safetensors to this SHA-256 (hex); every load fails loudly on
    /// mismatch. Air-gapped integrity hook.
    pub model_sha256: Option<String>,
    pub index: IndexMode,
    pub ivf_partitions: usize,
    pub top_k: usize,
    pub batch_size: usize,
    pub max_text_batch: usize,
    pub data_dir: PathBuf,
    pub lance_uri: String,
    pub port: u16,
    /// When set, every route except `/v1/health` requires a bearer API key.
    pub api_key: Option<String>,
    /// PEM cert/key paths; serving enables TLS when both are set.
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
}

impl Default for FerriteConfig {
    fn default() -> Self {
        Self {
            model_repo: "sentence-transformers/all-MiniLM-L6-v2".into(),
            model_revision: "refs/pr/21".into(),
            model_dir: std::env::var_os("FERRITE_MODEL_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/models")),
            model_sha256: None,
            index: IndexMode::Flat,
            ivf_partitions: 16,
            top_k: 10,
            batch_size: 32,
            max_text_batch: 256,
            data_dir: std::env::var_os("FERRITE_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data")),
            lance_uri: "data/lance".into(),
            port: 8080,
            api_key: None,
            tls_cert: None,
            tls_key: None,
        }
    }
}

impl FerriteConfig {
    pub fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let mut c = Self::default();
        if let Some(v) = lookup("FERRITE_INDEX") {
            c.index = match v.to_ascii_lowercase().as_str() {
                "flat" => IndexMode::Flat,
                "ivf_pq" => IndexMode::IvfPq,
                other => panic!("FERRITE_INDEX must be flat|ivf_pq, got {other}"),
            };
        }
        if let Some(v) = lookup("FERRITE_IVF_PARTITIONS") {
            c.ivf_partitions = v.parse().expect("FERRITE_IVF_PARTITIONS must be a usize");
        }
        if let Some(v) = lookup("FERRITE_TOP_K") {
            c.top_k = v.parse().expect("FERRITE_TOP_K must be a usize");
        }
        if let Some(v) = lookup("FERRITE_BATCH_SIZE") {
            c.batch_size = v.parse().expect("FERRITE_BATCH_SIZE must be a usize");
        }
        if let Some(v) = lookup("FERRITE_MAX_TEXT_BATCH") {
            c.max_text_batch = v.parse().expect("FERRITE_MAX_TEXT_BATCH must be a usize");
        }
        if let Some(v) = lookup("FERRITE_LANCE_URI") {
            c.lance_uri = v;
        }
        if let Some(v) = lookup("FERRITE_PORT") {
            c.port = v.parse().expect("FERRITE_PORT must be a u16");
        }
        if let Some(v) = lookup("FERRITE_MODEL_REPO") {
            c.model_repo = v;
        }
        if let Some(v) = lookup("FERRITE_MODEL_REVISION") {
            c.model_revision = v;
        }
        if let Some(v) = lookup("FERRITE_MODEL_SHA256") {
            c.model_sha256 = Some(v);
        }
        if let Some(v) = lookup("FERRITE_API_KEY") {
            c.api_key = Some(v);
        }
        if let Some(v) = lookup("FERRITE_TLS_CERT") {
            c.tls_cert = Some(PathBuf::from(v));
        }
        if let Some(v) = lookup("FERRITE_TLS_KEY") {
            c.tls_key = Some(PathBuf::from(v));
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_repo_and_revision_come_from_env() {
        let c = FerriteConfig::from_lookup(|key| match key {
            "FERRITE_MODEL_REPO" => Some("my-org/my-model".into()),
            "FERRITE_MODEL_REVISION" => Some("main".into()),
            _ => None,
        });
        assert_eq!(c.model_repo, "my-org/my-model");
        assert_eq!(c.model_revision, "main");
    }

    #[test]
    fn defaults_when_env_absent() {
        let c = FerriteConfig::from_lookup(|_| None);
        assert_eq!(c.model_repo, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(c.model_revision, "refs/pr/21");
        assert_eq!(c.top_k, 10);
        assert_eq!(c.port, 8080);
        assert_eq!(c.ivf_partitions, 16);
        assert_eq!(c.max_text_batch, 256);
    }

    #[test]
    fn ivf_partitions_and_max_text_batch_come_from_env() {
        let c = FerriteConfig::from_lookup(|key| match key {
            "FERRITE_IVF_PARTITIONS" => Some("64".into()),
            "FERRITE_MAX_TEXT_BATCH" => Some("128".into()),
            _ => None,
        });
        assert_eq!(c.ivf_partitions, 64);
        assert_eq!(c.max_text_batch, 128);
    }

    #[test]
    fn enterprise_env_vars_are_optional_and_parse() {
        let c = FerriteConfig::from_lookup(|_| None);
        assert!(c.api_key.is_none());
        assert!(c.tls_cert.is_none());
        assert!(c.tls_key.is_none());
        assert!(c.model_sha256.is_none());
        let c = FerriteConfig::from_lookup(|key| match key {
            "FERRITE_API_KEY" => Some("secret".into()),
            "FERRITE_TLS_CERT" => Some("/certs/t.pem".into()),
            "FERRITE_TLS_KEY" => Some("/certs/k.pem".into()),
            "FERRITE_MODEL_SHA256" => Some("deadbeef".into()),
            _ => None,
        });
        assert_eq!(c.api_key.as_deref(), Some("secret"));
        assert_eq!(
            c.tls_cert
                .as_deref()
                .map(|p| p.to_string_lossy().into_owned()),
            Some("/certs/t.pem".into())
        );
        assert_eq!(c.model_sha256.as_deref(), Some("deadbeef"));
    }
}
