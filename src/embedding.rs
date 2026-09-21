use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::config::FerriteConfig;
use crate::error::FerriteError;
use crate::pooling::{l2_normalize, mask_mean_pool};

pub const MODEL_MAX_LENGTH: usize = 256;
const FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];
const MANIFEST_FILE: &str = ".manifest.json";

/// Per-file SHA-256 manifest written by `download_model`/`ferrite prefetch`.
/// Presence + a matching hash for every file means the cached model is trusted
/// and the network is never touched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub repo: String,
    pub revision: String,
    pub files: Vec<(String, String)>,
}

fn sha256_hex(path: &Path) -> Result<String, FerriteError> {
    let bytes = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

fn hashed_files(dir: &Path) -> Result<Vec<(String, String)>, FerriteError> {
    FILES
        .iter()
        .map(|f| {
            let hash = sha256_hex(&dir.join(f))?;
            Ok((f.to_string(), hash))
        })
        .collect()
}

fn load_manifest(dir: &Path) -> Result<ModelManifest, FerriteError> {
    let text = std::fs::read_to_string(manifest_path(dir))?;
    serde_json::from_str(&text).map_err(|e| FerriteError::Model(format!("bad manifest: {e}")))
}

fn write_manifest(config: &FerriteConfig) -> Result<(), FerriteError> {
    let manifest = ModelManifest {
        repo: config.model_repo.clone(),
        revision: config.model_revision.clone(),
        files: hashed_files(&config.model_dir)?,
    };
    let text = serde_json::to_string(&manifest)
        .map_err(|e| FerriteError::Model(format!("manifest serialize: {e}")))?;
    std::fs::write(manifest_path(&config.model_dir), text).map_err(FerriteError::Io)?;
    Ok(())
}

fn verify_manifest(dir: &Path) -> Result<(), FerriteError> {
    let manifest = load_manifest(dir)?;
    let expect: Vec<(String, String)> = manifest
        .files
        .iter()
        .filter(|(f, _)| FILES.contains(&f.as_str()))
        .cloned()
        .collect();
    if expect.len() != FILES.len() {
        return Err(FerriteError::Model(
            "manifest does not cover all model files".into(),
        ));
    }
    for (file, expected) in &expect {
        let actual = sha256_hex(&dir.join(file))?;
        if &actual != expected {
            return Err(FerriteError::Model(format!(
                "integrity check failed for {file} ({actual} != {expected}); \
                 model may be tampered or corrupt"
            )));
        }
    }
    Ok(())
}

fn files_present(dir: &Path) -> bool {
    FILES.iter().all(|f| dir.join(f).is_file())
}

/// Fetch the model into `FERRITE_MODEL_DIR` (idempotent). When every file is
/// present and the per-file SHA-256 manifest matches, a *no-op that never
/// touches the network* — this is the air-gapped entry point. If
/// `FERRITE_MODEL_SHA256` is set, `model.safetensors` is also pinned to that
/// exact hash and mismatches fail loudly.
pub fn download_model(config: &FerriteConfig) -> Result<(), FerriteError> {
    let dir = &config.model_dir;
    if files_present(dir) {
        let pinned = config
            .model_sha256
            .as_ref()
            .map(|expected| {
                sha256_hex(&dir.join("model.safetensors")).map(|a| a.eq_ignore_ascii_case(expected))
            })
            .transpose()?;
        if pinned == Some(false) {
            return Err(FerriteError::Model(
                "model.safetensors does not match FERRITE_MODEL_SHA256".into(),
            ));
        }
        return match manifest_path(dir).is_file() {
            true => verify_manifest(dir),
            // Legacy cache side without a manifest: stamp it, then trust.
            false => write_manifest(config),
        };
    }
    if config.model_sha256.is_none() {
        tracing::warn!("FERRITE_MODEL_SHA256 unset: model integrity is not pinned");
    }
    std::fs::create_dir_all(dir).map_err(FerriteError::Io)?;
    let client = hf_hub::HFClientSync::new().map_err(|e| FerriteError::Model(e.to_string()))?;
    let (owner, name) = hf_hub::split_id(&config.model_repo);
    let repo = client.model(owner, name);
    for f in FILES {
        let src = repo
            .download_file()
            .filename(f)
            .maybe_revision(Some(config.model_revision.clone()))
            .send()
            .map_err(|e| FerriteError::Model(e.to_string()))?;
        let dst = dir.join(f);
        std::fs::copy(&src, &dst).map_err(FerriteError::Io)?;
        if f == "model.safetensors"
            && let Some(expected) = &config.model_sha256
            && !sha256_hex(&dst)?.eq_ignore_ascii_case(expected)
        {
            return Err(FerriteError::Model(
                "downloaded model.safetensors does not match FERRITE_MODEL_SHA256".into(),
            ));
        }
    }
    write_manifest(config)
}

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    batch_size: usize,
}

impl Embedder {
    pub fn load(config: &FerriteConfig) -> Result<Self, FerriteError> {
        download_model(config)?;
        let dir = &config.model_dir;
        let config_text =
            std::fs::read_to_string(dir.join("config.json")).map_err(FerriteError::Io)?;
        let bert_config: BertConfig = serde_json::from_str(&config_text)
            .map_err(|e| FerriteError::Model(format!("bad config.json: {e}")))?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DTYPE, &device)
                .map_err(|e| FerriteError::Model(e.to_string()))?
        };
        let model =
            BertModel::load(vb, &bert_config).map_err(|e| FerriteError::Model(e.to_string()))?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| FerriteError::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MODEL_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| FerriteError::Tokenizer(e.to_string()))?;
        Ok(Self {
            model,
            tokenizer,
            device,
            batch_size: config.batch_size.max(1),
        })
    }

    pub fn dim(&self) -> usize {
        384
    }

    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, FerriteError> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let encodings = self
                .tokenizer
                .encode_batch(chunk.to_vec(), true)
                .map_err(|e| FerriteError::Tokenizer(e.to_string()))?;
            let batch = encodings.len();
            let ids: Vec<u32> = encodings
                .iter()
                .flat_map(|e| e.get_ids().to_vec())
                .collect();
            let masks: Vec<u32> = encodings
                .iter()
                .flat_map(|e| e.get_attention_mask().to_vec())
                .collect();
            let seq = ids.len() / batch;
            let input_ids = Tensor::new(ids.as_slice(), &self.device)?.reshape((batch, seq))?;
            let attention_mask =
                Tensor::new(masks.as_slice(), &self.device)?.reshape((batch, seq))?;
            let token_type_ids = input_ids.zeros_like()?;
            let embeddings = self
                .model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))
                .map_err(|e| FerriteError::Model(e.to_string()))?;
            let pooled = mask_mean_pool(&embeddings, &attention_mask)?;
            let normalized = l2_normalize(&pooled)?;
            let rows: Vec<Vec<f32>> = normalized.to_vec2()?;
            for row in rows {
                out.push(row);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FerriteConfig;

    fn cfg() -> FerriteConfig {
        FerriteConfig {
            model_dir: std::path::PathBuf::from("data/models"),
            ..Default::default()
        }
    }

    #[test]
    fn embed_single_sentence_is_384dim_unit_norm() {
        let config = cfg();
        let embedder = Embedder::load(&config).unwrap();
        assert_eq!(embedder.dim(), 384);
        let out = embedder
            .embed(&["This is a test sentence.".to_string()])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 384);
        let norm: f32 = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm was {norm}");
        assert!(out[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn embed_batch_matches_singles() {
        let embedder = Embedder::load(&cfg()).unwrap();
        let texts = vec![
            "The cat sits outside.".to_string(),
            "I love pasta.".to_string(),
            "Do you like pizza?".to_string(),
        ];
        let batched = embedder.embed(&texts).unwrap();
        for (i, single) in texts.iter().enumerate() {
            let one = embedder.embed(std::slice::from_ref(single)).unwrap()[0].clone();
            let dot: f32 = batched[i].iter().zip(&one).map(|(a, b)| a * b).sum();
            assert!(dot > 0.999, "batch[{i}] vs single dot={dot}");
        }
    }

    #[test]
    fn embed_truncates_long_text() {
        let embedder = Embedder::load(&cfg()).unwrap();
        let long = "ferrite ".repeat(10_000);
        let out = embedder.embed(&[long]).unwrap();
        assert_eq!(out[0].len(), 384);
        assert!(out[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn embed_empty_string_yields_finite_384dim() {
        let embedder = Embedder::load(&cfg()).unwrap();
        let out = embedder.embed(&[String::new()]).unwrap();
        assert_eq!(out[0].len(), 384);
        assert!(out[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn embed_unicode_and_emoji() {
        let embedder = Embedder::load(&cfg()).unwrap();
        let texts = vec![
            "Привет, мир! Как дела?".to_string(),
            "🇺🇸 🇩🇪 🇫🇷 hello, café, ñoño".to_string(),
        ];
        let out = embedder.embed(&texts).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn embed_zero_texts_is_empty() {
        let embedder = Embedder::load(&cfg()).unwrap();
        let out = embedder.embed(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn manifest_roundtrip_verifies_and_detects_tampering() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        for f in FILES {
            std::fs::write(dir.join(f), f.as_bytes()).unwrap();
        }
        let config = FerriteConfig {
            model_dir: dir.to_path_buf(),
            model_repo: "r/o".into(),
            model_revision: "rev".into(),
            ..Default::default()
        };
        write_manifest(&config).unwrap();
        assert!(manifest_path(dir).is_file());
        verify_manifest(dir).unwrap();

        std::fs::write(dir.join("model.safetensors"), "tampered").unwrap();
        let err = verify_manifest(dir).unwrap_err();
        assert!(err.to_string().contains("integrity check failed"), "{err}");
    }

    #[test]
    fn pinned_sha256_rejects_mismatch_without_network() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        for f in FILES {
            std::fs::write(dir.join(f), f.as_bytes()).unwrap();
        }
        let config = FerriteConfig {
            model_dir: dir.to_path_buf(),
            model_sha256: Some("deadbeef".into()),
            ..Default::default()
        };
        // files are present, so the offline verify path runs: pinned hash differs
        let err = download_model(&config).unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");

        // matching pin also fails verification when no manifest exists for a
        // fresh dir whose pinned value is correct but files were "tampered"?
        // no — matching pin plus no manifest stamps it; use wrong hash above.
        let config = FerriteConfig {
            model_dir: dir.to_path_buf(),
            model_sha256: {
                let bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
                Some(format!("{:x}", Sha256::digest(&bytes)))
            },
            ..Default::default()
        };
        assert!(download_model(&config).is_ok());
    }
}
