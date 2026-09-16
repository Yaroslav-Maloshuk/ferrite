use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::config::FerriteConfig;
use crate::error::FerriteError;
use crate::pooling::{l2_normalize, mask_mean_pool};

pub const MODEL_MAX_LENGTH: usize = 256;
const FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

pub fn download_model(config: &FerriteConfig) -> Result<(), FerriteError> {
    let dir = &config.model_dir;
    if FILES.iter().all(|f| dir.join(f).is_file()) {
        return Ok(());
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
    }
    Ok(())
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
}
