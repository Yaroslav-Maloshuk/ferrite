use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::{FerriteConfig, IndexMode};
use crate::embedding::Embedder;
use crate::error::FerriteError;
use crate::store::{SearchHit, StoredItem, VectorStore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestItem {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub rows: u64,
    pub index: IndexMode,
    pub model: String,
    pub dim: usize,
}

pub struct Ferrite {
    embedder: Arc<Embedder>,
    store: Arc<VectorStore>,
    config: FerriteConfig,
}

impl Ferrite {
    pub async fn init(config: &FerriteConfig) -> Result<Self, FerriteError> {
        let config = config.clone();
        let embedder = tokio::task::spawn_blocking({
            let c = config.clone();
            move || Embedder::load(&c)
        })
        .await
        .map_err(|e| FerriteError::Other(format!("embedder task: {e}")))??;
        let store = VectorStore::open(&config).await?;
        store
            .create_index(config.index, config.ivf_partitions)
            .await?;
        Ok(Self {
            embedder: Arc::new(embedder),
            store: Arc::new(store),
            config,
        })
    }

    pub fn config_model(&self) -> String {
        format!("{}@{}", self.config.model_repo, self.config.model_revision)
    }

    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, FerriteError> {
        if texts.len() > self.config.max_text_batch {
            return Err(FerriteError::Other(format!(
                "batch of {} exceeds max {}",
                texts.len(),
                self.config.max_text_batch
            )));
        }
        let embedder = self.embedder.clone();
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || embedder.embed(&texts))
            .await
            .map_err(|e| FerriteError::Other(format!("embed task: {e}")))?
    }

    pub async fn ingest(&self, items: &[IngestItem]) -> Result<usize, FerriteError> {
        let texts: Vec<String> = items.iter().map(|i| i.text.clone()).collect();
        let embeddings = self.embed(&texts).await?;
        let stored: Vec<StoredItem> = items
            .iter()
            .map(|i| StoredItem {
                id: i.id.clone(),
                text: i.text.clone(),
                metadata: i.metadata.clone(),
            })
            .collect();
        self.store.add(&stored, &embeddings).await
    }

    pub async fn search_by_vector(
        &self,
        query_vec: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>, FerriteError> {
        self.store.search(query_vec, top_k).await
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>, FerriteError> {
        let embeddings = self.embed(&[query.to_string()]).await?;
        self.store.search(&embeddings[0], top_k).await
    }

    pub async fn stats(&self) -> Result<Stats, FerriteError> {
        let rows = self.store.count().await?;
        Ok(Stats {
            rows,
            index: self.config.index,
            model: self.config_model(),
            dim: self.embedder.dim(),
        })
    }
}
