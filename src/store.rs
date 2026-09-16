use std::sync::Arc;

use arrow_array::{FixedSizeListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::index::{Index, vector::IvfPqIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;

use crate::config::{FerriteConfig, IndexMode};
use crate::error::FerriteError;

pub const VECTOR_DIM: usize = 384;

#[derive(Debug, Clone)]
pub struct StoredItem {
    pub id: String,
    pub text: String,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub distance: f32,
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                VECTOR_DIM as i32,
            ),
            false,
        ),
    ]))
}

pub struct VectorStore {
    table: Table,
}

impl VectorStore {
    pub async fn open(config: &FerriteConfig) -> Result<Self, FerriteError> {
        let db = connect(&config.lance_uri)
            .execute()
            .await
            .map_err(|e| FerriteError::Database(e.to_string()))?;
        let table = match db.open_table("questions").execute().await {
            Ok(t) => t,
            Err(_) => {
                // create_table takes RecordBatch data (not a bare schema);
                // seed it with a zero-row batch carrying the right schema.
                use arrow_array::{ArrayRef, new_empty_array};
                let cols: Vec<ArrayRef> = schema()
                    .fields()
                    .iter()
                    .map(|f| new_empty_array(f.data_type()))
                    .collect();
                let empty = RecordBatch::try_new(schema(), cols)
                    .map_err(|e| FerriteError::Database(e.to_string()))?;
                db.create_table("questions", empty)
                    .execute()
                    .await
                    .map_err(|e| FerriteError::Database(e.to_string()))?
            }
        };
        Ok(Self { table })
    }

    pub async fn add(
        &self,
        items: &[StoredItem],
        embeddings: &[Vec<f32>],
    ) -> Result<usize, FerriteError> {
        if items.len() != embeddings.len() {
            return Err(FerriteError::Other(
                "items and embeddings length mismatch".into(),
            ));
        }
        let ids: StringArray = items.iter().map(|i| Some(i.id.as_str())).collect();
        let texts: StringArray = items.iter().map(|i| Some(i.text.as_str())).collect();
        let vectors =
            FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                embeddings
                    .iter()
                    .map(|v| Some(v.iter().map(|x| Some(*x)).collect::<Vec<_>>())),
                VECTOR_DIM as i32,
            );
        let batch = RecordBatch::try_new(
            schema(),
            vec![Arc::new(ids), Arc::new(texts), Arc::new(vectors)],
        )
        .map_err(|e| FerriteError::Database(e.to_string()))?;
        self.table
            .add(batch)
            .execute()
            .await
            .map_err(|e| FerriteError::Database(e.to_string()))?;
        Ok(items.len())
    }

    pub async fn search(
        &self,
        query_vec: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>, FerriteError> {
        let k = top_k.clamp(1, 1000);
        let stream = self
            .table
            .query()
            .limit(k)
            .nearest_to(query_vec.to_vec())
            .map_err(|e| FerriteError::Database(e.to_string()))?
            .execute()
            .await
            .map_err(|e| FerriteError::Database(e.to_string()))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| FerriteError::Database(e.to_string()))?;
        let mut hits = Vec::new();
        for batch in batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let texts = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let dists = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());
            let (Some(ids), Some(texts), Some(dists)) = (ids, texts, dists) else {
                return Err(FerriteError::Database("missing columns in results".into()));
            };
            for i in 0..batch.num_rows() {
                let distance = dists.value(i);
                hits.push(SearchHit {
                    id: ids.value(i).to_string(),
                    text: texts.value(i).to_string(),
                    score: 1.0 - distance,
                    distance,
                });
            }
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits.truncate(k);
        Ok(hits)
    }

    pub async fn count(&self) -> Result<u64, FerriteError> {
        // lancedb 0.38 `count_rows(filter)` returns `Result<usize>` directly.
        self.table
            .count_rows(None)
            .await
            .map(|n| n as u64)
            .map_err(|e| FerriteError::Database(e.to_string()))
    }

    pub async fn create_index(&self, mode: IndexMode) -> Result<(), FerriteError> {
        match mode {
            IndexMode::Flat => Ok(()),
            IndexMode::IvfPq => {
                let builder = IvfPqIndexBuilder::default().num_partitions(16);
                self.table
                    .create_index(&["vector"], Index::IvfPq(builder))
                    .execute()
                    .await
                    .map_err(|e| FerriteError::Database(e.to_string()))?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FerriteConfig;

    fn cfg(tmp: &std::path::Path) -> FerriteConfig {
        FerriteConfig {
            lance_uri: tmp.join("lance").display().to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn add_and_search_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&cfg(tmp.path())).await.unwrap();
        let items: Vec<StoredItem> = vec![
            StoredItem {
                id: "a".into(),
                text: "the cat sits outside".into(),
                metadata: None,
            },
            StoredItem {
                id: "b".into(),
                text: "a dog runs in the park".into(),
                metadata: None,
            },
            StoredItem {
                id: "c".into(),
                text: "pasta is delicious".into(),
                metadata: None,
            },
        ];
        // embeddings: pretend dim-384 vectors; component 0 encodes identity
        let embs: Vec<Vec<f32>> = vec![vec50(0.9f32, 384), vec50(0.5f32, 384), vec50(-0.9f32, 384)];
        let n = store.add(&items, &embs).await.unwrap();
        assert_eq!(n, 3);
        assert_eq!(store.count().await.unwrap(), 3);
        let hits = store.search(&vec50(0.88f32, 384), 3).await.unwrap();
        assert_eq!(hits[0].id, "a");
        assert_eq!(hits[0].text, "the cat sits outside");
        assert!(hits[0].distance < hits[1].distance);
    }

    // helper: all-`x` vector except index 0 dominated by bias
    fn vec50(x: f32, dim: usize) -> Vec<f32> {
        let mut v = vec![x * 2.0; dim];
        v[0] = x * 10.0;
        v
    }

    #[tokio::test]
    async fn search_empty_index_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&cfg(tmp.path())).await.unwrap();
        let hits = store.search(&vec![0.0; 384], 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn duplicate_texts_keep_distinct_rows_and_rank_stable() {
        let tmp = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&cfg(tmp.path())).await.unwrap();
        let item = StoredItem {
            id: "dup".into(),
            text: "same text twice".into(),
            metadata: None,
        };
        let emb = vec![0.25; 384];
        store
            .add(std::slice::from_ref(&item), std::slice::from_ref(&emb))
            .await
            .unwrap();
        store
            .add(std::slice::from_ref(&item), std::slice::from_ref(&emb))
            .await
            .unwrap();
        assert_eq!(store.count().await.unwrap(), 2);
        let hits = store.search(&vec![0.25; 384], 5).await.unwrap();
        assert_eq!(hits.len(), 2, "both duplicates returned");
        assert!(hits.iter().all(|h| h.text == "same text twice"));
    }

    #[tokio::test]
    async fn top_k_larger_than_rows_clamps_to_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let store = VectorStore::open(&cfg(tmp.path())).await.unwrap();
        store
            .add(
                &[StoredItem {
                    id: "x".into(),
                    text: "only one".into(),
                    metadata: None,
                }],
                &[vec![0.5; 384]],
            )
            .await
            .unwrap();
        let hits = store.search(&vec![0.5; 384], 1000).await.unwrap();
        assert_eq!(hits.len(), 1);
    }
}
