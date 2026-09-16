use std::sync::Arc;

use ferrite::config::FerriteConfig;
use ferrite::pipeline::{Ferrite, IngestItem};

fn cfg(tmp: &std::path::Path) -> FerriteConfig {
    FerriteConfig {
        lance_uri: tmp.join("lance").display().to_string(),
        ..Default::default()
    }
}

#[tokio::test]
async fn end_to_end_embed_ingest_search_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let ferrite = Arc::new(Ferrite::init(&cfg(tmp.path())).await.unwrap());

    let texts = [
        "the cat sits outside".to_string(),
        "a man is playing guitar".to_string(),
        "pasta with tomato sauce".to_string(),
    ];
    let items: Vec<IngestItem> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| IngestItem {
            id: format!("id{i}"),
            text: t.clone(),
            metadata: None,
        })
        .collect();
    let n = ferrite.ingest(&items).await.unwrap();
    assert_eq!(n, 3);

    let stats = ferrite.stats().await.unwrap();
    assert_eq!(stats.rows, 3);
    assert_eq!(stats.dim, 384);

    // probe with a query close to "the cat sits outside"
    let hits = ferrite.search("cat on a couch", 3).await.unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].text, "the cat sits outside");
    assert!(hits[0].score >= 0.0);

    // search_by_vector path
    let v = ferrite.embed(&["guitar player".to_string()]).await.unwrap();
    let by_vec = ferrite.search_by_vector(&v[0], 3).await.unwrap();
    assert_eq!(by_vec[0].text, "a man is playing guitar");
}

#[tokio::test]
async fn empty_texts_embed_is_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let ferrite = Ferrite::init(&cfg(tmp.path())).await.unwrap();
    assert!(ferrite.embed(&[]).await.unwrap().is_empty());
}
