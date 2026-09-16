use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::IndexMode;
use crate::pipeline::Ferrite;
use crate::store::VECTOR_DIM;

pub fn router(state: Arc<Ferrite>) -> Router {
    Router::new()
        .route("/v1/embed", post(v1_embed))
        .route("/v1/ingest", post(v1_ingest))
        .route("/v1/search", post(v1_search))
        .route("/v1/health", get(v1_health))
        .route("/v1/stats", get(v1_stats))
        .with_state(state)
}

#[derive(Deserialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct IngestRequest {
    items: Vec<crate::pipeline::IngestItem>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    top_k: Option<usize>,
}

fn err(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({ "error": msg.into() })))
}

fn internal_error(e: impl std::fmt::Display) -> (StatusCode, Json<Value>) {
    err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

async fn v1_embed(
    State(ferrite): State<Arc<Ferrite>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if req.texts.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "texts must be non-empty"));
    }
    if req.texts.len() > 256 {
        return Err(err(StatusCode::BAD_REQUEST, "texts batch exceeds 256"));
    }
    let start = std::time::Instant::now();
    let embeddings = ferrite.embed(&req.texts).await.map_err(internal_error)?;
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(json!({
        "embeddings": embeddings,
        "model": ferrite.config_model(),
        "dim": VECTOR_DIM,
        "latency_ms": latency_ms,
    })))
}

async fn v1_ingest(
    State(ferrite): State<Arc<Ferrite>>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if req.items.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "items must be non-empty"));
    }
    let n = ferrite.ingest(&req.items).await.map_err(internal_error)?;
    Ok(Json(json!({ "ingested": n })))
}

async fn v1_search(
    State(ferrite): State<Arc<Ferrite>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if req.query.trim().is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "query must be non-empty"));
    }
    let top_k = req.top_k.unwrap_or(10).clamp(1, 1000);
    let hits = ferrite
        .search(&req.query, top_k)
        .await
        .map_err(internal_error)?;
    let results: Vec<Value> = hits
        .iter()
        .map(|h| json!({ "id": h.id, "text": h.text, "score": h.score, "distance": h.distance }))
        .collect();
    Ok(Json(json!({ "results": results })))
}

async fn v1_health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn v1_stats(
    State(ferrite): State<Arc<Ferrite>>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let stats = ferrite.stats().await.map_err(internal_error)?;
    Ok(Json(json!({
        "rows": stats.rows,
        "index": match stats.index {
            IndexMode::Flat => "flat",
            IndexMode::IvfPq => "ivf_pq",
        },
        "model": stats.model,
        "dim": stats.dim,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FerriteConfig;
    use crate::pipeline::Ferrite;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use tower::ServiceExt;

    async fn app(tmp: &std::path::Path) -> axum::Router {
        let config = FerriteConfig {
            lance_uri: tmp.join("lance").display().to_string(),
            ..Default::default()
        };
        let ferrite = Arc::new(Ferrite::init(&config).await.unwrap());
        router(ferrite)
    }

    #[tokio::test]
    async fn embed_route_returns_embeddings() {
        let tmp = tempfile::tempdir().unwrap();
        let app = app(tmp.path()).await;
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embed")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"texts": ["hello world"]}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&body).unwrap();
        let embs = v["embeddings"].as_array().unwrap();
        assert_eq!(embs.len(), 1);
        assert_eq!(embs[0].as_array().unwrap().len(), 384);
        assert_eq!(v["dim"], 384);
    }

    #[tokio::test]
    async fn empty_texts_returns_400() {
        let tmp = tempfile::tempdir().unwrap();
        let app = app(tmp.path()).await;
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embed")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"texts": []}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn search_route_returns_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let config = FerriteConfig {
            lance_uri: tmp.path().join("lance").display().to_string(),
            ..Default::default()
        };
        let ferrite = Arc::new(Ferrite::init(&config).await.unwrap());
        let app = router(ferrite);
        // ingest first
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/ingest")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"items": [{"id":"1","text":"the cat sits outside"}]}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        // then search
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"query":"cat outside", "top_k": 5}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&body).unwrap();
        let hits = v["results"].as_array().unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0]["id"], "1");
    }
}
