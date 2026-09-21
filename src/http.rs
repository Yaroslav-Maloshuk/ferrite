use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::IndexMode;
use crate::pipeline::Ferrite;
use crate::store::VECTOR_DIM;

/// Build the service router. When `api_key` is `Some`, every route *except*
/// `/v1/health` requires either `Authorization: Bearer <key>` or
/// `X-API-Key: <key>`; each request is also written to the audit log.
pub fn router(state: Arc<Ferrite>, api_key: Option<String>) -> Router {
    Router::new()
        .route("/v1/embed", post(v1_embed))
        .route("/v1/ingest", post(v1_ingest))
        .route("/v1/search", post(v1_search))
        .route("/v1/health", get(v1_health))
        .route("/v1/stats", get(v1_stats))
        .layer(from_fn_with_state(Arc::new(api_key), require_api_key))
        .with_state(state)
}

async fn require_api_key(
    State(expected): State<Arc<Option<String>>>,
    req: Request,
    next: Next,
) -> Response {
    let start = std::time::Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let remote_ip = req
        .extensions()
        .get::<axum::extract::connect_info::ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0.ip().to_string())
        .unwrap_or_else(|| "-".into());

    let authed = match expected.as_ref() {
        // No key configured -> open service.
        None => true,
        // Health is intentionally public (healthchecks, LB probes).
        Some(_) if uri.path() == "/v1/health" => true,
        Some(key) => extract_api_key(req.headers())
            .map(|candidate| ct_eq(candidate, key))
            .unwrap_or(false),
    };

    let (status, res) = if authed {
        let res = next.run(req).await;
        (res.status().as_u16(), Some(res))
    } else {
        (StatusCode::UNAUTHORIZED.as_u16(), None)
    };
    tracing::info!(
        method = %method,
        uri = %uri,
        status,
        latency_ms = start.elapsed().as_secs_f64() * 1000.0,
        remote_ip = %remote_ip,
        authed,
        "audit",
    );
    match res {
        Some(res) => res,
        None => (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "unauthorized" })),
        )
            .into_response(),
    }
}

fn extract_api_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(v) = headers.get(AUTHORIZATION)
        && let Some(rest) = v.to_str().ok().and_then(|s| s.strip_prefix("Bearer "))
    {
        return Some(rest.trim());
    }
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok().map(str::trim))
}

/// Constant-time string comparison (length-checked) for API-key checks.
fn ct_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
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
        router(ferrite, None)
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
        let app = router(ferrite, None);
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

    async fn req(app: &axum::Router, bearer: Option<&str>) -> StatusCode {
        let mut b = Request::builder().method("GET").uri("/v1/stats");
        if let Some(k) = bearer {
            b = b.header("authorization", format!("Bearer {k}"));
        }
        let req = b.body(Body::from("")).unwrap();
        app.clone().oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn api_key_required_when_configured() {
        let tmp = tempfile::tempdir().unwrap();
        let config = FerriteConfig {
            lance_uri: tmp.path().join("lance").display().to_string(),
            ..Default::default()
        };
        let ferrite = Arc::new(Ferrite::init(&config).await.unwrap());
        let app = router(ferrite, Some("s3cret".into()));

        // /v1/health stays public for healthchecks.
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/health")
                    .body(Body::from(""))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // Without a key / with a wrong key -> 401.
        assert_eq!(req(&app, None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(req(&app, Some("nope")).await, StatusCode::UNAUTHORIZED);

        // Correct bearer and X-API-Key headers -> 200.
        assert_eq!(req(&app, Some("s3cret")).await, StatusCode::OK);
        let b = Request::builder()
            .method("GET")
            .uri("/v1/stats")
            .header("x-api-key", "s3cret")
            .body(Body::from(""))
            .unwrap();
        let res = app.clone().oneshot(b).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
}
