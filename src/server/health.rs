//! Liveness/readiness/capability-пробы (контракт источника: docs/API.md
//! §Health, liveness, readiness and metrics; crates/service/src/health.rs).
//! Готовность различает причину: draining имеет приоритет над остановленным
//! воркером.

use super::response::{json_err, json_ok};
use super::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;

pub(crate) async fn health(State(state): State<AppState>) -> Response {
    let inner = state.inner();
    json_ok(
        serde_json::json!({
            "status": "ok",
            "models": inner.pools.loaded_models(),
            "provider": inner.provider,
        })
        .to_string(),
    )
}

pub(crate) async fn livez() -> Response {
    json_ok(serde_json::json!({"status": "alive"}).to_string())
}

pub(crate) async fn readyz(State(state): State<AppState>) -> Response {
    let inner = state.inner();
    if inner.admission.is_draining() {
        // draining имеет приоритет: процесс уже останавливается по своей причине
        return json_err(StatusCode::SERVICE_UNAVAILABLE, "draining");
    }
    let stopped = [Some(&inner.pools.ctc), inner.pools.rnnt.as_ref(), inner.pools.whisper.as_ref(), inner.pools.parakeet.as_ref()]
        .into_iter()
        .flatten()
        .any(|p| p.is_stopped());
    if stopped {
        return json_err(StatusCode::SERVICE_UNAVAILABLE, "worker stopped");
    }
    json_ok(serde_json::json!({"status": "ready"}).to_string())
}
