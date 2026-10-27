//! HTTP-сервер `lighttrans-server`: batch-транскрибация `POST /v1/transcribe`.
//!
//! Контракты перенесены из gigaam-v3-runtime (docs/API.md §HTTP batch,
//! docs/OPERATIONS.md §Service limits / §Admission): тело = сырые байты
//! аудиофайла, строгий query, лимиты через окружение (кривое значение env
//! отказывает, а не подменяется дефолтом), graceful shutdown по SIGTERM/
//! SIGINT (Draining → новые запросы 503, активные дожидаются).
//!
//! Тёплый стейт: ONNX-сессии создаются один раз при старте (пул воркеров на
//! модель), на запрос сессии не создаются. Тексты ошибок HTTP — на английском
//! (контракт источника), логи — на русском (соглашения LightTrans).

pub mod admission;
pub mod config;
pub mod health;
pub mod http;
pub mod protocol;
pub mod response;
pub mod ws;

use crate::models::ModelPaths;
use crate::pipeline::WorkerPool;
use axum::extract::State;
use axum::response::IntoResponse;
use std::sync::Arc;

/// Пулы воркеров по моделям: сессии живут между запросами.
pub struct ModelPools {
    pub ctc: Arc<WorkerPool>,
    pub rnnt: Option<Arc<WorkerPool>>,
}

impl ModelPools {
    pub fn has_rnnt(&self) -> bool {
        self.rnnt.is_some()
    }

    pub fn worker_count(&self) -> usize {
        let rnnt = self.rnnt.as_ref().map(|p| p.len()).unwrap_or(0);
        self.ctc.len() + rnnt
    }
}

/// Общее состояние сервера (клон-дешёвое: всё внутри Arc/атомиков).
#[derive(Clone)]
pub struct AppState {
    inner: Arc<ServerInner>,
}

pub struct ServerInner {
    pub pools: ModelPools,
    pub admission: admission::AdmissionState,
    pub limits: config::ServerLimits,
    pub paths: ModelPaths,
    pub provider: &'static str,
    pub metrics: Arc<crate::metrics::Metrics>,
}

impl AppState {
    pub fn new(inner: ServerInner) -> Self {
        Self { inner: Arc::new(inner) }
    }

    pub fn inner(&self) -> &ServerInner {
        &self.inner
    }
}

/// Маршруты: batch-endpoint (граница тела — слой роутера) + пробы + метрики.
pub fn router(state: AppState) -> axum::Router {
    let body_limit = state.inner().limits.body_limit_bytes;
    axum::Router::new()
        .route("/v1/transcribe", axum::routing::post(http::transcribe))
        .with_state(state.clone())
        .route("/v1/stream", axum::routing::get(ws::stream).with_state(state.clone()))
        .route("/health", axum::routing::get(health::health).with_state(state.clone()))
        .route("/livez", axum::routing::get(health::livez))
        .route("/readyz", axum::routing::get(health::readyz).with_state(state.clone()))
        .route("/metrics", axum::routing::get(metrics_endpoint).with_state(state))
        .layer(axum::extract::DefaultBodyLimit::max(body_limit))
}

/// GET /metrics: текстовый рендер Prometheus.
async fn metrics_endpoint(State(state): State<AppState>) -> axum::response::Response {
    use axum::http::header;
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        state.inner().metrics.render(),
    )
        .into_response()
}
