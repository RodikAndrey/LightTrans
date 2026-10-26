//! Построение JSON-ответов `POST /v1/transcribe` по контракту источника
//! (docs/API.md §Successful response, crates/service/src/response.rs):
//! ровно одна текстовая проекция (text | channels_text), опциональные
//! words/segments/turns, округление duration_sec до 2 и rtf до 4 знаков.

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

pub const JSON_CONTENT_TYPE: &str = "application/json; charset=utf-8";

pub fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

pub fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

pub fn json_ok(body: String) -> Response {
    (
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        body,
    )
        .into_response()
}

pub fn json_err(code: StatusCode, message: impl Into<String>) -> Response {
    let body = serde_json::json!({ "error": message.into() }).to_string();
    (code, [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)], body).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_the_contract_projections() {
        assert_eq!(round2(0.12345), 0.12);
        assert_eq!(round2(59.234_999), 59.23);
        assert_eq!(round4(0.012_345_6), 0.0123);
        assert_eq!(round2(0.0), 0.0);
    }
}
