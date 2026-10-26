//! HTTP-адаптация batch-транскрибации (контракт: docs/API.md §HTTP batch).
//! Тело = сырые байты аудио; параметры — строгий query; каждый полезный
//! запрос получает `x-request-id`. 400 — вход/домен, 413 — граница тела
//! (слой axum), 503 — перегрузка/draining/очередь/ошибка обработки,
//! 504 — дедлайн обработки.

use super::admission::AdmissionRefusal;
use super::protocol::{QueryParameters, query_bool, query_positive_f32};
use super::response::{json_err, json_ok, round2, round4};
use super::AppState;
use crate::metrics::Metrics;
use axum::body::Bytes;
use axum::extract::{RawQuery, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use crate::audio;
use crate::channels::{self, ChannelWords};
use crate::pipeline::{
    CancelToken, ChannelMode, ChunkOutcome, MultiStream, QueueItem, SPLIT_BLOCK_FRAMES,
};
use crate::transcript::{Segment, Word};
use crate::worker::ModelKind;
use anyhow::anyhow;
use std::collections::BTreeMap;
use std::sync::mpsc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Дедлайн ожидания первого результата после постановки первого чанка в
/// очередь воркеров; превышение — 503 (контракт источника: queue-wait 60 c).
const QUEUE_WAIT: Duration = Duration::from_secs(60);

const TRANSCRIBE_QUERY_KEYS: &[&str] =
    &["model", "ext", "words", "segments", "turns", "turn_gap", "channels"];

const VALID_EXTENSIONS: &[&str] = &[
    "wav", "flac", "ogg", "opus", "mp3", "aac", "m4a", "mp4", "mkv", "vorbis", "pcm",
];

/// Идентификатор запроса: валидация заголовка + ответный заголовок.
struct RequestId {
    text: String,
    header: HeaderValue,
}

static REQ_SEQ: AtomicU64 = AtomicU64::new(0);

impl RequestId {
    fn from_headers(headers: &HeaderMap) -> Result<Self, String> {
        let Some(header) = headers.get("x-request-id") else {
            let text = new_req_id();
            let header = HeaderValue::from_str(&text)
                .expect("сгенерированный id содержит только видимый ASCII");
            return Ok(Self { text, header });
        };
        let text = header
            .to_str()
            .map_err(|_| "x-request-id must contain visible ASCII text")?
            .to_owned();
        if text.is_empty() {
            return Err("x-request-id must not be empty".into());
        }
        Ok(Self { text, header: header.clone() })
    }
}

fn new_req_id() -> String {
    let seq = REQ_SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("req-{nanos:x}-{seq:x}")
}

fn with_request_id(mut response: Response, request_id: &RequestId) -> Response {
    response
        .headers_mut()
        .insert("x-request-id", request_id.header.clone());
    response
}

/// Разобранный запрос; строгий домен значений.
#[derive(Debug, Clone)]
pub(crate) struct TranscribeQuery {
    pub model: ModelKind,
    pub ext: Option<String>,
    pub words: bool,
    pub segments: bool,
    pub turns: bool,
    pub turn_gap: f64,
    pub split: bool,
}

impl TranscribeQuery {
    fn parse(raw: Option<&str>) -> Result<Self, String> {
        let parameters = QueryParameters::parse(raw)?;
        parameters.reject_unknown(TRANSCRIBE_QUERY_KEYS)?;
        let model = match parameters.get("model") {
            None | Some("gigaam-v3") => ModelKind::GigaamV3,
            Some("gigaam-v3-rnnt") => ModelKind::GigaamV3Rnnt,
            Some(value) => {
                return Err(format!(
                    "model must be gigaam-v3 or gigaam-v3-rnnt, got {value:?}"
                ))
            }
        };
        let ext = match parameters.get("ext") {
            None => None,
            Some(value) if VALID_EXTENSIONS.contains(&value) => Some(value.to_owned()),
            Some(value) => {
                return Err(format!(
                    "ext must be one of {}, got {value:?}",
                    VALID_EXTENSIONS.join("|")
                ))
            }
        };
        let split = match parameters.get("channels") {
            None => false,
            Some("split") => true,
            Some(value) => return Err(format!("channels must be split, got {value:?}")),
        };
        Ok(Self {
            model,
            ext,
            words: query_bool(&parameters, "words", false)?,
            segments: query_bool(&parameters, "segments", false)?,
            turns: query_bool(&parameters, "turns", false)?,
            turn_gap: query_positive_f32(&parameters, "turn_gap", 1.0)? as f64,
            split,
        })
    }
}

#[derive(Debug)]
enum WorkError {
    /// 400: вход/домен запроса.
    BadRequest(String),
    /// 503: перегрузка очереди воркеров или ошибка обработки.
    Unavailable(String),
}

impl WorkError {
    fn status(&self) -> StatusCode {
        match self {
            WorkError::BadRequest(_) => StatusCode::BAD_REQUEST,
            WorkError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    fn message(&self) -> String {
        match self {
            WorkError::BadRequest(m) | WorkError::Unavailable(m) => m.clone(),
        }
    }
}

/// Итог блокирующей обработки запроса: собранный JSON-ответ.
type WorkResult = Result<serde_json::Value, WorkError>;

/// Лимит частоты дискретизации: приведённое к несократимому отношение
/// вход/модель должно оставаться ≤ 1000 с обеих сторон (контракт источника).
fn rate_compatible(rate: u32, model_rate: u32) -> bool {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let g = gcd(rate, model_rate).max(1);
    rate / g <= 1000 && model_rate / g <= 1000
}

/// Пермит admission + учёт активных запросов: метрика снимается ровно при
/// терминации работы (Drop), даже после ответа 504.
struct PermitWithMetrics {
    _permit: tokio::sync::OwnedSemaphorePermit,
    metrics: Arc<Metrics>,
}

impl Drop for PermitWithMetrics {
    fn drop(&mut self) {
        self.metrics.transcribe_finished();
    }
}

pub(crate) async fn transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(query_text): RawQuery,
    body: Bytes,
) -> Response {
    let started = Instant::now();
    let metrics = Arc::clone(&state.inner().metrics);
    metrics.transcribe_started();
    let request_id = match RequestId::from_headers(&headers) {
        Ok(id) => id,
        Err(error) => {
            metrics.transcribe_error();
            return json_err(StatusCode::BAD_REQUEST, error);
        }
    };
    if body.is_empty() {
        metrics.transcribe_error();
        return with_request_id(
            json_err(StatusCode::BAD_REQUEST, "empty body: audio expected"),
            &request_id,
        );
    }
    let query = match TranscribeQuery::parse(query_text.as_deref()) {
        Ok(query) => query,
        Err(error) => {
            metrics.transcribe_error();
            return with_request_id(
                json_err(StatusCode::BAD_REQUEST, error),
                &request_id,
            )
        }
    };
    // admission: перегрузка/draining → 503 до начала работы
    let permit = match state.inner().admission.admit_http() {
        Ok(permit) => permit,
        Err(AdmissionRefusal::Overloaded) => {
            metrics.overload();
            return with_request_id(
                json_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service overloaded, retry later",
                ),
                &request_id,
            )
        }
        Err(AdmissionRefusal::Draining) => {
            metrics.transcribe_error();
            return with_request_id(json_err(StatusCode::SERVICE_UNAVAILABLE, "draining"), &request_id)
        }
    };

    // блокирующая работа: декод → чанки → пул воркеров → слияние.
    // Пермит уезжает внутрь: ёмкость удерживается до терминации работы даже
    // после ответа 504 (кооперативная отмена, нативный вызов не прерывается).
    let server = state.clone();
    let cancel = CancelToken::new();
    let work_cancel = cancel.clone();
    let work_query = query.clone();
    let work_body = body.to_vec();
    let permit = PermitWithMetrics { _permit: permit, metrics: Arc::clone(&metrics) };
    let work = tokio::task::spawn_blocking(move || {
        do_transcribe(&server, &work_query, work_body, work_cancel, permit)
    });
    let timeout = state.inner().limits.request_timeout;
    let response = match tokio::time::timeout(timeout, work).await {
        Ok(Ok(Ok(json))) => json_ok(json.to_string()),
        Ok(Ok(Err(error))) => {
            metrics.transcribe_error();
            json_err(error.status(), error.message())
        }
        Ok(Err(_join)) => {
            metrics.transcribe_error();
            json_err(StatusCode::SERVICE_UNAVAILABLE, "internal worker error")
        }
        Err(_elapsed) => {
            metrics.timeout();
            cancel.cancel();
            json_err(StatusCode::GATEWAY_TIMEOUT, "processing timeout")
        }
    };
    metrics.observe_latency(started.elapsed().as_secs_f64());
    tracing::info!(
        request_id = %request_id.text,
        status = response.status().as_u16(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        bytes = body.len(),
        "transcribe завершён"
    );
    with_request_id(response, &request_id)
}

/// Итог продюсера: метрики входа и план каналов.
struct ProducerInfo {
    duration: f64,
    source_rate: u32,
    source_channels: usize,
    plan: Option<channels::ChannelPlan>,
}

/// Блокирующий путь запроса: декод → каналы → чанки → пул → слияние.
/// Пермит живёт до конца функции: ёмкость возвращается только после
/// терминации работы.
fn do_transcribe(
    server: &AppState,
    query: &TranscribeQuery,
    body: Vec<u8>,
    cancel: CancelToken,
    _permit: PermitWithMetrics,
) -> WorkResult {
    let inner = server.inner();
    let pool = match query.model {
        ModelKind::GigaamV3 => Arc::clone(&inner.pools.ctc),
        ModelKind::GigaamV3Rnnt => Arc::clone(inner.pools.rnnt.as_ref().ok_or_else(|| {
            WorkError::BadRequest("gigaam-v3-rnnt model is not loaded".into())
        })?),
        _ => {
            return Err(WorkError::BadRequest(format!(
                "model must be gigaam-v3 or gigaam-v3-rnnt, got {:?}",
                query.model.id()
            )))
        }
    };

    let proc_start = Instant::now();
    let (reply_tx, reply_rx) = mpsc::channel::<ChunkOutcome>();
    // канал-политика источника: channels=split или turns=true → раздельные
    // каналы; иначе mono-путь (микширование)
    let split = query.split || query.turns;
    let producer = {
        let pool = Arc::clone(&pool);
        let reply_tx = reply_tx.clone();
        let cancel = cancel.clone();
        let vad_path = inner.paths.silero_vad.clone();
        let req_label = "http".to_string();
        let query = query.clone();
        let metrics = Arc::clone(&inner.metrics);
        std::thread::spawn(move || -> Result<ProducerInfo, WorkError> {
            let target_channels = if split { 0 } else { 1 };
            let block_frames = if split {
                SPLIT_BLOCK_FRAMES
            } else {
                audio::TARGET_SR as usize
            };
            let mut dec = audio::Decoder::open_bytes(
                body,
                query.ext.as_deref(),
                audio::TARGET_SR,
                target_channels,
                block_frames,
            )
            .map_err(|e| WorkError::BadRequest(format!("invalid audio input: {e:#}")))?;
            let source_rate = dec.source_rate();
            if !(4000..=192_000).contains(&source_rate) {
                return Err(WorkError::BadRequest(format!(
                    "invalid sample rate {source_rate} Hz (4000..192000)"
                )));
            }
            if !rate_compatible(source_rate, audio::TARGET_SR) {
                return Err(WorkError::BadRequest(format!(
                    "sample rate {source_rate} Hz is not resampling-compatible with {} Hz",
                    audio::TARGET_SR
                )));
            }
            let nch = dec.channels();
            if nch == 0 {
                return Err(WorkError::BadRequest("input has no audio channels".into()));
            }
            let mode = if split { ChannelMode::Split } else { ChannelMode::Mix };
            let mut ms = MultiStream::new(
                mode,
                0,
                nch,
                req_label,
                30.0,
                false,
                vad_path,
                0.5,
                0.363,
            );
            let mut ready = Vec::new();
            let mut submitted = 0usize;
            let t_decode = Instant::now();
            let frames = dec
                .decode(|block| {
                    if cancel.cancelled() {
                        anyhow::bail!("request cancelled");
                    }
                    ready.clear();
                    ms.feed(block, &mut ready)
                        .map_err(|e| anyhow!("pipeline feed: {e:#}"))?;
                    for c in ready.drain(..) {
                        if !pool.submit(QueueItem {
                            chunk: c,
                            cancel: cancel.clone(),
                            reply: reply_tx.clone(),
                        }) {
                            anyhow::bail!("worker pool is closed");
                        }
                        submitted += 1;
                    }
                    Ok(())
                })
                .map_err(|e| {
                    if cancel.cancelled() {
                        WorkError::Unavailable("processing cancelled".into())
                    } else {
                        WorkError::BadRequest(format!("invalid audio input: {e:#}"))
                    }
                })?;
            ready.clear();
            ms.finish(&mut ready)
                .map_err(|e| WorkError::Unavailable(format!("pipeline finish: {e:#}")))?;
            for c in ready.drain(..) {
                if !pool.submit(QueueItem {
                    chunk: c,
                    cancel: cancel.clone(),
                    reply: reply_tx.clone(),
                }) {
                    return Err(WorkError::Unavailable("worker pool is closed".into()));
                }
                submitted += 1;
            }
            metrics.observe_decode(t_decode.elapsed().as_secs_f64());
            tracing::debug!("запрос: отправлено чанков {submitted}");
            Ok(ProducerInfo {
                duration: frames as f64 / audio::TARGET_SR as f64,
                source_rate,
                source_channels: nch,
                plan: ms.plan().cloned(),
            })
        })
    };
    drop(reply_tx); // канал закроется, когда продюсер и его клоны закончат

    // --- сборка результатов своего запроса ---
    let mut pending: BTreeMap<(usize, usize), (Vec<Segment>, Option<String>)> = BTreeMap::new();
    let mut first_error: Option<anyhow::Error> = None;
    let mut queue_deadline: Option<Instant> = None;
    let mut queue_exceeded = false;
    loop {
        let timeout = if queue_exceeded {
            None
        } else {
            Some(match queue_deadline {
                None => QUEUE_WAIT,
                Some(d) => d.saturating_duration_since(Instant::now()).max(Duration::from_millis(1)),
            })
        };
        let outcome = match timeout {
            Some(t) => match reply_rx.recv_timeout(t) {
                Ok(o) => Some(o),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // первый результат не пришёл за QUEUE_WAIT: очередь
                    // воркеров перегружена → отмена, дожидаемся терминации
                    // и отвечаем 503
                    cancel.cancel();
                    queue_exceeded = true;
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            },
            None => match reply_rx.recv() {
                Ok(o) => Some(o),
                Err(_) => break,
            },
        };
        let Some(ChunkOutcome { chunk, result }) = outcome else {
            continue;
        };
        if queue_deadline.is_none() {
            queue_deadline = Some(Instant::now() + QUEUE_WAIT);
        }
        match result {
            Ok((segs, lang)) => {
                pending.insert((chunk.channel, chunk.idx), (segs, lang));
            }
            Err(e) => {
                // отмена не считается ошибкой обработки
                if !cancel.cancelled() {
                    first_error.get_or_insert(e);
                }
            }
        }
        let _ = chunk; // сэмплы серверу не нужны
    }
    let info = match producer.join() {
        Ok(Ok(info)) => info,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(WorkError::Unavailable("decoder worker panicked".into())),
    };
    if let Some(e) = first_error {
        return Err(WorkError::Unavailable(format!("transcription failed: {e:#}")));
    }
    if queue_exceeded {
        return Err(WorkError::Unavailable(
            "worker queue wait exceeded 60 s".into(),
        ));
    }

    // --- многоканальная сборка: каналы = спикеры ---
    let selected = info
        .plan
        .as_ref()
        .map(|p| p.selected.clone())
        .unwrap_or_else(|| vec![0]);
    let multi = selected.len() > 1;
    let mut ch_words: BTreeMap<usize, Vec<Word>> = BTreeMap::new();
    for ((ch, _), (segs, _)) in &pending {
        for seg in segs {
            for w in &seg.words {
                ch_words.entry(*ch).or_default().push(w.clone());
            }
        }
    }
    let channels: Vec<ChannelWords> = ch_words
        .into_iter()
        .map(|(ch, words)| ChannelWords { channel: ch, words })
        .collect();
    let turn_list = channels::build_turns(&channels, query.turn_gap);
    let segment_list = channels::channel_turns(&channels, query.turn_gap);

    // --- JSON по контракту источника ---
    Ok(build_response(BuildInput {
        channels: &channels,
        turns: &turn_list,
        segments: &segment_list,
        options: query,
        duration_sec: info.duration,
        sample_rate_in: info.source_rate,
        source_channels: info.plan
            .as_ref()
            .map(|p| p.source_channels)
            .unwrap_or(info.source_channels),
        dual_mono: info.plan.as_ref().map(|p| p.dual_mono).unwrap_or(false),
        model: query.model.id(),
        rtf: proc_elapsed(proc_start, info.duration),
        multi,
    }))
}

/// Затраченное на обработку время, нормированное на длительность аудио.
fn proc_elapsed(started: Instant, duration: f64) -> f64 {
    let processing = started.elapsed().as_secs_f64();
    if duration > 0.0 {
        processing / duration
    } else {
        0.0
    }
}

struct BuildInput<'a> {
    channels: &'a [ChannelWords],
    turns: &'a [crate::transcript::Turn],
    segments: &'a [crate::transcript::Turn],
    options: &'a TranscribeQuery,
    duration_sec: f64,
    sample_rate_in: u32,
    source_channels: usize,
    dual_mono: bool,
    model: &'static str,
    rtf: f64,
    multi: bool,
}

/// Сборка JSON-ответа: ровно одна текстовая проекция; words/segments/turns —
/// по запросу; channel в words только для многоканального вывода.
fn build_response(input: BuildInput<'_>) -> serde_json::Value {
    let BuildInput {
        channels,
        turns,
        segments,
        options,
        duration_sec,
        sample_rate_in,
        source_channels,
        dual_mono,
        model,
        rtf,
        multi,
    } = input;

    let mut f = serde_json::Map::new();
    f.insert("duration_sec".into(), serde_json::json!(round2(duration_sec)));
    f.insert("sample_rate_in".into(), serde_json::json!(sample_rate_in));
    f.insert("channels".into(), serde_json::json!(source_channels));
    f.insert("dual_mono".into(), serde_json::json!(dual_mono));
    f.insert("model".into(), serde_json::json!(model));
    f.insert("rtf".into(), serde_json::json!(round4(rtf)));

    // одна текстовая проекция
    if options.split && multi {
        let texts: Vec<serde_json::Value> = channels
            .iter()
            .map(|c| serde_json::json!(crate::transcript::join_word_texts(&c.words)))
            .collect();
        f.insert("channels_text".into(), serde_json::Value::Array(texts));
    } else {
        let text = channels
            .first()
            .map(|c| crate::transcript::join_word_texts(&c.words))
            .unwrap_or_default();
        f.insert("text".into(), serde_json::json!(text));
    }

    if options.words {
        let arr: Vec<serde_json::Value> = channels
            .iter()
            .flat_map(|c| {
                c.words.iter().map(move |w| {
                    let mut o = serde_json::Map::new();
                    o.insert("start".into(), serde_json::json!(round2(w.start)));
                    o.insert("end".into(), serde_json::json!(round2(w.end)));
                    o.insert("text".into(), serde_json::json!(w.word));
                    if multi {
                        o.insert("channel".into(), serde_json::json!(c.channel));
                    }
                    serde_json::Value::Object(o)
                })
            })
            .collect();
        f.insert("words".into(), serde_json::Value::Array(arr));
    }
    if options.segments {
        let arr: Vec<serde_json::Value> = segments
            .iter()
            .map(|s| {
                serde_json::json!({
                    "start": round2(s.start),
                    "end": round2(s.end),
                    "channel": s.channel,
                    "text": s.text,
                })
            })
            .collect();
        f.insert("segments".into(), serde_json::Value::Array(arr));
    }
    if options.turns && multi {
        let arr: Vec<serde_json::Value> = turns
            .iter()
            .map(|t| {
                serde_json::json!({
                    "channel": t.channel,
                    "start": round2(t.start),
                    "end": round2(t.end),
                    "text": t.text,
                })
            })
            .collect();
        f.insert("turns".into(), serde_json::Value::Array(arr));
    }
    serde_json::Value::Object(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_compatibility_matches_the_contract_rule() {
        assert!(rate_compatible(16_000, 16_000));
        assert!(rate_compatible(44_100, 16_000)); // 441/160
        assert!(rate_compatible(8_000, 16_000)); // 1/2
        assert!(!rate_compatible(192_001, 16_000)); // вне диапазона всё равно ловится раньше
        // простое число рядом: несократимое отношение > 1000 с одной стороны
        assert!(!rate_compatible(16_007, 16_000));
    }

    #[test]
    fn query_parsing_refuses_unknown_and_keeps_documented_values() {
        let q = TranscribeQuery::parse(Some("model=gigaam-v3-rnnt&words=true&channels=split&turn_gap=1%2E5")).unwrap();
        assert_eq!(q.model, ModelKind::GigaamV3Rnnt);
        assert!(q.words);
        assert!(q.split);
        assert_eq!(q.turn_gap, 1.5);
        assert!(TranscribeQuery::parse(Some("model=whisper-turbo")).is_err());
        assert!(TranscribeQuery::parse(Some("channels=merge")).is_err());
        assert!(TranscribeQuery::parse(Some("ext=flac10")).is_err());
        assert!(TranscribeQuery::parse(Some("turn_gap=0")).is_err());
    }
}
