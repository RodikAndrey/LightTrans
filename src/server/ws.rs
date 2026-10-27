//! WebSocket-стриминг `GET /v1/stream` (Фаза 6 PORTING.md, перенос контракта
//! gigaam-v3-runtime: docs/API.md §WebSocket streaming, crates/service/src/ws.rs).
//!
//! MVP-срез: mono, `model=ctc` (GigaAM v3 CTC), `emit=words`; endpointing
//! «blank» (маска тишины CTC). Вне среза (отказ с явным сообщением до
//! upgrade): `channels>1`, `emit=turns|both`, `endpoint=vad`, `lock=true`,
//! `dedup` при mono (как в источнике).
//!
//! Протокол: бинарные кадры = интерливнутые сэмплы в `fmt` (pcm16/f32/alaw/
//! ulaw); граница кадра может резать группу сэмплов — неполный хвост
//! сохраняется до следующего кадра, терминальный неполный хвост = отказ.
//! Единственный текстовый control — `{"type":"end"}`. События сервера:
//! `words` (ревизия с `revise_from`), `stable`, `final`, `error`, `end`.
//! Кадр/сообщение > 16 МиБ — close 1009. Клиентский close/EOF завершает
//! локальную сессию без флаша. Дрейн (SIGTERM) дожидается активных сессий.

use super::admission::AdmissionRefusal;
use super::http::rate_compatible;
use super::protocol::{query_bool, query_finite_f32, query_positive_f32, QueryParameters};
use super::response::{round2, JSON_CONTENT_TYPE};
use super::AppState;
use crate::pipeline::StreamLease;
use crate::stream::{StreamConfig, StreamEvent};
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{RawQuery, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use std::sync::Arc;
use std::time::Duration;
use rubato::Resampler;
use tokio::sync::mpsc;

const STREAM_QUERY_KEYS: &[&str] = &[
    "model",
    "rate",
    "fmt",
    "channels",
    "emit",
    "endpoint",
    "horizon",
    "lock",
    "turn_gap",
    "dedup",
    "backchannel_max_ms",
];

/// Граница кадра/сообщения WebSocket (контракт источника) → close 1009.
const WS_MESSAGE_LIMIT_BYTES: usize = 16 * 1024 * 1024;

/// Ожидание свободного воркера после upgrade: занятые batch-работой воркеры
/// освобождаются только между чанками.
const LEASE_WAIT: Duration = Duration::from_secs(60);

/// Период опроса фазы admission (Draining) в цикле сессии.
const DRAIN_POLL: Duration = Duration::from_millis(200);

/// Входной чанк ресемплёра (rubato FftFixedIn, 64 мс @16 кГц).
const RESAMPLE_CHUNK: usize = 1024;

// ---------------------------------------------------------------------------
// Query (MVP-срез)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleFormat {
    Pcm16,
    F32,
    Alaw,
    Ulaw,
}

impl SampleFormat {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "pcm16" => Ok(Self::Pcm16),
            "f32" => Ok(Self::F32),
            "alaw" => Ok(Self::Alaw),
            "ulaw" => Ok(Self::Ulaw),
            other => Err(format!("fmt must be pcm16|f32|alaw|ulaw, got {other:?}")),
        }
    }

    fn bytes_per_sample(self) -> usize {
        match self {
            Self::Pcm16 => 2,
            Self::F32 => 4,
            Self::Alaw | Self::Ulaw => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamQuery {
    rate: u32,
    format: SampleFormat,
    config: StreamConfig,
}

impl StreamQuery {
    /// Строгий разбор query; всё вне MVP-среза — отказ с явным сообщением
    /// (400 до upgrade, никакой работы с моделями).
    fn parse(raw: Option<&str>) -> Result<Self, String> {
        let p = QueryParameters::parse(raw)?;
        p.reject_unknown(STREAM_QUERY_KEYS)?;
        match p.get("model") {
            None | Some("ctc") => {}
            Some(v) => return Err(format!("stream supports only model=ctc, got {v:?}")),
        }
        let rate_text = p
            .get("rate")
            .ok_or_else(|| "rate parameter is required (8000..192000)".to_string())?;
        let rate: u32 = rate_text
            .parse()
            .map_err(|_| format!("rate must be an integer in 8000..192000, got {rate_text:?}"))?;
        if !(8000..=192000).contains(&rate) {
            return Err("rate must be in 8000..192000".into());
        }
        if !rate_compatible(rate, crate::stream::MODEL_SR as u32) {
            return Err(format!(
                "rate {rate} Hz is not resampling-compatible with {} Hz",
                crate::stream::MODEL_SR
            ));
        }
        let format = match p.get("fmt") {
            None => SampleFormat::Pcm16,
            Some(v) => SampleFormat::parse(v)?,
        };
        match p.get("channels") {
            None => {}
            Some("1") => {}
            Some(v) => {
                return Err(format!(
                    "multichannel streaming is not in the MVP slice (mono only), got channels={v}"
                ))
            }
        }
        match p.get("emit") {
            None | Some("words") => {}
            Some(v) => {
                return Err(format!(
                    "emit must be words (MVP slice; turns/both are not implemented), got {v:?}"
                ))
            }
        }
        let horizon = query_positive_f32(&p, "horizon", 5.0)?;
        match p.get("endpoint") {
            None | Some("blank") => {}
            Some("vad") => {
                return Err(
                    "endpoint=vad is not implemented (MVP slice uses CTC-silence endpointing)"
                        .into(),
                )
            }
            Some(v) => return Err(format!("endpoint must be blank|vad, got {v:?}")),
        }
        if query_bool(&p, "lock", false)? {
            return Err("lock=true is not implemented (MVP slice)".into());
        }
        // turn_gap/backchannel_max_ms разбираются по контракту источника,
        // но в words-режиме не действуют (turns вне MVP-среза)
        let _turn_gap = query_positive_f32(&p, "turn_gap", 0.8)?;
        if query_finite_f32(&p, "backchannel_max_ms", 0.0)? < 0.0 {
            return Err("backchannel_max_ms must be non-negative".into());
        }
        // mono + явный dedup = отказ (как в источнике)
        if p.get("dedup").is_some() {
            return Err("dedup is valid only when channels is greater than one".into());
        }
        let config = StreamConfig { horizon_sec: horizon, ..Default::default() };
        Ok(Self { rate, format, config })
    }
}

// ---------------------------------------------------------------------------
// Протокол WS
// ---------------------------------------------------------------------------

fn pre_upgrade_refusal(status: StatusCode, message: impl Into<String>) -> axum::response::Response {
    (
        status,
        [(axum::http::header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        serde_json::json!({ "error": message.into() }).to_string(),
    )
        .into_response()
}

/// Строгий разбор control-кадра: только `{"type":"end"}` (с пробелами).
fn parse_control(text: &str) -> Result<(), String> {
    let mut remaining = text;
    for token in ["{", "\"type\"", ":", "\"end\"", "}"] {
        remaining = remaining
            .trim_start()
            .strip_prefix(token)
            .ok_or_else(|| "control frame must be exactly {\"type\":\"end\"}".to_string())?;
    }
    if remaining.trim().is_empty() {
        Ok(())
    } else {
        Err("control frame must be exactly {\"type\":\"end\"}".into())
    }
}

fn event_json(event: &StreamEvent) -> String {
    match event {
        StreamEvent::Words { at, revise_from, words } => serde_json::json!({
            "type": "words",
            "at": round2(f64::from(*at)),
            "revise_from": revise_from,
            "words": words.iter().map(|w| serde_json::json!({
                "start": round2(f64::from(w.start)),
                "end": round2(f64::from(w.end)),
                "text": w.text,
                "stable": w.stable,
            })).collect::<Vec<_>>(),
        }),
        StreamEvent::Stable { at, upto } => serde_json::json!({
            "type": "stable",
            "at": round2(f64::from(*at)),
            "upto": upto,
        }),
        StreamEvent::Final { at, upto, endpoint } => serde_json::json!({
            "type": "final",
            "at": round2(f64::from(*at)),
            "upto": upto,
            "endpoint": endpoint,
        }),
    }
    .to_string()
}

fn error_event(message: &str) -> String {
    serde_json::json!({ "type": "error", "message": message }).to_string()
}

fn end_event() -> String {
    "{\"type\":\"end\"}".to_string()
}

// ---------------------------------------------------------------------------
// Вход аудио: формат → f32 mono → ресемплинг 16 кГц (с состоянием)
// ---------------------------------------------------------------------------

/// G.711 → линейный PCM16 (стандартные разложения ITU/Sun).
fn alaw_to_f32(b: u8) -> f32 {
    let a = b ^ 0x55;
    let mut t: i32 = ((a & 0x0f) as i32) << 4;
    let seg = (a & 0x70) >> 4;
    t = match seg {
        0 => t + 8,
        1 => t + 0x108,
        _ => (t + 0x108) << (seg - 1),
    };
    let v = if a & 0x80 != 0 { t } else { -t };
    v as f32 / 32768.0
}

fn ulaw_to_f32(b: u8) -> f32 {
    let u = !b;
    const BIAS: i32 = 0x84;
    let mut t: i32 = (((u & 0x0f) as i32) << 3) + BIAS;
    t <<= (u & 0x70) >> 4;
    let v = if u & 0x80 != 0 { BIAS - t } else { t - BIAS };
    v as f32 / 32768.0
}

/// Конвейер входа одной WS-сессии: байты кадра → f32 mono (исходная частота)
/// → 16 кГц с сохранением состояния ресемплёра. Компенсирует задержку
/// ресемплёра (output_delay) вычитанием первых выходных сэмплов.
struct Ingest {
    format: SampleFormat,
    /// Неполная группа сэмплов (граница кадра может её резать).
    pending: Vec<u8>,
    resampler: Option<rubato::FftFixedIn<f32>>,
    rin: Vec<Vec<f32>>,
    rout: Vec<Vec<f32>>,
    delay: usize,
    dropped: usize,
}

impl Ingest {
    fn new(rate: u32, format: SampleFormat) -> Result<Self, String> {
        let (resampler, rin, rout, delay) = if rate as usize != crate::stream::MODEL_SR {
            let r = rubato::FftFixedIn::<f32>::new(
                rate as usize,
                crate::stream::MODEL_SR,
                RESAMPLE_CHUNK,
                1,
                1,
            )
            .map_err(|e| format!("resampler init {rate}→{}: {e}", crate::stream::MODEL_SR))?;
            let delay = r.output_delay();
            let rin = r.input_buffer_allocate(true);
            let rout = r.output_buffer_allocate(true);
            (Some(r), rin, rout, delay)
        } else {
            (None, Vec::new(), Vec::new(), 0)
        };
        Ok(Self { format, pending: Vec::new(), resampler, rin, rout, delay, dropped: 0 })
    }

    fn decode_samples(&mut self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        self.pending.extend_from_slice(bytes);
        let step = self.format.bytes_per_sample();
        let n = self.pending.len() / step;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let b = &self.pending[i * step..(i + 1) * step];
            let s = match self.format {
                SampleFormat::Pcm16 => {
                    i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0
                }
                SampleFormat::F32 => {
                    let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    if !v.is_finite() || !(-1.0..=1.0).contains(&v) {
                        return Err(format!(
                            "fmt=f32 sample must be finite in [-1, 1], got {v}"
                        ));
                    }
                    v
                }
                SampleFormat::Alaw => alaw_to_f32(b[0]),
                SampleFormat::Ulaw => ulaw_to_f32(b[0]),
            };
            out.push(s);
        }
        self.pending.drain(..n * step);
        Ok(out)
    }

    fn resample(&mut self, mono: Vec<f32>) -> Result<Vec<f32>, String> {
        let Some(r) = self.resampler.as_mut() else {
            return Ok(mono);
        };
        let mut out = Vec::with_capacity(mono.len() * 2);
        let mut pos = 0;
        while mono.len() - pos >= RESAMPLE_CHUNK {
            self.rin[0][..RESAMPLE_CHUNK].copy_from_slice(&mono[pos..pos + RESAMPLE_CHUNK]);
            pos += RESAMPLE_CHUNK;
            let (_, n_out) = r
                .process_into_buffer(&self.rin, &mut self.rout, None)
                .map_err(|e| format!("resample: {e}"))?;
            Self::take_delayed(self.delay, &mut self.dropped, &self.rout[0][..n_out], &mut out);
        }
        Ok(out)
    }

    /// Выход с компенсацией задержки: первые `delay` сэмплов отбрасываются.
    /// Свободная функция: resample держит раздельные займы полей структуры.
    fn take_delayed(delay: usize, dropped: &mut usize, chunk: &[f32], out: &mut Vec<f32>) {
        for &s in chunk {
            if *dropped < delay {
                *dropped += 1;
            } else {
                out.push(s);
            }
        }
    }

    /// Байты кадра → сэмплы 16 кГц mono. Ошибка — терминальная (сессия закрывается).
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        let mono = self.decode_samples(bytes)?;
        self.resample(mono)
    }

    /// Финальный флаш ресемплёра при `{"type":"end"}`: выгрузка внутренней
    /// линии задержки без нового входа.
    fn flush(&mut self) -> Result<Vec<f32>, String> {
        let Some(r) = self.resampler.as_mut() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        let (_, n_out) = r
            .process_partial_into_buffer(None::<&[Vec<f32>]>, &mut self.rout, None)
            .map_err(|e| format!("resample flush: {e}"))?;
        Self::take_delayed(self.delay, &mut self.dropped, &self.rout[0][..n_out], &mut out);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Сессия: блокирующая сторона (spawn_blocking) и транспорт
// ---------------------------------------------------------------------------

enum InMsg {
    Push(Vec<f32>),
    End,
}

enum OutMsg {
    Events(Vec<StreamEvent>),
    /// Терминальный маркер после флаша (End): все события уже в канале.
    Done,
    Failed(String),
}

/// Блокирующая сторона: обмен с арендованным воркером. Аренда освобождается
/// в любом выходе (Drop шлёт End воркеру).
fn run_stream_side(
    lease: StreamLease,
    config: StreamConfig,
    mut in_rx: mpsc::Receiver<InMsg>,
    out_tx: mpsc::Sender<OutMsg>,
) {
    if let Err(e) = lease.begin(config) {
        let _ = out_tx.blocking_send(OutMsg::Failed(e));
        return;
    }
    while let Some(msg) = in_rx.blocking_recv() {
        match msg {
            InMsg::Push(samples) => {
                tracing::debug!("стрим: push {} сэмплов", samples.len());
                match lease.push(samples) {
                    Ok(events) => {
                        tracing::debug!("стрим: {} событий", events.len());
                        if out_tx.blocking_send(OutMsg::Events(events)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = out_tx.blocking_send(OutMsg::Failed(e));
                        return;
                    }
                }
            }
            InMsg::End => {
                let out = match lease.flush() {
                    Ok(events) => OutMsg::Events(events),
                    Err(e) => OutMsg::Failed(e),
                };
                let _ = out_tx.blocking_send(out);
                let _ = out_tx.blocking_send(OutMsg::Done);
                return;
            }
        }
    }
    // вход закрыт без End (клиент закрылся): без флаша и без событий
}

async fn send_text(socket: &mut WebSocket, body: &str) -> bool {
    socket.send(Message::Text(body.into())).await.is_ok()
}

/// Точка входа ручки: строгий query → admission → upgrade.
pub(crate) async fn stream(
    State(state): State<AppState>,
    RawQuery(query_text): RawQuery,
    upgrade: WebSocketUpgrade,
) -> axum::response::Response {
    let metrics = Arc::clone(&state.inner().metrics);
    let query = match StreamQuery::parse(query_text.as_deref()) {
        Ok(q) => q,
        Err(e) => {
            metrics.ws_reject();
            return pre_upgrade_refusal(StatusCode::BAD_REQUEST, e);
        }
    };
    // admission: одна сессия = один слот ёмкости (контракт источника:
    // пермит на канал; MVP — mono, один пермит)
    let permit = match state.inner().admission.admit_http() {
        Ok(p) => p,
        Err(AdmissionRefusal::Draining) => {
            metrics.ws_reject();
            return pre_upgrade_refusal(StatusCode::SERVICE_UNAVAILABLE, "draining");
        }
        Err(AdmissionRefusal::Overloaded) => {
            metrics.ws_reject();
            return pre_upgrade_refusal(
                StatusCode::SERVICE_UNAVAILABLE,
                "too many active sessions, retry later",
            );
        }
    };
    upgrade
        .max_message_size(WS_MESSAGE_LIMIT_BYTES)
        .max_frame_size(WS_MESSAGE_LIMIT_BYTES)
        .on_upgrade(move |socket| handle_session(socket, state, query, permit))
        .into_response()
}

/// Арендовать свободного CTC-воркера с дедлайном (воркеры, занятые batch-чанками,
/// освобождаются между чанками). `None` — дедлайн истёк.
async fn wait_for_lease(state: &AppState) -> Option<StreamLease> {
    let deadline = tokio::time::Instant::now() + LEASE_WAIT;
    loop {
        if let Some(lease) = state.inner().pools.ctc.lease_stream() {
            return Some(lease);
        }
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn handle_session(
    mut socket: WebSocket,
    state: AppState,
    query: StreamQuery,
    _permit: tokio::sync::OwnedSemaphorePermit,
) {
    let metrics = Arc::clone(&state.inner().metrics);
    metrics.ws_session_started();
    tracing::info!(
        "ws stream открыт: rate={} fmt={:?} (MVP: mono/ctc/words)",
        query.rate,
        query.format
    );

    let Some(lease) = wait_for_lease(&state).await else {
        let _ = send_text(&mut socket, &error_event("no free recognition worker, retry later")).await;
        let _ = socket.send(Message::Close(None)).await;
        metrics.ws_error();
        metrics.ws_session_finished();
        return;
    };

    // каналы: транспорт (async) ↔ блокирующая сторона (spawn_blocking)
    let (in_tx, in_rx) = mpsc::channel::<InMsg>(32);
    let (out_tx, mut out_rx) = mpsc::channel::<OutMsg>(64);
    let blocking =
        tokio::task::spawn_blocking(move || run_stream_side(lease, query.config, in_rx, out_tx));

    let mut ingest = match Ingest::new(query.rate, query.format) {
        Ok(i) => i,
        Err(e) => {
            let _ = send_text(&mut socket, &error_event(&e)).await;
            let _ = socket.send(Message::Close(None)).await;
            metrics.ws_error();
            metrics.ws_session_finished();
            return;
        }
    };

    let mut drain = tokio::time::interval(DRAIN_POLL);
    drain.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    drain.tick().await; // первый тик мгновенный

    // терминация: Ok(()) — штатный end (флаш + end + close);
    // Err(None) — клиент закрылся (без флаша); Err(Some(code)) — ошибка
    enum Exit {
        End,
        Silent,
        Error(String),
    }
    #[allow(unused_assignments)]
    let mut exit: Exit = Exit::Silent;

    'outer: loop {
        tokio::select! {
            out = out_rx.recv() => match out {
                Some(OutMsg::Events(events)) => {
                    for ev in &events {
                        if !send_text(&mut socket, &event_json(ev)).await {
                            exit = Exit::Silent;
                            break 'outer;
                        }
                    }
                }
                Some(OutMsg::Failed(e)) => {
                    exit = Exit::Error(e);
                    break 'outer;
                }
                Some(OutMsg::Done) | None => {
                    // блокирующая сторона умерла без ответа (Done без End —
                    // не должно случаться; трактуем как обрыв сессии)
                    exit = Exit::Error("stream session ended unexpectedly".into());
                    break 'outer;
                }
            },
            frame = socket.recv() => match frame {
                None | Some(Ok(Message::Close(_))) => {
                    exit = Exit::Silent; // клиент закрылся: локальная сессия без флаша
                    break 'outer;
                }
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Binary(bytes))) => {
                    metrics.ws_frame();
                    match ingest.push(&bytes) {
                        Ok(samples) if !samples.is_empty() => {
                            if in_tx.send(InMsg::Push(samples)).await.is_err() {
                                exit = Exit::Error("stream session ended unexpectedly".into());
                                break 'outer;
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            exit = Exit::Error(e);
                            break 'outer;
                        }
                    }
                }
                Some(Ok(Message::Text(text))) => match parse_control(&text) {
                    Ok(()) => {
                        // терминальный неполный хвост группы сэмплов = отказ
                        // (контракт источника), иначе — флаш ресемплёра и End
                        if ingest.pending.is_empty() {
                            match ingest.flush() {
                                Ok(tail) => {
                                    if !tail.is_empty()
                                        && in_tx.send(InMsg::Push(tail)).await.is_err()
                                    {
                                        exit = Exit::Error(
                                            "stream session ended unexpectedly".into(),
                                        );
                                    } else {
                                        exit = Exit::End;
                                    }
                                }
                                Err(e) => exit = Exit::Error(e),
                            }
                        } else {
                            exit = Exit::Error(
                                "terminal incomplete sample group must not be discarded".into(),
                            );
                        }
                        break 'outer;
                    }
                    Err(e) => {
                        exit = Exit::Error(e);
                        break 'outer;
                    }
                },
                Some(Err(e)) => {
                    // 1009 — единственный структурированный отказ транспорта
                    let msg = e.to_string().to_ascii_lowercase();
                    if msg.contains("capacity") || msg.contains("too long") || msg.contains("too big") {
                        let _ = socket
                            .send(Message::Close(Some(CloseFrame {
                                code: 1009,
                                reason: "message too big".into(),
                            })))
                            .await;
                    }
                    exit = Exit::Silent;
                    break 'outer;
                }
            },
            _ = drain.tick() => {
                if state.inner().admission.is_draining() {
                    exit = Exit::End; // дрейн: штатный флаш + end + close
                    break 'outer;
                }
            }
        }
    }

    match exit {
        Exit::Silent => drop(in_tx),
        Exit::End => {
            // штатная терминация: End → дренировать все события (тики+флаш)
            // до терминального маркера → end → close
            let _ = in_tx.send(InMsg::End).await;
            drop(in_tx);
            let mut failed = false;
            while let Some(out) = out_rx.recv().await {
                match out {
                    OutMsg::Events(events) => {
                        for ev in &events {
                            if !send_text(&mut socket, &event_json(ev)).await {
                                failed = true;
                                break;
                            }
                        }
                    }
                    OutMsg::Failed(e) => {
                        metrics.ws_error();
                        let _ = send_text(&mut socket, &error_event(&e)).await;
                        failed = true;
                    }
                    OutMsg::Done => break,
                }
                if failed {
                    break;
                }
            }
            if !failed {
                let _ = send_text(&mut socket, &end_event()).await;
            }
            let _ = socket.send(Message::Close(None)).await;
        }
        Exit::Error(e) => {
            drop(in_tx);
            metrics.ws_error();
            let _ = send_text(&mut socket, &error_event(&e)).await;
            let _ = socket.send(Message::Close(None)).await;
        }
    }

    let _ = blocking.await;
    metrics.ws_session_finished();
    tracing::info!("ws stream закрыт");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::StreamWord;

    #[test]
    fn query_defaults_and_accepted_domain() {
        let q = StreamQuery::parse(Some("rate=16000")).unwrap();
        assert_eq!(q.rate, 16_000);
        assert_eq!(q.format, SampleFormat::Pcm16);
        assert!((q.config.horizon_sec - 5.0).abs() < 1e-6);
        let q = StreamQuery::parse(Some("rate=44100&fmt=alaw&horizon=2.5&endpoint=blank&emit=words&turn_gap=1%2E2&backchannel_max_ms=0"))
            .unwrap();
        assert_eq!(q.rate, 44_100);
        assert_eq!(q.format, SampleFormat::Alaw);
        assert!((q.config.horizon_sec - 2.5).abs() < 1e-6);
    }

    #[test]
    fn query_refuses_out_of_mvp_and_malformed_values() {
        // MVP-срез: явные отказы для многоканала/turns/vad/lock/dedup
        assert!(StreamQuery::parse(Some("rate=16000&channels=2")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&emit=turns")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&emit=both")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&endpoint=vad")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&lock=true")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&dedup=false")).is_err());
        // домен источника
        assert!(StreamQuery::parse(None).is_err(), "rate обязателен");
        assert!(StreamQuery::parse(Some("rate=7999")).is_err());
        assert!(StreamQuery::parse(Some("rate=192001")).is_err());
        assert!(StreamQuery::parse(Some("rate=16007")).is_err(), "ratio > 1000");
        assert!(StreamQuery::parse(Some("rate=16000&model=rnnt")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&fmt=opus")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&horizon=0")).is_err());
        assert!(StreamQuery::parse(Some("rate=16000&unknown=1")).is_err());
    }

    #[test]
    fn control_frame_is_strict() {
        assert!(parse_control("{\"type\":\"end\"}").is_ok());
        assert!(parse_control(" {\"type\":\"end\"} \n").is_ok());
        assert!(parse_control("{\"type\":\"close\"}").is_err());
        assert!(parse_control("").is_err());
    }

    #[test]
    fn event_json_matches_the_source_projection() {
        let words = StreamEvent::Words {
            at: 1.234,
            revise_from: 2,
            words: vec![StreamWord {
                text: "тест".into(),
                start: 0.123,
                end: 0.456,
                stable: false,
            }],
        };
        let parsed: serde_json::Value = serde_json::from_str(&event_json(&words)).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({
                "type": "words",
                "at": 1.23,
                "revise_from": 2,
                "words": [{"start": 0.12, "end": 0.46, "text": "тест", "stable": false}],
            })
        );
        let fin = StreamEvent::Final { at: 3.0, upto: 4, endpoint: true };
        assert!(event_json(&fin).contains("\"endpoint\":true"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&error_event("boom")).unwrap(),
            serde_json::json!({ "type": "error", "message": "boom" })
        );
        assert_eq!(end_event(), "{\"type\":\"end\"}");
    }
}
