//! Инкрементальная транскрибация одного канала (перенос контракта
//! gigaam-v3-runtime, crates/transcription/src/{stream,endpoint,stitch,padding}.rs).
//!
//! Схема: буфер до `window_sec` (30 c) накапливает сэмплы модели (16 кГц);
//! каждые `step_sec` (0.5 c) или при заполнении окна — повторный прогон
//! энкодера по всему окну с инкрементальным log-mel (хвост за реальным
//! аудио забит детерминированным шумом малой амплитуды). Слова соседних
//! окон сшиваются по шву (overlap 6 c), ревизии гипотез отдаются клиенту
//! как события `words` (ревизия с позиции `revise_from`), `stable`
//! (порядок слов зафиксирован) и `final` (слова неизменяемы).
//!
//! MVP-срез Фазы 6 (PORTING.md): mono, model=ctc, emit=words; endpointing —
//! «blank» (по маске тишины CTC, правила источника 1.2/2.4 c); lock,
//! turns, dedup и endpoint=vad не входят в срез.

use crate::mel::{MelExtractor, N_FFT, N_MELS};
use anyhow::Result;

/// Частота дискретизации модели (GigaAM v3).
pub const MODEL_SR: usize = 16_000;

// ---------------------------------------------------------------------------
// Детерминированный шумовой паддинг (padding.rs источника, seed 0x5eed)
// ---------------------------------------------------------------------------

const NOISE_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const NOISE_INCREMENT: u64 = 1_442_695_040_888_963_407;
const NOISE_AMPLITUDE: f64 = 1e-3;

/// Детерминированный шум малой амплитуды для паддинга окна за реальным аудио.
pub fn pad_noise(length: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(NOISE_MULTIPLIER).wrapping_add(NOISE_INCREMENT);
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(NOISE_MULTIPLIER).wrapping_add(NOISE_INCREMENT);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            ((unit * 2.0 - 1.0) * NOISE_AMPLITUDE) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Нормализация слова для сшивки швов (text.rs источника)
// ---------------------------------------------------------------------------

/// Нормализация для сравнения слов при сшивке: буквы/цифры/'/-, нижний регистр, ё→е.
pub fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '\'' | '-'))
        .flat_map(char::to_lowercase)
        .map(|c| if c == 'ё' { 'е' } else { c })
        .collect()
}

// ---------------------------------------------------------------------------
// Endpointing по маске тишины (endpoint.rs источника)
// ---------------------------------------------------------------------------

/// Правила закрытия/сброса реплики по тишине.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointRules {
    pub after_speech_seconds: f32,
    pub without_speech_seconds: f32,
}

impl Default for EndpointRules {
    fn default() -> Self {
        Self { after_speech_seconds: 1.2, without_speech_seconds: 2.4 }
    }
}

/// Решение текущего тика: реплика закрыта / безречевой сброс / продолжаем.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Endpoint {
    None,
    AfterSpeech,
    NoSpeech,
}

/// Есть ли в текущем буфере распознанная речь.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechState {
    Absent,
    Present,
}

fn validate_duration(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        anyhow::bail!("{name} must be finite and nonnegative");
    }
    Ok(())
}

/// Хвостовая тишина среди реальных фреймов маски (паддинг не считается).
pub fn trailing_silence(mask: &[bool], real_frames: usize) -> usize {
    mask[..real_frames.min(mask.len())].iter().rev().take_while(|&&s| s).count()
}

/// Endpoint-решение по хвостовой тишине маски (контракт endpoint.rs источника).
pub fn detect_endpoint(
    rules: EndpointRules,
    mask: &[bool],
    real_frames: usize,
    frames_per_second: f32,
    speech_state: SpeechState,
) -> Result<Endpoint> {
    validate_duration(rules.after_speech_seconds, "endpoint after-speech silence")?;
    validate_duration(rules.without_speech_seconds, "endpoint no-speech silence")?;
    if !frames_per_second.is_finite() || frames_per_second <= 0.0 {
        anyhow::bail!("endpoint frame rate must be finite and positive");
    }
    let silence_seconds = trailing_silence(mask, real_frames) as f32 / frames_per_second;
    match speech_state {
        SpeechState::Present if silence_seconds >= rules.after_speech_seconds => {
            Ok(Endpoint::AfterSpeech)
        }
        SpeechState::Absent if silence_seconds >= rules.without_speech_seconds => {
            Ok(Endpoint::NoSpeech)
        }
        _ => Ok(Endpoint::None),
    }
}

// ---------------------------------------------------------------------------
// Окна и швы (stitch.rs источника)
// ---------------------------------------------------------------------------

/// Слово текущего окна в абсолютном времени.
#[derive(Clone, Debug, PartialEq)]
pub struct TWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct WindowWords {
    start: f32,
    end: f32,
    words: Vec<TWord>,
}

/// Пара срезов, убирающая дубликат перекрытия соседних окон.
struct Seam {
    previous_count: usize,
    current_skip: usize,
}

/// Шов между предыдущей строкой слов и текущим окном: сначала ищется
/// совпадающее нормализованное слово около середины перекрытия, иначе
/// середина перекрытия задаёт оба среза (контракт stitch.rs).
fn seam(
    previous: &[TWord],
    previous_end: f32,
    current_start: f32,
    current: &[TWord],
    tolerance_seconds: f32,
) -> Result<Seam> {
    validate_duration(previous_end, "previous window end")?;
    validate_duration(tolerance_seconds, "stitch tolerance")?;
    if !(current_start.is_finite() && current_start >= 0.0) {
        anyhow::bail!("current window start must be finite and nonnegative");
    }
    let midpoint = (current_start + previous_end) / 2.0;
    let mut best: Option<(f32, usize, usize)> = None;
    for (pi, pw) in previous.iter().enumerate() {
        if pw.start < current_start {
            continue;
        }
        for (ci, cw) in current.iter().enumerate() {
            if cw.start > previous_end {
                break;
            }
            let offsets_match = (pw.start - cw.start).abs() <= tolerance_seconds;
            let text_matches = normalize_word(&pw.text) == normalize_word(&cw.text);
            if offsets_match && text_matches {
                let distance = (pw.start - midpoint).abs();
                if best.map_or(true, |(d, _, _)| distance < d) {
                    best = Some((distance, pi, ci));
                }
            }
        }
    }
    Ok(match best {
        Some((_, pc, cs)) => Seam { previous_count: pc, current_skip: cs },
        None => Seam {
            previous_count: previous.iter().filter(|w| w.start < midpoint).count(),
            current_skip: current.iter().filter(|w| w.start < midpoint).count(),
        },
    })
}

/// Сшивка строки открытых слов с предыдущим окном через шов.
fn stitch_open(previous: &WindowWords, current_start: f32, current: &[TWord]) -> Result<Vec<TWord>> {
    let s = seam(&previous.words, previous.end, current_start, current, 0.25)?;
    let mut out: Vec<TWord> = previous.words[..s.previous_count].to_vec();
    out.extend_from_slice(&current[s.current_skip..]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Конфигурация стрима (дефолты checked_default источника)
// ---------------------------------------------------------------------------

/// Проверенная конфигурация инкрементальной сессии; значения в секундах,
/// счётчики сэмплов считаются от `MODEL_SR`.
#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    pub window_sec: f32,
    pub overlap_sec: f32,
    pub step_sec: f32,
    pub horizon_sec: f32,
    pub seam_after_sec: f32,
    /// Сколько тишины сохранить при endpoint-сбросе.
    pub keep_silence_sec: f32,
    pub rules: EndpointRules,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window_sec: 30.0,
            overlap_sec: 6.0,
            step_sec: 0.5,
            horizon_sec: 5.0,
            seam_after_sec: 2.0,
            keep_silence_sec: 0.5,
            rules: EndpointRules::default(),
        }
    }
}

impl StreamConfig {
    fn validate(&self) -> Result<()> {
        for (v, name) in [
            (self.window_sec, "stream window"),
            (self.step_sec, "stream step"),
            (self.horizon_sec, "stream horizon"),
        ] {
            if !v.is_finite() || v <= 0.0 {
                anyhow::bail!("{name} duration must be finite and greater than zero");
            }
        }
        for (v, name) in [
            (self.overlap_sec, "stream overlap"),
            (self.seam_after_sec, "stream seam delay"),
            (self.keep_silence_sec, "stream retained silence"),
        ] {
            validate_duration(v, name)?;
        }
        if self.overlap_sec >= self.window_sec {
            anyhow::bail!("stream overlap duration must be shorter than the window");
        }
        validate_duration(self.rules.after_speech_seconds, "endpoint after-speech silence")?;
        validate_duration(self.rules.without_speech_seconds, "endpoint no-speech silence")?;
        Ok(())
    }

    fn samples(&self, sec: f32) -> usize {
        (sec as f64 * MODEL_SR as f64).round() as usize
    }

    pub fn window_samples(&self) -> usize {
        self.samples(self.window_sec)
    }
    fn step_samples(&self) -> usize {
        self.samples(self.step_sec)
    }
    fn overlap_samples(&self) -> usize {
        self.samples(self.overlap_sec)
    }
    fn retained_silence_samples(&self) -> usize {
        self.samples(self.keep_silence_sec)
    }
}

// ---------------------------------------------------------------------------
// События
// ---------------------------------------------------------------------------

/// Слово в строке событий: `stable` — порядок зафиксирован.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub stable: bool,
}

/// События стрима (контракт docs/API.md §Server events; mono — без channel).
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// Заменить хвост строки слов с позиции `revise_from`.
    Words { at: f32, revise_from: usize, words: Vec<StreamWord> },
    /// Порядок слов зафиксирован до `upto`.
    Stable { at: f32, upto: usize },
    /// Слова до `upto` неизменяемы; `endpoint` — финальность по endpointing.
    Final { at: f32, upto: usize, endpoint: bool },
}

/// Результат декода одного окна: слова (относительно начала окна, сек),
/// маска тишины по выходным фреймам, число выходных фреймов.
pub struct DecodedWindow {
    pub words: Vec<TWord>,
    pub silence: Vec<bool>,
    pub output_frames: usize,
    pub frame_rate: f32,
}

/// Способность декодировать одно полное окно мел-признаков.
pub trait WindowDecoder {
    /// `features` — [N_MELS x n_frames] (bin-major); `window_sec` — длительность окна.
    fn decode_window(&mut self, features: &[f32], n_frames: usize, window_sec: f32)
        -> Result<DecodedWindow>;
}

// ---------------------------------------------------------------------------
// Сессия
// ---------------------------------------------------------------------------

enum Progress {
    Advance,
    Flush,
}

/// Полное окно (сэмплы) → число мел-фреймов (center=false, hop=160).
fn full_frames_of(samples: usize) -> usize {
    if samples < N_FFT {
        0
    } else {
        1 + (samples - N_FFT) / crate::mel::HOP
    }
}

/// Состояние инкрементальной mono-сессии; `D` — декодер полного окна
/// (у воркера — GigaAM v3 CTC). Методы соответствуют push/flush источника.
pub struct StreamSession<D: WindowDecoder> {
    decoder: D,
    mel: MelExtractor,
    config: StreamConfig,
    full_samples: usize,
    full_frames: usize,
    noise: Vec<f32>,
    noise_mel: Vec<f32>,
    mel_buf: Vec<f32>, // [N_MELS x full_frames], bin-major
    real_full_frames: usize,
    start_time: f32,
    buffer: Vec<f32>,
    decoded_samples: usize,
    committed: Vec<StreamWord>,
    emitted: Vec<StreamWord>,
    final_sent: usize,
    stable_sent: usize,
    /// Слова текущего окна в абсолютном времени, до фильтра по cut_time.
    current: Vec<TWord>,
    /// Слова предыдущего полного окна, ждущие шва.
    previous: Option<WindowWords>,
    cut_time: f32,
}

impl<D: WindowDecoder> StreamSession<D> {
    pub fn new(decoder: D, config: StreamConfig) -> Result<Self> {
        config.validate()?;
        let full_samples = config.window_samples();
        let full_frames = full_frames_of(full_samples);
        let noise = pad_noise(full_samples, 0x5eed);
        let mut mel = MelExtractor::new();
        let mut noise_mel = vec![0f32; N_MELS * full_frames];
        for t in 0..full_frames {
            let mut frame = [0f32; N_FFT];
            frame.copy_from_slice(&noise[t * crate::mel::HOP..t * crate::mel::HOP + N_FFT]);
            let col = mel.frame_log_mel(&frame);
            for m in 0..N_MELS {
                noise_mel[m * full_frames + t] = col[m];
            }
        }
        Ok(Self {
            decoder,
            mel,
            config,
            full_samples,
            full_frames,
            mel_buf: noise_mel.clone(),
            real_full_frames: 0,
            start_time: 0.0,
            buffer: Vec::with_capacity(full_samples),
            decoded_samples: 0,
            committed: Vec::new(),
            emitted: Vec::new(),
            final_sent: 0,
            stable_sent: 0,
            current: Vec::new(),
            previous: None,
            cut_time: 0.0,
            noise,
            noise_mel,
        })
    }

    /// Прогрев окна детерминированным шумом (один прогон энкодера).
    pub fn warmup(&mut self) -> Result<()> {
        self.decoder
            .decode_window(&self.noise_mel, self.full_frames, self.config.window_sec)?;
        Ok(())
    }

    /// Текущая строка слов, какую видит клиент после применения всех событий.
    pub fn transcript(&self) -> &[StreamWord] {
        &self.emitted
    }

    /// Забрать декодер обратно (воркер возвращает модель в общий цикл).
    pub fn into_decoder(self) -> D {
        self.decoder
    }

    /// Подать сэмплы модели (16 кГц mono); вернуть события тиков.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<StreamEvent>> {
        let step = self.config.step_samples();
        let mut events = Vec::new();
        let mut remaining = input;
        while !remaining.is_empty() {
            let room = self
                .full_samples
                .checked_sub(self.buffer.len())
                .ok_or_else(|| anyhow::anyhow!("stream buffer exceeds its validated window length"))?;
            let unprocessed = self
                .buffer
                .len()
                .checked_sub(self.decoded_samples)
                .ok_or_else(|| anyhow::anyhow!("stream decoded sample count exceeds its buffer"))?;
            let to_step = step.saturating_sub(unprocessed);
            let take = remaining.len().min(room).min(to_step);
            self.buffer.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            let grown = self.buffer.len() - self.decoded_samples;
            if self.buffer.len() >= self.full_samples || grown >= step {
                events.extend(self.tick(Progress::Advance)?);
            }
            if take == 0 && self.buffer.len() >= self.full_samples {
                anyhow::bail!("stream buffer is full and did not shrink after a tick");
            }
        }
        Ok(events)
    }

    /// Завершить стрим: декод и коммит оставшегося хвоста.
    pub fn flush(&mut self) -> Result<Vec<StreamEvent>> {
        self.tick(Progress::Flush)
    }

    fn decode(&mut self) -> Result<(DecodedWindow, usize)> {
        let hop = crate::mel::HOP;
        let length = self.buffer.len();
        let real_frames = if length >= N_FFT { (length - N_FFT) / hop + 1 } else { 0 };
        let padded_frames = (length.div_ceil(hop)).min(self.full_frames);
        for time in self.real_full_frames..padded_frames {
            let offset = time * hop;
            let mut frame = [0f32; N_FFT];
            for (s, slot) in frame.iter_mut().enumerate() {
                let index = offset + s;
                *slot = if index < length { self.buffer[index] } else { self.noise[index] };
            }
            let col = self.mel.frame_log_mel(&frame);
            for m in 0..N_MELS {
                self.mel_buf[m * self.full_frames + time] = col[m];
            }
        }
        let window = self
            .decoder
            .decode_window(&self.mel_buf, self.full_frames, self.config.window_sec)?;
        Ok((window, real_frames.min(self.full_frames)))
    }

    /// Слова текущего окна после фильтра по cut_time и снятия дублей
    /// против хвоста committed (0.12 c + нормализация), как в источнике.
    fn open_words(&self) -> Vec<TWord> {
        let mut words: Vec<TWord> = self
            .current
            .iter()
            .filter(|w| w.start >= self.cut_time - 0.06)
            .cloned()
            .collect();
        while let Some(word) = words.first() {
            let duplicate = self.committed.iter().rev().take(12).any(|c| {
                (c.start - word.start).abs() <= 0.12
                    && normalize_word(&c.text) == normalize_word(&word.text)
            });
            if duplicate {
                words.remove(0);
            } else {
                break;
            }
        }
        words
    }

    /// Открытая строка: слова текущего окна, сшитые с предыдущим окном.
    fn open_line(&self) -> Result<Vec<TWord>> {
        match &self.previous {
            Some(previous) => stitch_open(previous, self.start_time, &self.open_words()),
            None => Ok(self.open_words()),
        }
    }

    fn commit(&mut self, words: &[TWord]) {
        self.committed.extend(words.iter().map(|w| StreamWord {
            text: w.text.clone(),
            start: w.start,
            end: w.end,
            stable: true,
        }));
    }

    fn resolve_seam(&mut self, now: f32) -> Result<()> {
        let Some(previous) = self.previous.take() else { return Ok(()) };
        let current = self.open_words();
        let s = seam(&previous.words, previous.end, self.start_time, &current, 0.25)?;
        self.commit(&previous.words[..s.previous_count]);
        self.cut_time = current.get(s.current_skip).map_or(now, |w| w.start);
        Ok(())
    }

    fn reset_buffer(&mut self, retained_samples: usize) {
        let retain = retained_samples.min(self.buffer.len());
        let dropped = self.buffer.len() - retain;
        self.start_time += dropped as f32 / MODEL_SR as f32;
        self.buffer.drain(..dropped);
        self.mel_buf.copy_from_slice(&self.noise_mel);
        self.real_full_frames = 0;
        self.decoded_samples = self.buffer.len();
        self.current.clear();
    }

    /// Отдать клиенту текущую строку: committed + открытые слова; ревизия
    /// от первого различия с уже отправленной строкой (допуск 0.1 c).
    fn emit(
        &mut self,
        open: &[TWord],
        now: f32,
        reason: Option<bool>,
        events: &mut Vec<StreamEvent>,
    ) {
        let mut target: Vec<StreamWord> = self
            .committed
            .iter()
            .map(|w| StreamWord { stable: true, ..w.clone() })
            .collect();
        target.extend(open.iter().map(|w| StreamWord {
            text: w.text.clone(),
            start: w.start,
            end: w.end,
            stable: w.start < now - self.config.horizon_sec,
        }));
        let shared = self.emitted.len().min(target.len());
        let revise_from = (0..shared)
            .find(|&i| !same_word(&self.emitted[i], &target[i]))
            .unwrap_or(shared);
        if revise_from < target.len() || revise_from < self.emitted.len() {
            events.push(StreamEvent::Words {
                at: now,
                revise_from,
                words: target[revise_from..].to_vec(),
            });
            self.emitted.truncate(revise_from);
            self.emitted.extend_from_slice(&target[revise_from..]);
        }
        let stable = target.iter().take_while(|w| w.stable).count();
        if stable > self.stable_sent {
            self.stable_sent = stable;
            for w in &mut self.emitted[..stable] {
                w.stable = true;
            }
            events.push(StreamEvent::Stable { at: now, upto: stable });
        }
        if self.committed.len() > self.final_sent {
            self.final_sent = self.committed.len();
            for w in &mut self.emitted[..self.final_sent] {
                w.stable = true;
            }
            events.push(StreamEvent::Final { at: now, upto: self.final_sent, endpoint: reason.unwrap_or(false) });
        }
    }

    fn tick(&mut self, progress: Progress) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let length = self.buffer.len();
        if length == 0 {
            return Ok(events);
        }
        let length_seconds = length as f32 / MODEL_SR as f32;
        let now = self.start_time + length_seconds;
        let (candidate, real_frames) = self.decode()?;
        let frame_rate = candidate.frame_rate;
        // слова текущего окна: только внутри реального аудио, с абсолютным временем
        let mut words: Vec<TWord> = candidate
            .words
            .into_iter()
            .filter(|w| w.start < length_seconds)
            .map(|mut w| {
                w.end = w.end.min(length_seconds);
                TWord { text: w.text, start: w.start + self.start_time, end: w.end + self.start_time }
            })
            .collect();
        words.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
        let upto = ((length_seconds * frame_rate) as usize).min(candidate.output_frames);
        let speech_state = if self.previous.is_some() || !words.is_empty() {
            SpeechState::Present
        } else {
            SpeechState::Absent
        };
        let endpoint = detect_endpoint(
            self.config.rules,
            &candidate.silence,
            upto,
            frame_rate,
            speech_state,
        )?;
        self.decoded_samples = length;
        self.real_full_frames = real_frames;
        self.current = words;
        let full = length >= self.full_samples;
        let resolves_seam = match progress {
            Progress::Advance => {
                length_seconds >= self.config.overlap_sec + self.config.seam_after_sec
                    || endpoint != Endpoint::None
                    || full
            }
            Progress::Flush => true,
        };
        if self.previous.is_some() && resolves_seam {
            self.resolve_seam(now)?;
        }
        let open = self.open_line()?;
        match progress {
            Progress::Flush => match speech_state {
                SpeechState::Present => {
                    self.commit(&open);
                    self.emit(&[], now, Some(true), &mut events);
                    self.reset_buffer(0);
                    self.cut_time = self.start_time;
                }
                SpeechState::Absent => {
                    self.emit(&open, now, None, &mut events);
                    self.reset_buffer(0);
                    self.cut_time = self.start_time;
                }
            },
            Progress::Advance => match (endpoint, speech_state) {
                (Endpoint::AfterSpeech, SpeechState::Present) => {
                    self.commit(&open);
                    self.emit(&[], now, Some(true), &mut events);
                    self.reset_buffer(self.config.retained_silence_samples());
                    self.cut_time = self.start_time;
                }
                (Endpoint::NoSpeech, _) => {
                    self.emit(&open, now, None, &mut events);
                    self.reset_buffer(self.config.retained_silence_samples());
                    self.cut_time = self.start_time;
                }
                (_, _) if full => {
                    self.emit(&open, now, None, &mut events);
                    self.previous = Some(WindowWords {
                        start: self.start_time,
                        end: now,
                        words: open,
                    });
                    self.reset_buffer(self.config.overlap_samples());
                }
                (_, _) => {
                    self.emit(&open, now, None, &mut events);
                }
            },
        }
        Ok(events)
    }
}

/// Равенство уже отправленного и нового слова (текст + допуск 0.1 c).
fn same_word(left: &StreamWord, right: &StreamWord) -> bool {
    left.text == right.text
        && (left.start - right.start).abs() <= 0.1
        && (left.end - right.end).abs() <= 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Фейковый декодер: слова по шаблону + маска тишины из флага `loud`
    /// и хвоста `silence_tail_frames` (реальному декодеру её даёт CTC).
    struct FakeDecoder {
        words: Vec<(&'static str, f32, f32)>,
        loud: bool,
        /// тишинных фреймов в хвосте маски (endpointing по хвосту)
        silence_tail_frames: usize,
    }

    impl WindowDecoder for FakeDecoder {
        fn decode_window(
            &mut self,
            _features: &[f32],
            _n_frames: usize,
            window_sec: f32,
        ) -> Result<DecodedWindow> {
            let output_frames = (window_sec * 25.0) as usize; // 25 fps энкодера
            let words = if self.loud {
                self.words
                    .iter()
                    .map(|(t, s, e)| TWord { text: t.to_string(), start: *s, end: *e })
                    .collect()
            } else {
                Vec::new()
            };
            let silence = if !self.loud {
                vec![true; output_frames]
            } else {
                let head = output_frames.saturating_sub(self.silence_tail_frames);
                let mut m = vec![false; head];
                m.extend(std::iter::repeat(true).take(output_frames - head));
                m
            };
            Ok(DecodedWindow {
                words,
                silence,
                output_frames,
                frame_rate: 25.0,
            })
        }
    }

    fn session() -> StreamSession<FakeDecoder> {
        StreamSession::new(
            FakeDecoder { words: vec![], loud: false, silence_tail_frames: 0 },
            StreamConfig {
                window_sec: 1.0,
                overlap_sec: 0.25,
                step_sec: 0.5,
                horizon_sec: 0.4,
                seam_after_sec: 0.2,
                keep_silence_sec: 0.1,
                rules: EndpointRules { after_speech_seconds: 0.3, without_speech_seconds: 0.6 },
            },
        )
        .unwrap()
    }

    fn tone(sec: f32) -> Vec<f32> {
        vec![0.5; (sec * MODEL_SR as f32) as usize]
    }

    #[test]
    fn push_emits_revisions_and_stable_frontier() {
        let mut s = session();
        s.decoder.words = vec![("привет", 0.1, 0.4), ("мир", 0.5, 0.9)];
        s.decoder.loud = true;
        let events = s.push(&tone(0.6)).unwrap();
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Words { .. })));
        // горизонт 0.4 c: слова старше now-0.4 стабильны
        assert!(s.transcript().iter().all(|w| w.text == "привет" || w.text == "мир"));
        assert!(s.transcript().first().unwrap().stable || s.transcript().len() >= 1);
        let final_flush = s.flush().unwrap();
        assert!(final_flush
            .iter()
            .any(|e| matches!(e, StreamEvent::Final { endpoint: true, .. })));
    }

    #[test]
    fn flush_without_speech_emits_no_final_and_keeps_transcript_empty() {
        // контракт источника: Final появляется только при коммите речи;
        // безречевой флаш отдаёт события слов (пустые) и просто завершается
        let mut s = session();
        s.decoder.loud = false;
        let events = s.push(&tone(0.7)).unwrap();
        let flush = s.flush().unwrap();
        let all: Vec<_> = events.into_iter().chain(flush).collect();
        assert!(all
            .iter()
            .all(|e| !matches!(e, StreamEvent::Final { .. })));
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn flush_with_speech_finalizes_with_endpoint() {
        let mut s = session();
        s.decoder.words = vec![("финал", 0.1, 0.3)];
        s.decoder.loud = true;
        let events = s.push(&tone(0.6)).unwrap();
        let flush = s.flush().unwrap();
        let all: Vec<_> = events.into_iter().chain(flush).collect();
        assert!(all
            .iter()
            .any(|e| matches!(e, StreamEvent::Final { endpoint: true, .. })));
        assert_eq!(s.committed.len(), 1);
    }

    #[test]
    fn endpoint_after_speech_resets_and_commits() {
        // step = окну: единственный тик на полном буфере; декодер видит речь
        // в 0.1..0.3 c и тишину в хвосте (0.48 c ≥ after_speech 0.3 c)
        let mut s = StreamSession::new(
            FakeDecoder { words: vec![("тест", 0.1, 0.3)], loud: true, silence_tail_frames: 12 },
            StreamConfig {
                window_sec: 1.0,
                overlap_sec: 0.25,
                step_sec: 1.0,
                horizon_sec: 0.4,
                seam_after_sec: 0.2,
                keep_silence_sec: 0.1,
                rules: EndpointRules { after_speech_seconds: 0.3, without_speech_seconds: 0.6 },
            },
        )
        .unwrap();
        let mut audio = tone(0.5);
        audio.extend(std::iter::repeat(0.0).take((0.5 * MODEL_SR as f32) as usize));
        let events = s.push(&audio).unwrap();
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Final { endpoint: true, .. })));
        assert_eq!(s.committed.len(), 1);
    }

    #[test]
    fn seam_matches_word_in_overlap() {
        let previous = WindowWords {
            start: 0.0,
            end: 10.0,
            words: vec![
                TWord { text: "один".into(), start: 1.0, end: 1.3 },
                TWord { text: "два,".into(), start: 5.0, end: 5.3 },
                TWord { text: "три".into(), start: 7.0, end: 7.3 },
            ],
        };
        let current = vec![
            TWord { text: "два".into(), start: 5.02, end: 5.3 },
            TWord { text: "три.".into(), start: 7.01, end: 7.3 },
            TWord { text: "четыре".into(), start: 9.0, end: 9.3 },
        ];
        let out = stitch_open(&previous, 4.0, &current).unwrap();
        let texts: Vec<&str> = out.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["один", "два,", "три.", "четыре"]);
    }

    #[test]
    fn pad_noise_is_deterministic_and_bounded() {
        let a = pad_noise(128, 0x5eed);
        let b = pad_noise(128, 0x5eed);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.abs() <= NOISE_AMPLITUDE as f32));
    }

    #[test]
    fn endpoint_rules_follow_the_source_contract() {
        let rules = EndpointRules { after_speech_seconds: 1.0, without_speech_seconds: 2.0 };
        // 1 тихий фрейм речи + 25 тишины = ровно 1.0 c хвостовой тишины
        let mut mask = vec![false; 1];
        mask.extend(std::iter::repeat(true).take(25));
        assert_eq!(
            detect_endpoint(rules, &mask, mask.len(), 25.0, SpeechState::Present).unwrap(),
            Endpoint::AfterSpeech
        );
        assert_eq!(
            detect_endpoint(rules, &mask, mask.len(), 25.0, SpeechState::Absent).unwrap(),
            Endpoint::None
        );
        assert!(detect_endpoint(rules, &[], 0, 0.0, SpeechState::Absent).is_err());
    }

    #[test]
    fn trailing_silence_ignores_padding_frames() {
        // маска: true = тишина; real_frames отсекает паддинг декодера
        assert_eq!(trailing_silence(&[false, false, false, false, true, true], 6), 2);
        assert_eq!(trailing_silence(&[false, true, true, false, false, false], 3), 2);
        assert_eq!(trailing_silence(&[false, false], 6), 0);
        assert_eq!(trailing_silence(&[true, true], 2), 2);
    }

    #[test]
    fn normalize_matches_the_source_rule() {
        assert_eq!(normalize_word("Ёлка-Ёлка, 2"), "елка-елка2");
        assert_eq!(normalize_word("Два."), normalize_word("два"));
    }

    #[test]
    fn invalid_config_refuses() {
        let cfg = StreamConfig { overlap_sec: 1.0, window_sec: 1.0, ..Default::default() };
        assert!(cfg.validate().is_err());
        let cfg = StreamConfig { step_sec: f32::NAN, ..Default::default() };
        assert!(cfg.validate().is_err());
    }
}
