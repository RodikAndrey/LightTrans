//! Потоковая сборка конвейера: VAD+чанкер на канал, выбор каналов (dual-mono)
//! и пул GPU-воркеров с очередью чанков. Общий код CLI и сервера.
//!
//! Инвариант порядка: результаты собираются строго по ключу
//! (file_idx, channel, idx) — порядок слов и тайминги не зависят от числа
//! воркеров и количества параллельных запросов.

use crate::channels::{channel_correlation, select_channels, ChannelPlan};
use crate::segments::{Chunk, Chunker};
use crate::transcript::Segment;
use crate::vad::SileroVad;
use crate::worker::Worker;
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

/// Режим обработки каналов входного потока.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
    /// Обычный mono-путь: декод уже смикширован (n_channels = 1).
    Mix,
    /// Каждый канал — отдельный спикер: раздельные VAD/чанкер/распознавание,
    /// dual-mono схлопывается в канал 0.
    Split,
}

/// Окно выбора каналов в фреймах на канал (≈30 c @16 кГц): за это время
/// набираем данные для корреляции dual-mono, дальше стримим по каналам.
pub const SELECTION_WINDOW_FRAMES: usize = 30 * 16_000;

/// Размер блока декодирования в фреймах для многоканального пути (100 мс).
pub const SPLIT_BLOCK_FRAMES: usize = 1_600;

/// Флаг кооперативной отмены (тайм-аут/разрыв HTTP-запроса): продюсер
/// перестаёт резать новые чанки, воркеры выбрасывают чанки отменённого
/// запроса. Уже идущий нативный вызов не прерывается.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Результат одного чанка, отправленный воркером обратно в свой запрос.
/// Сэмплы едут вместе с чанком (нужны фазе --w2v); сервер их сразу сбрасывает.
pub struct ChunkOutcome {
    pub chunk: Chunk,
    pub result: Result<(Vec<Segment>, Option<String>)>,
}

/// Элемент очереди воркеров: чанк + контекст запроса.
pub struct QueueItem {
    pub chunk: Chunk,
    pub cancel: CancelToken,
    pub reply: mpsc::Sender<ChunkOutcome>,
}

/// Параметры воркера на время жизни пула.
pub struct WorkerOpts {
    pub language: String,
    pub translate: bool,
    pub beam_size: usize,
    pub n_threads: usize,
}

/// Пул GPU-воркеров с общей очередью чанков. Ответы развозятся по
/// per-request каналам (`QueueItem.reply`), поэтому параллельные запросы
/// не смешивают результаты.
pub struct WorkerPool {
    chunk_tx: Mutex<Option<mpsc::Sender<QueueItem>>>,
    _handles: Vec<std::thread::JoinHandle<()>>,
}

impl WorkerPool {
    /// Запускает потоки-воркеры. Воркер, упавший с паникой, просто выходит:
    /// его чанки разбирают остальные, полная гибель пула видна как закрытие
    /// очереди (`submit` вернёт `false`).
    pub fn start(mut workers: Vec<Worker>, opts: WorkerOpts) -> Self {
        let (chunk_tx, chunk_rx) = mpsc::channel::<QueueItem>();
        let chunk_rx = Arc::new(Mutex::new(chunk_rx));
        let mut handles = Vec::with_capacity(workers.len());
        while let Some(mut w) = workers.pop() {
            let rx = Arc::clone(&chunk_rx);
            let language = opts.language.clone();
            let translate = opts.translate;
            let beam = opts.beam_size;
            let nthr = opts.n_threads;
            handles.push(std::thread::spawn(move || loop {
                let item = { rx.lock().unwrap().recv() };
                let Ok(item) = item else { break };
                // чанк отменённого запроса выбрасываем без инференса
                if item.cancel.cancelled() {
                    continue;
                }
                let t_inf = Instant::now();
                let res = w.transcribe(
                    &item.chunk.samples,
                    item.chunk.start,
                    &language,
                    translate,
                    beam,
                    nthr,
                );
                crate::stats::add(
                    "инференс",
                    &item.chunk.stats_label,
                    t_inf.elapsed().as_secs_f64(),
                );
                let _ = item.reply.send(ChunkOutcome { chunk: item.chunk, result: res });
            }));
        }
        Self { chunk_tx: Mutex::new(Some(chunk_tx)), _handles: handles }
    }

    /// Поставить чанк в очередь. `false` — пул закрыт (все воркеры умерли).
    pub fn submit(&self, item: QueueItem) -> bool {
        match self.chunk_tx.lock().unwrap().as_ref() {
            Some(tx) => tx.send(item).is_ok(),
            None => false,
        }
    }

    /// Число воркеров пула (для /readyz).
    pub fn len(&self) -> usize {
        self._handles.len()
    }

    /// Закрыть очередь: воркеры доработают активные чанки и завершатся.
    pub fn close(&self) {
        self.chunk_tx.lock().unwrap().take();
    }
}

/// VAD + чанкер одного канала входа.
pub struct ChunkSource {
    vad: Option<SileroVad>,
    chunker: Chunker,
}

impl ChunkSource {
    pub fn load_vad(vad_path: &std::path::Path, onset: f32, offset: f32) -> Result<SileroVad> {
        Ok(SileroVad::load(vad_path, 2)?.with_thresholds(onset, offset))
    }

    pub fn new(
        file_idx: usize,
        channel: usize,
        stats_label: String,
        chunk_size: f64,
        no_vad: bool,
        vad: Option<SileroVad>,
    ) -> Self {
        Self {
            vad,
            chunker: Chunker::new(file_idx, channel, stats_label, chunk_size, no_vad),
        }
    }

    /// Подать блок mono-сэмплов канала (16 кГц); готовые чанки — в `out`.
    pub fn feed(&mut self, block: &[f32], out: &mut Vec<Chunk>) -> Result<()> {
        let mut intervals = Vec::new();
        if let Some(v) = self.vad.as_mut() {
            v.feed(block, &mut intervals).context("VAD inference")?;
        }
        self.push(intervals, block, out)
    }

    /// Завершить поток канала: закрыть открытый интервал VAD и флаш чанкера.
    pub fn finish(&mut self, out: &mut Vec<Chunk>) {
        let mut intervals = Vec::new();
        if let Some(v) = self.vad.as_mut() {
            v.finish(&mut intervals);
        }
        let _ = self.push(intervals, &[], out);
        self.chunker.finish(out);
    }

    fn push(
        &mut self,
        intervals: Vec<crate::vad::VadSegment>,
        block: &[f32],
        out: &mut Vec<Chunk>,
    ) -> Result<()> {
        let secs: Vec<(f64, f64)> = intervals.iter().map(|s| (s.start, s.end)).collect();
        let mut ready = Vec::new();
        self.chunker.feed(block, &secs, &mut ready);
        out.extend(ready);
        Ok(())
    }
}

/// Потоковая обработка одного входа: интерливнутые блоки → деинтерливинг →
/// (выбор каналов) → VAD/чанкер на канал → чанки. Поток — 16 кГц.
///
/// В Split-режиме первые `SELECTION_WINDOW_FRAMES` фреймов каждого канала
/// буферизуются; по окну (или в `finish` для коротких входов) считается
/// корреляция и фиксируется `ChannelPlan`, буфер раздаётся источникам.
pub struct MultiStream {
    mode: ChannelMode,
    file_idx: usize,
    chunk_size: f64,
    no_vad: bool,
    vad_path: std::path::PathBuf,
    vad_onset: f32,
    vad_offset: f32,
    label_base: String,
    n_channels: usize,
    /// Собирать mono-поток целиком (диаризация в CLI).
    collect_mono: bool,
    mono: Vec<f32>,
    window: Vec<Vec<f32>>,
    window_filled: usize,
    plan: Option<ChannelPlan>,
    sources: Vec<ChunkSource>,
}

impl MultiStream {
    pub fn new(
        mode: ChannelMode,
        file_idx: usize,
        n_channels: usize,
        label_base: String,
        chunk_size: f64,
        no_vad: bool,
        vad_path: std::path::PathBuf,
        vad_onset: f32,
        vad_offset: f32,
    ) -> Self {
        let channels = match mode {
            ChannelMode::Mix => 1,
            ChannelMode::Split => n_channels.max(1),
        };
        Self {
            mode,
            file_idx,
            chunk_size,
            no_vad,
            vad_path,
            vad_onset,
            vad_offset,
            label_base,
            n_channels: channels,
            collect_mono: false,
            mono: Vec::new(),
            window: (0..channels).map(|_| Vec::new()).collect(),
            window_filled: 0,
            plan: None,
            sources: Vec::new(),
        }
    }

    /// Включить сборку mono-потока (для диаризации).
    pub fn with_mono_collection(mut self) -> Self {
        self.collect_mono = true;
        self
    }

    pub fn plan(&self) -> Option<&ChannelPlan> {
        self.plan.as_ref()
    }

    /// Собранный mono-поток (диаризация); забирает буфер.
    pub fn take_mono(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.mono)
    }

    /// Подать интерливнутый блок (n_channels декодера × фреймы);
    /// готовые чанки — в `out`. Поток — 16 кГц mono (Mix) или интерливнутый
    /// многоканальный (Split).
    pub fn feed(&mut self, interleaved: &[f32], out: &mut Vec<Chunk>) -> Result<()> {
        match self.mode {
            ChannelMode::Mix => {
                if self.collect_mono {
                    self.mono.extend_from_slice(interleaved);
                }
                if self.plan.is_none() {
                    self.resolve_plan(out)?;
                }
                if let Some(src) = self.sources.first_mut() {
                    src.feed(interleaved, out)?;
                }
            }
            ChannelMode::Split => {
                let n = self.n_channels;
                let frames = interleaved.len() / n.max(1);
                if self.plan.is_none() {
                    // окно выбора: буферизуем каждый исходный канал
                    let take = frames.min(SELECTION_WINDOW_FRAMES.saturating_sub(self.window_filled));
                    for (ch, buf) in self.window.iter_mut().enumerate() {
                        buf.reserve(take);
                        for f in 0..take {
                            buf.push(interleaved[f * n + ch]);
                        }
                    }
                    self.window_filled += take;
                    let rest = frames - take;
                    if rest == 0 {
                        return Ok(());
                    }
                    // окно набрано — фиксируем план и дожёвываем хвост блока
                    let tail: Vec<f32> = interleaved[take * n..].to_vec();
                    self.resolve_plan(out)?;
                    return self.feed(&tail, out);
                }
                // обычный стрим: деинтерливинг в канальные источники
                let selected = self
                    .plan
                    .as_ref()
                    .map(|p| p.selected.clone())
                    .unwrap_or_else(|| (0..self.sources.len()).collect());
                let mut bufs: Vec<Vec<f32>> = vec![Vec::with_capacity(frames); selected.len()];
                for (i, &ch) in selected.iter().enumerate() {
                    let buf = &mut bufs[i];
                    buf.reserve(frames);
                    for f in 0..frames {
                        buf.push(interleaved[f * n + ch]);
                    }
                }
                for (i, buf) in bufs.into_iter().enumerate() {
                    self.sources[i].feed(&buf, out)?;
                }
            }
        }
        Ok(())
    }

    /// Завершить поток: закрыть окно выбора (короткий вход), флаш источников.
    pub fn finish(&mut self, out: &mut Vec<Chunk>) -> Result<()> {
        if self.plan.is_none() {
            self.resolve_plan(out)?;
        }
        for src in &mut self.sources {
            src.finish(out);
        }
        Ok(())
    }

    /// Зафиксировать план выбора каналов и создать канальные источники,
    /// раздав им буферизованное окно (готовые чанки окна — в `out`).
    fn resolve_plan(&mut self, out: &mut Vec<Chunk>) -> Result<()> {
        let corr01 = if self.n_channels == 2 && self.window_filled > 0 {
            let (a, b) = (&self.window[0], &self.window[1]);
            if a.len() == b.len() && !a.is_empty() {
                match channel_correlation(a, b) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::warn!("корреляция каналов не удалась, считаю каналы разными: {e:#}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        let plan = select_channels(self.n_channels, corr01);
        if plan.dual_mono {
            tracing::info!(
                "dual-mono: каналы совпадают (корреляция {:.4}), распознаю только канал 0",
                corr01.unwrap_or(0.0)
            );
        } else if self.mode == ChannelMode::Split && plan.selected.len() > 1 {
            tracing::info!(
                "многоканальный вход: {} каналов как отдельные спикеры",
                plan.selected.len()
            );
        }
        let multi = self.mode == ChannelMode::Split && plan.selected.len() > 1;
        for &ch in &plan.selected {
            let label = if multi {
                format!("{}#ch{ch}", self.label_base)
            } else {
                self.label_base.clone()
            };
            let vad = if self.no_vad {
                None
            } else {
                Some(
                    ChunkSource::load_vad(&self.vad_path, self.vad_onset, self.vad_offset)
                        .with_context(|| format!("загрузка VAD для канала {ch}"))?,
                )
            };
            let mut src = ChunkSource::new(self.file_idx, ch, label, self.chunk_size, self.no_vad, vad);
            // раздаём накопленное окно (готовые чанки окна — в out)
            let buf = std::mem::take(&mut self.window[ch]);
            if !buf.is_empty() {
                src.feed(&buf, out)?;
            }
            self.sources.push(src);
        }
        self.window.clear();
        self.plan = Some(plan);
        Ok(())
    }
}
