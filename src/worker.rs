//! Воркеры инференса: один воркер = одна загруженная модель бэкенда.
//! Код перенесён из `main.rs`, чтобы CLI и сервер использовали общий пул.

use crate::ort_assignment::Policy;
use crate::stream::{StreamConfig, StreamSession, WindowDecoder};
use crate::{gigaam, gigaam_rnnt, models, parakeet, whisper};
use anyhow::Result;
use clap::ValueEnum;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModelKind {
    /// Whisper Large v3 Turbo (multilingual, ggml/whisper.cpp CUDA)
    WhisperTurbo,
    /// GigaAM v3 E2E CTC (Russian, ONNX Runtime CUDA, punctuation + ITN)
    GigaamV3,
    /// GigaAM v3 E2E RNN-T (Russian, ONNX Runtime CUDA, punctuation + ITN)
    GigaamV3Rnnt,
    /// Parakeet TDT 0.6B v3 (25 европейских языков, ONNX Runtime CUDA)
    ParakeetV3,
}

impl ModelKind {
    /// Строковый идентификатор модели (CLI/HTTP: `--model`, query `model`).
    pub fn id(self) -> &'static str {
        match self {
            ModelKind::WhisperTurbo => "whisper-turbo",
            ModelKind::GigaamV3 => "gigaam-v3",
            ModelKind::GigaamV3Rnnt => "gigaam-v3-rnnt",
            ModelKind::ParakeetV3 => "parakeet-v3",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "whisper-turbo" => Some(ModelKind::WhisperTurbo),
            "gigaam-v3" => Some(ModelKind::GigaamV3),
            "gigaam-v3-rnnt" => Some(ModelKind::GigaamV3Rnnt),
            "parakeet-v3" => Some(ModelKind::ParakeetV3),
            _ => None,
        }
    }
}

/// Backend used by a single inference worker.
pub enum Worker {
    Whisper(whisper::WhisperTurbo, whisper_rs::WhisperState),
    Gigaam(gigaam::GigaAmV3),
    GigaamRnnt(gigaam_rnnt::GigaAmRnnt),
    Parakeet(parakeet::ParakeetV3),
}

impl Worker {
    pub fn warmup(&mut self) -> Result<()> {
        let silence = vec![0.0f32; 16_000];
        let (segs, _) = self.transcribe(&silence, 0.0, "en", false, 1, 4)?;
        let _ = segs;
        Ok(())
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
        offset: f64,
        language: &str,
        translate: bool,
        beam_size: usize,
        n_threads: usize,
    ) -> Result<(Vec<crate::transcript::Segment>, Option<String>)> {
        match self {
            Worker::Whisper(model, state) => {
                let (segs, lang) = model.transcribe_chunk(
                    state,
                    samples,
                    offset,
                    &whisper::WhisperOpts { language, translate, beam_size, n_threads },
                )?;
                Ok((segs, Some(lang)))
            }
            Worker::Gigaam(g) => {
                let seg = g.transcribe_chunk(samples, offset)?;
                Ok((seg.map(|s| vec![s]).unwrap_or_default(), None))
            }
            Worker::GigaamRnnt(g) => {
                let seg = g.transcribe_chunk(samples, offset)?;
                Ok((seg.map(|s| vec![s]).unwrap_or_default(), None))
            }
            Worker::Parakeet(p) => {
                let seg = p.transcribe_chunk(samples, offset)?;
                Ok((seg.map(|s| vec![s]).unwrap_or_default(), None))
            }
        }
    }
}

/// `policy` — политика верификации CUDA-назначений (Фаза 4): сервер передаёт
/// разобранную политику, CLI — `None` (поведение CLI не меняется). Схема
/// покрывает роли CTC/RNNT; parakeet и whisper вне неё.
pub fn try_build_worker(
    paths: &models::ModelPaths,
    model: ModelKind,
    use_cuda: bool,
    intra: usize,
    dtw: bool,
    policy: Option<&Policy>,
) -> Result<Worker> {
    match model {
        ModelKind::WhisperTurbo => {
            let w = whisper::WhisperTurbo::load(&paths.whisper_turbo, use_cuda, dtw)?;
            let state = w.create_state()?;
            Ok(Worker::Whisper(w, state))
        }
        ModelKind::GigaamV3 => {
            let g = gigaam::GigaAmV3::load(
                &paths.gigaam_model,
                &paths.gigaam_vocab,
                use_cuda,
                intra,
                policy,
            )?;
            Ok(Worker::Gigaam(g))
        }
        ModelKind::GigaamV3Rnnt => {
            let g = gigaam_rnnt::GigaAmRnnt::load(&paths.gigaam_rnnt_dir, use_cuda, intra, policy)?;
            Ok(Worker::GigaamRnnt(g))
        }
        ModelKind::ParakeetV3 => {
            let p = parakeet::ParakeetV3::load(&paths.parakeet_dir, use_cuda, intra)?;
            Ok(Worker::Parakeet(p))
        }
    }
}

/// Auto worker count: CUDA — from free VRAM (NVML) and model size;
/// CPU — a single worker that uses all cores via intra-op threads.
pub fn auto_workers(use_cuda: bool, model: ModelKind) -> usize {
    if !use_cuda {
        return 1;
    }
    let free = nvml_wrapper::Nvml::init().ok().and_then(|nvml| {
        let dev = nvml.device_by_index(0).ok()?;
        let m = dev.memory_info().ok()?;
        Some(m.free)
    });
    let per_model: f64 = match model {
        ModelKind::WhisperTurbo => 2.2e9,
        // 1.5e9, not 1.3e9: 4 GigaAM sessions OOM on a 6 GB card (measured on
        // Linux with ~6.4 GB free); 3 is the documented optimum for 6 GB.
        ModelKind::GigaamV3 | ModelKind::GigaamV3Rnnt => 1.5e9,
        ModelKind::ParakeetV3 => 2.8e9,
    };
    match free {
        Some(free) => (((free as f64 - 0.8e9) / per_model).floor() as usize).clamp(1, 4),
        None => 2,
    }
}

// ---------------------------------------------------------------------------
// Стриминговая аренда (Фаза 6 PORTING.md)
// ---------------------------------------------------------------------------

/// CTC-декодер полного окна для StreamSession (владелец — GigaAM v3).
struct CtcStreamDecoder(gigaam::GigaAmV3);

impl WindowDecoder for CtcStreamDecoder {
    fn decode_window(
        &mut self,
        features: &[f32],
        n_frames: usize,
        window_sec: f32,
    ) -> Result<crate::stream::DecodedWindow> {
        self.0.decode_window(features, n_frames, window_sec)
    }
}

/// Построить стрим-сессию на воркере: модель уходит внутрь сессии до End.
fn begin_stream(
    worker: &mut Option<Worker>,
    config: StreamConfig,
) -> Result<StreamSession<CtcStreamDecoder>> {
    let g = match worker {
        Some(Worker::Gigaam(_)) => match worker.take() {
            Some(Worker::Gigaam(g)) => g,
            _ => unreachable!("проверено выше"),
        },
        Some(other) => {
            let _ = other;
            anyhow::bail!("текущая модель воркера не поддерживает стриминг (нужен GigaAM v3 CTC)");
        }
        None => anyhow::bail!("воркер без модели"),
    };
    let mut session = StreamSession::new(CtcStreamDecoder(g), config)?;
    session.warmup()?;
    Ok(session)
}

/// Цикл одного воркера: общая очередь batch-чанков + персональный канал
/// стриминговой аренды. Арендованный воркер обслуживает только свой стрим;
/// заявка на аренду подхватывается между чанками (poll 50 мс). При закрытии
/// очереди пула воркер дожидается завершения активной аренды и выходит.
pub fn run_worker_loop(
    w: Worker,
    rx: Arc<Mutex<mpsc::Receiver<crate::pipeline::QueueItem>>>,
    stream_rx: mpsc::Receiver<crate::pipeline::StreamMsg>,
    leased_flag: Arc<AtomicBool>,
    language: String,
    translate: bool,
    beam: usize,
    nthr: usize,
    metrics: Option<Arc<crate::metrics::Metrics>>,
) {
    let mut worker: Option<Worker> = Some(w);
    let mut leased: Option<StreamSession<CtcStreamDecoder>> = None;
    loop {
        // 1. арендован: эксклюзивное обслуживание стрим-сессии
        if let Some(session) = leased.as_mut() {
            match stream_rx.recv() {
                Ok(crate::pipeline::StreamMsg::Push { samples, reply }) => {
                    let t = Instant::now();
                    let res = session.push(&samples).map_err(|e| format!("{e:#}"));
                    let secs = t.elapsed().as_secs_f64();
                    crate::stats::add("инференс (стрим)", "stream", secs);
                    if let Some(m) = &metrics {
                        m.observe_inference(secs);
                    }
                    let _ = reply.send(res);
                }
                Ok(crate::pipeline::StreamMsg::Flush { reply }) => {
                    let res = session.flush().map_err(|e| format!("{e:#}"));
                    let _ = reply.send(res);
                }
                Ok(crate::pipeline::StreamMsg::Begin { .. }) => {
                    tracing::warn!("стрим: повторный Begin в арендованном воркере — игнорирую");
                }
                Ok(crate::pipeline::StreamMsg::End) | Err(_) => {
                    if let Some(session) = leased.take() {
                        worker = Some(Worker::Gigaam(session.into_decoder().0));
                    }
                    leased_flag.store(false, Ordering::SeqCst);
                }
            }
            continue;
        }

        // 2. заявка на аренду (флаг ставит lease_stream до отправки Begin):
        // выгребаем канал до Begin или пусто (могут быть устаревшие End)
        if leased_flag.load(Ordering::SeqCst) {
            let mut leased_now = false;
            loop {
                match stream_rx.try_recv() {
                    Ok(crate::pipeline::StreamMsg::Begin { config, reply }) => {
                        match begin_stream(&mut worker, config) {
                            Ok(session) => {
                                leased = Some(session);
                                // ответ после построения сессии и прогрева окна
                                let _ = reply.send(Ok(()));
                            }
                            Err(e) => {
                                leased_flag.store(false, Ordering::SeqCst);
                                let _ = reply.send(Err(format!("{e:#}")));
                            }
                        }
                        leased_now = leased.is_some();
                        break;
                    }
                    Ok(crate::pipeline::StreamMsg::Push { reply, .. })
                    | Ok(crate::pipeline::StreamMsg::Flush { reply, .. }) => {
                        let _ = reply.send(Err("stream push/flush before begin".into()));
                    }
                    Ok(crate::pipeline::StreamMsg::End) => {
                        // пустая аренда (арендатор умер до Begin) — ждём дальше
                    }
                    Err(_) => break,
                }
            }
            if leased_now {
                continue;
            }
        }

        // 3. общий очередь чанков; короткий тайм-аут, чтобы замечать заявки
        // на аренду (recv держит мьютекс очереди — как и раньше, один
        // ждущий воркер не мешает остальным: они блокируются на lock())
        let item = { rx.lock().unwrap().recv_timeout(Duration::from_millis(50)) };
        match item {
            Ok(item) => {
                // чанк отменённого запроса выбрасываем без инференса
                if item.cancel.cancelled() {
                    if let Some(m) = &metrics {
                        m.chunk_replied();
                    }
                    continue;
                }
                let t_inf = Instant::now();
                let res = worker
                    .as_mut()
                    .expect("не-арендованный воркер всегда держит модель")
                    .transcribe(
                        &item.chunk.samples,
                        item.chunk.start,
                        &language,
                        translate,
                        beam,
                        nthr,
                    );
                let secs = t_inf.elapsed().as_secs_f64();
                crate::stats::add("инференс", &item.chunk.stats_label, secs);
                if let Some(m) = &metrics {
                    m.observe_inference(secs);
                    m.chunk_replied();
                }
                let _ = item
                    .reply
                    .send(crate::pipeline::ChunkOutcome { chunk: item.chunk, result: res });
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // очередь пула закрыта: выходим, только если нет аренды;
                // активная аренда обслуживается до End (см. ветку 1)
                if !leased_flag.load(Ordering::SeqCst) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
