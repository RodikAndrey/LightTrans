//! Воркеры инференса: один воркер = одна загруженная модель бэкенда.
//! Код перенесён из `main.rs`, чтобы CLI и сервер использовали общий пул.

use crate::ort_assignment::Policy;
use crate::{gigaam, gigaam_rnnt, models, parakeet, whisper};
use anyhow::Result;
use clap::ValueEnum;

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
