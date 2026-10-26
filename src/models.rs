//! Model file locations.

use std::path::{Path, PathBuf};

pub const WHISPER_TURBO_FILE: &str = "ggml-large-v3-turbo.bin";
pub const GIGAAM_DIR: &str = "gigaam-v3";
pub const GIGAAM_MODEL_FILE: &str = "v3_e2e_ctc.onnx";
pub const GIGAAM_VOCAB_FILE: &str = "v3_e2e_ctc_vocab.txt";
pub const GIGAAM_RNNT_DIR: &str = "gigaam-v3-rnnt";
pub const SILERO_VAD_FILE: &str = "silero_vad.onnx";

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub whisper_turbo: PathBuf,
    pub gigaam_model: PathBuf,
    pub gigaam_vocab: PathBuf,
    pub gigaam_rnnt_dir: PathBuf,
    pub silero_vad: PathBuf,
    pub nemotron_diar: PathBuf,
    pub parakeet_dir: PathBuf,
}

/// Resolve model paths: `--models-dir` wins, otherwise look next to the project
/// (./models) or next to the executable.
pub fn resolve(explicit: Option<&Path>) -> anyhow::Result<ModelPaths> {
    let candidates: Vec<PathBuf> = match explicit {
        Some(p) => vec![p.to_path_buf()],
        None => {
            let mut v = vec![PathBuf::from("models")];
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    v.push(dir.join("models"));
                    // walk up a couple of levels (target/release -> project root)
                    let mut p = dir.to_path_buf();
                    for _ in 0..3 {
                        p = p.parent().map(|x| x.to_path_buf()).unwrap_or(p.clone());
                        v.push(p.join("models"));
                    }
                }
            }
            v
        }
    };

    let base = candidates
        .iter()
        .find(|p| p.join(WHISPER_TURBO_FILE).is_file() || p.join(GIGAAM_DIR).join(GIGAAM_MODEL_FILE).is_file())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "models not found. Expected {} and/or {}/{}/{}.\n\
                 Pass --models-dir or download models into ./models (see README).",
                WHISPER_TURBO_FILE, GIGAAM_DIR, GIGAAM_MODEL_FILE, GIGAAM_VOCAB_FILE
            )
        })?;

    let gigaam_dir = base.join(GIGAAM_DIR);
    Ok(ModelPaths {
        whisper_turbo: base.join(WHISPER_TURBO_FILE),
        gigaam_model: gigaam_dir.join(GIGAAM_MODEL_FILE),
        gigaam_vocab: gigaam_dir.join(GIGAAM_VOCAB_FILE),
        gigaam_rnnt_dir: base.join(GIGAAM_RNNT_DIR),
        silero_vad: base.join(SILERO_VAD_FILE),
        nemotron_diar: base.join("nemotron3-diarization"),
        parakeet_dir: base.join("parakeet-v3"),
    })
}
