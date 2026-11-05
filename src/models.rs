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

fn candidate_dirs(explicit: Option<&Path>) -> Vec<PathBuf> {
    match explicit {
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
    }
}

fn has_any_model(p: &Path) -> bool {
    p.join(WHISPER_TURBO_FILE).is_file()
        || p.join(GIGAAM_DIR).join(GIGAAM_MODEL_FILE).is_file()
        || p.join(SILERO_VAD_FILE).is_file()
}

/// Каталог models/, в который автозагрузка положит недостающее: как в
/// `resolve` — первый кандидат с любой моделью, иначе первый кандидат
/// (обычно ./models). Вызывается ДО resolve, чтобы файлы успели появиться
/// до строгой проверки манифестов.
pub fn base_dir(explicit: Option<&Path>) -> PathBuf {
    let candidates = candidate_dirs(explicit);
    candidates
        .iter()
        .find(|p| has_any_model(p))
        .cloned()
        .unwrap_or_else(|| candidates[0].clone())
}

/// Resolve model paths: `--models-dir` wins, otherwise look next to the project
/// (./models) or next to the executable.
pub fn resolve(explicit: Option<&Path>) -> anyhow::Result<ModelPaths> {
    let candidates = candidate_dirs(explicit);
    let base = candidates
        .iter()
        .find(|p| has_any_model(p))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "models not found. Expected {} and/or {}/{}/{}.\n\
                 Pass --models-dir or download models into ./models (see README).\n\
                 Auto-download runs on startup unless --no-download-models is passed.",
                WHISPER_TURBO_FILE, GIGAAM_DIR, GIGAAM_MODEL_FILE, GIGAAM_VOCAB_FILE
            )
        })?;

    // Фаза 3: манифесты моделей (если есть) проверяются до любой загрузки —
    // битые/подменённые файлы отсекаются с именем файла и расхождением.
    // Отсутствующие файлы семьи не отказывают (в git манифесты без весов):
    // присутствующие проверяются строго, недостающее докачивает автозагрузка
    // перед использованием семьи.
    let family_dirs = [
        base.clone(),
        base.join(GIGAAM_DIR),
        base.join(GIGAAM_RNNT_DIR),
        base.join("nemotron3-diarization"),
        base.join("parakeet-v3"),
        base.join("wav2vec2-align-ru"),
    ];
    // один кэш верификации на все семьи: грузим до проверки, сохраняем после
    let cache_path = base.join(crate::manifest::VERIFY_CACHE_FILE);
    let mut cache = crate::manifest::load_verify_cache(&cache_path);
    let cache_before = cache.clone();
    for dir in &family_dirs {
        verify_family(dir, &base, &mut cache)?;
    }
    if cache != cache_before {
        crate::manifest::save_verify_cache(&cache_path, &cache);
    }

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

/// Проверка манифеста одной семьи (если он там есть): присутствующие файлы —
/// строго (размер + sha256), отсутствующие — предупреждение. Полностью
/// нескачанная семья не должна валить чужие запуски: она либо будет
/// докачана автозагрузкой до использования, либо откажет уже на загрузке
/// сессии с понятным именем файла.
fn verify_family(dir: &Path, base: &Path, cache: &mut crate::manifest::VerifyCache) -> anyhow::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let Some(m) = crate::manifest::load_for(dir) else {
        return Ok(());
    };
    // кэш верификации: перечитывать ~7 ГБ весов под sha256 на каждый запуск
    // не нужно — хеш пересчитывается только после изменения файла или манифеста
    let prefix = dir
        .strip_prefix(base)
        .unwrap_or(dir)
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let missing = m.verify_files_cached(dir, &prefix, cache)?;
    if !missing.is_empty() {
        tracing::warn!(
            "манифест {}: отсутствуют файлы {:?} — семья установлена частично; \
             автозагрузка докачает её при использовании",
            dir.join(crate::manifest::MANIFEST_FILE).display(),
            missing
        );
    }
    Ok(())
}
