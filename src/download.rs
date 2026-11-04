//! Автозагрузка недостающих моделей (roadmap «Автозагрузка моделей с
//! HuggingFace»): таблица «файл в models/ → URL» + докачка с резюмом
//! (Range) и прогрессом. Запускается до `models::resolve` — манифесты
//! проверяются строго, поэтому файлы должны лежать на месте до верификации.
//!
//! Целостность: полная проверка всё равно идёт в resolve() по манифестам
//! (Фаза 3); здесь — только ранняя сверка sha256 свежескачанного файла,
//! если он входит в манифест своей семьи (файлы без манифеста — whisper,
//! parakeet — только предупреждают, проверить нечего).
//!
//! Что скачать нельзя: файлы `nemotron3-diarization/` генерируются один раз
//! скриптом `scripts/export_nemotron_diar.py` (нужен python с torch +
//! transformers + librosa) — при нехватке выдаётся подсказка, а не сетевая
//! ошибка. DeepFilterNet вшит в бинарник, ничего качать не нужно.

use anyhow::{bail, Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

/// Семейства моделей, которые можно запросить у автозагрузки.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Silero VAD (нужен конвейеру всегда)
    SileroVad,
    /// Whisper Large v3 Turbo (ggml)
    WhisperTurbo,
    /// GigaAM v3 CTC (дефолтная модель; она же — выравниватель `--w2v`)
    GigaamV3Ctc,
    /// GigaAM v3 RNN-T
    GigaamV3Rnnt,
    /// Parakeet TDT 0.6B v3
    ParakeetV3,
    /// Диаризация Nemotron — НЕ скачивается: генерируется локально скриптом
    NemotronDiar,
}

impl From<crate::worker::ModelKind> for Family {
    fn from(kind: crate::worker::ModelKind) -> Self {
        use crate::worker::ModelKind;
        match kind {
            ModelKind::WhisperTurbo => Family::WhisperTurbo,
            ModelKind::GigaamV3 => Family::GigaamV3Ctc,
            ModelKind::GigaamV3Rnnt => Family::GigaamV3Rnnt,
            ModelKind::ParakeetV3 => Family::ParakeetV3,
        }
    }
}

/// (семейство, путь от models/, URL). Локальные имена совпадают с именами
/// файлов в репозиториях — проверено range-запросами 0-0 по всем URL.
const SOURCES: &[(Family, &str, &str)] = &[
    (
        Family::WhisperTurbo,
        "ggml-large-v3-turbo.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    ),
    (
        Family::SileroVad,
        "silero_vad.onnx",
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    ),
    // gigaam-v3: CTC
    (
        Family::GigaamV3Ctc,
        "gigaam-v3/v3_e2e_ctc.onnx",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_ctc.onnx",
    ),
    (
        Family::GigaamV3Ctc,
        "gigaam-v3/v3_e2e_ctc.yaml",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_ctc.yaml",
    ),
    (
        Family::GigaamV3Ctc,
        "gigaam-v3/v3_e2e_ctc_vocab.txt",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_ctc_vocab.txt",
    ),
    // gigaam-v3-rnnt
    (
        Family::GigaamV3Rnnt,
        "gigaam-v3-rnnt/v3_e2e_rnnt_encoder.onnx",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_rnnt_encoder.onnx",
    ),
    (
        Family::GigaamV3Rnnt,
        "gigaam-v3-rnnt/v3_e2e_rnnt_decoder.onnx",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_rnnt_decoder.onnx",
    ),
    (
        Family::GigaamV3Rnnt,
        "gigaam-v3-rnnt/v3_e2e_rnnt_joint.onnx",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_rnnt_joint.onnx",
    ),
    (
        Family::GigaamV3Rnnt,
        "gigaam-v3-rnnt/v3_e2e_rnnt_vocab.txt",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_rnnt_vocab.txt",
    ),
    (
        Family::GigaamV3Rnnt,
        "gigaam-v3-rnnt/v3_e2e_rnnt.yaml",
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_rnnt.yaml",
    ),
    // parakeet-v3
    (
        Family::ParakeetV3,
        "parakeet-v3/encoder-model.onnx",
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx",
    ),
    (
        Family::ParakeetV3,
        "parakeet-v3/encoder-model.onnx.data",
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data",
    ),
    (
        Family::ParakeetV3,
        "parakeet-v3/decoder_joint-model.onnx",
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx",
    ),
    (
        Family::ParakeetV3,
        "parakeet-v3/nemo128.onnx",
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx",
    ),
    (
        Family::ParakeetV3,
        "parakeet-v3/vocab.txt",
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt",
    ),
    (
        Family::ParakeetV3,
        "parakeet-v3/config.json",
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/config.json",
    ),
];

/// Файлы диаризации: не распространяются, создаются export-скриптом.
const NEMOTRON_FILES: &[&str] = &[
    "nemotron3-diarization/diar_step.onnx",
    "nemotron3-diarization/mel_filters.f32",
    "nemotron3-diarization/silence_embeds.f32",
    "nemotron3-diarization/meta.json",
];

impl Family {
    /// Файлы семейства (пути от каталога models/).
    pub fn files(self) -> Vec<&'static str> {
        match self {
            Family::NemotronDiar => NEMOTRON_FILES.to_vec(),
            f => SOURCES
                .iter()
                .filter(|(x, ..)| *x == f)
                .map(|(_, rel, _)| *rel)
                .collect(),
        }
    }
}

const MAX_ATTEMPTS: usize = 4;
const PROGRESS_EVERY: Duration = Duration::from_secs(2);

/// Докачать недостающие файлы нужных семейств в `base`.
///
/// `enabled = false` — только диагностика: перечислить, чего не хватает,
/// и отказать с подсказкой (сеть не трогается). Файлы, которые уже есть,
/// не трогаются никогда; скачанное проверяется манифестом семьи, если он
/// этот файл описывает.
pub fn ensure(base: &Path, need: &[Family], enabled: bool) -> Result<()> {
    // Немотрон не скачивается: либо файлы сгенерированы, либо подсказка.
    if need.contains(&Family::NemotronDiar) {
        let missing: Vec<&str> = NEMOTRON_FILES
            .iter()
            .filter(|f| !base.join(f).is_file())
            .copied()
            .collect();
        if !missing.is_empty() {
            bail!(
                "диаризация: отсутствуют {} — эти файлы не распространяются и \
                 генерируются один раз скриптом scripts/export_nemotron_diar.py \
                 (нужен python с torch + transformers + librosa), см. README «Модели»",
                missing.join(", ")
            );
        }
    }

    let mut missing: Vec<&'static str> = Vec::new();
    let mut urls: Vec<&'static str> = Vec::new();
    for family in need {
        for (fam, rel, url) in SOURCES {
            if fam == family && !base.join(rel).is_file() {
                missing.push(rel);
                urls.push(url);
            }
        }
    }
    if missing.is_empty() {
        return Ok(());
    }

    if !enabled {
        bail!(
            "отсутствуют файлы моделей: {}; автозагрузка отключена — уберите \
             --no-download-models или скачайте вручную (README «Модели»)",
            missing.join(", ")
        );
    }

    tracing::info!(
        "нет {} файлов моделей — скачиваю в {} (повторный запуск продолжит недокачанное)",
        missing.len(),
        base.display()
    );
    let agent = agent();
    for (rel, url) in missing.iter().zip(urls.iter()) {
        let dest = base.join(rel);
        download_file(&agent, url, &dest)
            .with_context(|| format!("скачивание {rel} ({url})"))?;
        verify_against_manifest(base, rel)?;
    }
    Ok(())
}

/// Ранняя сверка свежескачанного файла с манифестом его каталога, если
/// манифест описывает этот файл. Полная проверка — в `models::resolve`.
fn verify_against_manifest(base: &Path, rel: &str) -> Result<()> {
    let path = base.join(rel);
    let dir = path.parent().unwrap_or(base);
    let Some(m) = crate::manifest::load_for(dir) else {
        tracing::warn!("у {rel} нет манифеста — sha256 не проверяется");
        return Ok(());
    };
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("имя файла без юникода")?;
    let Some(entry) = m.files.iter().find(|e| e.path == name) else {
        tracing::warn!("файла {rel} нет в манифесте — sha256 не проверяется");
        return Ok(());
    };
    let actual = crate::manifest::sha256_file(&path)?;
    if actual != entry.sha256.to_lowercase() {
        let _ = std::fs::remove_file(&path);
        bail!(
            "{rel}: sha256 {actual} не совпал с манифестом ({}), файл удалён — \
             запустите ещё раз для повторного скачивания",
            entry.sha256
        );
    }
    Ok(())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .user_agent("lighttrans")
        .build()
}

/// Скачать `url` в `dest` через временный `<dest>.part` (резюм по Range
/// между попытками и запусками), затем атомарно переименовать.
fn download_file(agent: &ureq::Agent, url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("создание каталога {}", dir.display()))?;
    }
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name().and_then(|s| s.to_str()).unwrap_or("file")
    ));
    let mut last_err = String::new();
    for attempt in 1..=MAX_ATTEMPTS {
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if attempt > 1 {
            tracing::warn!("повтор {attempt}/{MAX_ATTEMPTS} для {url} (с {} МБ)", mb(have));
        }
        match attempt_once(agent, url, &part, have) {
            Ok(total) => {
                std::fs::rename(&part, &dest).with_context(|| {
                    format!("переименование {} → {}", part.display(), dest.display())
                })?;
                tracing::info!("скачано {} ({} МБ)", dest.display(), mb(total));
                return Ok(());
            }
            Err(e) => last_err = format!("{e:#}"),
        }
    }
    bail!("не удалось скачать {url} за {MAX_ATTEMPTS} попыток: {last_err}")
}

/// Одна попытка: запрос (с Range от `have`), потоковое чтение в .part,
/// прогресс в лог. Возвращает итоговый размер файла. Часть сохраняется
/// при любой ошибке — следующая попытка продолжит с её размера.
fn attempt_once(agent: &ureq::Agent, url: &str, part: &Path, have: u64) -> Result<u64> {
    let mut req = agent.get(url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // часть уже размером с файл (или битая длина Range) — начать заново
        Err(ureq::Error::Status(416, _)) if have > 0 => {
            let _ = std::fs::remove_file(part);
            return attempt_once(agent, url, part, 0);
        }
        Err(ureq::Error::Status(code, _)) => bail!("HTTP {code}"),
        Err(e) => return Err(anyhow::anyhow!("{e}")),
    };
    // 206 — докачиваем с have; 200 — сервер Range проигнорировал (или часть
    // не существовала), пишем с нуля
    let from = match resp.status() {
        206 => have,
        200 => 0,
        code => bail!("неожиданный HTTP {code} (ожидался 200/206)"),
    };
    let total: Option<u64> = resp
        .header("content-length")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|rem| from + rem);
    if from == 0 {
        let _ = std::fs::remove_file(part);
    }
    if let Some(t) = total {
        tracing::info!("скачивание {url} → {} ({} МБ)", part.display(), mb(t));
    } else {
        tracing::info!("скачивание {url} → {} (размер неизвестен)", part.display());
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(part)
        .with_context(|| format!("открытие {}", part.display()))?;
    if from == 0 {
        file.set_len(0).with_context(|| format!("усечение {}", part.display()))?;
    }
    file.seek(SeekFrom::End(0))
        .with_context(|| format!("позиция в {}", part.display()))?;

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1024 * 1024];
    let mut written: u64 = 0;
    let mut last_log = Instant::now();
    loop {
        let n = reader.read(&mut buf).context("чтение тела ответа")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| format!("запись {}", part.display()))?;
        written += n as u64;
        if last_log.elapsed() >= PROGRESS_EVERY {
            last_log = Instant::now();
            match total {
                Some(t) => tracing::info!(
                    "  {} / {} МБ ({}%)",
                    mb(from + written),
                    mb(t),
                    (from + written) * 100 / t.max(1)
                ),
                None => tracing::info!("  {} МБ", mb(written)),
            }
        }
    }
    file.flush().context("flush .part")?;
    // сервер молча оборвал соединение — недокачанное увидим по размеру
    if let Some(t) = total {
        if from + written < t {
            bail!("соединение оборвалось на {} из {} МБ", mb(from + written), mb(t));
        }
    }
    Ok(from + written)
}

fn mb(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_cover_families_with_sane_paths_and_urls() {
        for family in [
            Family::SileroVad,
            Family::WhisperTurbo,
            Family::GigaamV3Ctc,
            Family::GigaamV3Rnnt,
            Family::ParakeetV3,
        ] {
            assert!(!family.files().is_empty(), "{family:?}: нет файлов в SOURCES");
        }
        for (_, rel, url) in SOURCES {
            assert!(!rel.starts_with('/') && !rel.contains("..") && !rel.contains('\\'), "{rel}");
            assert!(url.starts_with("https://"), "{url}");
            assert!(url.contains("huggingface.co") || url.contains("github.com"), "{url}");
        }
        // немотрон описан отдельно и в таблице источников отсутствует
        assert!(SOURCES.iter().all(|(f, ..)| *f != Family::NemotronDiar));
        assert_eq!(Family::NemotronDiar.files(), NEMOTRON_FILES.to_vec());
    }

    #[test]
    fn family_matches_model_kind() {
        use crate::worker::ModelKind;
        assert_eq!(Family::from(ModelKind::GigaamV3), Family::GigaamV3Ctc);
        assert_eq!(Family::from(ModelKind::WhisperTurbo), Family::WhisperTurbo);
        assert_eq!(Family::from(ModelKind::GigaamV3Rnnt), Family::GigaamV3Rnnt);
        assert_eq!(Family::from(ModelKind::ParakeetV3), Family::ParakeetV3);
    }

    #[test]
    fn ensure_lists_missing_without_network_when_disabled() {
        let dir = std::env::temp_dir().join("lt-download-disabled");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let err = format!(
            "{:#}",
            ensure(&dir, &[Family::SileroVad], false).unwrap_err()
        );
        assert!(err.contains("silero_vad.onnx"), "{err}");
        assert!(err.contains("--no-download-models"), "{err}");
        // при включённой автозагрузке здесь пошла бы сеть — проверяем только
        // ветку «файл уже есть» (качать нечего)
        std::fs::write(dir.join("silero_vad.onnx"), b"whatever").unwrap();
        assert!(ensure(&dir, &[Family::SileroVad], false).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Реальное скачивание мелкой модели и сверка sha с манифестом.
    #[test]
    #[ignore = "нужна сеть"]
    fn downloads_silero_and_matches_manifest() {
        let dir = std::env::temp_dir().join("lt-download-silero");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        ensure(&dir, &[Family::SileroVad], true).unwrap();
        let sha = crate::manifest::sha256_file(&dir.join("silero_vad.onnx")).unwrap();
        assert_eq!(sha, "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// «Тяжёлый» прогон пути HuggingFace (resolve → CDN) на семье со
    /// строгим манифестом: 886 МБ CTC + ранняя sha-сверка в ensure().
    #[test]
    #[ignore = "нужна сеть и ~886 МБ трафика"]
    fn downloads_gigaam_ctc_and_matches_manifest() {
        let dir = std::env::temp_dir().join("lt-download-gigaam");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        ensure(&dir, &[Family::GigaamV3Ctc], true).unwrap();
        let sha = crate::manifest::sha256_file(&dir.join("gigaam-v3/v3_e2e_ctc.onnx")).unwrap();
        assert_eq!(sha, "377701bd33568f4733feec2db5b2dc12544fd09a5a5dfa69ccf55d161f84027a");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
