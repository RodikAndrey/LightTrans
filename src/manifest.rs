//! Манифесты моделей (Фаза 3 PORTING.md, идея из crates/model-package
//! источника): для каталога семейства `manifest.json` — список файлов
//! (sha256 + размер) и ожидаемые IO-имена ONNX-графов.
//!
//! Правило: манифеста нет — загрузка идёт как раньше (совместимость);
//! манифест есть — проверка строгая, несоответствие = отказ старта с именем
//! файла и расхождением. Рассинхрон имён тензоров иначе ловится непонятной
//! ошибкой ORT в рантайме; битые LFS-загрузки давали 29-байтовые «onnx».

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Deserialize)]
pub struct ModelManifest {
    #[serde(default)]
    pub files: Vec<FileEntry>,
    /// Ожидаемые IO-имена ONNX-графа (проверяются после создания сессии).
    #[serde(default)]
    pub io: Option<IoSpec>,
}

#[derive(Debug, Deserialize)]
pub struct FileEntry {
    /// Путь относительно каталога семейства.
    pub path: String,
    pub sha256: String,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct IoSpec {
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
}

/// Загрузить манифест каталога; `None` — манифеста нет (проверять нечего).
pub fn load_for(dir: &Path) -> Option<ModelManifest> {
    let path = dir.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str(&text) {
        Ok(m) => Some(m),
        Err(e) => {
            // битый манифест — это не «нет манифеста»: отказ, а не пропуск
            tracing::error!("битый манифест {}: {e}", path.display());
            panic!("битый манифест модели {}: {e}", path.display());
        }
    }
}

/// SHA-256 файла (стриминг, чтобы не держать гигабайты в памяти).
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("не открыть {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).with_context(|| format!("чтение {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

impl ModelManifest {
    /// Проверить все файлы манифеста: существование, размер, sha256.
    /// Ошибка называет файл, ожидаемое и фактическое.
    pub fn verify_files(&self, dir: &Path) -> Result<()> {
        for entry in &self.files {
            let path = dir.join(&entry.path);
            let meta = std::fs::metadata(&path).with_context(|| {
                format!(
                    "манифест {}: файл отсутствует: {}",
                    dir.join(MANIFEST_FILE).display(),
                    path.display()
                )
            })?;
            if let Some(size) = entry.size {
                if meta.len() != size {
                    bail!(
                        "манифест {}: {}: размер {} байт, ожидается {} (файл обрезан или подменён)",
                        dir.join(MANIFEST_FILE).display(),
                        path.display(),
                        meta.len(),
                        size
                    );
                }
            }
            let actual = sha256_file(&path)?;
            if actual != entry.sha256.to_lowercase() {
                bail!(
                    "манифест {}: {}: sha256 {actual}, ожидается {} (файл подменён или повреждён)",
                    dir.join(MANIFEST_FILE).display(),
                    path.display(),
                    entry.sha256
                );
            }
        }
        Ok(())
    }

    /// Сверить фактические имена input/output ONNX-графа с манифестом.
    pub fn verify_io(&self, what: &str, inputs: &[String], outputs: &[String]) -> Result<()> {
        let Some(io) = &self.io else {
            return Ok(());
        };
        if !io.inputs.is_empty() {
            let missing: Vec<&str> = io
                .inputs
                .iter()
                .filter(|want| !inputs.contains(want))
                .map(|s| s.as_str())
                .collect();
            if !missing.is_empty() {
                bail!(
                    "{what}: входы ONNX-графа не совпадают с манифестом: нет {:?}; фактические {inputs:?}",
                    missing
                );
            }
        }
        if !io.outputs.is_empty() {
            let missing: Vec<&str> = io
                .outputs
                .iter()
                .filter(|want| !outputs.contains(want))
                .map(|s| s.as_str())
                .collect();
            if !missing.is_empty() {
                bail!(
                    "{what}: выходы ONNX-графа не совпадают с манифестом: нет {:?}; фактические {outputs:?}",
                    missing
                );
            }
        }
        Ok(())
    }
}

/// Проверить манифест одного каталога (если он там есть).
pub fn verify_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    match load_for(dir) {
        Some(m) => m.verify_files(dir),
        None => Ok(()),
    }
}

/// Пути манифестов (для генератора и тестов).
pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, data: &[u8]) {
        std::fs::write(path, data).unwrap();
    }

    #[test]
    fn verify_files_detects_substitution_and_truncation() {
        let dir = std::env::temp_dir().join("lt-manifest-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir.join("model.onnx"), b"good-bytes-here");
        write(&dir.join("vocab.txt"), b"a\nb\n");
        let good = sha256_file(&dir.join("model.onnx")).unwrap();
        let manifest = ModelManifest {
            files: vec![FileEntry {
                path: "model.onnx".into(),
                sha256: good,
                size: Some(15),
            }],
            io: None,
        };
        assert!(manifest.verify_files(&dir).is_ok());

        // подмена содержимого
        write(&dir.join("model.onnx"), b"evil-bytes-here");
        let err = format!("{:#}", manifest.verify_files(&dir).unwrap_err());
        assert!(err.contains("model.onnx") && err.contains("sha256"), "{err}");

        // обрезка
        write(&dir.join("model.onnx"), b"good");
        let err = format!("{:#}", manifest.verify_files(&dir).unwrap_err());
        assert!(err.contains("размер"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_io_names_missing_tensor() {
        let m = ModelManifest {
            files: vec![],
            io: Some(IoSpec {
                inputs: vec!["x".into(), "h.1".into()],
                outputs: vec!["y".into()],
            }),
        };
        assert!(m.verify_io("rnnt.decoder", &["x".into(), "h.1".into()], &["y".into()]).is_ok());
        let err = format!(
            "{:#}",
            m.verify_io("rnnt.decoder", &["x".into(), "hi".into()], &["y".into()]).unwrap_err()
        );
        assert!(err.contains("rnnt.decoder") && err.contains("h.1") && err.contains("hi"), "{err}");
    }
}
