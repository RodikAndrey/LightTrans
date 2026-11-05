//! Манифесты моделей (Фаза 3 PORTING.md, идея из crates/model-package
//! источника): для каталога семейства `manifest.json` — список файлов
//! (sha256 + размер) и ожидаемые IO-имена ONNX-графов.
//!
//! Правило: манифеста нет — загрузка идёт как раньше (совместимость);
//! манифест есть — проверка строгая, несоответствие = отказ старта с именем
//! файла и расхождением. Рассинхрон имён тензоров иначе ловится непонятной
//! ошибкой ORT в рантайме; битые LFS-загрузки давали 29-байтовые «onnx».

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub const MANIFEST_FILE: &str = "manifest.json";

/// Кэш результатов верификации (один файл в корне models/): путь →
/// подтверждённая сумма, привязанная к размеру+mtime файла и ожидаемой сумме
/// из манифеста. Полная установка весов — ~7 ГБ, и без кэша каждый запуск
/// перечитывал бы их под SHA-256 (~9 с на NVMe). Любое изменение файла
/// (mtime/размер) или правка манифеста (ожидаемая сумма) инвалидирует запись.
pub const VERIFY_CACHE_FILE: &str = ".verify-cache.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyCacheEntry {
    size: u64,
    /// mtime файла, нс от эпохи
    mtime: u128,
    /// ожидаемая (по манифесту) сумма, для которой SHA-256 был подтверждён
    sha256: String,
}

pub type VerifyCache = BTreeMap<String, VerifyCacheEntry>;

/// Загрузить кэш верификации (общий для всех семей вызова resolve): битый или
/// отсутствующий файл — просто пустой кэш.
pub fn load_verify_cache(path: &Path) -> VerifyCache {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Атомарная запись кэша; невозможность записать (read-only каталог и т.п.)
/// не валидирует запуск — верификация от этого не зависит.
pub fn save_verify_cache(path: &Path, cache: &VerifyCache) {
    let tmp = path.with_extension("json.tmp");
    let Ok(mut f) = std::fs::File::create(&tmp) else { return };
    let ok = serde_json::to_writer_pretty(&mut f, cache).is_ok();
    drop(f);
    if ok && std::fs::rename(&tmp, path).is_ok() {
        return;
    }
    let _ = std::fs::remove_file(&tmp);
}

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
        self.verify_files_impl(dir, false).map(|_| ())
    }

    /// Проверить строго только присутствующие файлы, отсутствующие вернуть
    /// списком (пути из манифеста). Присутствующий, но битый/подменённый
    /// файл по-прежнему отказывает. Отсутствие — не «расхождение»: в git
    /// коммитятся манифесты без весов, а недокачанные семьи автозагрузка
    /// доставляет перед использованием (src/download.rs).
    pub fn verify_files_partial(&self, dir: &Path) -> Result<Vec<String>> {
        self.verify_files_impl(dir, true)
    }

    /// Верификация присутствующих файлов с кэшем результатов (см.
    /// `VERIFY_CACHE_FILE`): SHA-256 пересчитывается, только если изменились
    /// размер+mtime файла или ожидаемая в манифесте сумма. Семантика отсутствия
    /// файлов — как в `verify_files_partial`. `cache` общий для всех семей
    /// одного запуска (загружается/сохраняется вызывающим кодом, чтобы записи
    /// семей не затирали друг друга). `prefix` — путь каталога семьи от корня
    /// models/ (пустой для самого корня), ключи кэша строятся с ним.
    pub fn verify_files_cached(
        &self,
        dir: &Path,
        prefix: &str,
        cache: &mut VerifyCache,
    ) -> Result<Vec<String>> {
        let mut missing = Vec::new();
        let mut verified: VerifyCache = BTreeMap::new();
        for entry in &self.files {
            let path = dir.join(&entry.path);
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => {
                    missing.push(entry.path.clone());
                    continue; // отсутствующие — не расхождение
                }
            };
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
            let expected = entry.sha256.to_lowercase();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let key = if prefix.is_empty() {
                entry.path.clone()
            } else {
                format!("{prefix}/{}", entry.path)
            };
            if let Some(c) = cache.get(&key) {
                if c.size == meta.len() && c.mtime == mtime && c.sha256 == expected {
                    verified.insert(key, c.clone());
                    continue;
                }
            }
            let actual = sha256_file(&path)?;
            if actual != expected {
                bail!(
                    "манифест {}: {}: sha256 {actual}, ожидается {} (файл подменён или повреждён)",
                    dir.join(MANIFEST_FILE).display(),
                    path.display(),
                    entry.sha256
                );
            }
            verified.insert(key, VerifyCacheEntry { size: meta.len(), mtime, sha256: expected });
        }
        // подтверждённые записи вычищают устаревшие (файл удалён, ожидаемая
        // сумма сменилась); записи других семей в cache не трогаем
        for (key, entry) in verified {
            cache.insert(key, entry);
        }
        Ok(missing)
    }

    fn verify_files_impl(&self, dir: &Path, skip_missing: bool) -> Result<Vec<String>> {
        let mut missing = Vec::new();
        for entry in &self.files {
            let path = dir.join(&entry.path);
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) if skip_missing => {
                    missing.push(entry.path.clone());
                    let _ = e;
                    continue;
                }
                Err(_) => {
                    bail!(
                        "манифест {}: файл отсутствует: {}",
                        dir.join(MANIFEST_FILE).display(),
                        path.display()
                    );
                }
            };
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
        Ok(missing)
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

    /// Записи разных семей копятся в одном кэше (раньше каждая семья
    /// перезаписывала кэш своим набором — оставалась только последняя).
    #[test]
    fn verify_files_cached_accumulates_across_families() {
        let dir = std::env::temp_dir().join("lt-manifest-cache-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir.join("a.onnx"), b"aaaa");
        let sha_a = sha256_file(&dir.join("a.onnx")).unwrap();
        let ma = ModelManifest {
            files: vec![FileEntry { path: "a.onnx".into(), sha256: sha_a, size: Some(4) }],
            io: None,
        };

        let mut cache = VerifyCache::new();
        ma.verify_files_cached(&dir, "", &mut cache).unwrap();

        // вторая «семья»: другой каталог и префикс, тот же кэш
        let dir_b = std::env::temp_dir().join("lt-manifest-cache-test-b");
        let _ = std::fs::remove_dir_all(&dir_b);
        std::fs::create_dir_all(&dir_b).unwrap();
        write(&dir_b.join("b.onnx"), b"bbbb");
        let sha_b = sha256_file(&dir_b.join("b.onnx")).unwrap();
        let mb = ModelManifest {
            files: vec![FileEntry { path: "b.onnx".into(), sha256: sha_b, size: Some(4) }],
            io: None,
        };
        let missing = mb.verify_files_cached(&dir_b, "family-b", &mut cache).unwrap();
        assert!(missing.is_empty());

        // обе записи в кэше, повторная проверка не теряет первую
        assert!(cache.contains_key("a.onnx"), "запись первой семьи потеряна");
        assert!(cache.contains_key("family-b/b.onnx"));

        // отсутствующий файл попадает в missing, а не в расхождение
        std::fs::remove_file(&dir.join("a.onnx")).unwrap();
        let missing = ma.verify_files_cached(&dir, "", &mut cache).unwrap();
        assert_eq!(missing, vec!["a.onnx".to_string()]);
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&dir_b).unwrap();
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
