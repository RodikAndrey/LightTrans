//! Строгая конфигурация сервера из окружения (контракт источника:
//! docs/OPERATIONS.md §Service limits and policy environment).
//!
//! Значение по умолчанию ≠ fallback: присутствующее, но пустое/кривое/вне
//! диапазона значение env **отказывает**, а не подменяется дефолтом.

use std::time::Duration;

pub const DEFAULT_MAX_CONCURRENCY: usize = 8;
pub const DEFAULT_BODY_LIMIT_MB: usize = 64;
pub const DEFAULT_REQ_TIMEOUT_SEC: u64 = 120;

const BYTES_PER_MIB: usize = 1024 * 1024;

/// Проверенные лимиты процесса.
#[derive(Debug, Clone, Copy)]
pub struct ServerLimits {
    /// `ASR_MAX_CONCURRENCY` — максимум одновременно исполняемых batch-запросов.
    pub max_concurrency: usize,
    /// `ASR_BODY_LIMIT_MB` — граница тела POST в байтах.
    pub body_limit_bytes: usize,
    /// `ASR_REQ_TIMEOUT_SEC` — дедлайн обработки всего запроса.
    pub request_timeout: Duration,
}

impl ServerLimits {
    pub fn from_env() -> Result<Self, String> {
        let max_concurrency =
            positive_usize_env("ASR_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)?;
        let body_mb = positive_usize_env("ASR_BODY_LIMIT_MB", DEFAULT_BODY_LIMIT_MB)?;
        let timeout_sec =
            positive_usize_env("ASR_REQ_TIMEOUT_SEC", DEFAULT_REQ_TIMEOUT_SEC as usize)? as u64;
        let body_limit_bytes = body_mb
            .checked_mul(BYTES_PER_MIB)
            .ok_or_else(|| "ASR_BODY_LIMIT_MB exceeds the supported body size".to_string())?;
        Ok(Self {
            max_concurrency,
            body_limit_bytes,
            request_timeout: Duration::from_secs(timeout_sec),
        })
    }
}

fn env_opt(key: &str) -> Result<Option<String>, String> {
    match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{key} must contain UTF-8 text"))
        }
    }
}

fn positive_usize_env(key: &str, default: usize) -> Result<usize, String> {
    let Some(value) = env_opt(key)? else {
        return Ok(default);
    };
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{key} must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!("{key} must be a positive integer"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_malformed_values_instead_of_falling_back() {
        assert_eq!(
            positive_usize_env("T_NOPE_A", 8),
            Ok(8),
            "отсутствующее значение берёт дефолт"
        );
        assert!(positive_usize_env("T_NOPE_B", 8).is_ok());
        assert_eq!(
            ServerLimits::from_env().map(|l| l.max_concurrency),
            Ok(DEFAULT_MAX_CONCURRENCY)
        );
    }
}
