//! Строгая грамматика query-строки (контракт источника: docs/API.md,
//! crates/service/src/protocol.rs). Непустой query разбирается как уникальные
//! непустые key=value: неизвестные имена, дубликаты, пустые значения, битые
//! percent-escape, не-UTF-8 и значения вне домена — отказ, а не дефолт.

use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QueryParameters {
    values: BTreeMap<String, String>,
}

impl QueryParameters {
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        let Some(raw) = raw else {
            return Ok(Self::default());
        };
        if raw.is_empty() {
            return Ok(Self::default());
        }
        let mut values = BTreeMap::new();
        for part in raw.split('&') {
            if part.is_empty() {
                return Err("query contains an empty parameter".into());
            }
            let (raw_key, raw_value) = part
                .split_once('=')
                .ok_or_else(|| format!("query parameter {part:?} must be key=value"))?;
            let key = percent_decode(raw_key, "query key")?;
            let value = percent_decode(raw_value, "query value")?;
            if key.is_empty() {
                return Err("query parameter name must not be empty".into());
            }
            if value.is_empty() {
                return Err(format!("query parameter {key:?} must not be empty"));
            }
            if values.insert(key.clone(), value).is_some() {
                return Err(format!("query parameter {key:?} may be specified only once"));
            }
        }
        Ok(Self { values })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    pub fn reject_unknown(&self, accepted: &[&str]) -> Result<(), String> {
        for key in self.values.keys() {
            if !accepted.contains(&key.as_str()) {
                return Err(format!("unknown query parameter {key:?}"));
            }
        }
        Ok(())
    }
}

fn percent_decode(value: &str, location: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            b'%' => {
                let high = *bytes
                    .get(index + 1)
                    .ok_or_else(|| format!("{location} has an incomplete percent escape"))?;
                let low = *bytes
                    .get(index + 2)
                    .ok_or_else(|| format!("{location} has an incomplete percent escape"))?;
                decoded.push((hex_value(high, location)? << 4) | hex_value(low, location)?);
                index += 3;
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(decoded).map_err(|_| format!("{location} must decode to UTF-8 text"))
}

fn hex_value(byte: u8, location: &str) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!("{location} has an invalid percent escape")),
    }
}

pub fn query_bool(parameters: &QueryParameters, key: &str, default: bool) -> Result<bool, String> {
    match parameters.get(key) {
        None => Ok(default),
        Some("1") | Some("true") => Ok(true),
        Some("0") | Some("false") => Ok(false),
        Some(value) => Err(format!("{key} must be 1|0|true|false, got {value:?}")),
    }
}

pub fn query_finite_f32(
    parameters: &QueryParameters,
    key: &str,
    default: f32,
) -> Result<f32, String> {
    let Some(value) = parameters.get(key) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("{key} must be a finite number, got {value:?}"))?;
    if !parsed.is_finite() {
        return Err(format!("{key} must be finite"));
    }
    Ok(parsed)
}

pub fn query_positive_f32(
    parameters: &QueryParameters,
    key: &str,
    default: f32,
) -> Result<f32, String> {
    let value = query_finite_f32(parameters, key, default)?;
    if value <= 0.0 {
        return Err(format!("{key} must be greater than zero"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_strictness() {
        assert!(QueryParameters::parse(Some("words=true&words=false")).is_err());
        assert!(QueryParameters::parse(Some("words")).is_err());
        assert!(QueryParameters::parse(Some("words=%ZZ")).is_err());
        assert!(QueryParameters::parse(Some("words=")).is_err());
        assert!(QueryParameters::parse(Some("=v")).is_err());
        assert!(QueryParameters::parse(Some("a=1&")).is_err());
        // неизвестные имена отсекаются отдельной проверкой домена
        let q = QueryParameters::parse(Some("unknown=value")).unwrap();
        assert!(q.reject_unknown(&["known"]).is_err());
        let q = QueryParameters::parse(Some("words=true&turn_gap=1%2E5")).unwrap();
        q.reject_unknown(&["words", "turn_gap"]).unwrap();
        assert!(query_bool(&q, "words", false).unwrap());
        assert_eq!(query_positive_f32(&q, "turn_gap", 1.0).unwrap(), 1.5);
    }
}
