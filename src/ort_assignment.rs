//! Инспекция назначения узлов графа ORT по execution provider'ам (API-24) и
//! канонический отпечаток CUDA-распределения — Фаза 4 PORTING.md.
//!
//! Порт контракта из gigaam-v3-runtime (`crates/recognition/src/ort_assignment.rs`):
//! перед возвратом CUDA-сессии ORT отчитывается, какие узлы на каком провайдере
//! исполняются; принимаются только CPU/CUDA-назначения и требуется хотя бы один
//! CUDA-узел, иначе ORT мог бы тихо оставить часть графа на CPU. Отпечаток —
//! sha256 канонического представления (домен сохранён как в источнике:
//! `gigaam-v3-runtime/cuda-assignment/v1`), поэтому референсные значения CTC/RNNT
//! совпадают между проектами при том же ORT (1.29) и файлах моделей.
//!
//! Для записи назначений сессия должна собираться с конфигом
//! `session.record_ep_graph_assignment_info=1` (см. gigaam.rs/gigaam_rnnt.rs).
//!
//! Схема верификации покрывает роли CTC и RNN-T (как в источнике): сервер
//! обслуживает только их; parakeet/whisper вне схемы (у parakeet нет
//! референсного отпечатка, whisper.cpp — не ONNX).

use ort::session::Session;
use ort::AsPointer;
use sha2::{Digest, Sha256};
use std::ffi::CStr;
use std::slice;

const MAX_ASSIGNMENT_RECORDS: usize = 1_000_000;
const FINGERPRINT_DOMAIN: &[u8] = b"gigaam-v3-runtime/cuda-assignment/v1";

/// Роль CUDA-энкодера, к которой относится отпечаток.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    Ctc,
    Rnnt,
}

impl Role {
    pub const fn as_str(self) -> &'static str {
        match self {
            Role::Ctc => "ctc",
            Role::Rnnt => "rnnt",
        }
    }
}

/// SHA-256 отпечаток канонического представления назначений узлов.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fingerprint([u8; 32]);

impl Fingerprint {
    /// Ровно 64 hex-символа.
    pub fn parse(hex: &str) -> Result<Self, String> {
        let text = hex.as_bytes();
        if text.len() != 64 {
            return Err(format!(
                "ожидается ровно 64 hex-символа SHA-256, получено {}",
                text.len()
            ));
        }
        let mut bytes = [0u8; 32];
        for (i, out) in bytes.iter_mut().enumerate() {
            let high = hex_nibble(text[2 * i])?;
            let low = hex_nibble(text[2 * i + 1])?;
            *out = high * 16 + low;
        }
        Ok(Self(bytes))
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub(crate) const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

fn hex_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err("SHA-256 отпечаток должен содержать только hex-символы".into()),
    }
}

/// Наблюдаемое распределение узлов одной CUDA-сессии (владельческая копия).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Evidence {
    fingerprint: Fingerprint,
    cpu_assignments: usize,
    cuda_assignments: usize,
}

impl Evidence {
    pub const fn fingerprint(&self) -> &Fingerprint {
        &self.fingerprint
    }

    pub const fn cpu_assignments(&self) -> usize {
        self.cpu_assignments
    }

    pub const fn cuda_assignments(&self) -> usize {
        self.cuda_assignments
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AssignmentRecord {
    provider: String,
    node: String,
    domain: String,
    operator_type: String,
}

/// Читает назначения узлов живой сессии и валидирует их.
pub fn inspect(session: &Session) -> Result<Evidence, String> {
    evidence_from_records(assignment_records(session)?)
}

/// Валидирует копии записей назначений и собирает владельческий Evidence:
/// только CPU/CUDA-провайдеры, минимум один CUDA-узел.
fn evidence_from_records(mut records: Vec<AssignmentRecord>) -> Result<Evidence, String> {
    if records.is_empty() {
        return Err("ORT не вернул ни одной записи назначения для CUDA-сессии".into());
    }
    let mut cpu_assignments = 0usize;
    let mut cuda_assignments = 0usize;
    for record in &records {
        match record.provider.as_str() {
            "CPUExecutionProvider" => {
                cpu_assignments = cpu_assignments
                    .checked_add(1)
                    .ok_or_else(|| "число CPU-назначений переполнило usize".to_string())?;
            }
            "CUDAExecutionProvider" => {
                cuda_assignments = cuda_assignments
                    .checked_add(1)
                    .ok_or_else(|| "число CUDA-назначений переполнило usize".to_string())?;
            }
            provider => {
                return Err(format!(
                    "узел {:?} назначен неподдерживаемому провайдеру {provider:?}",
                    record.node
                ));
            }
        }
    }
    if cuda_assignments == 0 {
        return Err("в назначении графа нет ни одного узла CUDAExecutionProvider".into());
    }
    records.sort_unstable();
    Ok(Evidence {
        fingerprint: fingerprint(&records)?,
        cpu_assignments,
        cuda_assignments,
    })
}

/// Читает метаданные назначений, пока сессия заимствована и жива.
///
/// # Safety
///
/// ORT 1.29 документирует эти объекты и C-строки как runtime-owned заимствования
/// живой сессии. Функция удерживает `&Session` на время всех FFI-вызовов,
/// проверяет каждый статус/указатель/длину, копирует строки во владение Rust
/// сразу после чтения, не хранит чужие указатели и не освобождает чужие объекты.
fn assignment_records(session: &Session) -> Result<Vec<AssignmentRecord>, String> {
    let api = ort::api();
    let mut raw_subgraphs: *const *const ort::sys::OrtEpAssignedSubgraph = std::ptr::null();
    let mut subgraph_count = 0usize;

    // SAFETY: session.ptr() ненулевой (AsPointer); оба out-указателя валидны на время вызова.
    let status = unsafe {
        (api.Session_GetEpGraphAssignmentInfo)(session.ptr(), &mut raw_subgraphs, &mut subgraph_count)
    };
    // SAFETY: статус получен из вызова ORT и его владение передано сюда.
    unsafe { ort::Error::result_from_status(status) }
        .map_err(|e| format!("чтение информации о назначении графа ORT: {e}"))?;

    let max_slice_len = usize::try_from(isize::MAX)
        .map_err(|_| "платформа не может представить максимальную длину FFI-среза".to_string())?;
    if subgraph_count > MAX_ASSIGNMENT_RECORDS || subgraph_count > max_slice_len {
        return Err("число подграфов назначения превышает границу безопасности".into());
    }
    if subgraph_count > 0 && raw_subgraphs.is_null() {
        return Err("ORT вернул нулевой указатель массива подграфов назначения".into());
    }
    let subgraphs: &[*const ort::sys::OrtEpAssignedSubgraph] = if subgraph_count == 0 {
        &[]
    } else {
        // SAFETY: ORT вернул ненулевой массив проверенной длины для живой сессии.
        unsafe { slice::from_raw_parts(raw_subgraphs, subgraph_count) }
    };

    let mut records = Vec::new();
    for &subgraph in subgraphs {
        if subgraph.is_null() {
            return Err("ORT вернул нулевой указатель подграфа назначения".into());
        }

        let mut provider_ptr: *const std::os::raw::c_char = std::ptr::null();
        // SAFETY: subgraph ненулевой и заимствован у живой сессии; out-указатель живёт до конца вызова.
        let status = unsafe { (api.EpAssignedSubgraph_GetEpName)(subgraph, &mut provider_ptr) };
        // SAFETY: статус получен из вызова ORT и его владение передано сюда.
        unsafe { ort::Error::result_from_status(status) }
            .map_err(|e| format!("чтение провайдера назначения ORT: {e}"))?;
        if provider_ptr.is_null() {
            return Err("ORT вернул пустое имя провайдера назначения".into());
        }
        // SAFETY: ORT вернул ненулевую NUL-терминированную строку; копируем до следующего FFI-вызова.
        let provider_bytes = unsafe { CStr::from_ptr(provider_ptr).to_bytes().to_vec() };
        let provider = String::from_utf8(provider_bytes)
            .map_err(|_| "имя провайдера назначения не UTF-8".to_string())?;

        let mut raw_nodes: *const *const ort::sys::OrtEpAssignedNode = std::ptr::null();
        let mut node_count = 0usize;
        // SAFETY: subgraph ненулевой и заимствован у живой сессии; оба out-указателя валидны на время вызова.
        let status =
            unsafe { (api.EpAssignedSubgraph_GetNodes)(subgraph, &mut raw_nodes, &mut node_count) };
        // SAFETY: статус получен из вызова ORT и его владение передано сюда.
        unsafe { ort::Error::result_from_status(status) }
            .map_err(|e| format!("чтение узлов назначения ORT: {e}"))?;
        if node_count > max_slice_len {
            return Err("число узлов назначения превышает границу платформы".into());
        }
        if node_count > 0 && raw_nodes.is_null() {
            return Err("ORT вернул нулевой указатель массива узлов назначения".into());
        }
        let next_count = records
            .len()
            .checked_add(node_count)
            .ok_or_else(|| "число записей назначения переполнило usize".to_string())?;
        if next_count > MAX_ASSIGNMENT_RECORDS {
            return Err("число записей назначения превышает границу безопасности".into());
        }
        let nodes: &[*const ort::sys::OrtEpAssignedNode] = if node_count == 0 {
            &[]
        } else {
            // SAFETY: ORT вернул ненулевой массив проверенной длины для живой сессии.
            unsafe { slice::from_raw_parts(raw_nodes, node_count) }
        };

        for &node in nodes {
            if node.is_null() {
                return Err("ORT вернул нулевой указатель узла назначения".into());
            }
            let node_name = node_string(api, node, NodeField::Name)?;
            let domain = node_string(api, node, NodeField::Domain)?;
            let operator_type = node_string(api, node, NodeField::OperatorType)?;
            records.push(AssignmentRecord {
                provider: provider.clone(),
                node: node_name,
                domain,
                operator_type,
            });
        }
    }
    Ok(records)
}

#[derive(Clone, Copy)]
enum NodeField {
    Name,
    Domain,
    OperatorType,
}

impl NodeField {
    const fn description(self) -> &'static str {
        match self {
            NodeField::Name => "имя",
            NodeField::Domain => "домен",
            NodeField::OperatorType => "тип оператора",
        }
    }
}

fn node_string(
    api: &ort::sys::OrtApi,
    node: *const ort::sys::OrtEpAssignedNode,
    field: NodeField,
) -> Result<String, String> {
    let mut ptr: *const std::os::raw::c_char = std::ptr::null();
    let status = match field {
        NodeField::Name => {
            // SAFETY: node проверен на ненулевое значение, заимствован у живой сессии; out-указатель живёт до конца вызова.
            unsafe { (api.EpAssignedNode_GetName)(node, &mut ptr) }
        }
        NodeField::Domain => {
            // SAFETY: node проверен на ненулевое значение, заимствован у живой сессии; out-указатель живёт до конца вызова.
            unsafe { (api.EpAssignedNode_GetDomain)(node, &mut ptr) }
        }
        NodeField::OperatorType => {
            // SAFETY: node проверен на ненулевое значение, заимствован у живой сессии; out-указатель живёт до конца вызова.
            unsafe { (api.EpAssignedNode_GetOperatorType)(node, &mut ptr) }
        }
    };
    // SAFETY: статус получен из непосредственно предшествующего вызова ORT.
    unsafe { ort::Error::result_from_status(status) }.map_err(|e| {
        format!(
            "чтение {} узла назначения ORT: {e}",
            field.description()
        )
    })?;
    if ptr.is_null() {
        return Err(format!(
            "ORT вернул пустое {} узла назначения",
            field.description()
        ));
    }
    // SAFETY: ORT вернул ненулевую NUL-терминированную строку; копируем во владение.
    let bytes = unsafe { CStr::from_ptr(ptr).to_bytes().to_vec() };
    String::from_utf8(bytes).map_err(|_| {
        format!(
            "{} узла назначения не UTF-8",
            field.description()
        )
    })
}

/// Канонический отпечаток: домен, число записей, затем (кратность, провайдер,
/// узел, домен, тип оператора) для каждой уникальной записи отсортированного
/// списка. Порядок чтения ORT не влияет на отпечаток.
fn fingerprint(records: &[AssignmentRecord]) -> Result<Fingerprint, String> {
    let total_count =
        u64::try_from(records.len()).map_err(|_| "число записей не влезает в u64".to_string())?;
    let mut hasher = Sha256::new();
    write_bytes(&mut hasher, FINGERPRINT_DOMAIN)?;
    write_u64(&mut hasher, total_count);

    let mut position = 0usize;
    while position < records.len() {
        let record = records
            .get(position)
            .ok_or_else(|| "запись назначения недоступна".to_string())?;
        let mut end = position
            .checked_add(1)
            .ok_or_else(|| "диапазон записей переполнил usize".to_string())?;
        while records.get(end) == Some(record) {
            end = end
                .checked_add(1)
                .ok_or_else(|| "диапазон записей переполнил usize".to_string())?;
        }
        let multiplicity = end
            .checked_sub(position)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| "кратность записи не влезает в u64".to_string())?;
        write_u64(&mut hasher, multiplicity);
        write_bytes(&mut hasher, record.provider.as_bytes())?;
        write_bytes(&mut hasher, record.node.as_bytes())?;
        write_bytes(&mut hasher, record.domain.as_bytes())?;
        write_bytes(&mut hasher, record.operator_type.as_bytes())?;
        position = end;
    }
    let digest: [u8; 32] = hasher.finalize().into();
    Ok(Fingerprint::from_bytes(digest))
}

fn write_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn write_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), String> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| "длина поля канонизации не влезает в u64".to_string())?;
    write_u64(hasher, length);
    hasher.update(bytes);
    Ok(())
}

/// Политика верификации CUDA-назначений (ASR_CUDA_ASSIGNMENT_POLICY).
/// Синтаксис проверяется до всякой работы с моделями: кривое значение отказывает
/// старт, а не подменяется.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Policy {
    /// Отпечаток каждой требуемой роли сверяется с заданным; несовпадение —
    /// отказ старта.
    Verified {
        ctc: Option<Fingerprint>,
        rnnt: Option<Fingerprint>,
    },
    /// Явный режим «только наблюдение»: проверки валидности назначения
    /// сохраняются, сравнение отпечатка пропускается, наблюдение печатается.
    AllowUnverified,
}

impl Policy {
    /// Читает три переменные окружения. `use_cuda` — выбран ли `--ep cuda`,
    /// `with_rnnt` — будет ли строиться RNN-T роль.
    pub fn from_env(use_cuda: bool, with_rnnt: bool) -> Result<Option<Policy>, String> {
        let policy = read_env("ASR_CUDA_ASSIGNMENT_POLICY")?;
        let ctc = read_env("ASR_CUDA_CTC_ASSIGNMENT_SHA256")?;
        let rnnt = read_env("ASR_CUDA_RNNT_ASSIGNMENT_SHA256")?;
        Self::parse(use_cuda, with_rnnt, policy.as_deref(), ctc.as_deref(), rnnt.as_deref())
    }

    /// Чистый разбор (env отделён для тестов).
    pub fn parse(
        use_cuda: bool,
        with_rnnt: bool,
        policy: Option<&str>,
        ctc_hex: Option<&str>,
        rnnt_hex: Option<&str>,
    ) -> Result<Option<Policy>, String> {
        if !use_cuda {
            if policy.is_some() || ctc_hex.is_some() || rnnt_hex.is_some() {
                return Err(
                    "ASR_CUDA_ASSIGNMENT_* требуют выбранного провайдера --ep cuda".into(),
                );
            }
            return Ok(None);
        }
        let ctc = ctc_hex
            .map(Fingerprint::parse)
            .transpose()
            .map_err(|e| format!("ASR_CUDA_CTC_ASSIGNMENT_SHA256: {e}"))?;
        let rnnt = rnnt_hex
            .map(Fingerprint::parse)
            .transpose()
            .map_err(|e| format!("ASR_CUDA_RNNT_ASSIGNMENT_SHA256: {e}"))?;
        match policy {
            None | Some("verified") => {
                // verified — поведение по умолчанию для CUDA: отпечаток обязателен
                // для каждой роли, которая будет построена
                if ctc.is_none() {
                    return Err(
                        "verified-политика CUDA требует ASR_CUDA_CTC_ASSIGNMENT_SHA256".into(),
                    );
                }
                if with_rnnt && rnnt.is_none() {
                    return Err(
                        "verified-политика CUDA требует ASR_CUDA_RNNT_ASSIGNMENT_SHA256 (или --no-rnnt)"
                            .into(),
                    );
                }
                Ok(Some(Policy::Verified { ctc, rnnt }))
            }
            Some("allow-unverified") => {
                if ctc.is_some() || rnnt.is_some() {
                    return Err(
                        "ASR_CUDA_ASSIGNMENT_POLICY=allow-unverified конфликтует с хешами назначения"
                            .into(),
                    );
                }
                Ok(Some(Policy::AllowUnverified))
            }
            Some(value) => Err(format!(
                "ASR_CUDA_ASSIGNMENT_POLICY: ожидается verified или allow-unverified, получено {value:?}"
            )),
        }
    }

    pub const fn is_allow_unverified(&self) -> bool {
        matches!(self, Policy::AllowUnverified)
    }

    /// Сверяет наблюдаемый отпечаток роли с ожидаемым (verified) или пропускает
    /// (allow-unverified). Валидность назначения уже проверена в `inspect`.
    pub fn check(&self, role: Role, observed: &Evidence) -> Result<(), String> {
        match self {
            Policy::AllowUnverified => Ok(()),
            Policy::Verified { ctc, rnnt } => {
                let expected = match role {
                    Role::Ctc => ctc.as_ref(),
                    Role::Rnnt => rnnt.as_ref(),
                }
                .ok_or_else(|| {
                    format!(
                        "для CUDA-роли {} не задан ожидаемый отпечаток назначения",
                        role.as_str()
                    )
                })?;
                if expected != observed.fingerprint() {
                    return Err(format!(
                        "несовпадение отпечатка CUDA-назначения ({}): ожидался {}, наблюдался {}",
                        role.as_str(),
                        expected.to_hex(),
                        observed.fingerprint().to_hex()
                    ));
                }
                Ok(())
            }
        }
    }
}

fn read_env(key: &str) -> Result<Option<String>, String> {
    match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{key} должен содержать UTF-8")),
    }
}

#[cfg(test)]
mod tests {
    use super::{evidence_from_records, AssignmentRecord};

    fn record(provider: &str, node: &str, domain: &str, operator_type: &str) -> AssignmentRecord {
        AssignmentRecord {
            provider: provider.into(),
            node: node.into(),
            domain: domain.into(),
            operator_type: operator_type.into(),
        }
    }

    #[test]
    fn evidence_не_зависит_от_порядка_и_чувствительна_к_каждому_полю() -> Result<(), String> {
        let cuda = record("CUDAExecutionProvider", "encoder/conv", "", "Conv");
        let cpu = record("CPUExecutionProvider", "encoder/cast", "ai.onnx", "Cast");
        let records = vec![cuda.clone(), cpu.clone(), cuda.clone()];
        let evidence = evidence_from_records(records.clone())?;
        assert_eq!(evidence.cpu_assignments(), 1);
        assert_eq!(evidence.cuda_assignments(), 2);

        let reordered = evidence_from_records(vec![cpu.clone(), cuda.clone(), cuda.clone()])?;
        assert_eq!(evidence.fingerprint(), reordered.fingerprint());

        let provider = evidence_from_records(vec![
            cuda.clone(),
            record("CUDAExecutionProvider", "encoder/cast", "ai.onnx", "Cast"),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), provider.fingerprint());

        let node = evidence_from_records(vec![
            record("CUDAExecutionProvider", "encoder/conv-2", "", "Conv"),
            cpu.clone(),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), node.fingerprint());

        let domain = evidence_from_records(vec![
            record(
                "CUDAExecutionProvider",
                "encoder/conv",
                "com.example",
                "Conv",
            ),
            cpu.clone(),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), domain.fingerprint());

        let operator_type = evidence_from_records(vec![
            record("CUDAExecutionProvider", "encoder/conv", "", "Relu"),
            cpu.clone(),
            cuda.clone(),
        ])?;
        assert_ne!(evidence.fingerprint(), operator_type.fingerprint());

        let without_duplicate = evidence_from_records(vec![cuda, cpu])?;
        assert_ne!(evidence.fingerprint(), without_duplicate.fingerprint());
        Ok(())
    }

    #[test]
    fn evidence_отвергает_пустые_неизвестные_и_cpu_only_назначения() {
        assert!(evidence_from_records(Vec::new()).is_err());
        assert!(evidence_from_records(vec![record(
            "TensorRTExecutionProvider",
            "encoder/node",
            "",
            "Conv",
        )])
        .is_err());
        assert!(evidence_from_records(vec![record(
            "CPUExecutionProvider",
            "encoder/node",
            "",
            "Conv",
        )])
        .is_err());
    }

    #[test]
    fn fingerprint_парсится_и_сверяется() {
        use super::Fingerprint;
        let hex = "aa".repeat(32);
        let fp = Fingerprint::parse(&hex).unwrap();
        assert_eq!(fp.to_hex(), hex);
        assert!(Fingerprint::parse(&hex[..63]).is_err());
        assert!(Fingerprint::parse(&format!("{hex}g")).is_err());

        // канонический отпечаток домен+количество+кратности: схема стабильна,
        // кратности учитываются (дубль CUDA-узла даёт другой отпечаток, чем пара)
        let evidence = evidence_from_records(vec![
            record("CUDAExecutionProvider", "n", "", "Conv"),
            record("CPUExecutionProvider", "m", "ai.onnx", "Cast"),
            record("CUDAExecutionProvider", "n", "", "Conv"),
        ])
        .unwrap();
        assert_eq!(evidence.fingerprint().to_hex().len(), 64);
        let no_dup = evidence_from_records(vec![
            record("CUDAExecutionProvider", "n", "", "Conv"),
            record("CPUExecutionProvider", "m", "ai.onnx", "Cast"),
        ])
        .unwrap();
        assert_ne!(evidence.fingerprint(), no_dup.fingerprint());
    }

    #[test]
    fn policy_parse_покрывает_контракт_источника() {
        use super::Policy;
        // CPU: любые настройки назначения — ошибка
        assert!(Policy::parse(false, true, Some("verified"), None, None).is_err());
        assert!(Policy::parse(false, true, None, Some(&"a".repeat(64)), None).is_err());
        assert!(Policy::parse(false, true, None, None, None).unwrap().is_none());

        // CUDA verified без хешей — отказ (поведение по умолчанию)
        assert!(Policy::parse(true, true, None, None, None).is_err());
        // verified: CTC обязателен, RNNT обязателен только при with_rnnt
        assert!(Policy::parse(true, false, None, Some(&"a".repeat(64)), None).is_ok());
        assert!(
            Policy::parse(true, true, None, Some(&"a".repeat(64)), None).is_err()
        );
        let verified = Policy::parse(
            true,
            true,
            Some("verified"),
            Some(&"a".repeat(64)),
            Some(&"b".repeat(64)),
        )
        .unwrap()
        .unwrap();
        assert!(!verified.is_allow_unverified());

        // allow-unverified конфликтует с хешами
        assert!(
            Policy::parse(true, true, Some("allow-unverified"), Some(&"a".repeat(64)), None)
                .is_err()
        );
        assert!(
            Policy::parse(true, true, Some("allow-unverified"), None, Some(&"b".repeat(64)))
                .is_err()
        );
        let au = Policy::parse(true, true, Some("allow-unverified"), None, None)
            .unwrap()
            .unwrap();
        assert!(au.is_allow_unverified());

        // неизвестное значение — отказ; не-hex хеш — отказ
        assert!(Policy::parse(true, true, Some("trust-me"), None, None).is_err());
        assert!(Policy::parse(true, true, Some("verified"), Some("xyz"), None).is_err());
    }
}
