//! Admission: монотонный жизненный цикл Running → Draining и ёмкость
//! параллельных HTTP-запросов (контракт источника: docs/OPERATIONS.md
//! §Admission, crates/service/src/admission.rs). Обратного перехода нет.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const PHASE_RUNNING: u8 = 0;
const PHASE_DRAINING: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionRefusal {
    /// Ёмкость исчерпана при фазе Running.
    Overloaded,
    /// Процесс в Draining: новые запросы не принимаются.
    Draining,
}

pub struct AdmissionState {
    phase: AtomicU8,
    http: Arc<Semaphore>,
    max_http: usize,
}

impl AdmissionState {
    pub fn new(max_http: usize) -> Self {
        Self {
            phase: AtomicU8::new(PHASE_RUNNING),
            http: Arc::new(Semaphore::new(max_http)),
            max_http,
        }
    }

    pub fn is_draining(&self) -> bool {
        self.phase.load(Ordering::SeqCst) == PHASE_DRAINING
    }

    /// Идемпотентный переход в Draining; true — переход произошёл сейчас.
    pub fn begin_draining(&self) -> bool {
        self.phase
            .compare_exchange(PHASE_RUNNING, PHASE_DRAINING, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Захватить слот запроса. Ёмкость удерживается до терминации работы
    /// (пермит уезжает в блокирующую работу), даже если клиент уже получил
    /// 504. Вторая проверка фазы закрывает гонку «дрейн начался между
    /// чтением фазы и захватом пермита».
    pub fn admit_http(&self) -> Result<OwnedSemaphorePermit, AdmissionRefusal> {
        if self.is_draining() {
            return Err(AdmissionRefusal::Draining);
        }
        let permit = match Arc::clone(&self.http).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                return Err(match self.is_draining() {
                    true => AdmissionRefusal::Draining,
                    false => AdmissionRefusal::Overloaded,
                })
            }
        };
        if self.is_draining() {
            drop(permit);
            return Err(AdmissionRefusal::Draining);
        }
        Ok(permit)
    }

    pub fn available_http(&self) -> usize {
        self.http.available_permits()
    }

    pub fn max_http(&self) -> usize {
        self.max_http
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn drain_is_monotonic_and_capacity_is_returned() {
        let state = AdmissionState::new(1);
        let permit = state.admit_http().expect("running capacity commits");
        assert!(matches!(state.admit_http(), Err(AdmissionRefusal::Overloaded)));
        assert!(state.begin_draining());
        assert!(!state.begin_draining(), "повторный дрейн идемпотентен");
        assert!(matches!(state.admit_http(), Err(AdmissionRefusal::Draining)));
        drop(permit);
        assert!(matches!(state.admit_http(), Err(AdmissionRefusal::Draining)));
    }
}
