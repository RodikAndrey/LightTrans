//! Метрики Prometheus в текстовом формате (Фаза 2 PORTING.md), без внешних
//! крейтов — по образцу crates/service/src/metrics.rs источника: фиксированные
//! бакеты, приём только конечных неотрицательных наблюдений, счётчик отказов.
//! Префикс `lighttrans_` (в отличие от `asr_` источника).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

const BUCKETS: [f64; 11] = [
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, f64::INFINITY,
];

#[derive(Default)]
struct HistogramState {
    buckets: [u64; 11], // накопительно по le
    sum_seconds: f64,
    count: u64,
    rejected_non_finite: u64,
    rejected_negative: u64,
}

impl HistogramState {
    fn observe(&mut self, seconds: f64) {
        for (index, &bound) in BUCKETS.iter().enumerate() {
            if seconds <= bound {
                self.buckets[index] = self.buckets[index].wrapping_add(1);
            }
        }
        self.sum_seconds += seconds;
        self.count = self.count.wrapping_add(1);
    }
}

#[derive(Default)]
struct Hist(Mutex<HistogramState>);

impl Hist {
    fn observe(&self, secs: f64) {
        if let Ok(mut st) = self.0.lock() {
            if !secs.is_finite() {
                st.rejected_non_finite += 1;
            } else if secs < 0.0 {
                st.rejected_negative += 1;
            } else {
                st.observe(secs);
            }
        }
    }

    fn render(&self, out: &mut String, name: &str) {
        let Ok(st) = self.0.lock() else { return };
        let _ = writeln!(out, "# TYPE {name} histogram");
        for (i, &b) in BUCKETS.iter().enumerate() {
            let le = if b.is_infinite() { "+Inf".to_string() } else { format!("{b}") };
            let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {}", st.buckets[i]);
        }
        let _ = writeln!(out, "{name}_sum {}", st.sum_seconds);
        let _ = writeln!(out, "{name}_count {}", st.count);
        let _ = writeln!(out, "# TYPE {name}_rejected_total counter");
        let _ = writeln!(out, "{name}_rejected_total{{reason=\"non_finite\"}} {}", st.rejected_non_finite);
        let _ = writeln!(out, "{name}_rejected_total{{reason=\"negative\"}} {}", st.rejected_negative);
    }
}

/// Счётчики/гейджи/гистограммы сервера. Всё через атомики/мьютексы — клон
/// Arc дешёвый, скрейп /metrics читает актуальные значения.
#[derive(Default)]
pub struct Metrics {
    transcribe_total: AtomicU64,
    transcribe_errors: AtomicU64,
    overload_total: AtomicU64,
    timeout_total: AtomicU64,
    /// Активные batch-запросы (захваченные admission-пермиты).
    active_transcribe: AtomicU64,
    /// Чанки, поставленные в очередь воркеров и ещё не отвеченные.
    worker_pending: AtomicU64,
    max_transcribe: AtomicU64,
    lat: Hist,
    lat_inference: Hist,
    lat_decode: Hist,
}

impl Metrics {
    pub fn new_arc() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set_max_transcribe(&self, v: usize) {
        self.max_transcribe.store(v as u64, Relaxed);
    }

    pub fn transcribe_started(&self) {
        self.transcribe_total.fetch_add(1, Relaxed);
        self.active_transcribe.fetch_add(1, Relaxed);
    }

    /// Активность уезжает вместе с пермитом: учитывается до терминации работы.
    pub fn transcribe_finished(&self) {
        self.active_transcribe.fetch_sub(1, Relaxed);
    }

    pub fn transcribe_error(&self) {
        self.transcribe_errors.fetch_add(1, Relaxed);
    }

    pub fn overload(&self) {
        self.overload_total.fetch_add(1, Relaxed);
    }

    pub fn timeout(&self) {
        self.timeout_total.fetch_add(1, Relaxed);
    }

    pub fn chunk_submitted(&self) {
        self.worker_pending.fetch_add(1, Relaxed);
    }

    pub fn chunk_replied(&self) {
        self.worker_pending.fetch_sub(1, Relaxed);
    }

    pub fn observe_latency(&self, secs: f64) {
        self.lat.observe(secs);
    }

    /// Время инференса чанка (воркер).
    pub fn observe_inference(&self, secs: f64) {
        self.lat_inference.observe(secs);
    }

    /// Время декодирования входа (+VAD).
    pub fn observe_decode(&self, secs: f64) {
        self.lat_decode.observe(secs);
    }

    /// Текстовый рендер Prometheus (text/plain; version=0.0.4).
    pub fn render(&self) -> String {
        let mut s = String::new();
        let c = |o: &AtomicU64| o.load(Relaxed);
        macro_rules! counter {
            ($n:expr, $v:expr) => {{
                let _ = writeln!(s, "# TYPE {} counter\n{} {}", $n, $n, $v);
            }};
        }
        macro_rules! gauge {
            ($n:expr, $v:expr) => {{
                let _ = writeln!(s, "# TYPE {} gauge\n{} {}", $n, $n, $v);
            }};
        }
        counter!("lighttrans_transcribe_requests_total", c(&self.transcribe_total));
        counter!("lighttrans_transcribe_errors_total", c(&self.transcribe_errors));
        counter!("lighttrans_overload_total", c(&self.overload_total));
        counter!("lighttrans_timeout_total", c(&self.timeout_total));
        gauge!("lighttrans_active_transcribe", c(&self.active_transcribe));
        gauge!("lighttrans_max_transcribe", c(&self.max_transcribe));
        gauge!("lighttrans_worker_pending", c(&self.worker_pending));
        let _ = writeln!(s, "# TYPE lighttrans_build_info gauge");
        let _ = writeln!(
            s,
            "lighttrans_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        );
        self.lat.render(&mut s, "lighttrans_transcribe_latency_seconds");
        self.lat_inference.render(&mut s, "lighttrans_inference_seconds");
        self.lat_decode.render(&mut s, "lighttrans_decode_seconds");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms_reject_non_finite_and_count_buckets_cumulatively() {
        let m = Metrics::default();
        m.observe_latency(0.3);
        m.observe_latency(0.02);
        m.observe_latency(f64::NAN);
        m.observe_latency(-1.0);
        let r = m.render();
        assert!(r.contains("lighttrans_transcribe_latency_seconds_count 2"));
        assert!(r.contains("lighttrans_transcribe_latency_seconds_rejected_total{reason=\"non_finite\"} 1"));
        assert!(r.contains("lighttrans_transcribe_latency_seconds_rejected_total{reason=\"negative\"} 1"));
        // бакеты накопительные: 0.02 и 0.3 оба ≤ 0.5
        assert!(r.contains("lighttrans_transcribe_latency_seconds_bucket{le=\"0.5\"} 2"));
        assert!(r.contains("lighttrans_transcribe_latency_seconds_bucket{le=\"0.1\"} 1"));
    }
}
