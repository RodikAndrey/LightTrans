//! Stage timing registry: every pipeline stage records its start time and
//! duration here; at the end of the run a summary table is printed
//! (stage -> when it started, how long it took).

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone)]
struct SpanRec {
    name: String,
    file: Option<String>,
    start: f64, // seconds since process start
    dur: f64,
}

/// Repeated work accumulated across calls (e.g. per-chunk inference): total
/// busy time, first start, last end, call count.
#[derive(Clone)]
struct AccumRec {
    name: String,
    file: String,
    count: u64,
    total: f64,
    first: Option<f64>,
    last: f64,
}

#[derive(Default)]
struct Stats {
    spans: Vec<SpanRec>,
    accum: Vec<AccumRec>,
    audio: BTreeMap<String, f64>, // file -> decoded audio seconds
}

static T0: OnceLock<Instant> = OnceLock::new();
static STATS: OnceLock<Mutex<Stats>> = OnceLock::new();

fn now() -> f64 {
    T0.get().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
}

fn with<R>(f: impl FnOnce(&mut Stats) -> R) -> Option<R> {
    STATS.get()?.lock().ok().map(|mut g| f(&mut g))
}

/// Call once at the very start of main; all times are measured from here.
pub fn init() {
    let _ = T0.set(Instant::now());
    let _ = STATS.set(Mutex::new(Stats::default()));
}

/// RAII span: records its duration into the registry when dropped.
pub struct SpanGuard {
    name: String,
    file: Option<String>,
    start: f64,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let dur = now() - self.start;
        with(|s| {
            s.spans.push(SpanRec {
                name: std::mem::take(&mut self.name),
                file: self.file.take(),
                start: self.start,
                dur,
            });
        });
    }
}

pub fn span(name: &str, file: Option<&str>) -> SpanGuard {
    SpanGuard { name: name.to_string(), file: file.map(Into::into), start: now() }
}

/// Accumulate a recurring duration (one call per chunk/work item).
pub fn add(name: &str, file: &str, dur: f64) {
    let t = now();
    with(|s| match s.accum.iter_mut().find(|a| a.name == name && a.file == file) {
        Some(a) => {
            a.count += 1;
            a.total += dur;
            a.last = t;
        }
        None => s.accum.push(AccumRec {
            name: name.to_string(),
            file: file.to_string(),
            count: 1,
            total: dur,
            first: Some(t),
            last: t,
        }),
    });
}

/// Record decoded audio length of an input file (for the RTF total).
pub fn set_audio(file: &str, secs: f64) {
    with(|s| {
        s.audio.insert(file.to_string(), secs);
    });
}

fn fmt_clock(secs: f64) -> String {
    let s = secs.max(0.0);
    let m = (s / 60.0).floor() as u64;
    let r = s - m as f64 * 60.0;
    format!("{m}:{r:04.1}")
}

/// Build the final summary table (returned as a string for the caller to print).
pub fn report() -> String {
    let total = now();
    let (spans, accum, audio) = with(|s| {
        (s.spans.clone(), s.accum.clone(), s.audio.clone())
    })
    .unwrap_or_default();

    let mut rows: Vec<(f64, String)> = Vec::new();
    for sp in spans {
        rows.push((sp.start, format!("{:<28} {:>8}  {:>7.2}s", sp.name, fmt_clock(sp.start), sp.dur)));
    }
    for ac in accum {
        let wall = ac.last - ac.first.unwrap_or(ac.last);
        rows.push((
            ac.first.unwrap_or(0.0),
            format!(
                "{:<28} {:>8}  {:>7.2}s  ({} чанков, окно {:.2}s)",
                ac.name,
                fmt_clock(ac.first.unwrap_or(0.0)),
                ac.total,
                ac.count,
                wall,
            ),
        ));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let audio_total: f64 = audio.values().sum();
    let mut out = String::new();
    out.push_str("\n════ СВОДКА ПО ВРЕМЕНИ ════\n");
    out.push_str(format!("{:<28} {:>8}  {:>9}\n", "этап", "старт", "длит.").as_str());
    out.push_str(&"-".repeat(64));
    out.push('\n');
    for (_, row) in rows {
        out.push_str(&row);
        out.push('\n');
    }
    out.push_str(&"-".repeat(64));
    out.push('\n');
    for (file, secs) in &audio {
        out.push_str(&format!("{:<28} {:>8}  {:>7.1}s\n", "аудио", file_short(file), secs));
    }
    out.push_str(&format!(
        "{:<28} {:>8}  {:>7.1}s\n",
        "ИТОГО (настенное время)",
        fmt_clock(total),
        total
    ));
    if audio_total > 0.0 {
        out.push_str(&format!(
            "аудио всего: {:.0}s, RTF общий: {:.3} ({}x быстрее реального времени)\n",
            audio_total,
            total / audio_total,
            format_rate(audio_total / total),
        ));
    }
    out
}

fn format_rate(x: f64) -> String {
    if x >= 10.0 {
        format!("{:.0}", x)
    } else if x >= 1.0 {
        format!("{:.1}", x)
    } else {
        format!("{:.2}", x)
    }
}

/// Shorten a path for the table: keep the file name only.
fn file_short(file: &str) -> String {
    let name = file.rsplit(['\\', '/']).next().unwrap_or(file);
    let mut s: String = name.chars().take(28).collect();
    if name.len() > s.len() {
        s.push('…');
    }
    s
}
