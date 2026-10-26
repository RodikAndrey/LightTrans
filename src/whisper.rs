//! Whisper large-v3-turbo (ggml) via whisper.cpp CUDA backend,
//! with WhisperX-like defaults: no previous-text conditioning, DTW token
//! timestamps for word-level results.

use crate::transcript::{Segment, Word};
use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

pub struct WhisperTurbo {
    ctx: WhisperContext,
    dtw: bool,
}

pub struct WhisperOpts<'a> {
    /// ISO code or "auto"
    pub language: &'a str,
    pub translate: bool,
    pub beam_size: usize,
    pub n_threads: usize,
}

impl WhisperTurbo {
    pub fn load(model: &Path, use_gpu: bool, dtw: bool) -> Result<Self> {
        // DTW token timestamps give accurate word-level timing but are
        // incompatible with flash attention. Without DTW (--w2v mode or no
        // alignment) whisper runs ~2.2x faster and flash attention speeds up
        // the encoder further; word timing is then either restored by the CTC
        // aligner or taken from the coarse segment token t0/t1.
        let mut cp = WhisperContextParameters::new();
        cp.use_gpu(use_gpu);
        cp.flash_attn(!dtw);
        if dtw {
            cp.dtw_parameters(whisper_rs::DtwParameters {
                mode: whisper_rs::DtwMode::TopMost { n_top: 1 },
                dtw_mem_size: 1024 * 1024 * 128,
            });
        }
        let ctx = WhisperContext::new_with_params(
            model
                .to_str()
                .context("model path is not valid UTF-8")?,
            cp,
        )
        .with_context(|| format!("load whisper model: {}", model.display()))?;
        Ok(Self { ctx, dtw })
    }

    pub fn create_state(&self) -> Result<WhisperState> {
        self.ctx.create_state().context("create whisper state")
    }

    /// Transcribe one chunk; returns segments with absolute timestamps and the
    /// detected language code.
    pub fn transcribe_chunk(
        &self,
        state: &mut WhisperState,
        samples: &[f32],
        offset: f64,
        opts: &WhisperOpts,
    ) -> Result<(Vec<Segment>, String)> {
        let mut params = FullParams::new(if opts.beam_size > 1 {
            SamplingStrategy::BeamSearch {
                beam_size: opts.beam_size as i32,
                patience: 1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        });
        params.set_language(Some(opts.language));
        // NOTE: language == "auto" triggers auto-detection inside whisper_full and
        // transcription continues; set_detect_language() would STOP after detection.
        params.set_translate(opts.translate);
        params.set_n_threads(opts.n_threads.max(1) as i32);
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.full(params, samples).context("whisper inference")?;
        tracing::debug!("whisper n_segments = {}", state.full_n_segments());
        let lang_id = state.full_lang_id_from_state();
        let lang = unsafe { lang_code(lang_id) };

        // collect segment texts and a flat token stream
        struct TokRow {
            text: String,
            t0: i64,
            t1: i64,
            tdw: i64,
            p: f32,
        }
        struct SegInfo {
            start: f64,
            end: f64,
            text: String,
            first_row: usize,
            last_row: usize,
            words: Vec<Word>,
        }
        const DTW_LAG_CS: i64 = 25; // decoder attention lags ~250 ms behind audio
        const DTW_END_LAG_CS: i64 = 35; // the switch to the token AFTER a word
                                        // (punctuation/next word) lags even more
        let mut rows: Vec<TokRow> = Vec::new();
        let mut segs: Vec<SegInfo> = Vec::new();
        for seg in state.as_iter() {
            // drop hallucinated segments on silence (whisperx-style filter)
            if seg.no_speech_probability() > 0.6 {
                tracing::debug!(
                    "dropping hallucinated segment [{:.2}-{:.2}] {:?}",
                    seg.start_timestamp() as f64 / 1000.0 + offset,
                    seg.end_timestamp() as f64 / 1000.0 + offset,
                    seg.to_str_lossy().unwrap_or_default().trim()
                );
                continue;
            }
            let text = seg.to_str_lossy().unwrap_or_default().trim().to_string();
            let t0 = offset + seg.start_timestamp() as f64 / 1000.0;
            let t1 = offset + seg.end_timestamp() as f64 / 1000.0;
            let first_row = rows.len();
            let n_tokens = seg.n_tokens().max(0) as usize;
            for i in 0..n_tokens {
                let Some(tok) = seg.get_token(i as i32) else { continue };
                let txt = tok.to_str_lossy().unwrap_or_default().to_string();
                // special/timestamp tokens: whisper.cpp renders them as "<|...|>"
                // or "[_BEG_]" / "[_TT_120]"
                if txt.starts_with("<|") || txt.starts_with("[_") {
                    continue;
                }
                let td = tok.token_data();
                rows.push(TokRow { text: txt, t0: td.t0, t1: td.t1, tdw: td.t_dtw, p: tok.token_probability() });
            }
            segs.push(SegInfo { start: t0, end: t1, text, first_row, last_row: rows.len(), words: Vec::new() });
        }

        // --- word assembly (rows are grouped per segment) ---
        let dtw_available = self.dtw && rows.iter().any(|r| r.tdw > 0);
        if std::env::var("LIGHTTRANS_DEBUG_TOKENS").is_ok() {
            for (k, r) in rows.iter().enumerate() {
                eprintln!(
                    "dtwrow {k} t0={:.2} t1={:.2} tdw={:.2} txt={:?}",
                    r.t0 as f64 / 100.0, r.t1 as f64 / 100.0, r.tdw as f64 / 100.0, r.text
                );
            }
        }
        // whisper.cpp's own segment t0/t1 are unreliable (often collapsed); word
        // boundaries from DTW are, so segment spans are rebuilt from their words.
        let mut prev_end_cs = 0i64;
        for seg in segs.iter_mut() {
            let seg_rows = &rows[seg.first_row..seg.last_row];
            let mut words: Vec<Word> = Vec::new();
            if dtw_available {
                // DTW: token t_dtw marks where the decoder switches to the token —
                // systematically ~250 ms after the token's audio actually begins
                // (attention lag), so correct it. A word starts at its first
                // letter token's corrected timestamp and ends at the corrected
                // timestamp of the row that follows its audio (trailing
                // punctuation or the next word); punctuation is part of the text.
                let fix = |tdw: i64| -> i64 { (tdw - DTW_LAG_CS).max(0) };
                let fix_end = |tdw: i64| -> i64 { (tdw - DTW_END_LAG_CS).max(0) };
                let mut cur = String::new();
                let mut cur_start: Option<i64> = None;
                let mut cur_end: Option<i64> = None;
                let mut probs: Vec<f32> = Vec::new();
                for row in seg_rows {
                    if row.text.starts_with(' ') && !cur.is_empty() {
                        // punctuation-only tokens (e.g. a lone "-") carry no
                        // spoken content and can inherit inflated switch points
                        // from echo/repetition; drop them from the word list
                        if cur.chars().any(|c| c.is_alphanumeric()) {
                            let start = cur_start.unwrap_or(prev_end_cs).max(prev_end_cs);
                            let end = (cur_end.unwrap_or(start + 1)).max(start + 1);
                            words.push(Word {
                                word: std::mem::take(&mut cur),
                                start: offset + start as f64 / 100.0,
                                end: offset + end as f64 / 100.0,
                                score: Some(score(&probs)),
                                speaker: None,
                                channel: None,
                            });
                            prev_end_cs = end;
                        } else {
                            cur.clear();
                        }
                        cur_start = None;
                        cur_end = None;
                        probs.clear();
                    }
                    let body = row.text.trim_start_matches(' ');
                    if body.is_empty() {
                        continue;
                    }
                    if cur.is_empty() && body.chars().any(|c| c.is_alphanumeric()) {
                        cur_start = Some(fix(row.tdw));
                    }
                    cur.push_str(body);
                    cur_end = Some(fix_end(row.tdw));
                    probs.push(row.p);
                }
                if !cur.is_empty() && cur.chars().any(|c| c.is_alphanumeric()) {
                    let start = cur_start.unwrap_or(prev_end_cs).max(prev_end_cs);
                    let end = (cur_end.unwrap_or(start + 1)).max(start + 1);
                    words.push(Word {
                        word: std::mem::take(&mut cur),
                        start: offset + start as f64 / 100.0,
                        end: offset + end as f64 / 100.0,
                        score: Some(score(&probs)),
                        speaker: None,
                        channel: None,
                    });
                    prev_end_cs = end;
                }
                cur.clear();
            } else {
                // fallback: classic token t0/t1 (cs) — one word per token
                for row in seg_rows {
                    words.push(Word {
                        word: row.text.trim_start_matches(' ').to_string(),
                        start: offset + row.t0 as f64 / 100.0,
                        end: (row.t1.max(row.t0 + 1)) as f64 / 100.0 + offset,
                        score: Some(row.p),
                        speaker: None,
                        channel: None,
                    });
                }
            }
            if !words.is_empty() {
                seg.start = words[0].start;
                seg.end = words.last().unwrap().end;
            }
            seg.words = words;
        }

        let out = segs
            .into_iter()
            .filter(|s| !s.text.is_empty())
            .map(|s| Segment { start: s.start, end: s.end, text: s.text, words: s.words, speaker: None, channel: None })
            .collect();
        Ok((out, lang))
    }
}

fn score(probs: &[f32]) -> f32 {
    if probs.is_empty() {
        return 0.0;
    }
    ((probs.iter().sum::<f32>() / probs.len() as f32) * 1000.0).round() / 1000.0
}

/// whisper.cpp language id -> ISO code (via whisper.cpp API, e.g. "ru", "en").
unsafe fn lang_code(id: i32) -> String {
    let ptr = whisper_rs::whisper_rs_sys::whisper_lang_str(id);
    if ptr.is_null() {
        return "unknown".into();
    }
    std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
}
