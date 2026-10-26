//! WhisperX-style word alignment: the ASR model runs WITHOUT word timestamps
//! (fast), then a small char-level CTC model restores word timing via forced
//! alignment (Viterbi trellis over its frame emissions — same scheme as
//! WhisperX's wav2vec2 aligner).
//!
//! Backend: GigaAM v3 E2E CTC (models/gigaam-v3, already required for the
//! default model). Its char alphabet covers Russian + digits + punctuation,
//! and its inference is ~10x cheaper than DTW inside whisper.cpp; the
//! wav2vec2-large aligner (models/wav2vec2-align-ru, exported by
//! scripts/export_w2v_align.py) costs ~263 s of GPU per 93 min of audio via
//! ORT vs ~20 s for GigaAM, so it is kept only as a fallback option.
//!
//! One inference per chunk; all segments of the chunk are aligned in a single
//! DP pass so words cannot drift into pauses away from their neighbours.

use crate::mel::MelExtractor;
use crate::transcript::{Segment, Word};
use anyhow::{Context, Result};
use ort::ep::{CUDA, ExecutionProviderDispatch};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;

pub struct WordAligner {
    session: Session,
    mel: MelExtractor,
    input_names: Vec<String>,
    logprobs_out: String,
    lengths_out: Option<String>,
    /// single-char vocab entries: char -> token id
    char_map: HashMap<char, usize>,
    /// id -> char (single-char entries; ' ' elsewhere)
    chars: Vec<char>,
    sep: usize,
    blank: usize,
}

impl WordAligner {
    pub fn load(model: &Path, vocab_path: &Path, use_cuda: bool, intra: usize) -> Result<Self> {
        let eps: Vec<ExecutionProviderDispatch> =
            if use_cuda { vec![CUDA::default().build()] } else { vec![] };
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(intra.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers(eps)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model)
            .with_context(|| format!("load aligner model: {}", model.display()))?;

        let (vocab, blank) = crate::gigaam::load_vocab(vocab_path)
            .with_context(|| format!("load aligner vocab: {}", vocab_path.display()))?;
        let mut char_map = HashMap::new();
        let mut chars = vec![' '; vocab.len()];
        for (id, piece) in vocab.iter().enumerate() {
            let mut pcs = piece.chars();
            if let (Some(ch), None) = (pcs.next(), pcs.next()) {
                chars[id] = ch;
                if id != blank {
                    char_map.insert(ch, id);
                }
            }
        }
        let sep = *char_map.get(&'\u{2581}').context("aligner vocab has no ▁")?;

        let input_names: Vec<String> =
            session.inputs().iter().map(|i| i.name().to_string()).collect();
        let output_names: Vec<String> =
            session.outputs().iter().map(|o| o.name().to_string()).collect();
        let (logprobs_out, lengths_out) = match output_names.iter().position(|n| n.contains("log")) {
            Some(i) => (
                output_names[i].clone(),
                output_names.iter().enumerate().find(|&(j, _)| j != i).map(|(_, n)| n.clone()),
            ),
            None => (output_names[0].clone(), output_names.get(1).cloned()),
        };

        Ok(Self {
            session,
            mel: MelExtractor::new(),
            input_names,
            logprobs_out,
            lengths_out,
            char_map,
            chars,
            sep,
            blank,
        })
    }

    /// Align all segments of one chunk in place. `samples` is the chunk audio
    /// (16 kHz), `offset` its absolute start; word times become absolute.
    pub fn align_chunk(&mut self, samples: &[f32], offset: f64, segs: &mut [Segment]) -> Result<()> {
        if samples.len() < 1600 {
            return Ok(()); // too short for meaningful alignment
        }
        // CTC emissions: log_probs [1, T_enc, C], already log-softmax
        let (t_len, classes, emission) = {
            let (features, n_frames) = self.mel.compute(samples);
            if n_frames == 0 {
                return Ok(());
            }
            let feats = Tensor::from_array((vec![1i64, 64, n_frames as i64], features))?;
            let lengths = Tensor::from_array((vec![1i64], vec![n_frames as i64]))?;
            let outputs = if self.input_names.len() >= 2 {
                let n0 = self.input_names[0].clone();
                let n1 = self.input_names[1].clone();
                self.session.run(ort::inputs![n0.as_str() => feats, n1.as_str() => lengths])?
            } else {
                self.session
                    .run(ort::inputs!["features" => feats, "feature_lengths" => lengths])?
            };
            let (lp_shape, logprobs) = outputs[self.logprobs_out.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("aligner emissions: {e}"))?;
            let t_enc = if let Some(len_out) = &self.lengths_out {
                let (_s, enc) = outputs[len_out.as_str()].try_extract_tensor::<i64>()?;
                (enc.first().copied().unwrap_or(0)).max(0) as usize
            } else {
                *lp_shape.get(1).context("log_probs rank < 2")? as usize
            };
            let classes = *lp_shape.last().context("log_probs rank < 2")? as usize;
            let t_len = (t_enc as usize).min(lp_shape.get(1).copied().unwrap_or(0) as usize);
            if t_len < 4 {
                return Ok(());
            }
            (t_len, classes, logprobs[..t_len * classes].to_vec())
        };
        let frame_s = samples.len() as f64 / 16_000.0 / t_len as f64;

        // One DP over the whole chunk (like WhisperX, which aligns a VAD
        // chunk in a single pass): the path is then constrained by ALL words,
        // so a word cannot drift into a pause away from its neighbours.
        // Flatten the words of all segments, align, write the times back.
        struct WRef {
            seg: usize,
            text: String,
            speaker: Option<String>,
            channel: Option<usize>,
        }
        let mut all_words: Vec<WRef> = Vec::new();
        for (si, seg) in segs.iter().enumerate() {
            for w in seg.text.split_whitespace() {
                all_words.push(WRef {
                    seg: si,
                    text: w.to_string(),
                    speaker: seg.words.first().and_then(|w| w.speaker.clone()),
                    channel: seg.channel,
                });
            }
        }
        if all_words.is_empty() {
            return Ok(());
        }

        // char sequence: word chars (lowercased, only chars present in the
        // vocab as single-char tokens), words separated by the ▁ separator
        let mut char_tok: Vec<usize> = Vec::new(); // token id per DP state
        let mut char_word: Vec<usize> = Vec::new(); // word index per DP state
        for (wi, w) in all_words.iter().enumerate() {
            if wi > 0 {
                char_tok.push(self.sep);
                char_word.push(wi);
            }
            for ch in w.text.chars() {
                if let Some(&id) = self.char_map.get(&ch) {
                    char_tok.push(id);
                    char_word.push(wi);
                }
            }
        }
        if char_tok.is_empty() {
            return Ok(());
        }
        let s_len = char_tok.len();
        let blank = self.blank;

        // CTC trellis (forced-alignment DP): state s = "currently on char s"
        // (entered by emitting char_tok[s]); staying emits blank. trellis[t][s]
        // = best score; via_change[t][s] marks transitions that EMIT char s
        // at frame t.
        let mut trellis = vec![f32::NEG_INFINITY; t_len * s_len];
        let mut via_change = vec![false; t_len * s_len];
        let em = |t: usize, tok: usize| emission[t * classes + tok];
        trellis[0] = em(0, blank);
        if s_len > 1 {
            // at t=0 the path may either sit on blank (state 0) or emit the
            // first char right away (state 1)
            trellis[s_len] = em(0, char_tok[0]);
            via_change[s_len] = true;
            for s in 2..s_len {
                trellis[s] = f32::NEG_INFINITY;
            }
        }
        for t in 1..t_len {
            let row = t * s_len;
            let prev = row - s_len;
            // state 0: stay on blank
            trellis[row] = trellis[prev] + em(t, blank);
            for s in 1..s_len {
                let stay = trellis[prev + s] + em(t, blank);
                let change = trellis[prev + s - 1] + em(t, char_tok[s]);
                if change > stay {
                    trellis[row + s] = change;
                    via_change[row + s] = true;
                } else {
                    trellis[row + s] = stay;
                }
            }
        }

        // backtrack: forced alignment must END at the last char (backtracking
        // from an argmax state would allow the path to stop early, which is
        // nearly free where the model emits confident blanks in silence)
        let final_row = (t_len - 1) * s_len;
        let mut s = s_len - 1;
        if trellis[final_row + s] == f32::NEG_INFINITY {
            s = (0..s_len)
                .max_by(|&a, &b| trellis[final_row + a].total_cmp(&trellis[final_row + b]))
                .unwrap();
        }
        if trellis[final_row + s] == f32::NEG_INFINITY {
            return Ok(());
        }
        // emission frame per char (None = char not reached)
        let mut char_at: Vec<Option<usize>> = vec![None; s_len];
        let mut t = t_len - 1;
        loop {
            if via_change[t * s_len + s] {
                char_at[s] = Some(t);
                if s == 0 || t == 0 {
                    break;
                }
                s -= 1;
                t -= 1;
            } else {
                if t == 0 {
                    break;
                }
                t -= 1;
            }
        }

        // word times: [first char frame, successor char frame); refine both
        // edges by the emission: a char "sounds" only while its probability
        // beats blank, so a pause absorbed into the DP span is trimmed off
        let mut wt: Vec<(f64, f64, Option<f32>)> = vec![(0.0, 0.0, None); all_words.len()];
        for wi in 0..all_words.len() {
            let idxs: Vec<usize> = (0..s_len)
                .filter(|&s| char_word[s] == wi && char_at[s].is_some())
                .collect();
            if idxs.is_empty() {
                continue;
            }
            // word TIMING comes from letter tokens only: e2e CTC models emit
            // punctuation (., —) in pauses after the word, which would stretch
            // the word's end into silence
            let first_alpha = idxs
                .iter()
                .find(|&&s| self.chars[char_tok[s]].is_alphanumeric())
                .or_else(|| idxs.first())
                .copied()
                .unwrap();
            let last_alpha = idxs
                .iter()
                .rev()
                .find(|&&s| self.chars[char_tok[s]].is_alphanumeric())
                .or_else(|| idxs.last())
                .copied()
                .unwrap();
            let t_start = char_at[first_alpha].unwrap();
            let t_end = match char_at.get(idxs.last().unwrap() + 1) {
                Some(Some(nt)) => *nt,
                _ => char_at[last_alpha].unwrap() + 1,
            };
            let c_first = char_tok[first_alpha];
            let c_last = char_tok[last_alpha];
            let mut a = t_start;
            for f in t_start..t_end.max(t_start + 1) {
                if em(f, c_first) > em(f, blank) {
                    a = f;
                    break;
                }
            }
            // word end: last frame where the final char beats blank
            let mut b = t_start + 1;
            for f in (t_start + 1..t_end).rev() {
                if em(f, c_last) > em(f, blank) {
                    b = f + 1;
                    break;
                }
            }
            let start = offset + a as f64 * frame_s;
            let end = offset + b.max(a + 1).min(t_len) as f64 * frame_s;
            let mut lp_sum = 0f32;
            for &s in &idxs {
                lp_sum += em(char_at[s].unwrap(), char_tok[s]);
            }
            wt[wi] = (
                start,
                end.max(start + frame_s),
                Some(((lp_sum / idxs.len() as f32).exp() * 1000.0).round() / 1000.0),
            );
        }

        // interpolate missing words between aligned neighbours (across segment
        // boundaries; trailing gaps stretch to the end of the chunk emission)
        let chunk_end = offset + t_len as f64 * frame_s;
        let mut last_end = offset;
        let mut gap: Vec<usize> = Vec::new();
        for wi in 0..=wt.len() {
            let aligned = wi < wt.len() && wt[wi].2.is_some();
            if !aligned {
                gap.push(wi);
                continue;
            }
            let next_start = wt[wi].0;
            let n = gap.len() as f64;
            for (k, gi) in gap.iter().enumerate() {
                if *gi >= wt.len() {
                    continue;
                }
                let a = last_end + (next_start - last_end) * (k as f64 + 1.0) / (n + 1.0);
                let b = last_end + (next_start - last_end) * (k as f64 + 2.0) / (n + 1.0);
                wt[*gi].0 = a;
                wt[*gi].1 = b.max(a + 0.01);
            }
            gap.clear();
            last_end = wt[wi].1;
        }
        if !gap.is_empty() {
            let n = gap.len() as f64;
            for (k, gi) in gap.iter().enumerate() {
                if *gi >= wt.len() {
                    continue;
                }
                let a = last_end + (chunk_end - last_end) * (k as f64 + 1.0) / (n + 1.0);
                let b = last_end + (chunk_end - last_end) * (k as f64 + 2.0) / (n + 1.0);
                wt[*gi].0 = a;
                wt[*gi].1 = b.max(a + 0.01);
            }
        }

        // distribute back into the segments (index-aligned with all_words)
        for (si, seg) in segs.iter_mut().enumerate() {
            let mut words: Vec<Word> = Vec::new();
            for (wi, w) in all_words.iter().enumerate() {
                if w.seg != si {
                    continue;
                }
                let (start, end, score) = wt[wi];
                words.push(Word {
                    word: w.text.clone(),
                    start,
                    end,
                    score,
                    speaker: w.speaker.clone(),
                    channel: w.channel,
                });
            }
            if words.is_empty() {
                continue;
            }
            if words.iter().all(|w| w.score.is_none()) {
                continue; // nothing aligned in this segment — keep original words
            }
            seg.words = words;
        }
        Ok(())
    }
}
