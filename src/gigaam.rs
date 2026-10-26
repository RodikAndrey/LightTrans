//! GigaAM v3 E2E CTC (Russian, ONNX) via ONNX Runtime with CUDA.
//!
//! Pipeline: 16 kHz mono f32 -> log-mel (64 bins, hop 10 ms) -> Conformer encoder
//! (ONNX) -> CTC greedy decode -> words with frame-level timestamps.

use crate::mel::MelExtractor;
use crate::transcript::{Segment, Word};
use anyhow::{Context, Result};
use ort::ep::{CUDA, ExecutionProviderDispatch};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

pub struct GigaAmV3 {
    session: Session,
    vocab: Vec<String>,
    blank_id: usize,
    mel: MelExtractor,
    input_names: Vec<String>,
    logprobs_out: String,
    lengths_out: Option<String>,
}

impl GigaAmV3 {
    pub fn load(model: &Path, vocab_path: &Path, use_cuda: bool, intra_threads: usize) -> Result<Self> {
        let eps: Vec<ExecutionProviderDispatch> = if use_cuda {
            vec![CUDA::default().build()]
        } else {
            vec![]
        };

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers(eps)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model)
            .with_context(|| format!("load gigaam model: {}", model.display()))?;

        let (vocab, blank_id) = load_vocab(vocab_path)
            .with_context(|| format!("load vocab: {}", vocab_path.display()))?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        // Фаза 3: сверка IO-имён ONNX-графа с манифестом (если он есть)
        if let Some(m) = crate::manifest::load_for(model.parent().unwrap_or(Path::new("."))) {
            m.verify_io("gigaam-v3 CTC", &input_names, &output_names)?;
        }

        // log_probs output: [1, T/4, num_classes]; optionally a second output
        // carries encoded_lengths (absent in some exports — then use shape[1])
        let (logprobs_out, lengths_out) = match output_names.iter().position(|n| n.contains("log")) {
            Some(i) => (
                output_names[i].clone(),
                output_names
                    .iter()
                    .enumerate()
                    .find(|&(j, _)| j != i)
                    .map(|(_, n)| n.clone()),
            ),
            None => (output_names[0].clone(), output_names.get(1).cloned()),
        };

        Ok(Self {
            session,
            vocab,
            blank_id,
            mel: MelExtractor::new(),
            input_names,
            logprobs_out,
            lengths_out,
        })
    }

    /// Transcribe one chunk; returns a segment with absolute timestamps.
    pub fn transcribe_chunk(&mut self, samples: &[f32], offset: f64) -> Result<Option<Segment>> {
        let (features, n_frames) = self.mel.compute(samples);
        if n_frames == 0 {
            return Ok(None);
        }

        let feats = Tensor::from_array((vec![1i64, 64, n_frames as i64], features))?;
        let lengths = Tensor::from_array((vec![1i64], vec![n_frames as i64]))?;

        let outputs = if self.input_names.len() >= 2 {
            let n0 = self.input_names[0].clone();
            let n1 = self.input_names[1].clone();
            self.session
                .run(ort::inputs![n0.as_str() => feats, n1.as_str() => lengths])?
        } else {
            self.session
                .run(ort::inputs!["features" => feats, "feature_lengths" => lengths])?
        };

        let (lp_shape, logprobs) = outputs[self.logprobs_out.as_str()].try_extract_tensor::<f32>()?;
        let enc_len = if let Some(len_out) = &self.lengths_out {
            let (_enc_shape, enc) = outputs[len_out.as_str()].try_extract_tensor::<i64>()?;
            (enc.first().copied().unwrap_or(0)).max(0) as usize
        } else {
            *lp_shape.get(1).context("log_probs rank < 2")? as usize
        };
        if enc_len == 0 {
            return Ok(None);
        }
        let classes = *lp_shape.last().context("log_probs rank < 2")? as usize;

        // CTC greedy: argmax per frame, collapse repeats, drop blanks
        let mut toks: Vec<(usize, usize, f32)> = Vec::new(); // (frame, id, prob)
        let mut prev: i64 = -1;
        for t in 0..enc_len {
            let row = &logprobs[t * classes..(t + 1) * classes];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            if best != self.blank_id && best as i64 != prev {
                toks.push((t, best, bv.exp()));
            }
            prev = best as i64;
        }
        if std::env::var("LIGHTTRANS_DEBUG_TOKENS").is_ok() {
            let shift_dbg = samples.len() as f64 / 16_000.0 / enc_len as f64;
            for (t, id, p) in &toks {
                eprintln!(
                    "tok frame={t} t={:.3}s piece={:?} p={p:.2}",
                    *t as f64 * shift_dbg, self.vocab[*id]
                );
            }
        }
        if toks.is_empty() {
            return Ok(None);
        }

        let dur = samples.len() as f64 / 16_000.0;
        let shift = dur / enc_len as f64; // seconds per encoder frame (~40 ms)

        let raw_text: String = toks
            .iter()
            .map(|&(_, id, _)| self.vocab[id].clone())
            .collect::<Vec<_>>()
            .join("")
            .replace('▁', " ");
        let text = raw_text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            return Ok(None);
        }

        let words = build_words(&self.vocab, &toks, shift, offset);

        Ok(Some(Segment {
            start: offset,
            end: offset + dur,
            text,
            words,
            speaker: None,
            channel: None,
        }))
    }
}

/// Word assembly shared by the CTC and RNN-T GigaAM backends: a new word starts
/// at a piece beginning with ▁. Word TIMING comes from letter tokens only: e2e
/// models emit punctuation (., —) in pauses after the word, which would
/// otherwise stretch the word's end into silence. `toks` are (frame, id, prob).
pub(crate) fn build_words(
    vocab: &[String],
    toks: &[(usize, usize, f32)],
    shift: f64,
    offset: f64,
) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut cur = String::new();
    let mut cs = 0usize;
    let mut ce = 0usize;
    let mut has_letters = false;
    let mut probs: Vec<f32> = Vec::new();
    let flush = |cur: &mut String,
                     cs: usize,
                     ce: usize,
                     has_letters: bool,
                     ps: &mut Vec<f32>,
                     words: &mut Vec<Word>| {
        if has_letters && !ps.is_empty() {
            let score = ps.iter().sum::<f32>() / ps.len() as f32;
            words.push(Word {
                word: std::mem::take(cur),
                start: offset + cs as f64 * shift,
                end: offset + (ce + 1) as f64 * shift,
                score: Some((score * 1000.0).round() / 1000.0),
                speaker: None,
                channel: None,
            });
        }
        cur.clear();
        ps.clear();
    };

    for &(t, id, p) in toks {
        let piece = &vocab[id];
        let starts_word = piece.starts_with('▁');
        if starts_word && !cur.is_empty() {
            flush(&mut cur, cs, ce, has_letters, &mut probs, &mut words);
            has_letters = false;
        }
        let body = piece.trim_start_matches('▁');
        if body.is_empty() {
            continue; // lone-space token: boundary only
        }
        cur.push_str(body);
        if body.chars().any(|c| c.is_alphanumeric()) {
            if !has_letters {
                cs = t;
                has_letters = true;
            }
            ce = t;
            probs.push(p);
        }
    }
    flush(&mut cur, cs, ce, has_letters, &mut probs, &mut words);
    words
}

/// Vocab file format: "<piece> <id>" per line (e.g. "▁ 1"). Pieces never contain
/// plain spaces (sentencepiece), so parse from the right. Returns (vocab, blank_id);
/// the blank token is the one with the highest id ("<blk> 256" in the official file).
pub(crate) fn load_vocab(path: &Path) -> Result<(Vec<String>, usize)> {
    let text = std::fs::read_to_string(path)?;
    let mut pairs: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        match line.rsplit_once(' ') {
            Some((piece, id)) => pairs.push((id.parse::<usize>()?, piece.to_string())),
            None => pairs.push((pairs.len(), line.to_string())),
        }
    }
    pairs.sort_by_key(|&(id, _)| id);
    let blank_id = pairs.last().map(|&(id, _)| id).context("empty vocab")?;
    let vocab = pairs.into_iter().map(|(_, p)| p).collect();
    Ok((vocab, blank_id))
}
