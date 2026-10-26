//! NVIDIA Parakeet TDT 0.6B v3 (multilingual FastConformer + TDT, ONNX).
//!
//! ONNX export: huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx (cc-by-4.0).
//! Three graphs: `nemo128.onnx` (16 kHz waveform -> 128-mel log features),
//! `encoder-model.onnx` (FastConformer, 8x subsampling -> [B, hidden, T_enc]),
//! `decoder_joint-model.onnx` (TDT prediction + joint net: one encoder frame and
//! the previously emitted token -> vocab logits + 5 duration logits, LSTM state).
//! Greedy TDT decoding and word assembly mirror the onnx-asr reference.

use crate::transcript::{Segment, Word};
use anyhow::{Context, Result};
use ort::ep::{CUDA, ExecutionProviderDispatch};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const ENCODER_FRAME_S: f64 = 0.08; // 10 ms mel hop * 8 subsampling
const DURATION_CLASSES: usize = 5;
const MAX_TOKENS_PER_STEP: usize = 10;
const N_MELS: usize = 128;
const PRED_HIDDEN: usize = 640;

pub struct ParakeetV3 {
    mel: Session,     // waveform -> features (CPU)
    encoder: Session, // features -> encodings (CUDA/CPU)
    joint: Session,   // TDT step (CPU: tiny graph, called per frame)
    vocab: Vec<String>,
    blank_id: usize,
    vocab_size: usize,
}

impl ParakeetV3 {
    pub fn load(model_dir: &Path, use_cuda: bool, _intra_threads: usize) -> Result<Self> {
        let mel = make_session(&model_dir.join("nemo128.onnx"), false, 2)?;
        let encoder = make_session(&model_dir.join("encoder-model.onnx"), use_cuda, 4)?;
        // the joint net is invoked once per encoder frame (sequential), so it
        // runs on CPU: per-call CUDA launch overhead would dominate
        let joint = make_session(&model_dir.join("decoder_joint-model.onnx"), false, 4)?;

        let vocab_path = model_dir.join("vocab.txt");
        let vocab = std::fs::read_to_string(&vocab_path)
            .with_context(|| format!("read {}", vocab_path.display()))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| match l.rsplit_once(' ') {
                Some((token, _id)) => token.to_string(),
                None => l.to_string(),
            })
            .collect::<Vec<_>>();
        let blank_id = vocab
            .iter()
            .position(|t| t == "<blk>")
            .context("vocab has no <blk> token")?;
        let vocab_size = vocab.len();
        anyhow::ensure!(blank_id + 1 == vocab_size, "unexpected vocab layout");

        Ok(Self { mel, encoder, joint, vocab, blank_id, vocab_size })
    }

    /// Transcribe one chunk; returns a segment with absolute word timestamps.
    pub fn transcribe_chunk(&mut self, samples: &[f32], offset: f64) -> Result<Option<Segment>> {
        if samples.is_empty() {
            return Ok(None);
        }

        // --- mel features ---
        let wave = Tensor::from_array((vec![1i64, samples.len() as i64], samples.to_vec()))?;
        let lens = Tensor::from_array((vec![1i64], vec![samples.len() as i64]))?;
        let mel_out = self.mel.run(ort::inputs!["waveforms" => wave, "waveforms_lens" => lens])?;
        let (feat_shape, features) = mel_out["features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("mel features: {e}"))?;
        let (_, feat_len) = mel_out["features_lens"]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow::anyhow!("mel lens: {e}"))?;
        anyhow::ensure!(feat_shape.len() == 3 && feat_shape[1] == N_MELS as i64);
        let t_mel = feat_len[0] as usize;
        if t_mel == 0 {
            return Ok(None);
        }

        // --- encoder ---
        let feats = Tensor::from_array((
            vec![1i64, N_MELS as i64, t_mel as i64],
            features[..t_mel * N_MELS].to_vec(),
        ))?;
        let enc_in_len = Tensor::from_array((vec![1i64], vec![t_mel as i64]))?;
        let enc_out = self
            .encoder
            .run(ort::inputs!["audio_signal" => feats, "length" => enc_in_len])?;
        let (enc_shape, encodings) = enc_out["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("encoder outputs: {e}"))?;
        let (_, enc_lens) = enc_out["encoded_lengths"]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow::anyhow!("encoder lens: {e}"))?;
        anyhow::ensure!(enc_shape.len() == 3, "unexpected encoder output shape");
        let t_enc = (enc_lens[0] as usize).min(enc_shape[2] as usize);
        let hidden = enc_shape[1] as usize;
        if t_enc == 0 {
            return Ok(None);
        }

        tracing::debug!("parakeet chunk: offset={offset} t_mel={t_mel} t_enc={t_enc} hidden={hidden}");
        // --- greedy TDT decoding (onnx-asr `_AsrWithTransducerDecoding`) ---
        let mut state1 = vec![0f32; 2 * PRED_HIDDEN];
        let mut state2 = vec![0f32; 2 * PRED_HIDDEN];
        struct Tok {
            id: usize,
            t: usize,
            /// duration frames the model consumed for this token (TDT semantics)
            dur: usize,
            logp: f32,
        }
        let mut tokens: Vec<Tok> = Vec::new();
        let mut prev_token = self.blank_id as i64;
        let mut t = 0usize;
        let mut emitted = 0usize;
        let mut frame_col = vec![0f32; hidden];
        while t < t_enc {
            // encoder_outputs: [1, hidden, 1] — column t of [1, hidden, t_enc]
            for h in 0..hidden {
                frame_col[h] = encodings[h * t_enc + t];
            }
            let enc_t = Tensor::from_array((vec![1i64, hidden as i64, 1i64], frame_col.clone()))?;
            // the export declares targets/target_length as int32
            let targets = Tensor::from_array((vec![1i64, 1i64], vec![prev_token as i32]))?;
            let tlen = Tensor::from_array((vec![1i64], vec![1i32]))?;
            let s1 = Tensor::from_array((vec![2i64, 1i64, PRED_HIDDEN as i64], state1.clone()))?;
            let s2 = Tensor::from_array((vec![2i64, 1i64, PRED_HIDDEN as i64], state2.clone()))?;
            let out = self.joint.run(ort::inputs![
                "encoder_outputs" => enc_t,
                "targets" => targets,
                "target_length" => tlen,
                "input_states_1" => s1,
                "input_states_2" => s2,
            ])?;

            let (_, logits) = out["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("joint outputs: {e}"))?;
            let row = &logits[logits.len() - self.vocab_size - DURATION_CLASSES..];
            let (tok_logits, dur_logits) = row.split_at(self.vocab_size);
            let token = argmax(tok_logits);
            let step = argmax(dur_logits);

            let (ns1, ns2) = extract_states(&out)?;
            if token != self.blank_id {
                state1 = ns1;
                state2 = ns2;
                prev_token = token as i64;
                let logp = log_softmax_at(tok_logits, token);
                tokens.push(Tok { id: token, t, dur: step.max(1), logp });
                emitted += 1;
            }

            if step > 0 {
                t += step;
                emitted = 0;
            } else if token == self.blank_id || emitted >= MAX_TOKENS_PER_STEP {
                t += 1;
                emitted = 0;
            }
        }
        if tokens.is_empty() {
            return Ok(None);
        }
        if std::env::var("LIGHTTRANS_DEBUG_TOKENS").is_ok() {
            for tok in tokens.iter().take(20) {
                eprintln!("ptok t={:3} ({:.2}s) {:?} logp={:.3}", tok.t, tok.t as f64 * ENCODER_FRAME_S, self.vocab[tok.id], tok.logp);
            }
        }

        // --- word assembly (sentencepiece: ▁ starts a word) ---
        let mut words: Vec<Word> = Vec::new();
        let mut cur = String::new();
        let mut cur_first_t: Option<f64> = None;
        let mut cur_body_start: Option<f64> = None;
        let mut cur_body_end: Option<f64> = None;
        let mut cur_scores: Vec<f32> = Vec::new();
        let flush = |words: &mut Vec<Word>,
                     cur: &mut String,
                     cur_first_t: &mut Option<f64>,
                     cur_body_start: &mut Option<f64>,
                     cur_body_end: &mut Option<f64>,
                     cur_scores: &mut Vec<f32>| {
            let start = cur_body_start.or(*cur_first_t).unwrap_or(0.0);
            let end = cur_body_end.unwrap_or(start + 0.01).max(start + 0.01);
            if std::env::var("LIGHTTRANS_DEBUG_TOKENS").is_ok() {
                eprintln!("pword offset={offset} start={start:.2} end={end:.2} cur={cur:?}");
            }
            words.push(Word {
                word: std::mem::take(cur),
                start: offset + start,
                end: offset + end,
                score: Some(score(cur_scores)),
                speaker: None,
            });
            *cur_first_t = None;
            *cur_body_start = None;
            *cur_body_end = None;
            cur_scores.clear();
        };
        for tok in tokens.iter() {
            let raw = &self.vocab[tok.id];
            if raw.trim().is_empty() || raw.starts_with('<') {
                continue; // special tokens (<|...|>) carry no text
            }
            let text = raw.replace('\u{2581}', " ");
            let t_start = tok.t as f64 * ENCODER_FRAME_S;
            // token span = its own duration frames; the next token may start
            // much later (silence is skipped by TDT durations), so the next
            // token's start must NOT stretch this token's end
            let t_end = t_start + tok.dur.max(1) as f64 * ENCODER_FRAME_S;
            if raw.starts_with('\u{2581}') && !cur.is_empty() {
                flush(
                    &mut words,
                    &mut cur,
                    &mut cur_first_t,
                    &mut cur_body_start,
                    &mut cur_body_end,
                    &mut cur_scores,
                );
            }
            if cur_first_t.is_none() {
                cur_first_t = Some(t_start);
            }
            let is_body = text.chars().any(|c| c.is_alphanumeric());
            if is_body {
                if cur_body_start.is_none() {
                    cur_body_start = Some(t_start);
                }
                cur_body_end = Some(t_end.max(t_start));
                cur_scores.push(tok.logp);
            }
            cur.push_str(&text);
        }
        if !cur.is_empty() {
            flush(
                &mut words,
                &mut cur,
                &mut cur_first_t,
                &mut cur_body_start,
                &mut cur_body_end,
                &mut cur_scores,
            );
        }
        if words.is_empty() {
            return Ok(None);
        }

        let text: String = tokens
            .iter()
            .map(|tok| self.vocab[tok.id].replace('\u{2581}', " "))
            .filter(|text| !text.trim().is_empty() && !text.starts_with('<'))
            .collect::<Vec<_>>()
            .join("")
            .trim()
            .to_string();
        let start = words[0].start;
        let end = words.last().unwrap().end;
        Ok(Some(Segment { start, end, text, words, speaker: None }))
    }
}

fn score(probs: &[f32]) -> f32 {
    if probs.is_empty() {
        return 0.0;
    }
    ((probs.iter().sum::<f32>() / probs.len() as f32) * 1000.0).round() / 1000.0
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

fn log_softmax_at(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    logits[idx] - max - sum.ln()
}

/// Copy the new prediction-network LSTM states out of the joint outputs.
fn extract_states(out: &ort::session::SessionOutputs<'_>) -> Result<(Vec<f32>, Vec<f32>)> {
    let (_, s1) = out["output_states_1"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("state1: {e}"))?;
    let (_, s2) = out["output_states_2"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("state2: {e}"))?;
    Ok((s1.to_vec(), s2.to_vec()))
}

fn make_session(model: &Path, use_cuda: bool, intra: usize) -> Result<Session> {
    let eps: Vec<ExecutionProviderDispatch> =
        if use_cuda { vec![CUDA::default().build()] } else { vec![] };
    Session::builder()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(intra.max(1))
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_execution_providers(eps)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(model)
        .with_context(|| format!("load parakeet model: {}", model.display()))
}
