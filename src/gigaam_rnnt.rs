//! GigaAM v3 E2E RNN-T (Russian, ONNX) via ONNX Runtime with CUDA.
//!
//! ONNX export: huggingface.co/istupakov/gigaam-v3-onnx (MIT). Three graphs:
//! `v3_e2e_rnnt_encoder.onnx` (64-mel -> Conformer -> [B, 768, T_enc] + lengths),
//! `v3_e2e_rnnt_decoder.onnx` (prediction LSTM: token + h/c state -> 320-d dec),
//! `v3_e2e_rnnt_joint.onnx` (enc 768 x dec 320 -> 1025 vocab logits, blank = 1024).
//! Greedy decoding and word assembly mirror the onnx-asr reference
//! (`onnx_asr.models.gigaam.GigaamV2Rnnt`): the prediction network runs only when
//! the last emitted token changes, its output is cached between blank steps.

use crate::gigaam::{build_words, load_vocab};
use crate::mel::MelExtractor;
use crate::ort_assignment::{self, Policy, Role};
use crate::transcript::{Segment, Word};
use anyhow::{Context, Result};
use ort::ep::{CUDA, ExecutionProviderDispatch};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const PRED_HIDDEN: usize = 320;
const HIDDEN: usize = 768;
/// RNN-T: at most this many tokens may be emitted per encoder frame.
const MAX_TOKENS_PER_STEP: usize = 3;

pub struct GigaAmRnnt {
    encoder: Session, // features -> encoded (CUDA/CPU)
    decoder: Session, // prediction LSTM (CPU: tiny, sequential)
    joint: Session,   // joint net (CPU: tiny, called per frame)
    vocab: Vec<String>,
    blank_id: usize,
    mel: MelExtractor,
}

impl GigaAmRnnt {
    pub fn load(
        model_dir: &Path,
        use_cuda: bool,
        intra_threads: usize,
        policy: Option<&Policy>,
    ) -> Result<Self> {
        let encoder = make_session(
            &model_dir.join("v3_e2e_rnnt_encoder.onnx"),
            use_cuda,
            intra_threads,
            policy.is_some(),
        )?;
        let decoder = make_session(
            &model_dir.join("v3_e2e_rnnt_decoder.onnx"),
            false,
            intra_threads,
            false,
        )?;
        let joint = make_session(
            &model_dir.join("v3_e2e_rnnt_joint.onnx"),
            false,
            intra_threads,
            false,
        )?;

        // Фаза 4: инспекция и верификация назначения графа — только энкодер
        // (decoder/joint всегда CPU, вне схемы)
        if let Some(policy) = policy {
            if use_cuda {
                let evidence =
                    ort_assignment::inspect(&encoder).map_err(anyhow::Error::msg)?;
                policy
                    .check(Role::Rnnt, &evidence)
                    .map_err(anyhow::Error::msg)?;
                if policy.is_allow_unverified() {
                    eprintln!(
                        "# CUDA assignment unverified role=rnnt sha256={} cpu_nodes={} cuda_nodes={}",
                        evidence.fingerprint().to_hex(),
                        evidence.cpu_assignments(),
                        evidence.cuda_assignments()
                    );
                }
            }
        }

        let vocab_path = model_dir.join("v3_e2e_rnnt_vocab.txt");
        let (vocab, blank_id) = load_vocab(&vocab_path)
            .with_context(|| format!("load vocab: {}", vocab_path.display()))?;

        // Фаза 3: сверка IO-имён с манифестом. io в манифесте — объединённая
        // поверхность семейства (encoder+decoder+joint), поэтому сверяем её
        // одним вызовом по сумме IO всех трёх графов (источник ловил здесь
        // рассинхрон вида «decoder inputs должны быть x,h.1,c.1, а не x,hi,ci»)
        if let Some(m) = crate::manifest::load_for(model_dir) {
            let mut ins: Vec<String> = Vec::new();
            let mut outs: Vec<String> = Vec::new();
            for s in [&encoder, &decoder, &joint] {
                ins.extend(s.inputs().iter().map(|i| i.name().to_string()));
                outs.extend(s.outputs().iter().map(|o| o.name().to_string()));
            }
            m.verify_io("gigaam-v3 RNN-T (encoder+decoder+joint)", &ins, &outs)?;
        }

        Ok(Self { encoder, decoder, joint, vocab, blank_id, mel: MelExtractor::new() })
    }

    /// Transcribe one chunk; returns a segment with absolute word timestamps.
    pub fn transcribe_chunk(&mut self, samples: &[f32], offset: f64) -> Result<Option<Segment>> {
        let (features, n_frames) = self.mel.compute(samples);
        if n_frames == 0 {
            return Ok(None);
        }

        let feats = Tensor::from_array((vec![1i64, 64, n_frames as i64], features))?;
        let lengths = Tensor::from_array((vec![1i64], vec![n_frames as i64]))?;
        // copy the encodings out before decoding: SessionOutputs borrows the
        // session, and the decode loop needs self.pred/self.join.
        // NOTE: the actual runtime layout is [B, 768, T_enc] (hidden-major,
        // NeMo-native) even though the symbolic graph dims suggest otherwise —
        // verified against the onnx-asr reference.
        let (encoded, t_enc) = {
            let enc_out = self
                .encoder
                .run(ort::inputs!["audio_signal" => feats, "length" => lengths])?;
            let (enc_shape, encoded) = enc_out["encoded"]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("encoder outputs: {e}"))?;
            let (_, enc_lens) = enc_out["encoded_len"]
                .try_extract_tensor::<i32>()
                .map_err(|e| anyhow::anyhow!("encoder lens: {e}"))?;
            anyhow::ensure!(
                enc_shape.len() == 3 && enc_shape[1] as usize == HIDDEN,
                "unexpected encoder output shape {:?} (expected [B, {HIDDEN}, T])",
                enc_shape
            );
            let t_enc = (enc_lens[0] as usize).min(enc_shape[2] as usize);
            (encoded.to_vec(), t_enc)
        };
        if t_enc == 0 {
            return Ok(None);
        }

        // --- greedy RNN-T decoding (onnx-asr `_AsrWithTransducerDecoding`) ---
        let mut h = vec![0f32; PRED_HIDDEN];
        let mut c = vec![0f32; PRED_HIDDEN];
        let mut dec: Option<Vec<f32>> = None; // cached prediction-network output
        let mut prev_token = self.blank_id as i64;
        let mut toks: Vec<(usize, usize, f32)> = Vec::new(); // (frame, id, prob)
        let mut frame_col = vec![0f32; HIDDEN];
        for t in 0..t_enc {
            // encoded is [1, 768, t_enc]: take column t (one frame, 768 channels)
            for h in 0..HIDDEN {
                frame_col[h] = encoded[h * t_enc + t];
            }
            let mut emitted = 0usize;
            loop {
                if dec.is_none() {
                    let (d, nh, nc) = self.pred(prev_token, &h, &c)?;
                    h = nh;
                    c = nc;
                    dec = Some(d);
                }
                let logits = self.join(&frame_col, dec.as_ref().unwrap())?;
                let token = argmax(&logits);
                if token != self.blank_id {
                    toks.push((t, token, prob_at(&logits, token)));
                    prev_token = token as i64;
                    dec = None; // prediction net re-runs for the new token
                    emitted += 1;
                }
                if token == self.blank_id || emitted >= MAX_TOKENS_PER_STEP {
                    // blank: the cached dec (same last token) stays valid
                    break;
                }
            }
        }
        if toks.is_empty() {
            return Ok(None);
        }
        if std::env::var("LIGHTTRANS_DEBUG_TOKENS").is_ok() {
            let shift_dbg = samples.len() as f64 / 16_000.0 / t_enc as f64;
            for (t, id, p) in toks.iter().take(20) {
                eprintln!(
                    "rtok frame={t} t={:.3}s piece={:?} p={p:.2}",
                    *t as f64 * shift_dbg, self.vocab[*id]
                );
            }
        }

        let dur = samples.len() as f64 / 16_000.0;
        let shift = dur / t_enc as f64; // seconds per encoder frame (~40 ms)
        let words = build_words(&self.vocab, &toks, shift, offset);
        if words.is_empty() {
            return Ok(None);
        }
        let text = words_to_text(&words);
        let start = words[0].start;
        let end = words.last().unwrap().end;
        Ok(Some(Segment { start, end, text, words, speaker: None, channel: None }))
    }

    /// Prediction network: last token + LSTM state -> 320-d embedding + new state.
    fn pred(&mut self, token: i64, h: &[f32], c: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let x = Tensor::from_array((vec![1i64, 1i64], vec![token]))?;
        let ht = Tensor::from_array((vec![1i64, 1i64, PRED_HIDDEN as i64], h.to_vec()))?;
        let ct = Tensor::from_array((vec![1i64, 1i64, PRED_HIDDEN as i64], c.to_vec()))?;
        let out = self.decoder.run(ort::inputs!["x" => x, "h.1" => ht, "c.1" => ct])?;
        let (_, dec) = out["dec"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("decoder dec: {e}"))?;
        let (_, nh) = out["h"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("decoder h: {e}"))?;
        let (_, nc) = out["c"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("decoder c: {e}"))?;
        Ok((dec.to_vec(), nh.to_vec(), nc.to_vec()))
    }

    /// Joint network: encoder frame (768) x prediction (320) -> 1025 logits.
    fn join(&mut self, enc_col: &[f32], dec: &[f32]) -> Result<Vec<f32>> {
        let enc = Tensor::from_array((vec![1i64, HIDDEN as i64, 1i64], enc_col.to_vec()))?;
        let dec_t = Tensor::from_array((vec![1i64, PRED_HIDDEN as i64, 1i64], dec.to_vec()))?;
        let out = self.joint.run(ort::inputs!["enc" => enc, "dec" => dec_t])?;
        let (_, joint) = out["joint"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("joint outputs: {e}"))?;
        Ok(joint.to_vec())
    }
}

/// Rebuild the segment text from assembled words (each word is a ▁-run, so a
/// single space between words restores the original spacing).
fn words_to_text(words: &[Word]) -> String {
    words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ")
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

/// Softmax probability of one joint-logits entry (joint output is unnormalized).
fn prob_at(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    ((logits[idx] - max).exp() / sum).max(0.0)
}

fn make_session(
    model: &Path,
    use_cuda: bool,
    intra: usize,
    record_assignment: bool,
) -> Result<Session> {
    let eps: Vec<ExecutionProviderDispatch> = if use_cuda {
        // error_on_failure: отказ при недоступном CUDA вместо тихого fallback на CPU (Фаза 4)
        vec![CUDA::default().build().error_on_failure()]
    } else {
        vec![]
    };
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(intra.max(1))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if record_assignment && use_cuda {
        // без записи назначений inspect() нечего читать (API-24, Фаза 4)
        builder = builder
            .with_config_entry("session.record_ep_graph_assignment_info", "1")
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    builder
        .with_execution_providers(eps)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(model)
        .with_context(|| format!("load gigaam-rnnt model: {}", model.display()))
}
