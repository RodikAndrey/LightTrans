//! NVIDIA Nemotron-3-Diarization (streaming Sortformer architecture) via ONNX Runtime.
//!
//! The exported graph (scripts/export_nemotron_diar.py) covers ONE chunk step of the
//! offline forward: cache embeds + FIFO embeds + chunk log-mel -> per-frame speaker
//! logits (10 ms frames, up to 8 speakers ordered by first arrival) + the chunk's
//! post-embedder embeds. The Arrival-Order Speaker Cache bookkeeping (FIFO overflow,
//! cache compression with per-speaker score boosting) is ported 1:1 from the HF
//! modeling code and lives here.
//!
//! Mel features (must match the HF `NemotronAsrStreamingFeatureExtractor` exactly):
//! 16 kHz mono -> preemphasis 0.97 -> STFT (n_fft 512, win 400 symmetric Hann zero-
//! padded to 512, hop 160, center=True with zero padding) -> power spectrum ->
//! librosa slaney filterbank (128 mels, fmin 0, fmax 8000, norm "slaney") ->
//! log(x + 2^-24). Valid frame count is floor(len / hop); the trailing STFT frame is
//! dropped, same as the HF attention-mask semantics.

use anyhow::{Context, Result};
use ort::ep::{CUDA, ExecutionProviderDispatch};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const N_FFT: usize = 512;
pub const WIN: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
pub const HIDDEN: usize = 512;
const LOG_GUARD: f32 = 5.960464477539063e-08; // 2^-24
/// Duration of one mel frame / one speaker-activity frame (10 ms).
pub const FRAME_S: f64 = HOP as f64 / 16_000.0;

#[derive(Debug, Clone, Deserialize)]
pub struct DiarMeta {
    pub chunk_length: usize,
    pub chunk_right_context: usize,
    pub fifo_length: usize,
    pub speaker_cache_length: usize,
    pub speaker_cache_update_period: usize,
    pub speaker_cache_silence_frames_per_speaker: usize,
    pub prediction_score_threshold: f32,
    pub latest_frames_score_boost: f32,
    pub min_positive_scores_rate: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub num_speakers: usize,
    pub subsampling_factor: usize,
}

// ---------------------------------------------------------------------------
// Mel features
// ---------------------------------------------------------------------------

pub struct NemotronMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>, // N_FFT: zeros(56) + symmetric hann(400) + zeros(56)
    fb: Vec<f32>,     // N_MELS x (N_FFT/2+1), row-major
    pre: Vec<f32>,    // preemphasized audio scratch
    frame: Vec<f32>,  // windowed frame scratch
    out: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl NemotronMel {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let fb_path = model_dir.join("mel_filters.f32");
        let fb_bytes = std::fs::read(&fb_path)
            .with_context(|| format!("read {}", fb_path.display()))?;
        let n_bins = N_FFT / 2 + 1;
        anyhow::ensure!(
            fb_bytes.len() == N_MELS * n_bins * 4,
            "{}: expected {} bytes, got {}",
            fb_path.display(),
            N_MELS * n_bins * 4,
            fb_bytes.len()
        );
        let fb: Vec<f32> = fb_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // torch.hann_window(400, periodic=False), zero-padded to n_fft=512
        let mut window = vec![0f32; N_FFT];
        for (i, w) in window[56..56 + WIN].iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (WIN - 1) as f64).cos()) as f32;
        }
        let scratch_len = fft.get_scratch_len();
        Ok(Self {
            fft,
            window,
            fb,
            pre: Vec::new(),
            frame: vec![0f32; N_FFT],
            out: vec![Complex::new(0.0, 0.0); n_bins],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        })
    }

    /// Returns (features frame-major [T x 128], T = floor(len / hop)).
    pub fn compute(&mut self, wav: &[f32]) -> (Vec<f32>, usize) {
        let n = wav.len();
        let n_frames = n / HOP;
        if n_frames == 0 {
            return (Vec::new(), 0);
        }
        self.pre.clear();
        self.pre.reserve(n);
        self.pre.push(wav[0]);
        for i in 1..n {
            self.pre.push(wav[i] - 0.97 * wav[i - 1]);
        }

        let n_bins = N_FFT / 2 + 1;
        let mut feats = vec![0f32; n_frames * N_MELS];
        for t in 0..n_frames {
            // centered window: samples [t*HOP - 256, t*HOP + 256), zero-padded outside
            for (i, buf) in self.frame.iter_mut().enumerate() {
                let si = t * HOP + i;
                let v = if si >= N_FFT / 2 && si < N_FFT / 2 + n {
                    self.pre[si - N_FFT / 2]
                } else {
                    0.0
                };
                *buf = v * self.window[i];
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.out, &mut self.scratch)
                .expect("fft failed");
            for m in 0..N_MELS {
                let row = &self.fb[m * n_bins..(m + 1) * n_bins];
                let mut e = 0f32;
                for (bin, w) in row.iter().enumerate() {
                    let c = self.out[bin];
                    e += w * (c.re * c.re + c.im * c.im);
                }
                feats[t * N_MELS + m] = (e + LOG_GUARD).ln();
            }
        }
        (feats, n_frames)
    }
}

// ---------------------------------------------------------------------------
// Result + postprocessing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct DiarSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

/// Per-frame speaker activity of one file (sigmoid probabilities).
pub struct Diarization {
    pub probs: Vec<f32>, // [num_frames, num_speakers] row-major
    pub num_frames: usize,
    pub num_speakers: usize,
}

impl Diarization {
    /// Speaker-activity segments, speakers ordered by first arrival (threshold 0.5,
    /// same boundary semantics as the HF processor's `extract_speaker_dict`).
    pub fn segments(&self, threshold: f32) -> Vec<DiarSegment> {
        let mut segs = Vec::new();
        for s in 0..self.num_speakers {
            let mut cur: Option<usize> = None;
            for f in 0..self.num_frames {
                let active = self.probs[f * self.num_speakers + s] > threshold;
                match (cur, active) {
                    (None, true) => cur = Some(f),
                    (Some(st), false) => {
                        segs.push(DiarSegment {
                            start: st as f64 * FRAME_S,
                            end: f as f64 * FRAME_S,
                            speaker: s,
                        });
                        cur = None;
                    }
                    _ => {}
                }
            }
            if let Some(st) = cur {
                segs.push(DiarSegment {
                    start: st as f64 * FRAME_S,
                    end: self.num_frames as f64 * FRAME_S,
                    speaker: s,
                });
            }
        }
        segs.sort_by(|a, b| {
            a.start.partial_cmp(&b.start).unwrap().then(a.speaker.cmp(&b.speaker))
        });
        segs
    }

    /// Speaker of a time span: highest summed probability over its frames.
    pub fn speaker_for_span(&self, start: f64, end: f64) -> Option<usize> {
        if self.num_frames == 0 || self.probs.is_empty() {
            return None;
        }
        let f0 = ((start / FRAME_S).floor() as i64).clamp(0, self.num_frames as i64 - 1) as usize;
        let f1 = ((end / FRAME_S).ceil() as i64).clamp(f0 as i64 + 1, self.num_frames as i64) as usize;
        let mut best = 0usize;
        let mut best_sum = f32::NEG_INFINITY;
        for s in 0..self.num_speakers {
            let mut acc = 0f32;
            for f in f0..f1 {
                acc += self.probs[f * self.num_speakers + s];
            }
            if acc > best_sum {
                best_sum = acc;
                best = s;
            }
        }
        Some(best)
    }
}

// ---------------------------------------------------------------------------
// Diarizer
// ---------------------------------------------------------------------------

pub struct NemotronDiarizer {
    session: Session,
    mel: NemotronMel,
    meta: DiarMeta,
    silence_embeds: Vec<f32>,
    // derived compression budget (HF semantics)
    min_positive_scores: usize,
    num_strong_boosted: usize,
    num_weak_boosted: usize,
    // AOSC state
    cache_embeds: Vec<f32>,
    cache_probs: Vec<f32>,
    fifo_embeds: Vec<f32>,
    num_cache: usize,
    num_fifo: usize,
    is_compressed: bool,
}

impl NemotronDiarizer {
    pub fn load(model_dir: &Path, use_cuda: bool, intra_threads: usize) -> Result<Self> {
        let meta: DiarMeta = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("meta.json"))
                .context("read diarization meta.json")?,
        )
        .context("parse diarization meta.json")?;

        let silence = std::fs::read(model_dir.join("silence_embeds.f32"))
            .context("read silence_embeds.f32")?;
        let silence_embeds: Vec<f32> = silence
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        anyhow::ensure!(silence_embeds.len() == HIDDEN, "bad silence_embeds size");

        let eps: Vec<ExecutionProviderDispatch> =
            if use_cuda { vec![CUDA::default().build()] } else { vec![] };
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers(eps)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_dir.join("diar_step.onnx"))
            .with_context(|| format!("load diarization model in {}", model_dir.display()))?;

        // budget = cache slots per speaker minus reserved silence slots
        let budget = meta.speaker_cache_length / meta.num_speakers
            - meta.speaker_cache_silence_frames_per_speaker;
        let min_positive_scores =
            (budget as f32 * meta.min_positive_scores_rate).floor() as usize;
        let num_strong_boosted = (budget as f32 * meta.strong_boost_rate).floor() as usize;
        let num_weak_boosted = (budget as f32 * meta.weak_boost_rate).floor() as usize;

        Ok(Self {
            session,
            mel: NemotronMel::load(model_dir)?,
            meta,
            silence_embeds,
            min_positive_scores,
            num_strong_boosted,
            num_weak_boosted,
            cache_embeds: Vec::new(),
            cache_probs: Vec::new(),
            fifo_embeds: Vec::new(),
            num_cache: 0,
            num_fifo: 0,
            is_compressed: false,
        })
    }

    /// Offline diarization of a whole file. `dump_dir` (debug): writes the mel
    /// features and per-frame probabilities as raw f32 for reference comparison.
    pub fn diarize(&mut self, samples: &[f32], dump: Option<&Path>) -> Result<Diarization> {
        let (mel, n_frames) = self.mel.compute(samples);
        let nspk = self.meta.num_speakers;
        let sub = self.meta.subsampling_factor;
        let mut out = vec![0f32; n_frames * nspk];

        if n_frames > 0 {
            let n_embeds = (n_frames + sub - 1) / sub;
            self.num_cache = 0;
            self.num_fifo = 0;
            self.is_compressed = false;
            self.cache_embeds.clear();
            self.cache_probs.clear();
            self.fifo_embeds.clear();

            let chunk_len = self.meta.chunk_length;
            let rc = self.meta.chunk_right_context;
            let mut start_idx = 0usize;
            let mut step_no = 0usize;
            while start_idx < n_embeds {
                let end_idx = (start_idx + chunk_len).min(n_embeds);
                let num_chunk_frames = end_idx - start_idx;
                let chunk_end = (end_idx + rc).min(n_embeds);
                let mel_lo = start_idx * sub;
                let n_mel_rows = (chunk_end - start_idx) * sub;
                // the mel buffer holds only the valid frames; the trailing padding
                // group (feature stacking) is zero-filled
                let mut chunk_mel = vec![0f32; n_mel_rows * N_MELS];
                let buf_from = mel_lo * N_MELS;
                if buf_from < mel.len() {
                    let copy = (mel.len() - buf_from).min(chunk_mel.len());
                    chunk_mel[..copy].copy_from_slice(&mel[buf_from..buf_from + copy]);
                }

                let (step_logits, step_embeds) = self.step(&chunk_mel, n_mel_rows)?;

                // copy this chunk's frame rows into the global output
                let cached = self.num_cache + self.num_fifo;
                for k in 0..num_chunk_frames * sub {
                    let mf = start_idx * sub + k;
                    if mf < n_frames {
                        let src = (cached * sub + k) * nspk;
                        out[mf * nspk..mf * nspk + nspk]
                            .copy_from_slice(&step_logits[src..src + nspk]);
                    }
                }

                self.update(&step_embeds, &step_logits, num_chunk_frames);
                start_idx += chunk_len;
                step_no += 1;
                if step_no % 20 == 0 {
                    tracing::debug!("диаризация: шаг {step_no}, {} энкодер-кадров", start_idx);
                }
            }
        }

        for v in out.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }

        if let Some(dir) = dump {
            let _ = std::fs::create_dir_all(dir);
            let stem = "diar_debug";
            let _ = write_f32(dir.join(format!("{stem}_mel.f32")), &mel);
            let _ = write_f32(dir.join(format!("{stem}_probs.f32")), &out);
            let meta = serde_json::json!({"num_frames": n_frames, "num_speakers": nspk,
                "n_mels": N_MELS, "hop": HOP});
            let _ = std::fs::write(dir.join(format!("{stem}_meta.json")), meta.to_string());
        }

        Ok(Diarization { probs: out, num_frames: n_frames, num_speakers: nspk })
    }

    /// One encoder step: cache embeds + fifo embeds + chunk mel -> logits + chunk embeds.
    fn step(&mut self, chunk_mel: &[f32], n_mel_rows: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let nspk = self.meta.num_speakers;
        let sub = self.meta.subsampling_factor;
        let s = self.num_cache + self.num_fifo + n_mel_rows / sub;
        let pos: Vec<i64> = (0..s as i64).collect();

        let cache_t = Tensor::from_array((
            vec![self.num_cache as i64, HIDDEN as i64],
            self.cache_embeds.clone(),
        ))?;
        let fifo_t = Tensor::from_array((
            vec![self.num_fifo as i64, HIDDEN as i64],
            self.fifo_embeds.clone(),
        ))?;
        let mel_t =
            Tensor::from_array((vec![n_mel_rows as i64, N_MELS as i64], chunk_mel.to_vec()))?;
        let pos_t = Tensor::from_array((vec![s as i64], pos))?;

        let outputs = self.session.run(ort::inputs![
            "cache_embeds" => cache_t,
            "fifo_embeds" => fifo_t,
            "chunk_mel" => mel_t,
            "position_ids" => pos_t,
        ])?;

        let (logits_shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("logits: {e}"))?;
        anyhow::ensure!(logits_shape.len() == 2 && logits_shape[1] == nspk as i64);
        let (emb_shape, embeds) = outputs["chunk_embeds"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("chunk_embeds: {e}"))?;
        anyhow::ensure!(emb_shape.len() == 2 && emb_shape[1] == HIDDEN as i64);

        Ok((logits.to_vec(), embeds.to_vec()))
    }

    /// HF `Nemotron3DiarizationSpeakerCache::update`, ported verbatim.
    fn update(&mut self, step_embeds: &[f32], step_logits: &[f32], num_chunk_frames: usize) {
        let nspk = self.meta.num_speakers;
        let sub = self.meta.subsampling_factor;
        let n_rows = step_logits.len() / (sub * nspk);

        // pooled per-embed-row speaker probabilities: mean of sigmoid over `sub` frames
        let mut probs = vec![0f32; n_rows * nspk];
        for r in 0..n_rows {
            for s in 0..nspk {
                let mut acc = 0f32;
                for k in 0..sub {
                    let v = step_logits[(r * sub + k) * nspk + s];
                    acc += 1.0 / (1.0 + (-v).exp());
                }
                probs[r * nspk + s] = acc / sub as f32;
            }
        }

        let n_cache = self.num_cache;
        let n_fifo = self.num_fifo;

        // step_embeds holds only the chunk's embeds (chunk + right context); the
        // chunk proper is its first `num_chunk_frames` rows. The pooled probs are
        // aligned with the FULL step input [cache; fifo; chunk] because the graph
        // concatenates internally before the encoder.
        let mut new_fifo = self.fifo_embeds.clone();
        new_fifo.extend_from_slice(&step_embeds[..num_chunk_frames * HIDDEN]);
        let nf_new = n_fifo + num_chunk_frames;

        let popped = if nf_new <= self.meta.fifo_length {
            0
        } else {
            self.meta
                .speaker_cache_update_period
                .max(nf_new - self.meta.fifo_length)
                .min(nf_new)
        };

        if popped == 0 {
            self.fifo_embeds = new_fifo;
            self.num_fifo = nf_new;
            return;
        }

        // probs rows: [0..n_cache) cache, [n_cache..n_cache+nf_new) fifo+chunk
        let fifo_probs = &probs[n_cache * nspk..(n_cache + nf_new) * nspk];
        let stored: Vec<f32> = if self.is_compressed {
            self.cache_probs[..n_cache * nspk].to_vec()
        } else {
            probs[..n_cache * nspk].to_vec()
        };

        let mut new_cache = self.cache_embeds.clone();
        new_cache.extend_from_slice(&new_fifo[..popped * HIDDEN]);
        let mut new_probs = stored;
        new_probs.extend_from_slice(&fifo_probs[..popped * nspk]);

        let rest_embeds = new_fifo[popped * HIDDEN..].to_vec();

        if new_cache.len() / HIDDEN > self.meta.speaker_cache_length {
            let (ce, cp) = self.compress(&new_cache, &new_probs);
            self.cache_embeds = ce;
            self.cache_probs = cp;
            self.is_compressed = true;
        } else {
            self.cache_embeds = new_cache;
            self.cache_probs = new_probs;
        }
        self.num_cache = self.cache_embeds.len() / HIDDEN;
        self.fifo_embeds = rest_embeds;
        self.num_fifo = self.fifo_embeds.len() / HIDDEN;
    }

    /// HF `_get_frame_scores` + `_boost_scores` + `_compress`, ported verbatim.
    fn compress(&self, embeds: &[f32], probs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = embeds.len() / HIDDEN;
        let nspk = self.meta.num_speakers;
        let thr = self.meta.prediction_score_threshold as f64;
        let ln_half = 0.5f64.ln();

        // 1. per-frame, per-speaker scores
        let mut scores = vec![f64::NEG_INFINITY; n * nspk];
        for i in 0..n {
            let mut sum_lc = 0f64;
            for s in 0..nspk {
                let p = probs[i * nspk + s] as f64;
                sum_lc += (1.0 - p).clamp(thr, 1.0).ln();
            }
            for s in 0..nspk {
                let p = probs[i * nspk + s] as f64;
                if p > 0.5 {
                    let lp = p.clamp(thr, 1.0).ln();
                    let lc = (1.0 - p).clamp(thr, 1.0).ln();
                    scores[i * nspk + s] = lp - lc + sum_lc - ln_half;
                }
            }
        }
        // 2. drop non-positive speech frames of speakers that have enough positives
        for s in 0..nspk {
            let positives: Vec<usize> = (0..n)
                .filter(|&i| scores[i * nspk + s] > 0.0)
                .collect();
            if positives.len() < self.min_positive_scores {
                continue;
            }
            for i in positives {
                if scores[i * nspk + s] <= 0.0 {
                    scores[i * nspk + s] = f64::NEG_INFINITY;
                }
            }
        }
        // 3. small boost for frames beyond the cache capacity (freshly popped)
        let cl = self.meta.speaker_cache_length;
        for i in cl..n {
            for s in 0..nspk {
                scores[i * nspk + s] += self.meta.latest_frames_score_boost as f64;
            }
        }
        // 4. strong then weak top-k boosting, per speaker
        for (count, boost) in [
            (self.num_strong_boosted, -2.0 * ln_half),
            (self.num_weak_boosted, -ln_half),
        ] {
            if count == 0 || n == 0 {
                continue;
            }
            let k = count.min(n);
            for s in 0..nspk {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_unstable_by(|&a, &b| {
                    scores[b * nspk + s]
                        .partial_cmp(&scores[a * nspk + s])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for &i in idx.iter().take(k) {
                    scores[i * nspk + s] += boost;
                }
            }
        }

        // 5. pad with reserved silence frames (+inf scores, silence embeds, zero probs)
        let sil = self.meta.speaker_cache_silence_frames_per_speaker;
        let n_scored = n + sil;
        let mut embeds_pad = embeds.to_vec();
        embeds_pad.extend_from_slice(&self.silence_embeds);
        let mut probs_pad = probs.to_vec();
        probs_pad.extend(std::iter::repeat(0f32).take(sil * nspk));

        // 6. top-k over the flat [speaker x frame] scores, order-preserving
        let mut flat = vec![f64::NEG_INFINITY; n_scored * nspk];
        for s in 0..nspk {
            for i in 0..n {
                flat[s * n_scored + i] = scores[i * nspk + s];
            }
            for i in n..n_scored {
                flat[s * n_scored + i] = f64::INFINITY;
            }
        }
        let k = cl.min(flat.len());
        let mut order: Vec<usize> = (0..flat.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            flat[b]
                .partial_cmp(&flat[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sentinel = n_scored * nspk;
        let mut topk: Vec<usize> = order[..k]
            .iter()
            .map(|&v| if flat[v] == f64::NEG_INFINITY { sentinel } else { v })
            .collect();
        topk.sort_unstable();

        let mut out_e = Vec::with_capacity(k * HIDDEN);
        let mut out_p = Vec::with_capacity(k * nspk);
        for idx in &topk {
            let frame = if *idx == sentinel { n } else { (*idx % n_scored).min(n) };
            out_e.extend_from_slice(&embeds_pad[frame * HIDDEN..(frame + 1) * HIDDEN]);
            out_p.extend_from_slice(&probs_pad[frame * nspk..(frame + 1) * nspk]);
        }
        (out_e, out_p)
    }
}

fn write_f32(path: PathBuf, data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
