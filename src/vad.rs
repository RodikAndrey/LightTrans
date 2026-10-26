//! Silero VAD v5 (ONNX) with WhisperX-style hysteresis binarization.
//! Streaming: feed blocks of samples, receive completed speech intervals.
//! Inference runs in batches of windows for throughput.

use anyhow::{Context, Result};
use ort::ep::CPU;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct VadSegment {
    pub start: f64,
    pub end: f64,
}

const WINDOW: usize = 512; // samples @16k = 32 ms
const CONTEXT: usize = 64; // prepended context samples (silero v5)
const SR: i64 = 16_000;
const BATCH: usize = 64; // windows per ONNX run (~2 s of audio)

pub struct SileroVad {
    session: Session,
    state: Vec<f32>,   // [2, BATCH, 128] flattened
    contexts: Vec<f32>, // [BATCH, CONTEXT] flattened
    onset: f32,
    offset: f32,
    min_duration_on: f64,
    min_duration_off: f64,
    // streaming state
    pending: Vec<f32>,
    next_win: usize, // absolute window index of pending[0]
    speech: bool,
    speech_start: usize,
    last_interval: Option<(f64, f64)>,
}

impl SileroVad {
    pub fn load(model_path: &std::path::Path, intra_threads: usize) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([CPU::default().build()])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .with_context(|| format!("load silero vad: {}", model_path.display()))?;
        Ok(Self {
            session,
            state: vec![0f32; 2 * BATCH * 128],
            contexts: vec![0f32; BATCH * CONTEXT],
            onset: 0.5,
            offset: 0.363,
            min_duration_on: 0.1,
            min_duration_off: 0.1,
            pending: Vec::new(),
            next_win: 0,
            speech: false,
            speech_start: 0,
            last_interval: None,
        })
    }

    pub fn with_thresholds(mut self, onset: f32, offset: f32) -> Self {
        self.onset = onset;
        self.offset = offset;
        self
    }

    /// Feed samples; completed speech intervals (absolute seconds) go to `out`.
    pub fn feed(&mut self, samples: &[f32], out: &mut Vec<VadSegment>) -> Result<()> {
        self.pending.extend_from_slice(samples);
        let n_full = self.pending.len() / WINDOW;
        if n_full == 0 {
            return Ok(());
        }
        let mut batch_start = 0usize;
        while batch_start < n_full {
            let count = (n_full - batch_start).min(BATCH);
            self.run_batch(batch_start, count, out)?;
            batch_start += count;
        }
        let consumed = n_full * WINDOW;
        self.pending.drain(..consumed);
        self.next_win += n_full;
        Ok(())
    }

    /// Close any open speech at end of stream and emit the final interval.
    pub fn finish(&mut self, out: &mut Vec<VadSegment>) {
        if self.speech {
            self.speech = false;
            self.close(self.speech_start, self.next_win + self.pending.len() / WINDOW, out);
        }
        if let Some(iv) = self.last_interval.take() {
            out.push(VadSegment { start: iv.0, end: iv.1 });
        }
    }

    fn close(&mut self, start_win: usize, end_win: usize, out: &mut Vec<VadSegment>) {
        let frame = WINDOW as f64 / SR as f64;
        let (s0, e0) = (start_win as f64 * frame, end_win as f64 * frame);
        if e0 - s0 < self.min_duration_on {
            return; // too short: drop
        }
        if let Some((ls, le)) = self.last_interval {
            if s0 - le < self.min_duration_off {
                self.last_interval = Some((ls, e0)); // merge across short silence
                return;
            }
            out.push(VadSegment { start: ls, end: le });
        }
        self.last_interval = Some((s0, e0));
    }

    fn run_batch(&mut self, first: usize, count: usize, out: &mut Vec<VadSegment>) -> Result<()> {
        let win_len = CONTEXT + WINDOW;
        let mut input = vec![0f32; count * win_len];
        for i in 0..count {
            input[i * win_len..i * win_len + CONTEXT]
                .copy_from_slice(&self.contexts[i * CONTEXT..(i + 1) * CONTEXT]);
            let w = first + i;
            input[i * win_len + CONTEXT..(i + 1) * win_len]
                .copy_from_slice(&self.pending[w * WINDOW..(w + 1) * WINDOW]);
        }

        // gather per-slot state [2, count, 128]
        let mut st = vec![0f32; 2 * count * 128];
        for s in 0..2 {
            for j in 0..count {
                let src = s * BATCH * 128 + j * 128;
                let dst = s * count * 128 + j * 128;
                st[dst..dst + 128].copy_from_slice(&self.state[src..src + 128]);
            }
        }

        let input_t = Tensor::from_array((vec![count as i64, win_len as i64], input))?;
        let state_t = Tensor::from_array((vec![2i64, count as i64, 128], st))?;
        // sr must stay a single-element tensor: the model's internal If node
        // reads it as a scalar condition
        let sr_t = Tensor::from_array((vec![1i64], vec![SR]))?;

        // copy results out while `outputs` borrows the session
        let (probs, new_state) = {
            let outputs = self
                .session
                .run(ort::inputs!["input" => input_t, "state" => state_t, "sr" => sr_t])?;
            let (_p_shape, probs) = outputs["output"].try_extract_tensor::<f32>()?;
            let (_s_shape, new_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
            (probs.to_vec(), new_state.to_vec())
        };

        // write back state + context for the used slots
        for s in 0..2 {
            for j in 0..count {
                let src = s * count * 128 + j * 128;
                let dst = s * BATCH * 128 + j * 128;
                self.state[dst..dst + 128].copy_from_slice(&new_state[src..src + 128]);
            }
        }

        let frame = WINDOW as f64 / SR as f64;
        let _ = frame;
        for j in 0..count {
            let p = probs[j];
            let abs_w = self.next_win + first + j;
            if !self.speech && p >= self.onset {
                self.speech = true;
                self.speech_start = abs_w;
            } else if self.speech && p < self.offset {
                self.speech = false;
                self.close(self.speech_start, abs_w, out);
            }
        }

        // update per-slot contexts with the tail of each processed window
        for i in 0..count {
            let w = first + i;
            self.contexts[i * CONTEXT..(i + 1) * CONTEXT]
                .copy_from_slice(&self.pending[w * WINDOW + WINDOW - CONTEXT..(w + 1) * WINDOW]);
        }
        Ok(())
    }
}
