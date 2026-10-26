//! Streaming chunking of speech into transcription chunks (WhisperX
//! `merge_chunks` semantics: accumulate speech into chunks not exceeding
//! `chunk_size`; long continuous speech is split at boundaries).

use crate::transcript::TimeInterval;

pub const SR: f64 = 16_000.0;

/// Extra audio appended after the last speech sample of a chunk, so quiet
/// word tails (e.g. trailing consonants) are not cut off by the VAD. The pad
/// never crosses into the next speech interval.
pub const END_PAD_S: f64 = 0.25;

/// Extra audio kept before the first speech interval of a file: the VAD onset
/// trails the true speech start by ~100-200 ms, and without pre-roll the first
/// word loses its beginning (its timestamp starts late). Only the first
/// interval needs it — later chunks begin at the previous chunk's end pad.
pub const START_PAD_S: f64 = 0.2;

#[derive(Debug)]
pub struct Chunk {
    pub file_idx: usize,
    pub idx: usize,
    pub start: f64,
    pub samples: Vec<f32>,
}

/// Streaming chunker: feed audio blocks + closed VAD intervals (absolute
/// seconds), receive ready chunks. In `fixed` mode (VAD disabled) audio is cut
/// into fixed windows.
pub struct Chunker {
    file_idx: usize,
    chunk_samples: usize,
    t0: usize,     // absolute sample index of current chunk start
    buf: Vec<f32>, // samples since t0
    started: bool, // seen first speech interval
    last_end: f64, // absolute seconds: end of last closed interval
    fed: usize,    // absolute samples fed so far
    idx: usize,
    fixed: bool,
}

impl Chunker {
    pub fn new(file_idx: usize, chunk_size: f64, fixed: bool) -> Self {
        Self {
            file_idx,
            chunk_samples: (chunk_size * SR) as usize,
            t0: 0,
            buf: Vec::new(),
            started: false,
            last_end: 0.0,
            fed: 0,
            idx: 0,
            fixed,
        }
    }

    pub fn feed(&mut self, block: &[f32], intervals: &[TimeInterval], out: &mut Vec<Chunk>) {
        self.buf.extend_from_slice(block);
        self.fed += block.len();
        if self.fixed {
            self.emit_fixed(out);
        } else {
            self.process_intervals(intervals, out);
        }
    }

    /// Process intervals that closed after the last audio block was fed
    /// (e.g. produced by `SileroVad::finish`).
    pub fn feed_intervals(&mut self, intervals: &[TimeInterval], out: &mut Vec<Chunk>) {
        if !self.fixed {
            self.process_intervals(intervals, out);
        }
    }

    pub fn finish(&mut self, out: &mut Vec<Chunk>) {
        if self.fixed {
            self.emit_fixed(out);
            if !self.buf.is_empty() {
                out.push(Chunk {
                    file_idx: self.file_idx,
                    idx: self.idx,
                    start: self.t0 as f64 / SR,
                    samples: std::mem::take(&mut self.buf),
                });
                self.idx += 1;
                self.t0 = self.fed;
            }
            return;
        }
        if self.started {
            // flush everything up to last speech end + pad
            let close_abs = ((self.last_end * SR).ceil() as usize)
                .saturating_add((END_PAD_S * SR) as usize);
            // clamp to the buffered remainder: `fed` can outrun `buf` when a
            // decoder flush re-emits tail samples, and slicing past it panics
            let close = close_abs
                .min(self.fed)
                .saturating_sub(self.t0)
                .min(self.buf.len());
            if close > 0 {
                let samples = self.buf[..close].to_vec();
                out.push(Chunk {
                    file_idx: self.file_idx,
                    idx: self.idx,
                    start: self.t0 as f64 / SR,
                    samples,
                });
                self.idx += 1;
            }
            self.buf.clear();
        } else {
            // no speech at all: fixed windows over the whole input
            let total = self.buf.len();
            let mut pos = 0usize;
            while pos < total {
                let take = self.chunk_samples.min(total - pos);
                let abs = self.fed - total + pos;
                out.push(Chunk {
                    file_idx: self.file_idx,
                    idx: self.idx,
                    start: abs as f64 / SR,
                    samples: self.buf[pos..pos + take].to_vec(),
                });
                self.idx += 1;
                pos += take;
            }
            self.buf.clear();
        }
    }

    fn emit_fixed(&mut self, out: &mut Vec<Chunk>) {
        while self.buf.len() >= self.chunk_samples {
            let samples = self.buf[..self.chunk_samples].to_vec();
            out.push(Chunk {
                file_idx: self.file_idx,
                idx: self.idx,
                start: self.t0 as f64 / SR,
                samples,
            });
            self.idx += 1;
            self.buf.drain(..self.chunk_samples);
            self.t0 += self.chunk_samples;
        }
    }

    fn process_intervals(&mut self, intervals: &[TimeInterval], out: &mut Vec<Chunk>) {
        for &(s, e) in intervals {
            if !self.started {
                self.started = true;
                // pre-roll: keep START_PAD_S of audio before the VAD onset
                let want = ((s * SR).floor() as usize).saturating_sub((START_PAD_S * SR) as usize);
                self.t0 = want.min(self.fed);
                self.buf.drain(..self.t0.min(self.buf.len()));
                self.last_end = e;
                continue;
            }
            if e <= self.last_end {
                continue; // stale/duplicate interval
            }
            // does this interval overflow the current chunk?
            if (e * SR) as usize > self.t0 + self.chunk_samples {
                // close point: last speech end + pad, but never into the next
                // word's audio
                let want = ((self.last_end * SR).ceil() as usize)
                    .saturating_add((END_PAD_S * SR) as usize);
                let limit = (s * SR).floor() as usize;
                let close_abs = want.min(limit);
                let close = close_abs.min(self.fed).saturating_sub(self.t0).min(self.buf.len());
                if close > 0 {
                    let samples = self.buf[..close].to_vec();
                    out.push(Chunk {
                        file_idx: self.file_idx,
                        idx: self.idx,
                        start: self.t0 as f64 / SR,
                        samples,
                    });
                    self.idx += 1;
                    self.buf.drain(..close);
                    self.t0 += close;
                }
            }
            self.last_end = e;
        }

        // hard split of long continuous speech (no interval boundaries)
        while self.buf.len() > self.chunk_samples + 8 * SR as usize {
            let samples = self.buf[..self.chunk_samples].to_vec();
            out.push(Chunk {
                file_idx: self.file_idx,
                idx: self.idx,
                start: self.t0 as f64 / SR,
                samples,
            });
            self.idx += 1;
            self.buf.drain(..self.chunk_samples);
            self.t0 += self.chunk_samples;
        }
    }
}
