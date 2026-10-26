//! Log-mel spectrogram matching GigaAM v3 / torchaudio MelSpectrogram exactly:
//! n_fft=320, win=320 (periodic Hann), hop=160, 64 HTK mel bins, center=false,
//! power=2.0, fmin=0, fmax=8000, natural log of clamp(x, 1e-9, 1e9).

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

pub const N_FFT: usize = 320;
pub const HOP: usize = 160;
pub const WIN: usize = 320;
pub const N_MELS: usize = 64;
pub const SR: f64 = 16_000.0;
const F_MIN: f64 = 0.0;
const F_MAX: f64 = SR / 2.0;

pub struct MelExtractor {
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    fb: Vec<f32>, // N_MELS x (N_FFT/2+1), row-major
    scratch_input: Vec<f32>,
    scratch_out: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

fn htk_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}
fn htk_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// HTK mel filterbank, no area normalization (torchaudio `mel_scale="htk"`, `norm=None`).
fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f64) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut fb = vec![0f32; n_mels * n_bins];

    let mel_min = htk_mel(F_MIN);
    let mel_max = htk_mel(F_MAX);
    // n_mels + 2 mel points, linear in mel space
    let mels: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .map(htk_hz)
        .collect();

    for bin in 0..n_bins {
        let f = bin as f64 * sr / n_fft as f64;
        for m in 0..n_mels {
            let (lower, center, upper) = (mels[m], mels[m + 1], mels[m + 2]);
            let rise = if center - lower > 0.0 {
                (f - lower) / (center - lower)
            } else {
                1.0
            };
            let fall = if upper - center > 0.0 {
                (upper - f) / (upper - center)
            } else {
                1.0
            };
            let w = rise.min(fall).max(0.0);
            fb[m * n_bins + bin] = w as f32;
        }
    }
    fb
}

impl MelExtractor {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // periodic Hann (torchaudio hann_window default)
        let window: Vec<f32> = (0..WIN)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / WIN as f64).cos()) as f32)
            .collect();
        let scratch_len = fft.get_scratch_len();
        Self {
            fft,
            window,
            fb: mel_filterbank(N_MELS, N_FFT, SR),
            scratch_input: vec![0f32; N_FFT],
            scratch_out: vec![Complex::new(0.0, 0.0); N_FFT / 2 + 1],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        }
    }

    /// Returns (features flattened as [N_MELS x T] row-major, T frame count).
    pub fn compute(&mut self, wav: &[f32]) -> (Vec<f32>, usize) {
        let n = wav.len();
        if n < WIN {
            return (Vec::new(), 0);
        }
        let n_frames = 1 + (n - WIN) / HOP; // center=false
        let n_bins = N_FFT / 2 + 1;

        let mut features = vec![0f32; N_MELS * n_frames];

        for t in 0..n_frames {
            let start = t * HOP;
            for (i, w) in self.window.iter().enumerate() {
                self.scratch_input[i] = wav[start + i] * w;
            }
            self.fft
                .process_with_scratch(
                    &mut self.scratch_input,
                    &mut self.scratch_out,
                    &mut self.scratch,
                )
                .expect("fft failed");

            for m in 0..N_MELS {
                let row = &self.fb[m * n_bins..(m + 1) * n_bins];
                let mut e = 0f32;
                for bin in 0..n_bins {
                    let c = self.scratch_out[bin];
                    let p = c.re * c.re + c.im * c.im; // power spectrum
                    e += row[bin] * p;
                }
                // torchaudio applies clamp AFTER mel-multiplication in GigaAM's log:
                // log(clamp(spec, 1e-9, 1e9))
                let e = e.clamp(1e-9, 1e9);
                features[m * n_frames + t] = e.ln();
            }
        }
        (features, n_frames)
    }
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}
