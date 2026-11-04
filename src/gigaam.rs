//! GigaAM v3 E2E CTC (Russian, ONNX) via ONNX Runtime with CUDA.
//!
//! Pipeline: 16 kHz mono f32 -> log-mel (64 bins, hop 10 ms) -> Conformer encoder
//! (ONNX) -> CTC greedy decode -> words with frame-level timestamps.

use crate::mel::MelExtractor;
use crate::ort_assignment::{self, Policy, Role};
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
    pub fn load(
        model: &Path,
        vocab_path: &Path,
        use_cuda: bool,
        intra_threads: usize,
        policy: Option<&Policy>,
    ) -> Result<Self> {
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
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if policy.is_some() && use_cuda {
            // без записи назначений inspect() нечего читать (API-24, Фаза 4)
            builder = builder
                .with_config_entry("session.record_ep_graph_assignment_info", "1")
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let session = builder
            .with_execution_providers(eps)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model)
            .with_context(|| format!("load gigaam model: {}", model.display()))?;

        if let Some(policy) = policy {
            if use_cuda {
                let evidence = ort_assignment::inspect(&session).map_err(anyhow::Error::msg)?;
                policy
                    .check(Role::Ctc, &evidence)
                    .map_err(anyhow::Error::msg)?;
                if policy.is_allow_unverified() {
                    eprintln!(
                        "# CUDA assignment unverified role=ctc sha256={} cpu_nodes={} cuda_nodes={}",
                        evidence.fingerprint().to_hex(),
                        evidence.cpu_assignments(),
                        evidence.cuda_assignments()
                    );
                }
            }
        }

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

    /// Прогон энкодера: (logprobs, enc_len, classes). Общий для batch-чанка
    /// и стримингового окна.
    fn run_encoder(&mut self, features: Vec<f32>, n_frames: usize) -> Result<(Vec<f32>, usize, usize)> {
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
        let logprobs = logprobs.to_vec();
        let enc_len = if let Some(len_out) = &self.lengths_out {
            let (_enc_shape, enc) = outputs[len_out.as_str()].try_extract_tensor::<i64>()?;
            (enc.first().copied().unwrap_or(0)).max(0) as usize
        } else {
            *lp_shape.get(1).context("log_probs rank < 2")? as usize
        };
        let classes = *lp_shape.last().context("log_probs rank < 2")? as usize;
        Ok((logprobs, enc_len, classes))
    }

    /// Прогон энкодера батчем: features — [B, 64, T_max], lengths — [B].
    /// Возвращает (logprobs [B x T_enc_max x classes], T_enc_max, classes);
    /// валидная длина каждого элемента — `enc_frames(lengths[i])`.
    fn run_encoder_batch(
        &mut self,
        features: Vec<f32>,
        lengths: &[i64],
        t_max: usize,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let b = lengths.len();
        let feats = Tensor::from_array((vec![b as i64, 64, t_max as i64], features))?;
        let lens = Tensor::from_array((vec![b as i64], lengths.to_vec()))?;

        let outputs = if self.input_names.len() >= 2 {
            let n0 = self.input_names[0].clone();
            let n1 = self.input_names[1].clone();
            self.session
                .run(ort::inputs![n0.as_str() => feats, n1.as_str() => lens])?
        } else {
            self.session
                .run(ort::inputs!["features" => feats, "feature_lengths" => lens])?
        };

        let (lp_shape, logprobs) = outputs[self.logprobs_out.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("log_probs: {e}"))?;
        anyhow::ensure!(
            lp_shape.len() == 3 && lp_shape[0] as usize == b,
            "log_probs: неожиданная форма {lp_shape:?} для батча {b}"
        );
        let logprobs = logprobs.to_vec();
        let enc_max = *lp_shape.get(1).context("log_probs rank < 2")? as usize;
        let classes = *lp_shape.last().context("log_probs rank < 2")? as usize;
        Ok((logprobs, enc_max, classes))
    }

    /// CTC greedy: (фрейм, id, prob) по непустым ненаглым токенам.
    fn greedy_tokens(&self, logprobs: &[f32], enc_len: usize, classes: usize) -> Vec<(usize, usize, f32)> {
        let mut toks: Vec<(usize, usize, f32)> = Vec::new();
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
        toks
    }

    /// Transcribe one chunk; returns a segment with absolute timestamps.
    pub fn transcribe_chunk(&mut self, samples: &[f32], offset: f64) -> Result<Option<Segment>> {
        let (features, n_frames) = self.mel.compute(samples);
        if n_frames == 0 {
            return Ok(None);
        }
        let (logprobs, enc_len, classes) = self.run_encoder(features, n_frames)?;
        if enc_len == 0 {
            return Ok(None);
        }

        let toks = self.greedy_tokens(&logprobs, enc_len, classes);

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

        Ok(toks_to_segment(&self.vocab, &toks, shift, offset, dur))
    }

    /// Батч-инференс (roadmap «Batched-инференс GigaAM»): несколько чанков за
    /// один прогон ONNX (динамический batch-размер экспорта). Чанки добиваются
    /// нулями до максимальной длины в батче, результат каждого режется по
    /// своей mel-длине (формула субдискретизации ×4, проверена тестом).
    ///
    /// Семантика как у WhisperX batch_size: из-за паддинга хвост чанка рядом
    /// с нулями может минимально отличаться от одиночного прогона
    /// (неказуальные свёртки конформера); текст/тайминги в середине чанка
    /// бит-в-бит совпадают.
    pub fn transcribe_chunks(
        &mut self,
        chunks: &[&[f32]],
        offsets: &[f64],
    ) -> Result<Vec<Option<Segment>>> {
        anyhow::ensure!(chunks.len() == offsets.len(), "chunks/offsets длины разъехались");
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        if chunks.len() == 1 {
            // одиночный чанк — старый путь без паддинга (бит-в-бит как раньше)
            let seg = self.transcribe_chunk(chunks[0], offsets[0])?;
            return Ok(vec![seg]);
        }

        let mels: Vec<(Vec<f32>, usize)> = chunks
            .iter()
            .map(|&s| self.mel.compute(s))
            .collect();
        let t_max = mels.iter().map(|(_, n)| *n).max().unwrap_or(0);
        if t_max == 0 {
            return Ok(vec![None; chunks.len()]);
        }

        // [B, 64, T_max]: добивка в конце каждой временной оси
        let mut feats = vec![0f32; chunks.len() * 64 * t_max];
        for (e, (features, n)) in mels.iter().enumerate() {
            for m in 0..64 {
                let src = &features[m * n..m * n + *n];
                let dst = &mut feats[e * 64 * t_max + m * t_max..e * 64 * t_max + m * t_max + *n];
                dst.copy_from_slice(src);
            }
        }
        let lengths: Vec<i64> = mels.iter().map(|(_, n)| *n as i64).collect();

        let (logprobs, enc_lens_max, classes) = self.run_encoder_batch(feats, &lengths, t_max)?;

        let mut out = Vec::with_capacity(chunks.len());
        for (e, (_, n)) in mels.iter().enumerate() {
            let enc_len = enc_frames(*n).min(enc_lens_max);
            if enc_len == 0 {
                out.push(None);
                continue;
            }
            let rows = &logprobs[e * enc_lens_max * classes..(e + 1) * enc_lens_max * classes];
            let toks = self.greedy_tokens(rows, enc_len, classes);
            let dur = chunks[e].len() as f64 / 16_000.0;
            let shift = dur / enc_len as f64;
            out.push(toks_to_segment(&self.vocab, &toks, shift, offsets[e], dur));
        }
        Ok(out)
    }

    /// Декод одного полного окна мел-признаков для стриминга (src/stream.rs):
    /// слова в секундах от начала окна + маска тишины (argmax == blank,
    /// контракт sources' CTC silence mask для endpointing «blank»).
    pub fn decode_window(
        &mut self,
        features: &[f32],
        n_frames: usize,
        window_sec: f32,
    ) -> Result<crate::stream::DecodedWindow> {
        let (logprobs, enc_len, classes) =
            self.run_encoder(features.to_vec(), n_frames)?;
        let frame_rate = enc_len as f32 / window_sec.max(1e-6);
        let silence: Vec<bool> = if enc_len == 0 {
            Vec::new()
        } else {
            (0..enc_len)
                .map(|t| {
                    let row = &logprobs[t * classes..(t + 1) * classes];
                    let mut best = 0usize;
                    let mut bv = f32::NEG_INFINITY;
                    for (i, &v) in row.iter().enumerate() {
                        if v > bv {
                            bv = v;
                            best = i;
                        }
                    }
                    best == self.blank_id
                })
                .collect()
        };
        let words = if enc_len == 0 {
            Vec::new()
        } else {
            let toks = self.greedy_tokens(&logprobs, enc_len, classes);
            let shift = window_sec as f64 / enc_len as f64;
            build_words(&self.vocab, &toks, shift, 0.0)
                .into_iter()
                .map(|w| crate::stream::TWord {
                    text: w.word,
                    start: w.start as f32,
                    end: w.end as f32,
                })
                .collect()
        };
        Ok(crate::stream::DecodedWindow { words, silence, output_frames: enc_len, frame_rate })
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

/// Число энкодерных фреймов для n мел-фреймов: субдискретизация ×4
/// (мел 100 fps → энкодер ~25 fps, ~40 мс на фрейм). Формула сверена
/// с фактической длиной выхода модели на тестовых длинах — см. тест
/// `subsampling_formula_matches_model` (нужна модель).
pub(crate) fn enc_frames(n_mel: usize) -> usize {
    n_mel.div_ceil(4)
}

/// Сборка сегмента из greedy-токенов: нормализация текста (▁ → пробел),
/// слова с таймингами. Общий для одиночного и батч-путей.
fn toks_to_segment(
    vocab: &[String],
    toks: &[(usize, usize, f32)],
    shift: f64,
    offset: f64,
    dur: f64,
) -> Option<Segment> {
    if toks.is_empty() {
        return None;
    }
    let raw_text: String = toks
        .iter()
        .map(|&(_, id, _)| vocab[id].clone())
        .collect::<Vec<_>>()
        .join("")
        .replace('▁', " ");
    let text = raw_text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return None;
    }
    let words = build_words(vocab, toks, shift, offset);
    Some(Segment {
        start: offset,
        end: offset + dur,
        text,
        words,
        speaker: None,
        channel: None,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn load_model() -> Result<GigaAmV3> {
        let dir = Path::new("models/gigaam-v3");
        GigaAmV3::load(
            &dir.join(GIGAAM_ONNX),
            &dir.join("v3_e2e_ctc_vocab.txt"),
            false, // CPU EP: детерминированно и без требования GPU в тестах
            4,
            None,
        )
    }

    const GIGAAM_ONNX: &str = "v3_e2e_ctc.onnx";

    /// Детерминированный псевдошум (LCG) — контент не важен, важна
    /// воспроизводимость: сравниваем одиночный и батч-прогоны одной модели.
    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// Формула субдискретизации ×4 обязана совпадать с фактической длиной
    /// выхода модели на всём диапазоне длин (иначе батч-путь порежет
    /// log_probs не там).
    #[test]
    #[ignore = "нужна модель models/gigaam-v3"]
    fn subsampling_formula_matches_model() {
        let mut g = load_model().unwrap();
        for n_samples in [16_000usize, 37_333, 160_000, 301_111, 479_900]
            .into_iter()
            .chain((96..=104).chain(996..=1004).chain(2996..=3004).map(|nf| (nf - 1) * 160 + 400)) {
            let (features, n_frames) = g.mel.compute(&noise(n_samples, 7));
            let (logprobs, enc_len, classes) = g.run_encoder(features, n_frames).unwrap();
            assert_eq!(logprobs.len(), enc_len * classes);
            assert_eq!(
                enc_len,
                enc_frames(n_frames),
                "n_mel={n_frames}: модель дала {enc_len}, формула {}",
                enc_frames(n_frames)
            );
        }
    }

    /// Батч из реальных речевых чанков разной длины (фикстуры tests/_dlg_*,
    /// склеенные до разной длительности): текст и тайминги слов обязаны
    /// совпасть с одиночными прогонами. На реальной речи аргмаксы уверенные —
    /// паддинг не должен их переворачивать (шум не годится: там CTC на грани
    /// и Any сравнение бессмысленно).
    #[test]
    #[ignore = "нужна модель models/gigaam-v3 и фикстуры tests/_dlg_*.wav"]
    fn batch_matches_single_runs() {
        let mut g = load_model().unwrap();
        let decode = |name: &str| -> Vec<f32> {
            let mut out = Vec::new();
            let path = std::path::Path::new("tests").join(name);
            crate::audio::decode_streaming(&path, crate::audio::TARGET_SR, 16_000, |block| {
                out.extend_from_slice(block);
                Ok(())
            })
            .unwrap();
            out
        };
        // 30 c из двух длинных реплик, ~17 c, ~5 c, ~22 c
        let d0 = decode("_dlg_0.wav");
        let d1 = decode("_dlg_1.wav");
        let d2 = decode("_dlg_2.wav");
        let d3 = decode("_dlg_3.wav");
        let mut long = d0.clone();
        long.extend_from_slice(&d1);
        long.extend_from_slice(&d0);
        long.truncate(30 * 16_000);
        let chunks: Vec<Vec<f32>> = vec![
            long,
            {
                let mut v = d1.clone();
                v.extend_from_slice(&d2);
                v.truncate((17.3 * 16_000.0) as usize);
                v
            },
            d2.clone(),
            {
                let mut v = d3.clone();
                v.extend_from_slice(&d0);
                v.truncate((22.7 * 16_000.0) as usize);
                v
            },
        ];
        let offsets: Vec<f64> = vec![0.0, 100.0, 250.5, 999.0];

        let singles: Vec<Option<Segment>> = chunks
            .iter()
            .zip(&offsets)
            .map(|(c, &o)| g.transcribe_chunk(c, o).unwrap())
            .collect();
        assert!(
            singles.iter().filter_map(|s| s.as_ref()).all(|s| !s.words.is_empty()),
            "фикстуры должны давать слова"
        );
        let refs: Vec<&[f32]> = chunks.iter().map(|c| c.as_slice()).collect();
        let batched = g.transcribe_chunks(&refs, &offsets).unwrap();

        for (i, (single, batch)) in singles.iter().zip(&batched).enumerate() {
            let (Some(a), Some(b)) = (single, batch) else {
                panic!("чанк {i}: Some/None разъехались");
            };
            assert_eq!(a.text, b.text, "чанк {i}: текст разъехался");
            assert_eq!(a.words.len(), b.words.len(), "чанк {i}: число слов");
            for (wa, wb) in a.words.iter().zip(&b.words) {
                assert_eq!(wa.word, wb.word, "чанк {i}");
                assert!(
                    (wa.start - wb.start).abs() < 0.05 && (wa.end - wb.end).abs() < 0.05,
                    "чанк {i}: тайминг «{}»: {}..{} vs {}..{}",
                    wa.word, wa.start, wa.end, wb.start, wb.end
                );
            }
        }
    }
}
