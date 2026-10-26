//! Многоканальный вход: выбор каналов (dual-mono) и сборка диалога (turns).
//! Семантика перенесена из gigaam-v3-runtime
//! (`crates/transcription/src/{channel_selection,turns}.rs`,
//! `crates/audio/src/channel_analysis.rs`): каждый канал входного файла —
//! отдельный спикер.

use crate::transcript::{Turn, Word, join_word_texts};

/// Порог корреляции Пирсона, выше которого два канала считаются копиями
/// друг друга (dual-mono).
pub const DUPLICATE_THRESHOLD: f64 = 0.98;

/// Корреляция Пирсона двух каналов равной ненулевой длины.
/// Постоянный канал (нулевой дисперсии) даёт 0 — это ограниченное
/// «не положительно коррелирует», а не ошибка.
pub fn channel_correlation(left: &[f32], right: &[f32]) -> anyhow::Result<f64> {
    if left.is_empty() || right.is_empty() {
        anyhow::bail!("корреляция каналов требует непустых каналов");
    }
    if left.len() != right.len() {
        anyhow::bail!(
            "корреляция каналов требует равной длины: {} и {}",
            left.len(),
            right.len()
        );
    }
    let count = left.len() as f64;
    let left_mean = left.iter().map(|s| f64::from(*s)).sum::<f64>() / count;
    let right_mean = right.iter().map(|s| f64::from(*s)).sum::<f64>() / count;
    let (mut numerator, mut left_energy, mut right_energy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&l, &r) in left.iter().zip(right.iter()) {
        let ld = f64::from(l) - left_mean;
        let rd = f64::from(r) - right_mean;
        numerator += ld * rd;
        left_energy += ld * ld;
        right_energy += rd * rd;
    }
    let denominator = (left_energy * right_energy).sqrt();
    if denominator == 0.0 {
        return Ok(0.0);
    }
    let value = numerator / denominator;
    // допуск только на округление f64 у математической границы [-1, 1]
    const TOLERANCE: f64 = 1e-12;
    if !value.is_finite() || !(-(1.0 + TOLERANCE)..=(1.0 + TOLERANCE)).contains(&value) {
        anyhow::bail!("корреляция каналов вышла за границы [-1, 1]");
    }
    Ok(value.clamp(-1.0, 1.0))
}

/// Итог выбора каналов для одного входа.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelPlan {
    /// Каналов в исходном файле.
    pub source_channels: usize,
    /// Классификация dual-mono (только для ровно двух каналов).
    pub dual_mono: bool,
    /// Отобранные каналы в порядке возрастания исходных номеров.
    pub selected: Vec<usize>,
}

/// Выбор каналов по политике split: dual-mono схлопывается в канал 0,
/// иначе распознаются все каналы. `corr01` — корреляция каналов 0 и 1
/// (None, если каналов не два или окна не хватило).
pub fn select_channels(source_channels: usize, corr01: Option<f64>) -> ChannelPlan {
    let dual_mono = source_channels == 2
        && corr01.map(|c| c > DUPLICATE_THRESHOLD).unwrap_or(false);
    let selected = if dual_mono {
        vec![0]
    } else {
        (0..source_channels).collect()
    };
    ChannelPlan { source_channels, dual_mono, selected }
}

/// Слова одного канала в хронологическом порядке.
pub struct ChannelWords {
    pub channel: usize,
    pub words: Vec<Word>,
}

fn turn_from_words(channel: usize, words: &[Word]) -> Turn {
    let start = words[0].start;
    let end = words.iter().map(|w| w.end).fold(start, f64::max);
    Turn {
        channel,
        start,
        end,
        text: join_word_texts(words),
    }
}

/// Реплики одного канала: разрыв ≥ `gap` секунд между end слова и start
/// следующего режет реплику. Каналы идут в порядке `channels`.
pub fn channel_turns(channels: &[ChannelWords], gap: f64) -> Vec<Turn> {
    let mut out = Vec::new();
    for ch in channels {
        let mut current: Vec<Word> = Vec::new();
        for word in &ch.words {
            let cut = match current.last() {
                Some(last) => word.start - last.end >= gap,
                None => false,
            };
            if cut {
                out.push(turn_from_words(ch.channel, &current));
                current.clear();
            }
            current.push(word.clone());
        }
        if !current.is_empty() {
            out.push(turn_from_words(ch.channel, &current));
        }
    }
    out
}

/// Реплики всех каналов, упорядоченные по (start, channel) — хронологический
/// диалог.
pub fn build_turns(channels: &[ChannelWords], gap: f64) -> Vec<Turn> {
    let mut turns = channel_turns(channels, gap);
    turns.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then_with(|| a.channel.cmp(&b.channel))
    });
    turns
}

/// Слияние слов всех каналов в единый хронологический поток:
/// сортировка по (start, channel) — как `turns::merge` в источнике.
pub fn merge_words(channels: &[ChannelWords]) -> Vec<Word> {
    let mut words: Vec<Word> = channels
        .iter()
        .flat_map(|ch| ch.words.iter().cloned())
        .collect();
    words.sort_by(|a, b| a.start.total_cmp(&b.start).then_with(|| {
        a.channel.unwrap_or(0).cmp(&b.channel.unwrap_or(0))
    }));
    words
}

/// Проверка turn_gap: конечное неотрицательное число секунд.
pub fn validate_turn_gap(gap: f64) -> anyhow::Result<f64> {
    if !gap.is_finite() || gap < 0.0 {
        anyhow::bail!("turn_gap должен быть конечным неотрицательным числом секунд");
    }
    Ok(gap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64, channel: usize) -> Word {
        Word {
            word: text.into(),
            start,
            end,
            score: None,
            speaker: None,
            channel: Some(channel),
        }
    }

    #[test]
    fn dual_mono_selection_collapses_to_channel_zero() {
        let plan = select_channels(2, Some(0.99));
        assert!(plan.dual_mono);
        assert_eq!(plan.selected, vec![0]);
        let plan = select_channels(2, Some(0.97));
        assert!(!plan.dual_mono);
        assert_eq!(plan.selected, vec![0, 1]);
        let plan = select_channels(3, Some(0.99));
        assert!(!plan.dual_mono);
        assert_eq!(plan.selected, vec![0, 1, 2]);
    }

    #[test]
    fn correlation_boundaries() {
        let a = [0.0_f32, 0.5, -0.5, 0.25];
        let b = [0.0_f32, 0.5, -0.5, 0.25];
        assert!((channel_correlation(&a, &b).unwrap() - 1.0).abs() < 1e-9);
        let c = [0.0_f32, -0.5, 0.5, -0.25];
        assert!((channel_correlation(&a, &c).unwrap() + 1.0).abs() < 1e-9);
        assert!(channel_correlation(&[0.0], &a).is_err());
    }

    #[test]
    fn turns_split_on_gap_and_sort_by_start_then_channel() {
        let ch0 = ChannelWords {
            channel: 0,
            words: vec![
                word("hello", 0.0, 0.4, 0),
                word("yes", 0.5, 0.8, 0),
                word("bye", 3.4, 3.9, 0),
            ],
        };
        let ch1 = ChannelWords {
            channel: 1,
            words: vec![word("hi", 1.0, 1.5, 1)],
        };
        let turns = build_turns(&[ch0, ch1], 1.0);
        let got: Vec<(usize, &str)> = turns
            .iter()
            .map(|t| (t.channel, t.text.as_str()))
            .collect();
        assert_eq!(got, vec![(0, "hello yes"), (1, "hi"), (0, "bye")]);
        assert_eq!(turns[0].end, 0.8);
        assert_eq!(turns[2].start, 3.4);
    }
}
