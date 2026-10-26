//! Core result types shared by all backends (WhisperX-compatible schema).

#[derive(Debug, Clone, serde::Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// Speaker label (e.g. "SPEAKER_00") when diarization is enabled, or the
    /// channel-derived label (`SPEAKER_0N`) in split-channels mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Исходный номер канала входа (только в режиме разделения каналов).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
    /// Majority speaker of the segment's words when diarization is enabled
    /// or channels are split into speakers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Исходный номер канала сегмента (в режиме разделения каналов).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<usize>,
}

/// Реплика диалога одного канала (режим разделения каналов).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Turn {
    pub channel: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Transcript {
    pub language: String,
    pub segments: Vec<Segment>,
    /// Реплики диалога (только в режиме разделения каналов с --turns).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub turns: Vec<Turn>,
}

impl Default for Transcript {
    fn default() -> Self {
        Self { language: String::new(), segments: Vec::new(), turns: Vec::new() }
    }
}

impl Transcript {
    pub fn words_flat(&self) -> Vec<&Word> {
        self.segments.iter().flat_map(|s| s.words.iter()).collect()
    }

    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Rebuild cue text from word entries. Words from some backends carry no
/// spacing (GigaAM strips the ▁ marker, whisper drops the token's leading
/// space), so a plain concatenation would glue everything together — insert a
/// space between adjacent word-ish tokens, but attach trailing punctuation
/// (`,.!?;:…»)%-`) to the previous word and don't space after opening
/// punctuation (`«(`).
pub fn join_word_texts(words: &[Word]) -> String {
    let mut s = String::new();
    for w in words {
        let t = w.word.trim();
        if t.is_empty() {
            continue;
        }
        let prev_wordish = s.ends_with(|c: char| {
            c.is_alphanumeric()
                || matches!(c, ',' | '.' | '!' | '?' | ';' | ':' | '…' | '»' | ')' | '%' | '—' | '’')
        });
        let cur_attached =
            t.starts_with(|c: char| matches!(c, ',' | '.' | '!' | '?' | ';' | ':' | '…' | '»' | ')' | '%' | '-'));
        if !s.is_empty() && prev_wordish && !cur_attached {
            s.push(' ');
        }
        s.push_str(t);
    }
    s
}

/// Half-open speech interval in seconds.
pub type TimeInterval = (f64, f64);
