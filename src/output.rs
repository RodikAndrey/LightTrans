//! Output writers: WhisperX-compatible json, plus srt/vtt/txt/tsv.

use crate::transcript::Transcript;
use anyhow::Result;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutFormat {
    Txt,
    Srt,
    Vtt,
    Tsv,
    Json,
}

impl OutFormat {
    pub fn ext(&self) -> &'static str {
        match self {
            OutFormat::Txt => "txt",
            OutFormat::Srt => "srt",
            OutFormat::Vtt => "vtt",
            OutFormat::Tsv => "tsv",
            OutFormat::Json => "json",
        }
    }
}

pub fn parse_formats(spec: &str) -> Vec<OutFormat> {
    if spec.eq_ignore_ascii_case("all") {
        return vec![
            OutFormat::Txt,
            OutFormat::Srt,
            OutFormat::Vtt,
            OutFormat::Tsv,
            OutFormat::Json,
        ];
    }
    spec.split(',')
        .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
            "txt" => Some(OutFormat::Txt),
            "srt" => Some(OutFormat::Srt),
            "vtt" => Some(OutFormat::Vtt),
            "tsv" => Some(OutFormat::Tsv),
            "json" => Some(OutFormat::Json),
            _ => None,
        })
        .collect()
}

pub fn write_outputs(
    t: &Transcript,
    stem: &Path,
    formats: &[OutFormat],
) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    for f in formats {
        let path = stem.with_extension(f.ext());
        let mut file = std::fs::File::create(&path)?;
        match f {
            OutFormat::Txt => write!(file, "{}\n", t.text())?,
            OutFormat::Srt => write_srt(&mut file, t)?,
            OutFormat::Vtt => write_vtt(&mut file, t)?,
            OutFormat::Tsv => write_tsv(&mut file, t)?,
            OutFormat::Json => write_json(&mut file, t)?,
        }
        written.push(path);
    }
    Ok(written)
}

fn write_json(file: &mut std::fs::File, t: &Transcript) -> Result<()> {
    let word_segments: Vec<&crate::transcript::Word> = t.words_flat();
    let doc = serde_json::json!({
        "language": t.language,
        "segments": t.segments,
        "word_segments": word_segments,
    });
    serde_json::to_writer_pretty(&mut *file, &doc)?;
    writeln!(file)?;
    Ok(())
}

fn fmt_hms(secs: f64, ms_sep: char) -> String {
    let s = secs.max(0.0);
    let h = (s / 3600.0).floor() as u64;
    let m = ((s - h as f64 * 3600.0) / 60.0).floor() as u64;
    let sec = (s - h as f64 * 3600.0 - m as f64 * 60.0).floor() as u64;
    let ms = ((s - (h as f64 * 3600.0 + m as f64 * 60.0 + sec as f64)) * 1000.0).round() as u64;
    format!("{h:02}:{m:02}:{sec:02}{ms_sep}{ms:03}")
}

fn write_srt(file: &mut std::fs::File, t: &Transcript) -> Result<()> {
    for (i, seg) in t.segments.iter().enumerate() {
        let prefix = match &seg.speaker {
            Some(s) => format!("[{s}] "),
            None => String::new(),
        };
        writeln!(file, "{}", i + 1)?;
        writeln!(file, "{} --> {}", fmt_hms(seg.start, ','), fmt_hms(seg.end, ','))?;
        writeln!(file, "{prefix}{}\n", seg.text)?;
    }
    Ok(())
}

fn write_vtt(file: &mut std::fs::File, t: &Transcript) -> Result<()> {
    writeln!(file, "WEBVTT\n")?;
    for seg in &t.segments {
        let prefix = match &seg.speaker {
            Some(s) => format!("[{s}] "),
            None => String::new(),
        };
        writeln!(file, "{} --> {}", fmt_hms(seg.start, '.'), fmt_hms(seg.end, '.'))?;
        writeln!(file, "{prefix}{}\n", seg.text)?;
    }
    Ok(())
}

fn write_tsv(file: &mut std::fs::File, t: &Transcript) -> Result<()> {
    writeln!(file, "start\tend\ttext")?;
    for w in t.words_flat() {
        writeln!(
            file,
            "{}\t{}\t{}",
            (w.start * 1000.0) as u64,
            (w.end * 1000.0) as u64,
            w.word
        )?;
    }
    Ok(())
}
