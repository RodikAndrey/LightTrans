//! Офлайн-диагностика стрим-сессии на реальной модели: batch-декод того же
//! аудио против инкрементального пути. Запуск: cargo run --release --example stream_smoke

use lighttrans::gigaam::GigaAmV3;
use lighttrans::stream::{StreamConfig, StreamSession, WindowDecoder};
use std::path::Path;

struct Ctc(GigaAmV3);

impl WindowDecoder for Ctc {
    fn decode_window(
        &mut self,
        features: &[f32],
        n_frames: usize,
        window_sec: f32,
    ) -> anyhow::Result<lighttrans::stream::DecodedWindow> {
        self.0.decode_window(features, n_frames, window_sec)
    }
}

fn main() -> anyhow::Result<()> {
    let wav = std::env::args().nth(1).unwrap_or_else(|| "/tmp/ws_test.wav".into());
    // читаем wav 16k mono pcm16
    let raw = std::fs::read(&wav)?;
    let data_start = raw.iter().position(|b| *b == b'd').unwrap() + 8;
    let samples: Vec<f32> = raw[data_start..]
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    println!("samples: {}", samples.len());

    let dir = Path::new("models/gigaam-v3");
    let mut model = GigaAmV3::load(
        &dir.join("v3_e2e_ctc.onnx"),
        &dir.join("v3_e2e_ctc_vocab.txt"),
        true,
        4,
        None,
    )?;

    // batch
    if let Some(seg) = model.transcribe_chunk(&samples, 0.0)? {
        println!("BATCH: {}", seg.text);
    } else {
        println!("BATCH: <empty>");
    }

    // stream
    let mut s = StreamSession::new(Ctc(model), StreamConfig::default())?;
    s.warmup()?;
    let frame = 3200; // 0.2 c
    let mut total_events = 0;
    for chunk in samples.chunks(frame) {
        let ev = s.push(chunk)?;
        if !ev.is_empty() {
            println!("tick events: {}", ev.len());
            total_events += ev.len();
        }
    }
    let fin = s.flush()?;
    total_events += fin.len();
    println!("total events: {total_events}");
    let line: Vec<String> = s.transcript().iter().map(|w| w.text.clone()).collect();
    println!("STREAM: {}", line.join(" "));
    Ok(())
}
