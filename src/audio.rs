//! Audio decoding via ffmpeg libraries: any input format -> 16 kHz mono f32.

use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Sample;
use ffmpeg_next::software::resampling;
use ffmpeg_next::util::channel_layout::ChannelLayout;
use ffmpeg_next::util::frame::Audio as AudioFrame;
use std::path::Path;

pub const TARGET_SR: u32 = 16_000;

fn collect(out: &AudioFrame, acc: &mut Vec<f32>) -> Result<()> {
    if out.samples() == 0 {
        return Ok(());
    }
    // target layout is mono; packed or planar both expose a single plane
    let plane = out.plane::<f32>(0);
    acc.extend_from_slice(plane);
    Ok(())
}

fn drain_decoder(
    decoder: &mut ffmpeg::decoder::Audio,
    resampler: &mut Option<resampling::Context>,
    acc: &mut Vec<f32>,
    target_sr: u32,
) -> Result<()> {
    let mut frame = AudioFrame::empty();
    loop {
        match decoder.receive_frame(&mut frame) {
            Ok(_) => {
                // WAV/PCM frames may carry an UNSPEC channel order; swr requires
                // an exact layout match, so normalize to the default (native)
                // layout for the channel count
                if frame.channel_layout().is_empty() {
                    frame.set_channel_layout(ChannelLayout::default(frame.channels() as i32));
                }
                // build the resampler from the first real frame: decoder-reported
                // parameters may disagree with what frames actually carry
                if resampler.is_none() {
                    tracing::debug!(
                        frame_format = ?frame.format(),
                        frame_rate = frame.rate(),
                        frame_channels = frame.channels(),
                        layout_channels = frame.channel_layout().channels(),
                        "first decoded frame"
                    );
                    *resampler = Some(
                        ffmpeg::software::resampler(
                            (frame.format(), frame.channel_layout(), frame.rate()),
                            (
                                Sample::F32(ffmpeg_next::format::sample::Type::Packed),
                                ChannelLayout::MONO,
                                target_sr,
                            ),
                        )
                        .context("swresample context")?,
                    );
                }
                let rs = resampler.as_mut().unwrap();
                // pre-allocate generously (covers upsampled rates, e.g. 8k -> 16k)
                let mut out = AudioFrame::empty();
                unsafe {
                    out.alloc(
                        Sample::F32(ffmpeg_next::format::sample::Type::Packed),
                        frame.samples() * 8 + 1024,
                        ChannelLayout::MONO,
                    );
                }
                rs.run(&frame, &mut out).context("resampler run")?;
                collect(&out, acc)?;
            }
            Err(ffmpeg::util::error::Error::Other { errno: ffmpeg::util::error::EAGAIN }) => break,
            Err(e) if e == ffmpeg::util::error::Error::Eof => break,
            Err(e) => return Err(e).context("ffmpeg decode"),
        }
    }
    Ok(())
}

/// Decode any audio/video file to mono f32 using libavformat/libavcodec/
/// libswresample. Streams results in `block_samples`-sized blocks through `cb`;
/// returns the total decoded duration in seconds.
pub fn decode_streaming(
    path: &Path,
    target_sr: u32,
    block_samples: usize,
    mut cb: impl FnMut(&[f32]) -> Result<()>,
) -> Result<f64> {
    ffmpeg::init().context("ffmpeg init")?;

    let mut ictx = ffmpeg::format::input(path)
        .with_context(|| format!("ffmpeg cannot open: {}", path.display()))?;

    let fmt_name = ictx.format().name().to_string();
    let duration = ictx.duration() as f64 / 1_000_000.0;

    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .context("no audio stream found in file")?;
    let stream_index = stream.index();

    let mut codec = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .context("stream codec context")?;
    let codec_name = codec
        .codec()
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "?".into());
    // multithreaded decoding where the codec supports it
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    codec.set_threading(ffmpeg_next::codec::threading::Config {
        kind: ffmpeg_next::codec::threading::Type::Frame,
        count: (cores / 2).clamp(1, 8),
    });
    let mut decoder = codec.decoder().audio().context("audio decoder")?;

    tracing::info!(
        file = %path.display(),
        container = %fmt_name,
        codec = %codec_name,
        container_duration_s = format!("{duration:.1}"),
        "decoding"
    );

    let mut resampler: Option<resampling::Context> = None;
    let mut acc: Vec<f32> = Vec::new();
    let mut total = 0usize;
    let mut last_progress = 0usize;

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet).context("send_packet")?;
        drain_decoder(&mut decoder, &mut resampler, &mut acc, target_sr)?;
        while acc.len() >= block_samples {
            cb(&acc[..block_samples])?;
            total += block_samples;
            acc.drain(..block_samples);
        }
        if total - last_progress >= target_sr as usize * 60 {
            last_progress = total;
            tracing::info!("декодировано {}с...", total / target_sr as usize);
        }
    }
    decoder.send_eof().context("send_eof")?;
    drain_decoder(&mut decoder, &mut resampler, &mut acc, target_sr)?;

    // flush resampler tail
    if let Some(rs) = resampler.as_mut() {
        loop {
            let mut out = AudioFrame::empty();
            unsafe {
                out.alloc(
                    Sample::F32(ffmpeg_next::format::sample::Type::Packed),
                    32_768,
                    ChannelLayout::MONO,
                );
            }
            match rs.flush(&mut out) {
                Ok(_) => {
                    let got = out.samples();
                    collect(&out, &mut acc)?;
                    if got == 0 {
                        break;
                    }
                }
                Err(e) => return Err(e).context("resampler flush"),
            }
        }
    }
    // emit the remainder
    if !acc.is_empty() {
        cb(&acc)?;
        total += acc.len();
    }

    Ok(total as f64 / target_sr as f64)
}
