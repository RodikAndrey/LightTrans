//! Audio decoding via ffmpeg libraries: any input format → f32 at the target
//! sample rate, mono (mixed) or with all source channels preserved
//! (interleaved). Files and in-memory readers (custom AVIO) are supported.

use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Sample;
use ffmpeg_next::software::resampling;
use ffmpeg_next::util::channel_layout::ChannelLayout;
use ffmpeg_next::util::frame::Audio as AudioFrame;
use std::io::{Read, Seek};
use std::path::Path;

pub const TARGET_SR: u32 = 16_000;

/// Read + Seek + Send одним трейтом: dyn-объект не может нести два не-auto
/// трейта, поэтому объединяем их в супертрейт.
pub trait ReadSeekSend: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeekSend for T {}

/// Источник входа: путь к файлу или уже прочитанное в память тело
/// (для HTTP-запросов — custom AVIO без временных файлов).
pub enum Source<'a> {
    Path(&'a Path),
    /// Тело в памяти + подсказка формата (`ext`): участвует в детекте
    /// контейнера, как имя файла у ffmpeg.
    Bytes(Vec<u8>, Option<String>),
    /// Чужой reader — сервер декодирует тело без лишних копий.
    Reader(Box<dyn ReadSeekSend>, Option<String>),
}

/// Параметры вывода и счётчики декодера (отдельно от libav-контекстов,
/// чтобы их можно было заимствовать независимо от `ictx`).
struct DecodeState {
    target_sr: u32,
    /// 1 — микс в mono (прежнее поведение), 0 — сохранить исходные каналы.
    target_channels: u32,
    block_frames: usize,
    acc: Vec<f32>, // интерливнутые сэмплы до отправки блока
    channels: usize,
    source_rate: u32,
    total_frames: usize,
    last_progress: usize,
}

/// Стриминговый декодер: отдаёт интерливнутые блоки по `block_frames` фреймов
/// (фрейм = один сэмпл каждого канала).
pub struct Decoder {
    ictx: ffmpeg::format::context::Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Audio,
    resampler: Option<resampling::Context>,
    state: DecodeState,
    done: bool,
}

impl Decoder {
    pub fn open(
        source: Source<'_>,
        target_sr: u32,
        target_channels: u32,
        block_frames: usize,
    ) -> Result<Self> {
        ffmpeg::init().context("ffmpeg init")?;
        let (ictx, fmt_name, duration, label) = match source {
            Source::Path(path) => {
                let ictx = ffmpeg::format::input(path)
                    .with_context(|| format!("ffmpeg cannot open: {}", path.display()))?;
                let name = ictx.format().name().to_string();
                let dur = ictx.duration() as f64 / 1_000_000.0;
                (ictx, name, dur, path.display().to_string())
            }
            Source::Bytes(body, hint) => {
                let io = ffmpeg::format::context::StreamIo::from_read_seek(std::io::Cursor::new(body))
                    .context("custom AVIO context")?;
                let ictx = ffmpeg::format::input_from_stream(io, hint.as_deref(), None)
                    .context("ffmpeg cannot open in-memory audio")?;
                let name = ictx.format().name().to_string();
                let dur = ictx.duration() as f64 / 1_000_000.0;
                (ictx, name, dur, hint.unwrap_or_else(|| "<memory>".into()))
            }
            Source::Reader(reader, hint) => {
                let io = ffmpeg::format::context::StreamIo::from_read_seek(reader)
                    .context("custom AVIO context")?;
                let ictx = ffmpeg::format::input_from_stream(io, hint.as_deref(), None)
                    .context("ffmpeg cannot open streamed audio")?;
                let name = ictx.format().name().to_string();
                let dur = ictx.duration() as f64 / 1_000_000.0;
                (ictx, name, dur, hint.unwrap_or_else(|| "<reader>".into()))
            }
        };

        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .context("no audio stream found in input")?;
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
        let decoder = codec.decoder().audio().context("audio decoder")?;
        // параметры потока нужны ДО декодирования (число каналов для
        // деинтерливинга); сверяются с первым реальным фреймом ниже
        let channels = decoder.channels().max(1) as usize;
        let source_rate = decoder.rate();

        tracing::info!(
            input = %label,
            container = %fmt_name,
            codec = %codec_name,
            channels,
            source_rate,
            container_duration_s = format!("{duration:.1}"),
            "decoding"
        );

        Ok(Self {
            ictx,
            stream_index,
            decoder,
            resampler: None,
            state: DecodeState {
                target_sr,
                target_channels,
                block_frames,
                acc: Vec::new(),
                channels,
                source_rate,
                total_frames: 0,
                last_progress: 0,
            },
            done: false,
        })
    }

    pub fn open_path(
        path: &Path,
        target_sr: u32,
        target_channels: u32,
        block_frames: usize,
    ) -> Result<Self> {
        Self::open(Source::Path(path), target_sr, target_channels, block_frames)
    }

    /// Декодировать тело из памяти. `ext` — подсказка формата (None = autodetect).
    pub fn open_bytes(
        body: Vec<u8>,
        ext: Option<&str>,
        target_sr: u32,
        target_channels: u32,
        block_frames: usize,
    ) -> Result<Self> {
        let hint = ext.map(|e| format!("audio.{e}"));
        Self::open(Source::Bytes(body, hint), target_sr, target_channels, block_frames)
    }

    /// Число каналов исходного аудио (по параметрам потока).
    pub fn channels(&self) -> usize {
        self.state.channels
    }

    /// Частота дискретизации исходного аудио, Гц.
    pub fn source_rate(&self) -> u32 {
        self.state.source_rate
    }

    /// Длительность из контейнера (может отсутствовать).
    pub fn container_duration(&self) -> Option<f64> {
        let d = self.ictx.duration();
        if d > 0 {
            Some(d as f64 / 1_000_000.0)
        } else {
            None
        }
    }

    /// Декодировать остаток потока, отправляя интерливнутые блоки ровно по
    /// `block_frames × каналы` сэмплов через `cb` (последний блок может быть
    /// неполным). Возвращает число декодированных фреймов на target rate.
    pub fn decode(&mut self, mut cb: impl FnMut(&[f32]) -> Result<()>) -> Result<usize> {
        if self.done {
            anyhow::bail!("повторный decode не поддерживается");
        }
        // заёмы полей по отдельности: итератор держит `ictx`, остальное
        // доступно через независимые поля
        {
            let ictx = &mut self.ictx;
            for (stream, packet) in ictx.packets() {
                if stream.index() != self.stream_index {
                    continue;
                }
                self.decoder.send_packet(&packet).context("send_packet")?;
                drain_decoder(&mut self.decoder, &mut self.resampler, &mut self.state, &mut cb)?;
            }
        }
        self.decoder.send_eof().context("send_eof")?;
        drain_decoder(&mut self.decoder, &mut self.resampler, &mut self.state, &mut cb)?;
        flush_resampler(&mut self.resampler, &mut self.state, &mut cb)?;
        // emit the remainder
        if !self.state.acc.is_empty() {
            cb(&self.state.acc)?;
            let nch = if self.state.target_channels == 0 {
                self.state.channels
            } else {
                self.state.target_channels as usize
            };
            self.state.total_frames += self.state.acc.len() / nch.max(1);
            self.state.acc.clear();
        }
        self.done = true;
        Ok(self.state.total_frames)
    }
}

fn target_channel_count(state: &DecodeState, frame: &AudioFrame) -> i32 {
    if state.target_channels == 0 {
        frame.channels().max(1) as i32
    } else {
        state.target_channels as i32
    }
}

fn make_resampler(state: &DecodeState, frame: &AudioFrame) -> Result<resampling::Context> {
    // target layout: mono (микс) или исходная раскладка каналов
    let target_channels = target_channel_count(state, frame);
    let target_layout = if target_channels == 1 {
        ChannelLayout::MONO
    } else {
        ChannelLayout::default(target_channels)
    };
    ffmpeg::software::resampler(
        (frame.format(), frame.channel_layout(), frame.rate()),
        (
            Sample::F32(ffmpeg_next::format::sample::Type::Packed),
            target_layout,
            state.target_sr,
        ),
    )
    .context("swresample context")
}

fn drain_decoder(
    decoder: &mut ffmpeg::decoder::Audio,
    resampler: &mut Option<resampling::Context>,
    state: &mut DecodeState,
    cb: &mut impl FnMut(&[f32]) -> Result<()>,
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
                    if state.target_channels == 0 && frame.channels() as usize != state.channels {
                        anyhow::bail!(
                            "число каналов изменилось посреди потока: {} → {}",
                            state.channels,
                            frame.channels()
                        );
                    }
                    *resampler = Some(make_resampler(state, &frame)?);
                }
                let rs = resampler.as_mut().unwrap();
                // pre-allocate generously (covers upsampled rates, e.g. 8k -> 16k)
                let out_channels = target_channel_count(state, &frame);
                let mut out = AudioFrame::empty();
                unsafe {
                    out.alloc(
                        Sample::F32(ffmpeg_next::format::sample::Type::Packed),
                        frame.samples() as usize * 8 + 1024,
                        if out_channels == 1 {
                            ChannelLayout::MONO
                        } else {
                            ChannelLayout::default(out_channels)
                        },
                    );
                }
                rs.run(&frame, &mut out).context("resampler run")?;
                collect(&out, state, cb)?;
            }
            Err(ffmpeg::util::error::Error::Other { errno: ffmpeg::util::error::EAGAIN }) => break,
            Err(e) if e == ffmpeg::util::error::Error::Eof => break,
            Err(e) => return Err(e).context("ffmpeg decode"),
        }
    }
    Ok(())
}

fn flush_resampler(
    resampler: &mut Option<resampling::Context>,
    state: &mut DecodeState,
    cb: &mut impl FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    let Some(rs) = resampler.as_mut() else {
        return Ok(());
    };
    let out_channels = if state.target_channels == 0 {
        state.channels as i32
    } else {
        state.target_channels as i32
    };
    loop {
        let mut out = AudioFrame::empty();
        unsafe {
            out.alloc(
                Sample::F32(ffmpeg_next::format::sample::Type::Packed),
                32_768,
                if out_channels == 1 {
                    ChannelLayout::MONO
                } else {
                    ChannelLayout::default(out_channels)
                },
            );
        }
        match rs.flush(&mut out) {
            Ok(_) => {
                let got = out.samples();
                collect(&out, state, cb)?;
                if got == 0 {
                    break;
                }
            }
            Err(e) => return Err(e).context("resampler flush"),
        }
    }
    Ok(())
}

fn collect(
    out: &AudioFrame,
    state: &mut DecodeState,
    cb: &mut impl FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    if out.samples() == 0 {
        return Ok(());
    }
    // packed layout: все каналы в одной плоскости, интерливнуто
    let plane = out.plane::<f32>(0);
    state.acc.extend_from_slice(plane);
    let block = state.block_frames * out.channels().max(1) as usize;
    while state.acc.len() >= block {
        cb(&state.acc[..block])?;
        state.total_frames += state.block_frames;
        state.acc.drain(..block);
    }
    if state.total_frames - state.last_progress >= state.target_sr as usize * 60 {
        state.last_progress = state.total_frames;
        tracing::info!("декодировано {}с...", state.total_frames / state.target_sr as usize);
    }
    Ok(())
}

/// Decode any audio/video file to mono f32 (16 kHz by default) using
/// libavformat/libavcodec/libswresample. Streams results in
/// `block_samples`-sized blocks through `cb`; returns the total decoded
/// duration in seconds.
pub fn decode_streaming(
    path: &Path,
    target_sr: u32,
    block_samples: usize,
    mut cb: impl FnMut(&[f32]) -> Result<()>,
) -> Result<f64> {
    let mut dec = Decoder::open_path(path, target_sr, 1, block_samples)?;
    let frames = dec.decode(|block| cb(block))?;
    Ok(frames as f64 / target_sr as f64)
}
