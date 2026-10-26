//! lighttrans — fast GPU audio/video transcription (WhisperX-style pipeline).
//!
//! Streaming pipeline: ffmpeg decode (any format) -> Silero VAD (streaming,
//! batched) -> chunker -> N worker threads (GPU inference) -> ordered merge ->
//! output writers. Worker count is auto-detected from free VRAM (CUDA) or CPU
//! cores, or set explicitly with --workers.

mod align;
mod audio;
mod df;
mod diarize;
mod gigaam;
mod gigaam_rnnt;
mod mel;
mod parakeet;
mod models;
mod output;
mod segments;
mod stats;
mod transcript;
mod vad;
mod whisper;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, ValueEnum};
use segments::Chunker;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use transcript::{join_word_texts, Segment, Transcript, Word};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ModelKind {
    /// Whisper Large v3 Turbo (multilingual, ggml/whisper.cpp CUDA)
    WhisperTurbo,
    /// GigaAM v3 E2E CTC (Russian, ONNX Runtime CUDA, punctuation + ITN)
    GigaamV3,
    /// GigaAM v3 E2E RNN-T (Russian, ONNX Runtime CUDA, punctuation + ITN)
    GigaamV3Rnnt,
    /// Parakeet TDT 0.6B v3 (25 европейских языков, ONNX Runtime CUDA)
    ParakeetV3,
}

#[derive(Parser)]
#[command(
    name = "lighttrans",
    version,
    disable_help_flag = true,
    term_width = 120,
    about = "Транскрибация аудио/видео на GPU: GigaAM v3 CTC/RNN-T, Whisper Turbo, Parakeet v3",
    after_help = "Примеры:
  lighttrans video.mp4                        gigaam-v3, русский
  lighttrans talk.mkv -m whisper-turbo -l ru --w2v   пословные таймстампы
  lighttrans rec.mp3 --diarize --denoise      спикеры + шумоподавление"
)]
struct Args {
    /// Входные аудио/видео файлы
    files: Vec<PathBuf>,

    /// модель: gigaam-v3|gigaam-v3-rnnt|whisper-turbo|parakeet-v3
    #[arg(short, long, value_enum, hide_possible_values = true, default_value = "gigaam-v3")]
    model: ModelKind,

    /// cuda | cpu
    #[arg(short, long, default_value = "cuda")]
    device: String,

    /// язык (ru, en, ...) или auto; gigaam — только ru
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// transcribe | translate (whisper-turbo)
    #[arg(long, default_value = "transcribe")]
    task: String,

    /// без VAD — нарезать фиксированными окнами
    #[arg(long)]
    no_vad: bool,

    /// порог начала речи
    #[arg(long, default_value_t = 0.5)]
    vad_onset: f32,

    /// порог окончания речи
    #[arg(long, default_value_t = 0.363)]
    vad_offset: f32,

    /// макс. длина чанка, с
    #[arg(long, default_value_t = 30.0)]
    chunk_size: f64,

    /// beam для whisper
    #[arg(long, default_value_t = 1)]
    beam_size: usize,

    /// GPU-воркеров (0 = авто)
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// каталог с моделями
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// каталог результатов
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// txt,srt,vtt,tsv,json | all
    #[arg(short = 'f', long, default_value = "all")]
    output_format: String,

    /// разделение спикеров (Nemotron, до 8)
    #[arg(long)]
    diarize: bool,

    /// пословные таймстампы GigaAM-CTC (whisper-turbo)
    #[arg(long, conflicts_with = "dtw")]
    w2v: bool,

    /// пословные таймстампы DTW (whisper-turbo)
    #[arg(long)]
    dtw: bool,

    /// шумоподавление DeepFilterNet 3
    #[arg(long)]
    denoise: bool,

    /// предел ослабления шума, dB
    #[arg(long, default_value_t = 100.0)]
    denoise_atten_lim: f32,

    /// потоки шумодава (0 = авто)
    #[arg(long, default_value_t = 0)]
    denoise_threads: usize,

    /// не печатать текст в stdout
    #[arg(short, long)]
    quiet: bool,

    /// показать справку и выйти
    #[arg(short = 'h', long)]
    help: bool,
}

/// Backend used by a single inference worker.
enum Worker {
    Whisper(whisper::WhisperTurbo, whisper_rs::WhisperState),
    Gigaam(gigaam::GigaAmV3),
    GigaamRnnt(gigaam_rnnt::GigaAmRnnt),
    Parakeet(parakeet::ParakeetV3),
}

impl Worker {
    fn warmup(&mut self) -> Result<()> {
        let silence = vec![0.0f32; 16_000];
        let (segs, _) = self.transcribe(&silence, 0.0, "en", false, 1, 4)?;
        let _ = segs;
        Ok(())
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        offset: f64,
        language: &str,
        translate: bool,
        beam_size: usize,
        n_threads: usize,
    ) -> Result<(Vec<Segment>, Option<String>)> {
        match self {
            Worker::Whisper(model, state) => {
                let (segs, lang) = model.transcribe_chunk(
                    state,
                    samples,
                    offset,
                    &whisper::WhisperOpts { language, translate, beam_size, n_threads },
                )?;
                Ok((segs, Some(lang)))
            }
            Worker::Gigaam(g) => {
                let seg = g.transcribe_chunk(samples, offset)?;
                Ok((seg.map(|s| vec![s]).unwrap_or_default(), None))
            }
            Worker::GigaamRnnt(g) => {
                let seg = g.transcribe_chunk(samples, offset)?;
                Ok((seg.map(|s| vec![s]).unwrap_or_default(), None))
            }
            Worker::Parakeet(p) => {
                let seg = p.transcribe_chunk(samples, offset)?;
                Ok((seg.map(|s| vec![s]).unwrap_or_default(), None))
            }
        }
    }
}

/// Печатает текст постранично, если терминал ниже текста: страница помещается
/// на экран, дальше — «нажмите клавишу» (q — выход). Если вывод не терминал
/// (пайп, файл) или текст помещается — печатает целиком.
fn print_paged(text: &str) {
    use std::io::{stdout, Write};
    let mut out = stdout().lock();
    let lines: Vec<&str> = text.lines().collect();
    let rows = console_rows().unwrap_or(24);
    if !stdin_is_console() || rows < 5 || lines.len() <= rows - 1 {
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
        return;
    }
    let per_page = rows - 2; // строка на приглашение
    for (i, chunk) in lines.chunks(per_page).enumerate() {
        for l in chunk {
            let _ = writeln!(out, "{l}");
        }
        if i * per_page + chunk.len() < lines.len() {
            let _ = write!(out, "── ещё {}/{} ── любая клавиша — дальше, q — выход ─ ", i * per_page + chunk.len(), lines.len());
            let _ = out.flush();
            if let Some(c) = read_key() {
                if matches!(c, 'q' | 'Q' | 'й' | 'Й') {
                    let _ = writeln!(out);
                    return;
                }
            }
            let _ = writeln!(out);
        }
    }
}

#[cfg(windows)]
mod console {
    const STD_OUTPUT_HANDLE: u32 = -11i64 as u32;
    const STD_INPUT_HANDLE: u32 = -10i64 as u32;

    #[repr(C)]
    #[derive(Default)]
    struct Coord { x: u16, y: u16 }
    #[repr(C)]
    #[derive(Default)]
    struct SmallRect { l: i16, t: i16, r: i16, b: i16 }
    #[repr(C)]
    #[derive(Default)]
    struct ConsoleScreenBufferInfo {
        size: Coord,
        cursor: Coord,
        attr: u16,
        window: SmallRect,
        max_window: Coord,
    }

    extern "system" {
        fn GetStdHandle(n_std_handle: u32) -> isize;
        fn GetConsoleMode(handle: isize, mode: *mut u32) -> i32;
        fn GetConsoleScreenBufferInfo(handle: isize, info: *mut ConsoleScreenBufferInfo) -> i32;
    }

    extern "C" {
        fn _getch() -> i32;
    }

    /// Высота окна консоли в строках (None, если stdout не консоль).
    pub fn rows() -> Option<usize> {
        let h = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        let mut info = ConsoleScreenBufferInfo::default();
        if unsafe { GetConsoleScreenBufferInfo(h, &mut info) } != 0 {
            Some((info.window.b - info.window.t + 1) as usize)
        } else {
            None
        }
    }

    /// Подключён ли stdin к консоли (иначе клавишу не прочитать).
    pub fn input_is_console() -> bool {
        let h = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
        let mut mode = 0u32;
        unsafe { GetConsoleMode(h, &mut mode) != 0 }
    }

    /// Ждёт нажатия клавиши и возвращает её символ.
    pub fn read_key() -> Option<char> {
        let c = unsafe { _getch() };
        char::from_u32(c as u32)
    }
}

#[cfg(not(windows))]
mod console {
    pub fn rows() -> Option<usize> {
        std::env::var("LINES").ok().and_then(|v| v.parse().ok())
    }
    pub fn input_is_console() -> bool {
        true
    }
    pub fn read_key() -> Option<char> {
        use std::io::BufRead;
        let mut s = String::new();
        std::io::stdin().lock().read_line(&mut s).ok()?;
        s.trim().chars().next()
    }
}

fn console_rows() -> Option<usize> {
    console::rows()
}

fn stdin_is_console() -> bool {
    console::input_is_console()
}

fn read_key() -> Option<char> {
    console::read_key()
}

fn main() -> Result<()> {
    // без аргументов или с -h/--help — компактная справка с постраничной
    // прокруткой, если она не влезает в экран
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print_paged(&Args::command().render_help().to_string());
        return Ok(());
    }

    stats::init();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn,onnxruntime=warn".into()),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    if args.files.is_empty() {
        anyhow::bail!("укажите хотя бы один входной файл");
    }
    if args.task == "translate" && args.model != ModelKind::WhisperTurbo {
        anyhow::bail!("translate доступен только для whisper-turbo");
    }
    let translate = args.task == "translate";
    if (args.w2v || args.dtw) && args.model != ModelKind::WhisperTurbo {
        tracing::warn!("ключи --w2v/--dtw применяются только к whisper-turbo — игнорирую");
    }
    if args.w2v && args.language != "auto" && args.language != "ru" {
        tracing::warn!(
            "модель wav2vec2-выравнивания русская (--language {}): тайминги могут быть плохими",
            args.language
        );
    }

    let paths = models::resolve(args.models_dir.as_deref())?;
    let use_cuda_requested = match args.device.as_str() {
        "cuda" | "gpu" => true,
        "cpu" => false,
        other => anyhow::bail!("неизвестное устройство: {other} (cuda|cpu)"),
    };

    std::fs::create_dir_all(&args.output_dir)?;
    let formats = output::parse_formats(&args.output_format);

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    // --- build worker pool once (loads in parallel, CUDA -> CPU fallback) ---
    let _load_span = stats::span("загрузка ASR-модели", None);
    let mut use_cuda = use_cuda_requested;
    let workers = if args.workers > 0 {
        args.workers
    } else {
        auto_workers(use_cuda, args.model)
    };
    let intra = (cores / workers.max(1)).clamp(1, 8);

    let build_all = |use_cuda: bool| -> Vec<std::thread::JoinHandle<Result<Worker>>> {
        (0..workers)
            .map(|_| {
                let paths = paths.clone();
                let model = args.model;
                std::thread::spawn(move || try_build_worker(&paths, model, use_cuda, intra, args.dtw))
            })
            .collect()
    };
    let join_all = |handles: Vec<std::thread::JoinHandle<Result<Worker>>>,
                        out: &mut Vec<Worker>|
     -> Result<bool> {
        let mut ok = true;
        for h in handles {
            match h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("panic while loading model"))) {
                Ok(w) => out.push(w),
                Err(e) => {
                    ok = false;
                    tracing::warn!("ошибка загрузки воркера: {e:#}");
                }
            }
        }
        Ok(ok)
    };

    let mut workers_vec: Vec<Worker> = Vec::new();
    if use_cuda {
        let handles = build_all(true);
        if !join_all(handles, &mut workers_vec)? {
            workers_vec.clear();
        }
        // validate CUDA EP really executes (warm up the first worker)
        if !workers_vec.is_empty() {
            if let Some(w) = workers_vec.first_mut() {
                if let Err(e) = w.warmup() {
                    tracing::warn!("CUDA недоступна ({e:#}); переключаю всех воркеров на CPU");
                    workers_vec.clear();
                }
            }
        }
    }
    if workers_vec.is_empty() {
        if use_cuda_requested {
            tracing::warn!("переключаю всех воркеров на CPU");
        }
        use_cuda = false;
        let handles = build_all(false);
        join_all(handles, &mut workers_vec)?;
    }
    if workers_vec.is_empty() {
        anyhow::bail!("не удалось создать ни одного воркера");
    }
    tracing::info!(
        "{} воркеров ({}), intra-потоков на воркер: {intra}",
        workers_vec.len(),
        if use_cuda { "CUDA" } else { "CPU" },
    );
    drop(_load_span);

    // --- optional speaker diarization (Nemotron-3-Diarization) ---
    let diarizer: Option<Arc<Mutex<diarize::NemotronDiarizer>>> = if args.diarize {
        let _g = stats::span("загрузка модели диаризации", None);
        let d = diarize::NemotronDiarizer::load(&paths.nemotron_diar, use_cuda, intra)
            .context("загрузка модели диаризации (models/nemotron3-diarization)")?;
        tracing::info!(
            "модель диаризации загружена ({})",
            if use_cuda { "CUDA" } else { "CPU" }
        );
        Some(Arc::new(Mutex::new(d)))
    } else {
        None
    };
    let dump_diar = std::env::var("LIGHTTRANS_DUMP_DIAR").ok().map(PathBuf::from);
    // отладка: дамп очищенного --denoise потока 16 кГц (f32le) для проверки выравнивания
    let dump_denoised = std::env::var("LIGHTTRANS_DUMP_DENOISED").ok().map(PathBuf::from);

    // --- optional word alignment (whisper without DTW + cheap CTC aligner) ---
    // The aligner is NOT loaded here: it would compete with the whisper
    // workers for the GPU the whole run (measured ~4x slowdown of the
    // alignment phase). Instead transcription finishes first, the workers are
    // dropped, and alignment then runs on the idle GPU (see phase 2 below).
    let is_whisper = args.model == ModelKind::WhisperTurbo;
    let w2v_active = args.w2v && is_whisper;
    if w2v_active && !paths.gigaam_model.exists() {
        anyhow::bail!(
            "--w2v: не найдена модель выравнивания (нужна models/gigaam-v3/v3_e2e_ctc.onnx): {}",
            paths.gigaam_model.display()
        );
    }

    // --- channels shared across all input files ---
    let (chunk_tx, chunk_rx) = mpsc::channel::<segments::Chunk>();
    // result: (file, chunk, segments, chunk samples + absolute start — the
    // samples ride along for the deferred alignment phase and are dropped
    // immediately when --w2v is off)
    let (res_tx, res_rx) = mpsc::channel::<(
        usize,
        usize,
        anyhow::Result<(Vec<Segment>, Option<String>)>,
        Vec<f32>,
        f64,
    )>();
    let chunk_rx = Arc::new(Mutex::new(chunk_rx));

    let nthr = if use_cuda { 4 } else { cores };
    let lang_arg = args.language.clone();
    let beam = args.beam_size;
    let file_names: Arc<Vec<String>> = Arc::new(
        args.files
            .iter()
            .map(|f| f.display().to_string())
            .collect(),
    );
    for mut w in workers_vec {
        let rx = Arc::clone(&chunk_rx);
        let tx = res_tx.clone();
        let lang_arg = lang_arg.clone();
        let file_names = Arc::clone(&file_names);
        std::thread::spawn(move || loop {
            let chunk = { rx.lock().unwrap().recv() };
            let Ok(chunk) = chunk else { break };
            let t_inf = Instant::now();
            let res = w.transcribe(&chunk.samples, chunk.start, &lang_arg, translate, beam, nthr);
            stats::add("инференс", &file_names[chunk.file_idx], t_inf.elapsed().as_secs_f64());
            let _ = tx.send((chunk.file_idx, chunk.idx, res, chunk.samples, chunk.start));
        });
    }
    drop(res_tx); // res_rx закроется, когда все воркеры завершатся

    let n_files = args.files.len();
    let diar_results: Arc<Mutex<Vec<Option<anyhow::Result<diarize::Diarization>>>>> =
        Arc::new(Mutex::new((0..n_files).map(|_| None).collect()));
    let mut transcripts: Vec<Transcript> = args
        .files
        .iter()
        .map(|_| Transcript { language: String::new(), segments: Vec::new() })
        .collect();
    if args.model == ModelKind::GigaamV3 || args.model == ModelKind::GigaamV3Rnnt {
        for t in &mut transcripts {
            t.language = "ru".into();
        }
    } else if args.model == ModelKind::ParakeetV3 && args.language == "auto" {
        for t in &mut transcripts {
            t.language = "multi".into();
        }
    } else if args.language != "auto" {
        for t in &mut transcripts {
            t.language = args.language.clone();
        }
    }

    // --- producer: decode -> [denoise] -> VAD -> chunker, sequentially over input files ---
    let producer = {
        let chunk_tx = chunk_tx;
        let files = args.files.clone();
        let vad_path = paths.silero_vad.clone();
        let no_vad = args.no_vad;
        let denoise = args.denoise;
        let denoise_atten_lim = args.denoise_atten_lim;
        let denoise_threads = args.denoise_threads;
        let vad_onset = args.vad_onset;
        let vad_offset = args.vad_offset;
        let chunk_size = args.chunk_size;
        let diarizer = diarizer.clone();
        let diar_results = Arc::clone(&diar_results);
        std::thread::spawn(move || -> Result<()> {
            let mut diar_handles = Vec::new();
            // DeepFilterNet 3 работает в 48 кГц — при включённом шумоподавлении
            // декодируем сразу в него; декодер питает пул шумоподавления
            // прямо из колбэка, готовые полосы идут в VAD/чанкер по порядку
            let mut df_pool = if denoise {
                Some(df::StripPool::new(denoise_atten_lim, denoise_threads))
            } else {
                None
            };
            let target_sr: u32 = if denoise { df::MODEL_SR } else { audio::TARGET_SR };
            let mut dump_denoised_file = dump_denoised.as_ref().and_then(|p| {
                std::fs::File::create(p)
                    .map_err(|e| tracing::warn!("не открыть LIGHTTRANS_DUMP_DENOISED: {e}"))
                    .ok()
            });
            for (file_idx, file) in files.iter().enumerate() {
                let mut vad = if no_vad {
                    None
                } else {
                    Some(
                        vad::SileroVad::load(&vad_path, 2)?
                            .with_thresholds(vad_onset, vad_offset),
                    )
                };
                let mut chunker = Chunker::new(file_idx, chunk_size, no_vad);
                let mut intervals: Vec<vad::VadSegment> = Vec::new();
                let mut collected: Vec<f32> = Vec::new();
                let file_disp = file.display().to_string();
                let _dec_span = stats::span("декодирование + VAD", Some(&file_disp));
                let t_dec = Instant::now();
                let mut feed_block = |block: &[f32]| -> Result<()> {
                    if diarizer.is_some() {
                        collected.extend_from_slice(block);
                    }
                    intervals.clear();
                    if let Some(v) = vad.as_mut() {
                        v.feed(block, &mut intervals).context("VAD inference")?;
                    }
                    let secs: Vec<(f64, f64)> =
                        intervals.iter().map(|s| (s.start, s.end)).collect();
                    let mut ready = Vec::new();
                    chunker.feed(block, &secs, &mut ready);
                    for c in ready {
                        let _ = chunk_tx.send(c);
                    }
                    Ok(())
                };
                // дамп очищенного потока + передача в VAD/чанкер
                let mut emit16 = |out: &[f32]| -> Result<()> {
                    if let Some(f) = dump_denoised_file.as_mut() {
                        use std::io::Write;
                        for s in out {
                            let _ = f.write_all(&s.to_le_bytes());
                        }
                    }
                    feed_block(out)
                };
                let dur =
                    audio::decode_streaming(file, target_sr, target_sr as usize, |block| {
                        match df_pool.as_mut() {
                            Some(pool) => pool.feed(block, &mut emit16),
                            None => emit16(block),
                        }
                    })
                    .with_context(|| format!("декодирование {}", file.display()))?;
                if let Some(pool) = df_pool.as_mut() {
                    pool.finish(&mut emit16)?;
                }
                stats::set_audio(&file_disp, dur);
                tracing::info!(
                    "декод завершён [{file_idx}]: {:.1}с аудио за {:.2}с (RTF декода {:.3})",
                    dur,
                    t_dec.elapsed().as_secs_f64(),
                    if dur > 0.0 { t_dec.elapsed().as_secs_f64() / dur } else { 0.0 }
                );
                intervals.clear();
                if let Some(v) = vad.as_mut() {
                    v.finish(&mut intervals);
                }
                let secs: Vec<(f64, f64)> = intervals.iter().map(|s| (s.start, s.end)).collect();
                let mut ready = Vec::new();
                chunker.feed_intervals(&secs, &mut ready);
                chunker.finish(&mut ready);
                for c in ready {
                    let _ = chunk_tx.send(c);
                }
                tracing::info!("файл [{file_idx}] отправлен в очередь");

                if let Some(dz) = diarizer.as_ref() {
                    let dz = Arc::clone(dz);
                    let diar_results = Arc::clone(&diar_results);
                    let file_name = file.display().to_string();
                    let dump = dump_diar.clone();
                    diar_handles.push(std::thread::spawn(move || {
                        let mut d = dz.lock().unwrap();
                        let _g = stats::span("диаризация", Some(&file_name));
                        let t0 = Instant::now();
                        let res = d.diarize(&collected, dump.as_deref());
                        tracing::info!(
                            "диаризация [{file_name}]: {:.2}с аудио за {:.2}с",
                            collected.len() as f64 / 16_000.0,
                            t0.elapsed().as_secs_f64()
                        );
                        diar_results.lock().unwrap()[file_idx] = Some(res);
                    }));
                }
            }
            for h in diar_handles {
                if let Err(e) = h.join() {
                    tracing::error!("поток диаризации паниковал: {e:?}");
                }
            }
            Ok(())
        })
    };
    // chunk_tx переехал в producer: чанк-канал закроется, когда producer
    // закончит, и воркеры завершатся

    // --- collect results ---
    let _collect_span = stats::span("сборка результатов", None);
    let mut pending: BTreeMap<(usize, usize), (Vec<f32>, f64, Vec<Segment>)> = BTreeMap::new();
    let mut received = 0usize;
    let mut first_error: Option<anyhow::Error> = None;
    for (file_idx, idx, res, samples, start) in res_rx {
        received += 1;
        match res {
            Ok((segs, lang)) => {
                let t = &mut transcripts[file_idx];
                if t.language.is_empty() {
                    if let Some(l) = lang {
                        t.language = l;
                    }
                }
                pending.insert((file_idx, idx), (samples, start, segs));
                if received % 20 == 0 {
                    tracing::info!("обработано чанков: {received}");
                }
            }
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    producer
        .join()
        .map_err(|_| anyhow::anyhow!("паника в потоке декодирования"))??;

    if let Some(e) = first_error {
        return Err(e.context("ошибка инференса в одном из чанков"));
    }
    drop(_collect_span);

    if w2v_active {
        // --- phase 2: word alignment on the now-idle GPU ---
        // All whisper workers have exited by now (their CUDA context is
        // freed), so the aligner sessions get the whole GPU: measured ~4x
        // faster than aligning inline in the worker threads. Two sessions
        // overlap the GPU encoder with the CPU mel/DP work.
        let n_al = if use_cuda { 2 } else { 1 };
        let aligners: Vec<Arc<Mutex<align::WordAligner>>> = {
            let _g = stats::span("загрузка модели выравнивания", None);
            let mut v = Vec::new();
            for _ in 0..n_al {
                match align::WordAligner::load(&paths.gigaam_model, &paths.gigaam_vocab, use_cuda, intra)
                    .or_else(|e| {
                        if use_cuda {
                            tracing::warn!("выравнивание на CUDA недоступно ({e:#}); пробую CPU");
                            align::WordAligner::load(
                                &paths.gigaam_model,
                                &paths.gigaam_vocab,
                                false,
                                intra,
                            )
                        } else {
                            Err(e)
                        }
                    }) {
                    Ok(a) => v.push(Arc::new(Mutex::new(a))),
                    Err(e) => {
                        if v.is_empty() {
                            anyhow::bail!("--w2v: не удалось загрузить выравнивщик: {e:#}");
                        }
                        tracing::warn!("второй выравнивщик не загрузился, работаю в один поток: {e:#}");
                    }
                }
            }
            v
        };
        let _align_span = stats::span("выравнивание (w2v)", None);
        let (al_tx, al_rx) = mpsc::channel::<(usize, usize, Vec<Segment>)>();
        let al_rx = Arc::new(Mutex::new(al_rx));
        let queue: Arc<Mutex<std::collections::VecDeque<((usize, usize), (Vec<f32>, f64, Vec<Segment>))>>> =
            Arc::new(Mutex::new(pending.into_iter().collect()));
        for al in aligners {
            let queue = Arc::clone(&queue);
            let tx = al_tx.clone();
            let file_names = Arc::clone(&file_names);
            std::thread::spawn(move || loop {
                let Some(item) = queue.lock().unwrap().pop_front() else { break };
                let ((fi, idx), (samples, start, mut segs)) = item;
                let t_al = Instant::now();
                if let Err(e) = al.lock().unwrap().align_chunk(&samples, start, &mut segs) {
                    tracing::warn!("выравнивание w2v не удалось: {e:#}");
                }
                stats::add("выравнивание w2v", &file_names[fi], t_al.elapsed().as_secs_f64());
                drop(samples);
                let _ = tx.send((fi, idx, segs));
            });
        }
        drop(al_tx);
        let mut aligned: BTreeMap<(usize, usize), Vec<Segment>> = BTreeMap::new();
        {
            let rx = al_rx.lock().unwrap();
            while let Ok((fi, idx, segs)) = rx.recv() {
                aligned.insert((fi, idx), segs);
            }
        }
        for ((fi, _), segs) in aligned {
            transcripts[fi].segments.extend(segs);
        }
    } else {
        for ((fi, _), (_, _, segs)) in pending {
            transcripts[fi].segments.extend(segs);
        }
    }

    // --- attach speakers from diarization ---
    let diar_slots: Vec<Option<anyhow::Result<diarize::Diarization>>> =
        std::mem::take(&mut *diar_results.lock().unwrap());
    for (file_idx, slot) in diar_slots.into_iter().enumerate() {
        let d = match slot {
            None => continue, // diarization disabled
            Some(Ok(d)) => d,
            Some(Err(e)) => return Err(e.context("ошибка диаризации")),
        };
        let used: std::collections::HashSet<usize> =
            d.segments(0.5).iter().map(|s| s.speaker).collect();
        tracing::info!(
            "[{}] диаризация: {} спикеров",
            file_idx,
            used.len()
        );
        let file_disp = args
            .files
            .get(file_idx)
            .map(|f| f.display().to_string())
            .unwrap_or_default();
        let _attach_span = stats::span("привязка спикеров", Some(&file_disp));
        let t = &mut transcripts[file_idx];
        for seg in &mut t.segments {
            for w in &mut seg.words {
                if let Some(spk) = d.speaker_for_span(w.start, w.end) {
                    w.speaker = Some(format!("SPEAKER_{spk:02}"));
                }
            }
        }
        // one subtitle cue per speaker run; tiny runs merge into their neighbour
        t.segments = rebuild_segments_by_speaker(std::mem::take(&mut t.segments));
    }

    // --- outputs ---
    for (file, t) in args.files.iter().zip(&mut transcripts) {
        if t.language.is_empty() {
            t.language = "auto".into();
        }
        let file_disp = file.display().to_string();
        let _write_span = stats::span("запись результатов", Some(&file_disp));
        let stem = args.output_dir.join(
            file.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
        );
        let written = output::write_outputs(t, &stem, &formats)?;
        println!(
            "saved: {}",
            written.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        );
        if !args.quiet {
            println!("--- {} ---\n{}", file.display(), t.text());
        }
    }
    tracing::info!("всего чанков обработано: {received}");
    println!("{}", stats::report());

    Ok(())
}


/// Split segments at speaker changes and merge tiny runs, so subtitle cues
/// align with who is speaking (segments can span 30 s chunks otherwise).
fn rebuild_segments_by_speaker(segs: Vec<Segment>) -> Vec<Segment> {
    fn make_seg(words: Vec<Word>, speaker: Option<String>) -> Segment {
        let start = words[0].start;
        let end = words.last().unwrap().end;
        // words from some backends carry no spacing — rebuild with spacing rules
        let text = join_word_texts(&words);
        Segment { start, end, text, words, speaker }
    }

    // 1. split into per-speaker runs
    let mut runs: Vec<Segment> = Vec::new();
    for seg in segs {
        if seg.words.is_empty() {
            runs.push(seg);
            continue;
        }
        let mut cur: Vec<Word> = Vec::new();
        let mut cur_spk: Option<String> = None;
        for w in seg.words {
            if cur_spk.is_some() && w.speaker.is_some() && w.speaker != cur_spk {
                runs.push(make_seg(std::mem::take(&mut cur), cur_spk.take()));
            }
            if cur_spk.is_none() {
                cur_spk = w.speaker.clone();
            }
            cur.push(w);
        }
        if !cur.is_empty() {
            runs.push(make_seg(cur, cur_spk));
        }
    }

    // 2. merge tiny runs into the previous cue (spurious single-word flips)
    let mut out: Vec<Segment> = Vec::new();
    for r in runs {
        let tiny = r.words.len() < 3 || (r.end - r.start) < 0.6;
        if tiny {
            if let Some(prev) = out.last_mut() {
                if prev.speaker.is_some() && prev.speaker == r.speaker {
                    prev.end = r.end;
                    prev.words.extend(r.words);
                    prev.text = join_word_texts(&prev.words);
                    continue;
                }
            }
        }
        out.push(r);
    }
    out
}

fn try_build_worker(
    paths: &models::ModelPaths,
    model: ModelKind,
    use_cuda: bool,
    intra: usize,
    dtw: bool,
) -> Result<Worker> {
    match model {
        ModelKind::WhisperTurbo => {
            let w = whisper::WhisperTurbo::load(&paths.whisper_turbo, use_cuda, dtw)?;
            let state = w.create_state()?;
            Ok(Worker::Whisper(w, state))
        }
        ModelKind::GigaamV3 => {
            let g = gigaam::GigaAmV3::load(&paths.gigaam_model, &paths.gigaam_vocab, use_cuda, intra)?;
            Ok(Worker::Gigaam(g))
        }
        ModelKind::GigaamV3Rnnt => {
            let g = gigaam_rnnt::GigaAmRnnt::load(&paths.gigaam_rnnt_dir, use_cuda, intra)?;
            Ok(Worker::GigaamRnnt(g))
        }
        ModelKind::ParakeetV3 => {
            let p = parakeet::ParakeetV3::load(&paths.parakeet_dir, use_cuda, intra)?;
            Ok(Worker::Parakeet(p))
        }
    }
}

/// Auto worker count: CUDA — from free VRAM (NVML) and model size;
/// CPU — a single worker that uses all cores via intra-op threads.
fn auto_workers(use_cuda: bool, model: ModelKind) -> usize {
    if !use_cuda {
        return 1;
    }
    let free = nvml_wrapper::Nvml::init().ok().and_then(|nvml| {
        let dev = nvml.device_by_index(0).ok()?;
        let m = dev.memory_info().ok()?;
        Some(m.free)
    });
    let per_model: f64 = match model {
        ModelKind::WhisperTurbo => 2.2e9,
        // 1.5e9, not 1.3e9: 4 GigaAM sessions OOM on a 6 GB card (measured on
        // Linux with ~6.4 GB free); 3 is the documented optimum for 6 GB.
        ModelKind::GigaamV3 | ModelKind::GigaamV3Rnnt => 1.5e9,
        ModelKind::ParakeetV3 => 2.8e9,
    };
    match free {
        Some(free) => (((free as f64 - 0.8e9) / per_model).floor() as usize).clamp(1, 4),
        None => 2,
    }
}
