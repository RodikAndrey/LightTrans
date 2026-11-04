//! lighttrans — fast GPU audio/video transcription (WhisperX-style pipeline).
//!
//! Streaming pipeline: ffmpeg decode (any format) -> Silero VAD (streaming,
//! batched) -> chunker -> N worker threads (GPU inference) -> ordered merge ->
//! output writers. Worker count is auto-detected from free VRAM (CUDA) or CPU
//! cores, or set explicitly with --workers.
//!
//! Общая логика конвейера живёт в библиотечных модулях (`pipeline.rs`,
//! `worker.rs` и др.): их использует и сервер (`lighttrans-server`).

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use lighttrans::channels::ChannelWords;
use lighttrans::pipeline::{
    CancelToken, ChannelMode, ChunkOutcome, MultiStream, QueueItem, WorkerPool,
};
use lighttrans::segments::Chunk;
use lighttrans::transcript::{join_word_texts, Segment, Transcript, Word};
use lighttrans::worker::{auto_workers, try_build_worker, ModelKind, Worker};
use lighttrans::{align, audio, channels, df, diarize, download, models, output, pipeline, stats, transcript, vad};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
  lighttrans rec.mp3 --diarize --denoise      спикеры + шумоподавление
  lighttrans conf.wav --split-channels --turns  каждый канал — спикер, диалог
  lighttrans vad meeting.mp4 -o intervals.json   интервалы речи (Silero), без распознавания
  lighttrans bench video.mp4 -m gigaam-v3        этапы и RTF, машиночитаемый json"
)]
struct Args {
    /// vad — интервалы речи Silero (json); bench — бенчмарк этапов и RTF;
    /// без подкоманды — транскрибация
    #[command(subcommand)]
    command: Option<Cmd>,

    #[command(flatten)]
    tr: TranscribeArgs,
}

/// Флаги транскрибации: тело запуска без подкоманды и тело `bench`.
#[derive(clap::Args, Clone)]
struct TranscribeArgs {
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
    #[arg(long, conflicts_with_all = ["split_channels", "turns"])]
    diarize: bool,

    /// каждый канал входа — отдельный спикер (многоканальный диалог)
    #[arg(long, conflicts_with = "denoise")]
    split_channels: bool,

    /// реплики диалога по каналам (подразумевает --split-channels)
    #[arg(long)]
    turns: bool,

    /// пауза, режущая реплику в канале, с (--turns)
    #[arg(long, default_value_t = 1.0)]
    turn_gap: f64,

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

    /// чанков на один прогон CTC-GigaAM (1 = почанково; whisper/RNN-T/
    /// parakeet игнорируют)
    #[arg(long, default_value_t = 4)]
    batch: usize,

    /// отключить автозагрузку недостающих моделей из сети
    #[arg(long)]
    no_download_models: bool,

    /// показать справку и выйти
    #[arg(short = 'h', long)]
    help: bool,
}

#[derive(clap::Subcommand)]
enum Cmd {
    /// интервалы речи Silero VAD без распознавания, json на выход
    Vad {
        /// Входные аудио/видео файлы
        files: Vec<PathBuf>,

        /// выход: файл (при одном входе) или каталог; по умолчанию <вход>.vad.json
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// порог начала речи
        #[arg(long, default_value_t = 0.5)]
        vad_onset: f32,

        /// порог окончания речи
        #[arg(long, default_value_t = 0.363)]
        vad_offset: f32,

        /// каталог с моделями
        #[arg(long)]
        models_dir: Option<PathBuf>,

        /// отключить автозагрузку недостающих моделей из сети
        #[arg(long)]
        no_download_models: bool,
    },
    /// прогон транскрибации с машиночитаемой сводкой этапов и RTF
    /// (json в stdout, таблица — в stderr, текст не печатается)
    Bench {
        #[command(flatten)]
        tr: TranscribeArgs,
    },
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

/// Текст диалога: `[channel N] текст` на каждую реплику (как CLI источника).
fn dialog_text(turns: &[transcript::Turn]) -> String {
    turns
        .iter()
        .map(|t| format!("[channel {}] {}", t.channel, t.text))
        .collect::<Vec<_>>()
        .join("\n")
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
    let cli = Args::parse();
    // bench печатает в stdout машиночитаемый json — все логи уходят в stderr
    let bench = matches!(cli.command, Some(Cmd::Bench { .. }));
    init_tracing(bench);

    match cli.command {
        Some(Cmd::Vad { files, output, vad_onset, vad_offset, models_dir, no_download_models }) => {
            return run_vad(
                &files,
                output.as_deref(),
                vad_onset,
                vad_offset,
                models_dir.as_deref(),
                no_download_models,
            );
        }
        Some(Cmd::Bench { tr }) => {
            // bench — это транскрибация без печати текста и с json-сводкой
            let mut tr = tr;
            tr.quiet = true;
            return run_transcribe(tr, true);
        }
        None => {}
    }
    run_transcribe(cli.tr, false)
}

/// Инициализация tracing: `stderr = true` — все логи в stderr (bench держит
/// stdout чистым под json).
fn init_tracing(stderr: bool) {
    let sub = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn,onnxruntime=warn".into()),
        )
        .with_target(false);
    if stderr {
        sub.with_writer(std::io::stderr).init();
    } else {
        sub.with_writer(std::io::stdout).init();
    }
}

/// Подкоманда `vad` (Фаза 5 PORTING.md): декод → Silero VAD → интервалы речи
/// в json. Распознавание не запускается, ASR-модели не загружаются.
fn run_vad(
    files: &[PathBuf],
    output: Option<&std::path::Path>,
    vad_onset: f32,
    vad_offset: f32,
    models_dir: Option<&std::path::Path>,
    no_download_models: bool,
) -> Result<()> {
    if files.is_empty() {
        anyhow::bail!("укажите хотя бы один входной файл");
    }
    download::ensure(
        &models::base_dir(models_dir),
        &[download::Family::SileroVad],
        !no_download_models,
    )
    .with_context(|| "автозагрузка моделей")?;
    let paths = models::resolve(models_dir)?;
    let many = files.len() > 1;
    for file in files {
        let out_path = match output {
            Some(p) if many => p.join(format!(
                "{}.vad.json",
                file.file_stem().and_then(|s| s.to_str()).unwrap_or("audio")
            )),
            Some(p) => p.to_path_buf(),
            None => file.with_extension("vad.json"),
        };
        let file_disp = file.display().to_string();
        let mut vad = pipeline::ChunkSource::load_vad(&paths.silero_vad, vad_onset, vad_offset)?;
        let mut intervals: Vec<vad::VadSegment> = Vec::new();
        let duration = audio::decode_streaming(file, audio::TARGET_SR, 16_000, |block| {
            vad.feed(block, &mut intervals)
        })
        .with_context(|| format!("декодирование {file_disp}"))?;
        vad.finish(&mut intervals);
        let doc = serde_json::json!({
            "file": file_disp,
            "duration_sec": (duration * 1000.0).round() / 1000.0,
            "sample_rate": audio::TARGET_SR,
            "vad": { "onset": round4(vad_onset), "offset": round4(vad_offset) },
            "intervals": intervals
                .iter()
                .map(|i| serde_json::json!({
                    "start": (i.start * 1000.0).round() / 1000.0,
                    "end": (i.end * 1000.0).round() / 1000.0,
                }))
                .collect::<Vec<_>>(),
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&doc)? + "\n")
            .with_context(|| format!("запись {}", out_path.display()))?;
        println!("saved: {} ({} интервалов)", out_path.display(), intervals.len());
    }
    Ok(())
}

fn round4(x: f32) -> f64 {
    (f64::from(x) * 10_000.0).round() / 10_000.0
}

/// Весь конвейер транскрибации (общее тело запуска без подкоманды и `bench`).
/// В bench-режиме текст не печатается (quiet уже выставлен), человекочитаемая
/// сводка уходит в stderr, а в stdout — стабильный json этапов и RTF.
fn run_transcribe(args: TranscribeArgs, bench: bool) -> Result<()> {
    if args.files.is_empty() {
        anyhow::bail!("укажите хотя бы один входной файл");
    }
    if args.task == "translate" && args.model != ModelKind::WhisperTurbo {
        anyhow::bail!("translate доступен только для whisper-turbo");
    }
    let translate = args.task == "translate";
    // --turns подразумевает раздельные каналы
    let split_active = args.split_channels || args.turns;
    channels::validate_turn_gap(args.turn_gap)?;
    if (args.w2v || args.dtw) && args.model != ModelKind::WhisperTurbo {
        tracing::warn!("ключи --w2v/--dtw применяются только к whisper-turbo — игнорирую");
    }
    if args.w2v && args.language != "auto" && args.language != "ru" {
        tracing::warn!(
            "модель wav2vec2-выравнивания русская (--language {}): тайминги могут быть плохими",
            args.language
        );
    }

    // автозагрузка до resolve: манифесты проверяются строго, файлы должны
    // лежать на месте до верификации
    let mut need = vec![
        download::Family::SileroVad,
        download::Family::from(args.model),
    ];
    if args.diarize {
        need.push(download::Family::NemotronDiar);
    }
    if args.w2v && args.model == ModelKind::WhisperTurbo {
        // выравниватель --w2v — это GigaAM CTC (см. align.rs)
        need.push(download::Family::GigaamV3Ctc);
    }
    download::ensure(
        &models::base_dir(args.models_dir.as_deref()),
        &need,
        !args.no_download_models,
    )
    .with_context(|| "автозагрузка моделей")?;
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
                std::thread::spawn(move || {
                    try_build_worker(&paths, model, use_cuda, intra, args.dtw, None)
                })
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

    // --- worker pool: общая очередь чанков, ответы развозятся по чанкам ---
    let pool = Arc::new(WorkerPool::start(
        workers_vec,
        pipeline::WorkerOpts {
            language: args.language.clone(),
            translate,
            beam_size: args.beam_size,
            n_threads: if use_cuda { 4 } else { cores },
            batch: args.batch,
            metrics: None,
        },
    ));
    let (reply_tx, reply_rx) = mpsc::channel::<ChunkOutcome>();
    let cancel = CancelToken::new();

    let n_files = args.files.len();
    let diar_results: Arc<Mutex<Vec<Option<anyhow::Result<diarize::Diarization>>>>> =
        Arc::new(Mutex::new((0..n_files).map(|_| None).collect()));
    let mut transcripts: Vec<Transcript> = (0..n_files)
        .map(|_| Transcript::default())
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
        let pool = Arc::clone(&pool);
        let reply_tx = reply_tx.clone();
        let cancel = cancel.clone();
        let files = args.files.clone();
        let vad_path = paths.silero_vad.clone();
        let no_vad = args.no_vad;
        let split_active = split_active;
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
                let file_disp = file.display().to_string();
                let _dec_span = stats::span("декодирование + VAD", Some(&file_disp));
                let t_dec = Instant::now();
                let mut ready: Vec<Chunk> = Vec::new();
                // отправить готовый чанк в пул воркеров
                let submit = |c: Chunk| {
                    let ok = pool.submit(QueueItem {
                        chunk: c,
                        cancel: cancel.clone(),
                        reply: reply_tx.clone(),
                    });
                    if !ok {
                        anyhow::bail!("пул воркеров закрыт (воркеры погибли)");
                    }
                    Ok(())
                };
                let dur: f64 = if split_active {
                    // --- многоканальный путь: декод с сохранением каналов ---
                    let mut dec = audio::Decoder::open_path(
                        file,
                        audio::TARGET_SR,
                        0, // сохранить все каналы
                        pipeline::SPLIT_BLOCK_FRAMES,
                    )
                    .with_context(|| format!("декодирование {}", file.display()))?;
                    let nch = dec.channels();
                    let mut ms = MultiStream::new(
                        ChannelMode::Split,
                        file_idx,
                        nch,
                        file_disp.clone(),
                        chunk_size,
                        no_vad,
                        vad_path.clone(),
                        vad_onset,
                        vad_offset,
                    );
                    let frames = dec.decode(|block| {
                        ready.clear();
                        ms.feed(block, &mut ready)?;
                        for c in ready.drain(..) {
                            submit(c)?;
                        }
                        Ok(())
                    })?;
                    ready.clear();
                    ms.finish(&mut ready)?;
                    for c in ready.drain(..) {
                        submit(c)?;
                    }
                    let dur = frames as f64 / audio::TARGET_SR as f64;
                    stats::set_audio(&file_disp, dur);
                    if let Some(plan) = ms.plan() {
                        tracing::info!(
                            "каналы [{file_idx}]: исходных {}, выбрано {:?} (dual_mono={})",
                            plan.source_channels,
                            plan.selected,
                            plan.dual_mono
                        );
                    }
                    dur
                } else {
                    // --- обычный mono-путь (микширование в декодере) ---
                    let mut ms = MultiStream::new(
                        ChannelMode::Mix,
                        file_idx,
                        1,
                        file_disp.clone(),
                        chunk_size,
                        no_vad,
                        vad_path.clone(),
                        vad_onset,
                        vad_offset,
                    );
                    if diarizer.is_some() {
                        ms = ms.with_mono_collection();
                    }
                    let mut feed_block = |block: &[f32]| -> Result<()> {
                        ready.clear();
                        ms.feed(block, &mut ready)?;
                        for c in ready.drain(..) {
                            submit(c)?;
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
                    ready.clear();
                    ms.finish(&mut ready)?;
                    for c in ready.drain(..) {
                        submit(c)?;
                    }
                    stats::set_audio(&file_disp, dur);
                    if diarizer.is_some() {
                        let collected = ms.take_mono();
                        let dz = Arc::clone(diarizer.as_ref().unwrap());
                        let diar_results = Arc::clone(&diar_results);
                        let file_name = file_disp.clone();
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
                    dur
                };
                tracing::info!(
                    "декод завершён [{file_idx}]: {:.1}с аудио за {:.2}с (RTF декода {:.3})",
                    dur,
                    t_dec.elapsed().as_secs_f64(),
                    if dur > 0.0 { t_dec.elapsed().as_secs_f64() / dur } else { 0.0 }
                );
                tracing::info!("файл [{file_idx}] отправлен в очередь");
            }
            for h in diar_handles {
                if let Err(e) = h.join() {
                    tracing::error!("поток диаризации паниковал: {e:?}");
                }
            }
            Ok(())
        })
    };
    // основная копия reply_tx больше не нужна: канал закроется, когда
    // закончит producer и его клоны
    drop(reply_tx);

    // --- collect results, строго по порядку (file_idx, channel, idx) ---
    let _collect_span = stats::span("сборка результатов", None);
    let mut pending: BTreeMap<(usize, usize, usize), (Vec<f32>, f64, Vec<Segment>)> = BTreeMap::new();
    let mut received = 0usize;
    let mut first_error: Option<anyhow::Error> = None;
    for ChunkOutcome { chunk, result } in reply_rx {
        received += 1;
        match result {
            Ok((segs, lang)) => {
                let t = &mut transcripts[chunk.file_idx];
                if t.language.is_empty() {
                    if let Some(l) = lang {
                        t.language = l;
                    }
                }
                pending.insert(
                    (chunk.file_idx, chunk.channel, chunk.idx),
                    (chunk.samples, chunk.start, segs),
                );
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
        let file_names: Arc<Vec<String>> = Arc::new(
            args.files
                .iter()
                .map(|f| f.display().to_string())
                .collect(),
        );
        let (al_tx, al_rx) = mpsc::channel::<(usize, usize, usize, Vec<Segment>)>();
        let al_rx = Arc::new(Mutex::new(al_rx));
        let queue: Arc<Mutex<std::collections::VecDeque<((usize, usize, usize), (Vec<f32>, f64, Vec<Segment>))>>> =
            Arc::new(Mutex::new(pending.into_iter().collect()));
        for al in aligners {
            let queue = Arc::clone(&queue);
            let tx = al_tx.clone();
            let file_names = Arc::clone(&file_names);
            std::thread::spawn(move || loop {
                let Some(item) = queue.lock().unwrap().pop_front() else { break };
                let (key, (samples, start, mut segs)) = item;
                let (fi, ch, idx) = key;
                let t_al = Instant::now();
                if let Err(e) = al.lock().unwrap().align_chunk(&samples, start, &mut segs) {
                    tracing::warn!("выравнивание w2v не удалось: {e:#}");
                }
                stats::add(
                    "выравнивание w2v",
                    &format!("{}#ch{ch}", file_names[fi]),
                    t_al.elapsed().as_secs_f64(),
                );
                drop(samples);
                let _ = tx.send((fi, ch, idx, segs));
            });
        }
        drop(al_tx);
        let mut aligned: BTreeMap<(usize, usize, usize), Vec<Segment>> = BTreeMap::new();
        {
            let rx = al_rx.lock().unwrap();
            while let Ok((fi, ch, idx, segs)) = rx.recv() {
                aligned.insert((fi, ch, idx), segs);
            }
        }
        for ((fi, _, _), segs) in aligned {
            transcripts[fi].segments.extend(segs);
        }
    } else if split_active {
        // --- многоканальная сборка: каналы = спикеры + turns ---
        for fi in 0..n_files {
            // слова по каналам в порядке чанков
            let mut ch_words: BTreeMap<usize, Vec<Word>> = BTreeMap::new();
            for ((_f, ch, _), (_, _, segs)) in pending.range((fi, 0, 0)..(fi + 1, 0, 0)) {
                for seg in segs {
                    for w in &seg.words {
                        ch_words.entry(*ch).or_default().push(w.clone());
                    }
                }
            }
            let channels: Vec<ChannelWords> = ch_words
                .into_iter()
                .map(|(ch, mut words)| {
                    for w in &mut words {
                        w.channel = Some(ch);
                        w.speaker = Some(format!("SPEAKER_{ch:02}"));
                    }
                    ChannelWords { channel: ch, words }
                })
                .collect();
            let turns = channels::build_turns(&channels, args.turn_gap);
            let merged = channels::merge_words(&channels);
            let mut segs: Vec<Segment> = Vec::new();
            if !merged.is_empty() {
                let start = merged[0].start;
                let end = merged.last().unwrap().end;
                let one = Segment {
                    start,
                    end,
                    text: join_word_texts(&merged),
                    words: merged,
                    speaker: None,
                    channel: None,
                };
                segs = rebuild_segments_by_speaker(vec![one]);
                for s in &mut segs {
                    s.channel = s.words.first().and_then(|w| w.channel);
                }
            }
            transcripts[fi].segments = segs;
            transcripts[fi].turns = turns;
        }
        // pending больше не нужен
        pending.clear();
    } else {
        for ((fi, _, _), (_, _, segs)) in pending {
            transcripts[fi].segments.extend(segs);
        }
    }

    // --- attach speakers from diarization (только mono-путь) ---
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
        let saved = written.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ");
        if bench {
            eprintln!("saved: {saved}");
        } else {
            println!("saved: {saved}");
        }
        if !args.quiet {
            let text = if split_active && !t.turns.is_empty() {
                dialog_text(&t.turns)
            } else {
                t.text()
            };
            println!("--- {} ---\n{}", file.display(), text);
        }
    }
    tracing::info!("всего чанков обработано: {received}");
    if bench {
        eprint!("{}", stats::report());
        println!("{}", stats::report_json());
    } else {
        println!("{}", stats::report());
    }

    pool.close();
    Ok(())
}

/// Split segments at speaker changes and merge tiny runs, so subtitle cues
/// align with who is speaking (segments can span 30 s chunks otherwise).
fn rebuild_segments_by_speaker(segs: Vec<Segment>) -> Vec<Segment> {
    fn make_seg(words: Vec<Word>, speaker: Option<String>, channel: Option<usize>) -> Segment {
        let start = words[0].start;
        let end = words.last().unwrap().end;
        // words from some backends carry no spacing — rebuild with spacing rules
        let text = join_word_texts(&words);
        Segment { start, end, text, words, speaker, channel }
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
                let spk = cur_spk.take();
                let ch = cur.first().and_then(|x| x.channel);
                runs.push(make_seg(std::mem::take(&mut cur), spk, ch));
            }
            if cur_spk.is_none() {
                cur_spk = w.speaker.clone();
            }
            cur.push(w);
        }
        if !cur.is_empty() {
            let spk = cur_spk;
            let ch = cur.first().and_then(|x| x.channel);
            runs.push(make_seg(cur, spk, ch));
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
