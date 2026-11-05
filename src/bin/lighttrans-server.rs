//! lighttrans-server — HTTP-сервис batch-транскрибации (Фаза 1 PORTING.md).
//!
//! Тёплый стейт: модели загружаются один раз при старте, пул воркеров живёт
//! между запросами. SIGTERM/SIGINT → Draining: новые запросы получают 503,
//! активные дожидаются завершения.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use lighttrans::download;
use lighttrans::metrics::Metrics;
use lighttrans::models;
use lighttrans::ort_assignment::Policy;
use lighttrans::pipeline::WorkerPool;
use lighttrans::server::{self, AppState, ModelPools, ServerInner};
use lighttrans::stats;
use lighttrans::worker::{self, ModelKind, Worker};
use std::sync::Arc;
use std::time::Instant;

/// Провайдер исполнения ONNX-графов. Выбирается явно, fallback запрещён:
/// недоступный CUDA — отказ старта, а не тихая работа на CPU (Фаза 4).
/// tensorrt этой сборкой не поддерживается (ort без фичи tensorrt).
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Ep {
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(
    name = "lighttrans-server",
    version,
    about = "HTTP-сервис транскрибации: POST /v1/transcribe (GigaAM v3 CTC/RNN-T, Whisper Turbo, Parakeet v3)",
    after_help = "Модели --models: gigaam-v3-ctc (gc), gigaam-v3-rnnt (gr),\n\
                  whisper-large-v3-turbo (wt), parakeet-v3 (pr); CTC-пул обязателен.\n\
                  Лимиты окружением (неверное значение отказывает, не подменяется):\n\
                  ASR_MAX_CONCURRENCY=8 ASR_BODY_LIMIT_MB=64 ASR_REQ_TIMEOUT_SEC=120\n\
                  Верификация CUDA-назначений (Фаза 4; только с --ep cuda, роли CTC/RNNT):\n\
                  ASR_CUDA_ASSIGNMENT_POLICY=verified|allow-unverified\n\
                  ASR_CUDA_CTC_ASSIGNMENT_SHA256 ASR_CUDA_RNNT_ASSIGNMENT_SHA256"
)]
struct Args {
    /// порт HTTP-слушателя
    #[arg(long, default_value_t = 9898)]
    port: u16,

    /// каталог с моделями
    #[arg(long)]
    models_dir: Option<std::path::PathBuf>,

    /// провайдер исполнения: cpu | cuda (без fallback)
    #[arg(long, value_enum, default_value_t = Ep::Cuda)]
    ep: Ep,

    /// загружаемые модели через запятую: gc|gr|wt|pr и полные имена
    /// (см. --help); CTC-пул (gc) обязателен — его арендует WS-стрим
    #[arg(long, default_value = "gc,gr")]
    models: String,

    /// не загружать RNN-T (только CTC); сокращение для --models gc
    #[arg(long)]
    no_rnnt: bool,

    /// отключить автозагрузку недостающих моделей из сети
    #[arg(long)]
    no_download_models: bool,

    /// воркеров на модель (0 = авто по свободной VRAM)
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// чанков на один прогон CTC-GigaAM (1 = почанково; RNN-T игнорирует)
    #[arg(long, default_value_t = 4)]
    batch: usize,
}

fn main() -> Result<()> {
    stats::init();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn,onnxruntime=warn".into()),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    // состав пулов: алиасы как в CLI, дубликаты отбрасываются, CTC обязателен
    let mut wanted: Vec<ModelKind> = args
        .models
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            ModelKind::parse(s)
                .ok_or_else(|| anyhow::anyhow!("--models: неизвестная модель {s:?} (допустимо: gc|gr|wt|pr и полные имена)"))
        })
        .collect::<Result<_>>()?;
    wanted.dedup();
    if args.no_rnnt {
        wanted.retain(|k| *k != ModelKind::GigaamV3Rnnt);
    }
    if !wanted.contains(&ModelKind::GigaamV3) {
        anyhow::bail!("--models: CTC-пул (gc) обязателен — его арендует /v1/stream и требует health");
    }
    // конфигурация окружения проверяется до всякой работы с моделями:
    // кривое значение env отказывает, а не подменяется дефолтом
    let limits = server::config::ServerLimits::from_env()
        .map_err(|e| anyhow::anyhow!("конфигурация окружения: {e}"))?;
    let use_cuda = args.ep == Ep::Cuda;
    // политика верификации CUDA-назначений — тоже до загрузки моделей;
    // Arc: политики живут в потоках-воркерах ('static). Схема покрывает
    // только роли CTC/RNNT — whisper/parakeet вне неё
    let with_rnnt = wanted.contains(&ModelKind::GigaamV3Rnnt);
    let policy: Option<Arc<Policy>> = Policy::from_env(use_cuda, with_rnnt)
        .map_err(|e| anyhow::anyhow!("конфигурация окружения: {e}"))?
        .map(Arc::new);
    // автозагрузка до resolve: манифесты проверяются строго, файлы должны
    // лежать на месте до верификации
    let mut need = vec![download::Family::SileroVad];
    for kind in &wanted {
        need.push(download::Family::from(*kind));
    }
    download::ensure(
        &models::base_dir(args.models_dir.as_deref()),
        &need,
        !args.no_download_models,
    )
    .with_context(|| "автозагрузка моделей")?;
    let paths = models::resolve(args.models_dir.as_deref())?;

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    // --- загрузка моделей: пул на каждую выбранную модель ---
    let _load_span = stats::span("загрузка ASR-модели", None);
    let load_model = |kind: ModelKind, use_cuda: bool| -> Result<Vec<Worker>> {
        let count = if args.workers > 0 {
            args.workers
        } else {
            worker::auto_workers(use_cuda, kind)
        };
        let intra = (cores / count.max(1)).clamp(1, 8);
        let mut handles = Vec::new();
        for _ in 0..count {
            let paths = paths.clone();
            let policy = policy.clone();
            handles.push(std::thread::spawn(move || {
                worker::try_build_worker(&paths, kind, use_cuda, intra, false, policy.as_deref())
            }));
        }
        let mut out = Vec::new();
        for h in handles {
            match h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("panic while loading model"))) {
                Ok(w) => out.push(w),
                Err(e) => tracing::warn!("ошибка загрузки воркера: {e:#}"),
            }
        }
        if out.is_empty() {
            anyhow::bail!("не удалось загрузить ни одного воркера {kind:?}");
        }
        // проверка, что CUDA EP реально исполняет (прогрев первого воркера)
        let mut ok = Vec::with_capacity(out.len());
        for (i, mut w) in out.into_iter().enumerate() {
            if i == 0 {
                if let Err(e) = w.warmup() {
                    tracing::warn!("прогрев воркера не удался: {e:#}");
                    return Err(anyhow::anyhow!("warmup failed: {e:#}"));
                }
            }
            ok.push(w);
        }
        Ok(ok)
    };

    let metrics = Metrics::new_arc();
    metrics.set_max_transcribe(limits.max_concurrency);
    // Выбор провайдера явный и окончательный: при ошибке — отказ (Фаза 4),
    // fallback на CPU убран сознательно. Язык: gigaam — русский контракт
    // сервиса, whisper/parakeet — автоопределение.
    let build_pool = |kind: ModelKind, use_cuda: bool| -> Result<WorkerPool> {
        let opts = lighttrans::pipeline::WorkerOpts {
            language: if matches!(kind, ModelKind::GigaamV3 | ModelKind::GigaamV3Rnnt) {
                "ru".into()
            } else {
                "auto".into()
            },
            translate: false,
            beam_size: 1,
            n_threads: if use_cuda { 4 } else { cores },
            batch: args.batch,
            metrics: Some(Arc::clone(&metrics)),
        };
        let workers = load_model(kind, use_cuda)?;
        tracing::info!(
            "{}: {} воркеров ({})",
            kind.id(),
            workers.len(),
            if use_cuda { "CUDA" } else { "CPU" }
        );
        Ok(WorkerPool::start(workers, opts))
    };
    let build_optional = |kind: ModelKind| -> Result<Option<Arc<WorkerPool>>> {
        if wanted.contains(&kind) {
            Ok(Some(Arc::new(build_pool(kind, use_cuda)?)))
        } else {
            Ok(None)
        }
    };

    let ctc = Arc::new(build_pool(ModelKind::GigaamV3, use_cuda)?);
    let rnnt = build_optional(ModelKind::GigaamV3Rnnt)?;
    let whisper = build_optional(ModelKind::WhisperTurbo)?;
    let parakeet = build_optional(ModelKind::ParakeetV3)?;
    drop(_load_span);
    let pools = ModelPools { ctc, rnnt, whisper, parakeet };
    tracing::info!("воркеров всего: {}", pools.worker_count());

    let admission = server::admission::AdmissionState::new(limits.max_concurrency);
    let state = AppState::new(ServerInner {
        pools,
        admission,
        limits,
        paths,
        provider: if use_cuda { "cuda" } else { "cpu" },
        metrics,
    });

    // --- HTTP-слушатель + graceful shutdown ---
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    // при любом выходе не дропаем тёплый стейт: разрушение ORT/CUDA-сессий
    // в этой среде периодически падает (segfault в teardown при гонке
    // нескольких сессий на одном контексте) — состояние и так неживое,
    // поэтому завершаем процесс немедленно, воркеры не присоединяем
    match runtime.block_on(serve(state, args.port)) {
        Ok(()) => {
            tracing::info!("сервер остановлен");
            std::process::exit(0);
        }
        Err(e) => {
            tracing::error!("{e:#}");
            eprintln!("Error: {e:#}");
            std::process::exit(1);
        }
    }
}

async fn serve(state: AppState, port: u16) -> Result<()> {
    let app = server::router(state.clone());
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("bind 0.0.0.0:{port}"))?;
    tracing::info!(
        "lighttrans-server слушает 0.0.0.0:{port} (models: {}, лимит {}, тело {} МБ, тайм-аут {} c)",
        state.inner().pools.loaded_models().join(", "),
        state.inner().limits.max_concurrency,
        state.inner().limits.body_limit_bytes / (1024 * 1024),
        state.inner().limits.request_timeout.as_secs(),
    );
    let shutdown_state = state.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_state))
        .await
        .context("http server")?;
    Ok(())
}

/// SIGTERM/SIGINT → Draining (новые запросы 503, /readyz 503 — фаза 2),
/// активные запросы дожидаются завершения; обратного перехода нет.
async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("установить обработчик SIGINT");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("установить обработчик SIGTERM")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    let started = Instant::now();
    let began = state.inner().admission.begin_draining();
    if began {
        tracing::warn!(
            "SIGTERM/SIGINT: draining — новые запросы отклоняются (503), активные дожидаются"
        );
    }
    // активные batch-запросы держат соединения: axum дожидается их
    // терминации в with_graceful_shutdown
    while state.inner().admission.available_http() < state.inner().admission.max_http() {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let _ = started;
}
