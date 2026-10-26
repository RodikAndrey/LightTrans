//! lighttrans-server — HTTP-сервис batch-транскрибации (Фаза 1 PORTING.md).
//!
//! Тёплый стейт: модели загружаются один раз при старте, пул воркеров живёт
//! между запросами. SIGTERM/SIGINT → Draining: новые запросы получают 503,
//! активные дожидаются завершения.

use anyhow::{Context, Result};
use clap::Parser;
use lighttrans::metrics::Metrics;
use lighttrans::models;
use lighttrans::pipeline::WorkerPool;
use lighttrans::server::{self, AppState, ModelPools, ServerInner};
use lighttrans::stats;
use lighttrans::worker::{self, ModelKind, Worker};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "lighttrans-server",
    version,
    about = "HTTP-сервис транскрибации: POST /v1/transcribe (GigaAM v3 CTC/RNN-T)",
    after_help = "Лимиты окружением (неверное значение отказывает, не подменяется):\n\
                  ASR_MAX_CONCURRENCY=8 ASR_BODY_LIMIT_MB=64 ASR_REQ_TIMEOUT_SEC=120"
)]
struct Args {
    /// порт HTTP-слушателя
    #[arg(long, default_value_t = 9898)]
    port: u16,

    /// каталог с моделями
    #[arg(long)]
    models_dir: Option<std::path::PathBuf>,

    /// cuda | cpu
    #[arg(long, default_value = "cuda")]
    device: String,

    /// не загружать RNN-T (только CTC)
    #[arg(long)]
    no_rnnt: bool,

    /// воркеров на модель (0 = авто по свободной VRAM)
    #[arg(long, default_value_t = 0)]
    workers: usize,
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
    // конфигурация окружения проверяется до всякой работы с моделями:
    // кривое значение env отказывает, а не подменяется дефолтом
    let limits = server::config::ServerLimits::from_env()
        .map_err(|e| anyhow::anyhow!("конфигурация окружения: {e}"))?;

    let use_cuda_requested = match args.device.as_str() {
        "cuda" | "gpu" => true,
        "cpu" => false,
        other => anyhow::bail!("неизвестное устройство: {other} (cuda|cpu)"),
    };
    let paths = models::resolve(args.models_dir.as_deref())?;

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    // --- загрузка моделей: пул CTC всегда, пул RNN-T без --no-rnnt ---
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
            handles.push(std::thread::spawn(move || {
                worker::try_build_worker(&paths, kind, use_cuda, intra, false)
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
    let build_pool = |kind: ModelKind, use_cuda: bool| -> Result<WorkerPool> {
        let mk_opts = |use_cuda: bool| lighttrans::pipeline::WorkerOpts {
            language: "ru".into(),
            translate: false,
            beam_size: 1,
            n_threads: if use_cuda { 4 } else { cores },
            metrics: Some(Arc::clone(&metrics)),
        };
        match load_model(kind, use_cuda) {
            Ok(workers) => {
                tracing::info!(
                    "{}: {} воркеров ({})",
                    kind.id(),
                    workers.len(),
                    if use_cuda { "CUDA" } else { "CPU" }
                );
                Ok(WorkerPool::start(workers, mk_opts(use_cuda)))
            }
            Err(e) if use_cuda => {
                tracing::warn!("{}: CUDA недоступна ({e:#}); переключаю на CPU", kind.id());
                let workers = load_model(kind, false)?;
                tracing::info!("{}: {} воркеров (CPU)", kind.id(), workers.len());
                Ok(WorkerPool::start(workers, mk_opts(false)))
            }
            Err(e) => Err(e),
        }
    };

    let use_cuda = use_cuda_requested;
    let ctc = Arc::new(build_pool(ModelKind::GigaamV3, use_cuda)?);
    let rnnt = if args.no_rnnt {
        None
    } else {
        Some(Arc::new(build_pool(ModelKind::GigaamV3Rnnt, use_cuda)?))
    };
    drop(_load_span);
    tracing::info!("воркеров всего: {}", ctc.len() + rnnt.as_ref().map(|p| p.len()).unwrap_or(0));

    let admission = server::admission::AdmissionState::new(limits.max_concurrency);
    let state = AppState::new(ServerInner {
        pools: ModelPools { ctc, rnnt },
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
        "lighttrans-server слушает 0.0.0.0:{port} (models: gigaam-v3{}, лимит {}, тело {} МБ, тайм-аут {} c)",
        if state.inner().pools.has_rnnt() { ", gigaam-v3-rnnt" } else { "" },
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
