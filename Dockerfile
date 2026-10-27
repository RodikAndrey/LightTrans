# syntax=docker/dockerfile:1
# Multi-stage (PORTING.md Фаза 7): сборка в полном образе (nvcc для
# whisper.cpp, dev-заголовки ffmpeg, rust), рантайм — только рантайм-библиотеки.
#
# CUDA 13/cuDNN 9 рантайм-библиотеки берутся из базового образа nvidia/cuda:
# deps/cuda13 на хосте нужен лишь потому, что системный nvcc — 12.x; в
# контейнере версии совпадают с ORT CUDA EP. Модели в образ не зашиваются —
# монтируются томом: -v $PWD/models:/app/models.
#
# Сборка:  docker build -t lighttrans .
# Запуск:  docker run --rm --gpus all -p 9898:9898 -v $PWD/models:/app/models lighttrans
# CPU:     docker run --rm -p 9898:9898 -v $PWD/models:/app/models lighttrans --ep cpu
#
# Драйвер NVIDIA на хосте предоставляется nvidia-container-toolkit.

ARG UBUNTU=24.04
ARG CUDA=13.0.2

# --- builder: полный тулчейн -------------------------------------------------
FROM nvidia/cuda:${CUDA}-cudnn-devel-ubuntu${UBUNTU} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates clang libclang-dev curl xz-utils \
        git python3 pkg-config libssl-dev cmake \
    && rm -rf /var/lib/apt/lists/*

# rust stable (rustup)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH}
ADD https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init /tmp/rustup-init
RUN chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal \
        --default-host x86_64-unknown-linux-gnu --default-toolchain stable \
    && rm /tmp/rustup-init \
    && cargo --version

# ffmpeg 9 shared (dev: заголовки + .so) — тот же дистрибутив, что deps/ на хосте
ARG FFMPEG_BUILD=ffmpeg-n9.0-latest-linux64-gpl-shared-9.0
ADD https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FFMPEG_BUILD}.tar.xz /tmp/ffmpeg.tar.xz
RUN mkdir -p /opt/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
    && rm /tmp/ffmpeg.tar.xz
ENV FFMPEG_DIR=/opt/ffmpeg \
    LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY scripts ./scripts
COPY src ./src
COPY examples ./examples

# libonnxruntime статична в бинарнике; динамические — только CUDA-провайдеры,
# ort-sys ищет их рядом с бинарником, поэтому кладём в /out/bin/
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    bash scripts/patch_whisper_dtw.sh \
    && cargo fetch --locked \
    && cargo build --locked --release --bins \
    && mkdir -p /out/bin /out/ffmpeg/lib \
    && install -m755 target/release/lighttrans-server /out/bin/ \
    && install -m755 target/release/lighttrans /out/bin/ \
    && cp -aL target/release/libonnxruntime_providers_cuda.so \
            target/release/libonnxruntime_providers_shared.so /out/bin/ \
    && cp -a /opt/ffmpeg/lib/libavcodec.so* /opt/ffmpeg/lib/libavdevice.so* \
             /opt/ffmpeg/lib/libavfilter.so* /opt/ffmpeg/lib/libavformat.so* \
             /opt/ffmpeg/lib/libavutil.so* /opt/ffmpeg/lib/libswresample.so* \
             /opt/ffmpeg/lib/libswscale.so* /out/ffmpeg/lib/

# --- runtime: только рантайм --------------------------------------------------
FROM nvidia/cuda:${CUDA}-cudnn-runtime-ubuntu${UBUNTU}

# рантайм-зависимости ffmpeg-библиотек BtbN (кодеки/X11/vdpau линкуются
# внешне, в lib/ их нет) + сертификаты; CUDA 13/cuDNN 9 — из базового образа
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
        libmp3lame0 libmpg123-0t64 libopus0 libvdpau1 libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 lighttrans \
    && useradd --uid 10001 --gid lighttrans --no-create-home --home-dir /app \
        --shell /usr/sbin/nologin lighttrans

WORKDIR /app
COPY --from=builder --chown=0:0 --chmod=755 /out/bin/ /app/bin/
COPY --from=builder /out/ffmpeg/ /app/ffmpeg/

# CUDA 13/cuDNN 9 — из базового образа (ldconfig); провайдеры ORT лежат
# рядом с бинарником; добавляем в поиск только ffmpeg
ENV LD_LIBRARY_PATH=/app/ffmpeg/lib \
    ASR_MAX_CONCURRENCY=8 \
    ASR_BODY_LIMIT_MB=64 \
    ASR_REQ_TIMEOUT_SEC=120

EXPOSE 9898
USER lighttrans
ENTRYPOINT ["/app/bin/lighttrans-server"]
CMD ["--port", "9898", "--models-dir", "/app/models"]

LABEL org.opencontainers.image.title="LightTrans" \
      org.opencontainers.image.description="GPU audio/video transcription service (GigaAM v3 CTC/RNN-T): HTTP batch + WebSocket streaming, ONNX Runtime CUDA"
