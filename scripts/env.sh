# Source this before building/running: sets toolchain + library search paths.
export PATH="$HOME/.cargo/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

if [ "$(uname -s)" = "Linux" ]; then
    # FFmpeg 9 shared dev libraries (headers + .so), BtbN linux64 gpl shared
    export FFMPEG_DIR="$ROOT/deps/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0"

    # CUDA toolkit: Debian/Ubuntu package -> /usr/lib/nvidia-cuda-toolkit,
    # upstream installer -> /usr/local/cuda (or CUDA_PATH from the environment)
    if [ -z "${CUDA_PATH:-}" ]; then
        if [ -d /usr/local/cuda ]; then
            export CUDA_PATH=/usr/local/cuda
        elif [ -d /usr/lib/nvidia-cuda-toolkit ]; then
            export CUDA_PATH=/usr/lib/nvidia-cuda-toolkit
        fi
    fi
    [ -d "$CUDA_PATH/bin" ] && export PATH="$CUDA_PATH/bin:$PATH"
    export CUDACXX="${CUDACXX:-$(command -v nvcc || true)}"

    # nvcc 12.x rejects GCC >= 14 as host compiler; pin gcc-13 when present
    if command -v gcc-13 >/dev/null 2>&1; then
        export CUDAHOSTCXX="$(command -v gcc-13)"
        export NVCC_CCBIN="$(command -v gcc-13)"
    fi

    # libclang for ffmpeg-sys-next bindgen
    if [ -z "${LIBCLANG_PATH:-}" ] && [ -d /usr/lib/x86_64-linux-gnu ]; then
        export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
    fi

    # Runtime .so search: CUDA 13 runtime for the ORT CUDA EP (deps/cuda13) +
    # ffmpeg + cuDNN 13 (wheel) + ort providers (dlopen).
    # The system nvcc 12.4 (whisper.cpp build) uses the system cublas/cudart 12.
    CUDA13_LIB="$ROOT/deps/cuda13/sys/usr/local/cuda-13.3/targets/x86_64-linux/lib"
    CUDA13_CUDNN="$ROOT/deps/cuda13/sys/nvidia/cudnn/lib"
    ORT_DIR="$(find "$ROOT/target/release/build" -maxdepth 5 -type d -name lib -path '*ort-sys*' 2>/dev/null | head -1)"
    CUDA_LIB=""
    [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH/lib64" ] && CUDA_LIB="$CUDA_PATH/lib64"
    export LD_LIBRARY_PATH="$CUDA13_LIB:$CUDA13_CUDNN:$FFMPEG_DIR/lib${CUDA_LIB:+:$CUDA_LIB}${ORT_DIR:+:$ORT_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    # FFmpeg 9 shared dev libraries (headers + import libs + DLLs)
    export FFMPEG_DIR="$ROOT/deps/ffmpeg-n9.0-latest-win64-gpl-shared-9.0"

    # CUDA toolkit (adjust version if different)
    CUDA_ROOT="${CUDA_PATH:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3}"
    export CUDA_PATH="$CUDA_ROOT"
    # CUDA MSBuild integration resolves the toolkit via the versioned env var
    export CUDA_PATH_V13_3="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3"

    # Runtime DLL search: ffmpeg bin + CUDA runtime + cuDNN (installed via pip) + ort-sys prebuilt dir
    CUDNN_BIN="C:/Python313/Lib/site-packages/nvidia/cudnn/bin"
    ORT_DIR="$(find "$ROOT/target" -name 'onnxruntime.dll' -path '*release*' 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null)"
    export PATH="$ROOT/deps/ffmpeg-n9.0-latest-win64-gpl-shared-9.0/bin:$CUDA_ROOT/bin:$CUDA_ROOT/bin/x64:$CUDNN_BIN:$ORT_DIR:$PATH"

    # libclang for ffmpeg-sys-next bindgen
    if [ -d "/c/Program Files/LLVM/bin" ]; then
        export LIBCLANG_PATH="C:\\Program Files\\LLVM\\bin"
    fi
fi
