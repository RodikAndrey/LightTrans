#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"

if [ "$(uname -s)" = "Linux" ]; then
    # fetch first so whisper-rs-sys is in the registry, then patch whisper.cpp
    # (the sys build script copies it from there), then build
    cargo fetch
    bash scripts/patch_whisper_dtw.sh
    cargo build --release "$@"
    exit 0
fi

# whisper.cpp CUDA needs cl.exe/nvcc on PATH -> run cargo under vcvars64
VCVARS="F:\\VS2022BT\\VC\\Auxiliary\\Build\\vcvars64.bat"
if [ ! -f "/f/VS2022BT/VC/Auxiliary/Build/vcvars64.bat" ]; then
    VCVARS="C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
fi

BAT="$ROOT/scripts/_vcbuild.bat"
{
    echo "@echo off"
    echo "call \"$VCVARS\" >nul"
    echo "bash scripts/patch_whisper_dtw.sh"
    echo "cargo build --release $*"
    echo "exit /b %ERRORLEVEL%"
} > "$BAT"
cmd //c "$(cygpath -w "$BAT")"

# deploy runtime DLLs next to the exe (self-contained build)
BIN="$ROOT/target/release"
cp -n "$ROOT/deps/ffmpeg-n9.0-latest-win64-gpl-shared-9.0/bin/"*.dll "$BIN/" 2>/dev/null || true
cp -n "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64/cublas64_13.dll" \
      "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64/cublasLt64_13.dll" \
      "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64/cudart64_13.dll" "$BIN/" 2>/dev/null || true
cp -n "C:/Python313/Lib/site-packages/nvidia/cudnn/bin/"*.dll "$BIN/" 2>/dev/null || true
echo "DLL deployment done"
