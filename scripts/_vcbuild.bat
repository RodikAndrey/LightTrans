@echo off
call F:\VS2022BT\VC\Auxiliary\Buildcvars64.bat >nul
bash -c "source scripts/env.sh && cargo build --release 2>&1"
