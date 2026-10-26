#!/usr/bin/env bash
set -e
source "$(dirname "$0")/env.sh"
cd "$ROOT"
if [ "$(uname -s)" = "Linux" ]; then
    exec "$ROOT/target/release/lighttrans" "$@"
fi
exec "$ROOT/target/release/lighttrans.exe" "$@"
