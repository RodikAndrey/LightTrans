#!/usr/bin/env bash
# Patches the vendored whisper.cpp inside whisper-rs-sys to fix the DTW
# median-filter assert (upstream whisper.cpp issue #2301): a decode that ends
# early leaves a near-zero-length window, where ne[2] <= filter_width.
# The guard passes the data through unfiltered in that case.
# Idempotent; run before `cargo build`. build.sh calls it automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

REGISTRY_SRC="$(ls -d ~/.cargo/registry/src/index.crates.io-*/whisper-rs-sys-* 2>/dev/null || true)"
TARGETS="$REGISTRY_SRC/whisper.cpp/src/whisper.cpp target/release/build/whisper-rs-sys-*/out/whisper.cpp/src/whisper.cpp"

for f in $TARGETS; do
    [ -f "$f" ] || continue
    if grep -q "issue #2301" "$f"; then
        echo "patch_whisper_dtw: already patched: $f"
        continue
    fi
    PY="$(command -v python3 || command -v python)"
    "$PY" - "$f" <<'EOF'
import sys
p = sys.argv[1]
old = """    int filter_width = ((median_filter_user_data *) userdata)->filter_width;
    WHISPER_ASSERT(filter_width < a->ne[2]);"""
new = """    int filter_width = ((median_filter_user_data *) userdata)->filter_width;
    // Guard (upstream whisper.cpp issue #2301): with DTW, a decode that ends
    // early can leave a near-zero-length window, where ne[2] <= filter_width.
    // The filter cannot be applied there; pass the data through unfiltered.
    if (filter_width >= a->ne[2]) {
        memcpy((char *) dst->data, (const char *) a->data, ggml_nbytes(a));
        return;
    }
    WHISPER_ASSERT(filter_width < a->ne[2]);"""
s = open(p, encoding='utf-8').read()
if old not in s:
    raise SystemExit(f"patch_whisper_dtw: pattern not found in {p} — whisper.cpp version changed?")
open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new, 1))
print(f"patch_whisper_dtw: patched {p}")
EOF
done
