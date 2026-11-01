#!/usr/bin/env bash
# Build the wasm example.
#
#   ./build.sh            # static site: one .js + one .html, no server needed
#   ./build.sh node       # for testing under node against the real filesystem
#
# Needs: `source <emsdk>/emsdk_env.sh` and `rustup target add wasm32-unknown-emscripten`.
set -euo pipefail

MODE="${1:-static}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/target/wasm32-unknown-emscripten/release"
DIST="$ROOT/target/wasm-dist"

if [[ -z "${EMSDK:-}" ]]; then
  echo "EMSDK is not set. Run: source /path/to/emsdk/emsdk_env.sh" >&2
  exit 1
fi

#   ALLOW_MEMORY_GROWTH: the model load is the largest allocation and its size
#     is not known until the GGUF header is read.
#   TOTAL_STACK: llama.cpp's graph build recurses deeper than the 64KB default.
#   EXIT_RUNTIME: let main() return so destructors run.
#
# Deliberately no -pthread. Emscripten's thread pool needs SharedArrayBuffer,
# which browsers only grant a cross-origin-isolated page — and that takes
# COOP/COEP response headers a static host cannot send. So this is
# single-threaded by construction rather than by accident.
FLAGS=(
  -C link-arg=-sALLOW_MEMORY_GROWTH=1
  -C link-arg=-sTOTAL_STACK=8MB
  -C link-arg=-sEXIT_RUNTIME=1
)

case "$MODE" in
  static)
    FLAGS+=(
      # Inline the wasm into the .js as base64. Costs ~33% in file size, and
      # buys a page that needs no second request — so it works from file:// and
      # from any static host, neither of which can be relied on to serve
      # application/wasm or allow the fetch.
      -C link-arg=-sSINGLE_FILE=1
      # The model arrives from an <input type="file">, written into MEMFS by
      # the page, so the filesystem and FS must exist at runtime. No
      # --preload-file: that would bake a model into the build and need a
      # separate .data fetch.
      -C link-arg=-sFORCE_FILESYSTEM=1
      -C link-arg=-sEXPORTED_RUNTIME_METHODS=callMain,FS
      -C link-arg=-sINVOKE_RUN=0
      # Export a factory instead of writing a global `Module`. The global form
      # relies on `var Module = typeof Module !== "undefined" ? ... ` picking up
      # a pre-set global, which silently fails under any module system because
      # `var` hoisting makes the name local. A factory returns a promise, so
      # "runtime is up" is awaited rather than guessed at.
      -C link-arg=-sMODULARIZE=1
      -C link-arg=-sEXPORT_NAME=createLlama
    )
    ;;
  node)
    FLAGS+=(-C link-arg=-sNODERAWFS=1)
    ;;
  *) echo "unknown mode: $MODE (want static|node)" >&2; exit 1 ;;
esac

echo "building ($MODE)…"
# `linker_messages` fires on `WASM_BIGINT is deprecated`, which comes from
# rustc's own emscripten target spec rather than anything set here.
RUSTFLAGS="${FLAGS[*]} -A linker_messages" cargo build --release \
  --target wasm32-unknown-emscripten -p wasm-example

rm -rf "$DIST" && mkdir -p "$DIST"
cp "$OUT/llama-wasm.js" "$DIST/"
# SINGLE_FILE inlines the wasm, so there is nothing else to copy in that mode.
[[ -f "$OUT/llama_wasm.wasm" && "$MODE" == node ]] && cp "$OUT/llama_wasm.wasm" "$DIST/"
[[ "$MODE" == static ]] && cp "$HERE/index.html" "$HERE/harness.cjs" "$DIST/"

# `stat` is not portable: BSD/macOS wants `-f%z`, GNU wants `-c%s`, and the
# uutils build on the CI runners rejects the BSD form outright. `wc -c` is
# POSIX and needs no dialect check; awk does the division so `bc` is not
# needed either.
file_size_mb() {
  awk -v bytes="$(wc -c < "$1")" 'BEGIN { printf "%.2f", bytes / 1048576 }'
}

echo
echo "dist: $DIST"
for f in "$DIST"/*; do
  printf '  %8s MB  %s\n' "$(file_size_mb "$f")" "$(basename "$f")"
done

echo
if [[ "$MODE" == static ]]; then
  echo "Two files, no server required. Open it directly:"
  echo "  open $DIST/index.html"
  echo "or upload index.html + llama-wasm.js to any static host."
  echo
  echo "Headless check of the same code path (factory -> MEMFS -> callMain):"
  echo "  node $DIST/harness.cjs <model.gguf> \"Once upon a time\" 16"
else
  echo "Run it:"
  echo "  node $DIST/llama-wasm.js <model.gguf> \"Once upon a time\" 24"
fi
