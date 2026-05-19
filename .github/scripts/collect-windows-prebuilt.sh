#!/usr/bin/env bash
# Collect llama/ggml prebuilt artifacts on Windows CI.
# Static builds install .lib files under %LOCALAPPDATA%\llcb\<hash>\lib (see build.rs).
# Dynamic builds also emit .dll files under target/.

set -euo pipefail

ROOT="${1:?output root}"
LIBRARY_TYPE="${2:?static or dynamic}"

mkdir -p "$ROOT/lib" "$ROOT/lib64" "$ROOT/bin"

is_llama_lib() {
  local base="$1"
  case "$base" in
    llama*.lib|ggml*.lib|common*.lib|mtmd*.lib) return 0 ;;
    *) return 1 ;;
  esac
}

is_llama_dll() {
  local base="$1"
  case "$base" in
    llama*.dll|ggml*.dll|common*.dll|mtmd*.dll) return 0 ;;
    *) return 1 ;;
  esac
}

collect_from() {
  local search_root="$1"
  [[ -d "$search_root" ]] || return 0

  while IFS= read -r -d '' f; do
    local base
    base="$(basename "$f")"
    if [[ "$LIBRARY_TYPE" == "static" ]] && [[ "$base" == *.lib ]] && is_llama_lib "$base"; then
      cp -f "$f" "$ROOT/lib/$base"
    elif [[ "$LIBRARY_TYPE" == "dynamic" ]] && [[ "$base" == *.dll ]] && is_llama_dll "$base"; then
      cp -f "$f" "$ROOT/bin/$base"
    fi
  done < <(find "$search_root" -type f \( -name '*.lib' -o -name '*.dll' \) -print0 2>/dev/null)
}

# Cargo build tree (dynamic DLLs and occasional import libs).
collect_from target

# CMake install prefix used by llama-cpp-sys-4 on Windows (static .lib files).
if [[ -n "${LOCALAPPDATA:-}" ]]; then
  collect_from "$LOCALAPPDATA/llcb"
fi

# Fallback: cmake build tree inside OUT_DIR under target.
while IFS= read -r -d '' out_lib; do
  collect_from "$(dirname "$out_lib")"
done < <(find target -path '*/out/lib/*.lib' -print0 2>/dev/null)

COUNT="$(find "$ROOT" -type f ! -name 'metadata.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "Collected $COUNT file(s) for library_type=$LIBRARY_TYPE"
ls -la "$ROOT/lib/" "$ROOT/bin/" 2>/dev/null || true

if [[ "$COUNT" -lt 1 ]]; then
  echo "::error::No libraries collected (library_type=$LIBRARY_TYPE)"
  echo "::group::Debug: sample paths under LOCALAPPDATA/llcb"
  find "${LOCALAPPDATA:-/missing}/llcb" -type f -name '*.lib' 2>/dev/null | head -40 || true
  echo "::endgroup::"
  echo "::group::Debug: sample paths under target"
  find target -type f \( -name 'llama*.lib' -o -name 'ggml*.lib' -o -name 'llama*.dll' -o -name 'ggml*.dll' \) 2>/dev/null | head -40 || true
  echo "::endgroup::"
  exit 1
fi
