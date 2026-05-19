#!/usr/bin/env bash
# Collect llama/ggml prebuilt artifacts on Linux CI.
# Static .a files live under target/**/llama-cmake-cache/*/lib/ after install.
# Dynamic builds emit versioned sonames (libllama.so.0.0.N) and symlinks
# (libllama.so -> libllama.so.0); plain `find -type f -name 'lib*.so'` misses both.

set -euo pipefail

ROOT="${1:?output root}"
LIBRARY_TYPE="${2:?static or dynamic}"

mkdir -p "$ROOT/lib" "$ROOT/lib64" "$ROOT/bin"

is_llama_lib_name() {
  local base="$1"
  case "$base" in
    libllama*|libggml*|libcommon*|libmtmd*) return 0 ;;
    *) return 1 ;;
  esac
}

collect_static() {
  local search_root="$1"
  [[ -d "$search_root" ]] || return 0

  while IFS= read -r -d '' f; do
    local base
    base="$(basename "$f")"
    if is_llama_lib_name "$base" && [[ "$base" == *.a ]]; then
      cp -f "$f" "$ROOT/lib/$base"
    fi
  done < <(find "$search_root" -type f -name 'lib*.a' -print0 2>/dev/null)
}

collect_dynamic() {
  local search_root="$1"
  [[ -d "$search_root" ]] || return 0

  # Include symlinks: libfoo.so -> libfoo.so.0 -> libfoo.so.0.0.N
  while IFS= read -r -d '' f; do
    local base
    base="$(basename "$f")"
    if ! is_llama_lib_name "$base"; then
      continue
    fi
    case "$base" in
      *.so|*.so.*)
        cp -f "$f" "$ROOT/lib/$base"
        ;;
    esac
  done < <(
    find "$search_root" \( -type f -o -type l \) \
      \( -name 'lib*.so' -o -name 'lib*.so.*' \) -print0 2>/dev/null
  )
}

if [[ "$LIBRARY_TYPE" == "static" ]]; then
  collect_static target
elif [[ "$LIBRARY_TYPE" == "dynamic" ]]; then
  collect_dynamic target
else
  echo "::error::Unknown library_type=$LIBRARY_TYPE (expected static or dynamic)"
  exit 1
fi

COUNT="$(find "$ROOT/lib" -type f 2>/dev/null | wc -l | tr -d ' ')"
# Symlinks copied into ROOT/lib count as files only when they are regular files;
# also count symlinks we intentionally ship.
if [[ "$COUNT" -lt 1 ]]; then
  COUNT="$(find "$ROOT/lib" \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' ')"
fi

echo "Collected $COUNT file(s) for library_type=$LIBRARY_TYPE"
ls -la "$ROOT/lib/" 2>/dev/null || true

if [[ "$COUNT" -lt 1 ]]; then
  echo "::error::No libraries collected (library_type=$LIBRARY_TYPE)"
  echo "::group::Debug: sample shared libs under target"
  find target \( -type f -o -type l \) \
    \( -name 'libllama*.so' -o -name 'libllama*.so.*' -o -name 'libggml*.so*' \) \
    2>/dev/null | head -40 || true
  echo "::endgroup::"
  exit 1
fi
