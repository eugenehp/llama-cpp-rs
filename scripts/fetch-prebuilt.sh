#!/usr/bin/env bash
# Download prebuilt llama.cpp libraries for the current host into the Cargo cache.
#
# Usage:
#   ./scripts/fetch-prebuilt.sh
#   LLAMA_PREBUILT_TAG=v0.4.0 ./scripts/fetch-prebuilt.sh
#   ./scripts/fetch-prebuilt.sh --features "metal,mtmd"
#
# After download, build with:
#   cargo build -p llama-cpp-4 --features prebuilt
#
# Or point directly at the extracted directory:
#   export LLAMA_PREBUILT_DIR=target/llama-prebuilt-cache/0.4.0/llama-prebuilt-...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="${LLAMA_PREBUILT_TAG:-v0.4.0}"
REPO="${LLAMA_PREBUILT_REPO:-eugenehp/llama-cpp-rs}"
FEATURES="${LLAMA_PREBUILT_FEATURES:-mtmd}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,14p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

TARGET="$(rustc -vV | sed -n 's/^host: //p')"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
  linux) PLATFORM=linux ;;
  darwin) PLATFORM=macos ;;
  mingw*|msys*|cygwin*|windows*) PLATFORM=windows ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

suffix="cpu"
if [[ ",$FEATURES," == *",metal,"* ]] && [[ "$PLATFORM" == "macos" ]]; then
  suffix="metal"
elif [[ ",$FEATURES," == *",vulkan,"* ]]; then
  suffix="vulkan"
elif [[ ",$FEATURES," == *",blas,"* ]]; then
  suffix="blas"
fi

library_type="static"
if [[ ",$FEATURES," == *",dynamic-link,"* ]]; then
  library_type="dynamic"
fi

ASSET="llama-prebuilt-${PLATFORM}-${TARGET}-${suffix}-${library_type}.tar.gz"
CACHE="$ROOT/target/llama-prebuilt-cache/${TAG#v}/$ASSET"
CACHE="${CACHE%.tar.gz}"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

if [[ -d "$CACHE/lib" ]] && ls "$CACHE/lib"/*llama* &>/dev/null; then
  echo "Prebuilt already cached: $CACHE"
  echo "export LLAMA_PREBUILT_DIR=$CACHE"
  exit 0
fi

mkdir -p "$CACHE"
TMP="$(mktemp -t llama-prebuilt.XXXXXX.tar.gz)"
trap 'rm -f "$TMP"' EXIT

echo "Downloading $URL"
curl -fsSL "$URL" -o "$TMP"
tar -xzf "$TMP" -C "$CACHE"

echo "Prebuilt libraries extracted to: $CACHE"
echo "export LLAMA_PREBUILT_DIR=$CACHE"
