#!/usr/bin/env bash
# Download a tiny GGUF used by llama-cpp-4 integration tests and CI.
#
# Usage:
#   ./scripts/fetch-test-model.sh
#
# The file is cached at target/test-models/stories260K.gguf (same model as
# llama.cpp server tests: ggml-org/models tinyllamas/stories260K.gguf).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/target/test-models"
FILE="$DEST/stories260K.gguf"
URL="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"

mkdir -p "$DEST"

if [[ -f "$FILE" ]]; then
  echo "Test model already present: $FILE"
  exit 0
fi

echo "Downloading stories260K.gguf (~1 MB)…"
curl -fsSL "$URL" -o "$FILE"
echo "Saved: $FILE"
