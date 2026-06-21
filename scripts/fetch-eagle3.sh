#!/usr/bin/env bash
#
# Download + convert an EAGLE-3 target/draft pairing to GGUF for the `eagle`
# example. Defaults to Red Hat's fully-open Qwen3-8B pairing:
#
#   target : Qwen/Qwen3-8B                       (Apache-2.0, non-gated)
#   draft  : RedHatAI/Qwen3-8B-speculator.eagle3 (Apache-2.0, 1.0B EAGLE-3 head)
#
# Override either with env vars, e.g. the upstream-validated Llama pairing
# (the target is gated — accept its license on HF and `hf auth login` first):
#
#   TARGET_REPO=meta-llama/Llama-3.1-8B-Instruct \
#   DRAFT_REPO=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
#   scripts/fetch-eagle3.sh ./models/eagle3-llama
#
# Requirements:
#   - Hugging Face CLI:  pip install -U "huggingface_hub[cli]"   (provides `hf`)
#   - convert deps:      pip install -r <llama.cpp>/requirements/requirements-convert_hf_to_gguf.txt
#   - ~40 GB free disk for the Qwen3-8B pairing (16 GB download + GGUFs).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONVERT="$REPO_ROOT/llama-cpp-sys-4/llama.cpp/convert_hf_to_gguf.py"

TARGET_REPO="${TARGET_REPO:-Qwen/Qwen3-8B}"
DRAFT_REPO="${DRAFT_REPO:-RedHatAI/Qwen3-8B-speculator.eagle3}"
OUTDIR="${1:-$REPO_ROOT/models/eagle3}"
TARGET_OUTTYPE="${TARGET_OUTTYPE:-bf16}" # set to q8_0 for a smaller (~8.5 GB) target

if ! command -v hf >/dev/null 2>&1; then
    echo "error: the 'hf' CLI is not on PATH." >&2
    echo "       install it with: pip install -U \"huggingface_hub[cli]\"" >&2
    exit 1
fi
if [ ! -f "$CONVERT" ]; then
    echo "error: convert script not found at $CONVERT" >&2
    echo "       run: git submodule update --init --recursive" >&2
    exit 1
fi

TARGET_DIR="$OUTDIR/target"
DRAFT_DIR="$OUTDIR/draft"
TARGET_GGUF="$OUTDIR/target.gguf"
DRAFT_GGUF="$OUTDIR/draft-eagle3.gguf"
mkdir -p "$OUTDIR"

# Each step is skipped if its output already exists, so re-running is cheap.
if [ -f "$TARGET_DIR/config.json" ]; then
    echo "==> Target already downloaded ($TARGET_DIR)"
else
    echo "==> Downloading target  $TARGET_REPO"
    hf download "$TARGET_REPO" --local-dir "$TARGET_DIR"
fi

if [ -f "$DRAFT_DIR/config.json" ]; then
    echo "==> Draft already downloaded ($DRAFT_DIR)"
else
    echo "==> Downloading draft   $DRAFT_REPO"
    hf download "$DRAFT_REPO" --local-dir "$DRAFT_DIR"
fi

if [ -f "$TARGET_GGUF" ]; then
    echo "==> Target GGUF already exists ($TARGET_GGUF)"
else
    echo "==> Converting target -> $TARGET_GGUF (outtype=$TARGET_OUTTYPE)"
    python3 "$CONVERT" "$TARGET_DIR" --outtype "$TARGET_OUTTYPE" --outfile "$TARGET_GGUF"
fi

# The EAGLE-3 draft is a standalone head: it needs the target model's metadata
# (tokenizer, hidden size, the 3 extract-layer ids) baked in at convert time.
if [ -f "$DRAFT_GGUF" ]; then
    echo "==> Draft GGUF already exists ($DRAFT_GGUF)"
else
    echo "==> Converting draft  -> $DRAFT_GGUF (with --target-model-dir)"
    python3 "$CONVERT" "$DRAFT_DIR" --outtype f16 --target-model-dir "$TARGET_DIR" --outfile "$DRAFT_GGUF"
fi

echo
echo "Done. Run the example with:"
echo
echo "  cargo run --release -p eagle --features metal -- \\"
echo "      --predict 64 --n-draft-max 8 --p-min 0.5 \\"
echo "      \"$TARGET_GGUF\" \"$DRAFT_GGUF\""
echo
echo "(Optional) shrink the target for Metal with the quantize example:"
echo "  cargo run --release -p quantize -- \"$TARGET_GGUF\" \"$OUTDIR/target-q4_k_m.gguf\" q4_k_m"
