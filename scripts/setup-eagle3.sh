#!/usr/bin/env bash
#
# One command to get everything needed to run the `eagle` example:
# creates a Python conversion env (first run only), then downloads and converts
# the EAGLE-3 target + draft weights to GGUF.
#
#   scripts/setup-eagle3.sh [OUTDIR]      # default OUTDIR: ./models/eagle3
#
# Defaults to the open Qwen3-8B + RedHatAI/Qwen3-8B-speculator.eagle3 pairing.
# Override the repos like fetch-eagle3.sh, e.g.:
#   TARGET_REPO=... DRAFT_REPO=... scripts/setup-eagle3.sh
#
# Needs ~40 GB free disk and `uv` (https://docs.astral.sh/uv/).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv-convert"
REQ="$REPO_ROOT/llama-cpp-sys-4/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
OUTDIR="${1:-$REPO_ROOT/models/eagle3}"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is required. Install it: https://docs.astral.sh/uv/ (or: pipx install uv)" >&2
    exit 1
fi

# 1. One-time conversion env: hf CLI + torch/transformers/gguf.
if [ -x "$VENV/bin/hf" ] && [ -x "$VENV/bin/python" ]; then
    echo "==> Using existing conversion env at $VENV"
else
    echo "==> Creating conversion env at $VENV (one-time, downloads torch etc.)"
    uv venv --python 3.12 "$VENV"
    # --index-strategy unsafe-best-match: the requirements file adds the pytorch
    # index, and uv otherwise refuses to look past it for transformers/etc.
    uv pip install --python "$VENV" --index-strategy unsafe-best-match -r "$REQ"
    uv pip install --python "$VENV" "huggingface_hub[cli]"
fi

# 2. Download + convert, with this env's hf/python on PATH.
echo "==> Fetching + converting weights into $OUTDIR"
PATH="$VENV/bin:$PATH" "$REPO_ROOT/scripts/fetch-eagle3.sh" "$OUTDIR"
