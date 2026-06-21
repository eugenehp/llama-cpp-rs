# eagle — EAGLE-3 speculative decoding

Drives **EAGLE-3** speculative decoding from Rust via
[`llama_cpp_4::eagle::Eagle3Session`]. It pairs a **target** model with a
small, separately-trained **EAGLE-3 draft** model that predicts the next tokens
from hidden states extracted out of the target.

Unlike MTP (one model, special draft context), EAGLE-3 needs **two GGUFs**: the
full target model and an EAGLE-3 draft model trained for it. No pre-converted
EAGLE-3 GGUFs are published, so you convert a HF pairing once (below).

## Verified

Run end-to-end on an Apple M4 Pro (Metal) with `Qwen/Qwen3-8B` (q8_0 target)
and `RedHatAI/Qwen3-8B-speculator.eagle3` (draft):

```
List three primary colors. The primary colors are red, blue, and yellow. ...
generated 49 tokens in 1.60s = 30.7 tok/s
EAGLE-3: 24 draft calls, 34 drafts proposed, 24 accepted (70.6% acceptance)
```

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (the setup script uses it to build a
  one-off Python env with `hf` + torch/transformers/gguf). Install: `pipx
  install uv`.
- ~40 GB free disk for the Qwen3-8B pairing (16 GB download + GGUFs).

## 1. Download + convert weights (once)

One command, from the repo root — creates the conversion env on first run, then
downloads and converts both models to GGUF (re-running is cheap; finished steps
are skipped):

```sh
scripts/setup-eagle3.sh ./models/eagle3
```

Defaults to the fully-open `Qwen/Qwen3-8B` + `RedHatAI/Qwen3-8B-speculator.eagle3`.
Override the repos for the upstream-validated (gated) Llama pairing:

```sh
TARGET_REPO=meta-llama/Llama-3.1-8B-Instruct \
DRAFT_REPO=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
scripts/setup-eagle3.sh ./models/eagle3-llama
```

> If you already have `hf` and the convert deps on your `PATH`, you can call the
> lower-level `scripts/fetch-eagle3.sh ./models/eagle3` directly (it does the
> download + convert without the env bootstrap).

## 2. Run

```sh
# smoke test: loads both models + creates the session (validates the draft)
cargo run --release -p eagle --features metal -- \
    models/eagle3/target.gguf models/eagle3/draft-eagle3.gguf

# generate through the draft/verify/accept loop
cargo run --release -p eagle --features metal -- \
    --predict 96 --n-draft-max 8 --p-min 0.5 \
    --prompt "Explain what speculative decoding is in two sentences." \
    models/eagle3/target.gguf models/eagle3/draft-eagle3.gguf
```

Swap `--features metal` for `cuda` / `vulkan` as appropriate.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--n-draft-max` | `8` | Max tokens drafted per round (`--spec-draft-n-max`) |
| `--p-min` | `0.5` | Draft probability floor (`--spec-draft-p-min`) |
| `--n-rs-seq` | `0` | Recurrent-state snapshots; set `>= n-draft-max` for hybrid/recurrent targets |
| `-c, --ctx-size` | `2048` | Context size |
| `--predict N` | — | Generate N tokens (omit for smoke test) |
| `--prompt` | "The capital of France is" | Prompt when `--predict` is set |

## Notes

- **Use `cargo run`.** With the default `dynamic-link` feature, the built binary
  links shared libs without an rpath, so invoking `target/release/eagle`
  directly fails with `Library not loaded: @rpath/libggml-base...`. `cargo run`
  sets the dylib search path for you. (To run the raw binary, prefix it with
  `DYLD_LIBRARY_PATH="$PWD/target/release"`.)
- EAGLE-3 has known issues with some hybrid targets (e.g. Qwen3.6 / `qwen3_5`,
  llama.cpp issue #24541). The Qwen3-8B and Llama-3.1-8B pairings above avoid it.
