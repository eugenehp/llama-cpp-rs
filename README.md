# 🦙 llama-cpp-rs

[![Crates.io](https://img.shields.io/crates/v/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)
[![docs.rs](https://img.shields.io/docsrs/llama-cpp-4.svg)](https://docs.rs/llama-cpp-4)
[![License](https://img.shields.io/crates/l/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)

Safe Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp), tracking upstream closely.

| Crate | Description | crates.io |
|---|---|---|
| [`llama-cpp-4`](llama-cpp-4/) | Safe high-level API | [![](https://img.shields.io/crates/v/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4) |
| [`llama-cpp-sys-4`](llama-cpp-sys-4/) | Raw bindgen bindings | [![](https://img.shields.io/crates/v/llama-cpp-sys-4.svg)](https://crates.io/crates/llama-cpp-sys-4) |

**llama.cpp version:** c30e01225 (April 2026) — includes [TurboQuant (PR #21038)](#turboQuant--attention-rotation)

---

## Examples

| Package name | Directory | Description |
|---|---|---|
| `simple` | [`examples/simple/`](examples/simple/) | Single-turn text completion from CLI or Hugging Face |
| `chat` | [`examples/chat/`](examples/chat/) | Interactive multi-turn chat REPL |
| `embeddings` | [`examples/embeddings/`](examples/embeddings/) | Batch embedding with cosine similarity |
| `split-model-example` | [`examples/split_model/`](examples/split_model/) | Load sharded / split GGUF files |
| `openai-server` | [`examples/server/`](examples/server/) | OpenAI-compatible HTTP server with streaming and tool calling |
| `mtmd` | [`examples/mtmd/`](examples/mtmd/) | Multimodal (vision / audio) inference (requires `--features mtmd`) |
| `quantize` | [`examples/quantize/`](examples/quantize/) | Quantize a GGUF model with full typed API |
| `turbo-quant` | [`examples/turbo-quant/`](examples/turbo-quant/) | TurboQuant demo — compare attn rotation on/off |

---

## Quick start

```bash
git clone --recursive https://github.com/eugenehp/llama-cpp-rs
cd llama-cpp-rs
```

### Interactive chat

```bash
cargo run -p chat -- \
    hf-model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### OpenAI-compatible server

```bash
# Starts on http://127.0.0.1:8080
cargo run -p openai-server -- \
    hf-model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### Using prebuilt native libraries (skip CMake compile)

`llama-cpp-sys-4` can consume precompiled llama/ggml libraries via env vars.
This is useful for CI pipelines that publish native artifacts once and reuse
them in downstream repos (for example, speeding up a separate app build).

```bash
# Directory containing prebuilt libs in one of:
#   <dir>, <dir>/lib, <dir>/lib64, <dir>/bin
export LLAMA_PREBUILT_DIR=/path/to/prebuilt

# Optional: force dynamic linking mode for prebuilt artifacts.
# Defaults to the crate's normal link mode for the active feature set.
# export LLAMA_PREBUILT_SHARED=1

cargo build -p your-app --features "q1,vulkan"
```

Notes:
- `q1` compatibility is determined by the prebuilt artifact itself — publish
  separate artifacts per feature/backend tuple (`q1+vulkan`, `q1+metal`, ...).
- `build.rs` still generates Rust bindings, but skips the expensive CMake
  compile when `LLAMA_PREBUILT_DIR` is set.

Backend feature coverage (practical targets):
- `metal`  → macOS (Apple Silicon and Intel Macs)
- `vulkan` → Linux/Windows (cross-vendor desktop GPUs)
- `webgpu` → Linux/Windows (experimental; requires Dawn/WebGPU-native stack)
- `cuda`   → Linux/Windows with NVIDIA CUDA toolkit (experimental in CI)
- `hip`    → Linux ROCm/HIP environments (experimental in CI)
- `opencl` → Linux/Windows with OpenCL SDK/runtime (experimental in CI)
- `blas`   → CPU acceleration (Linux/macOS/Windows)

```bash
# Chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}], "max_tokens":128}'

# Streaming
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count to 5"}], "stream":true}'

# Embeddings
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "Bonjour le monde"]}'
```

### Text generation (library)

```rust
use llama_cpp_4::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    context::params::LlamaContextParams,
    sampling::LlamaSampler,
};
use std::num::NonZeroU32;

let backend = LlamaBackend::init()?;
let model = LlamaModel::load_from_file(&backend, "model.gguf", &LlamaModelParams::default())?;
let mut ctx = model.new_context(&backend, LlamaContextParams::default())?;

let tokens = model.str_to_token("Hello, world!", AddBos::Always)?;
let mut batch = LlamaBatch::new(512, 1);
for (i, &tok) in tokens.iter().enumerate() {
    batch.add(tok, i as i32, &[0], i == tokens.len() - 1)?;
}
ctx.decode(&mut batch)?;

let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
// ... decode loop
```

---

## Quantization

The `llama_cpp_4::quantize` module provides a fully typed Rust API for all
quantization options.

```rust
use llama_cpp_4::quantize::{GgmlType, LlamaFtype, QuantizeParams, TensorTypeOverride};

// Basic — quantize to Q4_K_M
let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM)
    .with_nthread(8)
    .with_quantize_output_tensor(true);

llama_cpp_4::model_quantize("model-f16.gguf", "model-q4km.gguf", &params).unwrap();

// Advanced — keep output tensor in F16, prune layers 28-31
let params = QuantizeParams::new(LlamaFtype::MostlyQ5KM)
    .with_tensor_type_override(TensorTypeOverride::new("output", GgmlType::F16).unwrap())
    .with_pruned_layers(28..=31);

llama_cpp_4::model_quantize("model-f16.gguf", "model-q5km-pruned.gguf", &params).unwrap();
```

From the CLI:

```bash
# List all available quantization types
cargo run -p quantize -- --list-types

# Quantize with auto output name
cargo run -p quantize -- model-f16.gguf Q4_K_M

# Override a specific tensor type
cargo run -p quantize -- --tensor-type output=F16 model-f16.gguf Q5_K_M

# Dry-run: show size without writing
cargo run -p quantize -- --dry-run model-f16.gguf Q4_K_M
```

---

## TurboQuant — attention rotation

**TurboQuant** (llama.cpp [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038))
applies a [Hadamard rotation](https://en.wikipedia.org/wiki/Hadamard_matrix) to the Q, K,
and V tensors before they are stored in the KV cache.

### Why it matters

Attention activations have large outlier values on some dimensions that make
quantization hard.  The rotation spreads these outliers evenly so the KV cache
can be stored in aggressive formats (Q4_0, Q5_0) with drastically less quality
loss:

| KV cache type | Without TurboQuant | With TurboQuant | VRAM vs F16 |
|:---:|:---:|:---:|:---:|
| F16 (baseline) | — | — | 100% |
| Q8_0 | +0.003 PPL | +0.003 PPL | 53% |
| Q5_1 | +61.70 PPL | **+0.44 PPL** | 37% |
| Q5_0 | +17.28 PPL | **+0.55 PPL** | 34% |
| Q4_1 | +212.5 PPL | **+8.65 PPL** | 31% |
| Q4_0 | +62.02 PPL | **+32.6 PPL** | 28% |

*PPL delta vs F16 baseline on Qwen3 0.6B BF16 — source: llama.cpp PR #21038.*

### Measured KV-cache space savings

Numbers below come from a benchmark run against **Qwen2.5-0.5B-Instruct**
(24 layers, 2 KV heads, 64 head-dim), obtained by calling `ggml_row_size()`
directly against the compiled GGML library in this repo's build tree.

```
Model : Qwen2.5-0.5B-Instruct  (24 layers, 2 KV heads, 64 head-dim)

Config                 B/row  B/elem     KV @2K      KV @32K  Saved@32K  Ratio
--------------------  ------  ------  ---------  ----------  ---------  -----
F16  (baseline)          128  2.0000   24.00 MB   384.00 MB      —       1.00x
Q8_0 + TurboQuant         68  1.0625   12.75 MB   204.00 MB  180.0 MB   1.88x
Q5_1 + TurboQuant         48  0.7500    9.00 MB   144.00 MB  240.0 MB   2.67x
Q5_0 + TurboQuant         44  0.6875    8.25 MB   132.00 MB  252.0 MB   2.91x  ← sweet spot
Q4_1 + TurboQuant         40  0.6250    7.50 MB   120.00 MB  264.0 MB   3.20x
Q4_0 + TurboQuant         36  0.5625    6.75 MB   108.00 MB  276.0 MB   3.56x
```

The ratios are pure GGML block geometry and **scale identically to larger
models** — for a 7B model (32 layers, 8 KV heads, 128 head-dim) multiply
every MB figure by ~85×; the ratios and % savings are the same.

#### Sweet spot: Q5_0 + TurboQuant

- **2.91× smaller** KV cache than vanilla F16 (saves **252 MB per 32 K
  context window** on the 0.5B model, ~21 GB on a 70B model at 32 K ctx)
- Only **+0.55 PPL** delta — essentially indistinguishable from F16 in practice
- The same Q5_0 *without* TurboQuant gives +17.28 PPL (noticeably wrong output)
- Q8_0 is the conservative zero-risk choice (1.88×, near-zero PPL cost)
- Q4_0 gives maximum compression (3.56×) at the price of measurable but
  tolerable quality loss with rotation on

### Key properties

- **Enabled automatically** for any model whose head dimension is a power of two
  (covers essentially all modern transformers).
- **No GGUF changes required** — it is a runtime transform of the KV cache only.
- **Reversible** — the rotation is applied before storing and reversed before
  computing attention, so results are mathematically identical to F16.
- **Controlled via the `LLAMA_ATTN_ROT_DISABLE` env var** — set to `1` to opt out.

### Using TurboQuant from Rust

```rust
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::quantize::GgmlType;

// TurboQuant is ON by default — just set a quantized KV cache type:
let ctx_params = LlamaContextParams::default()
    .with_cache_type_k(GgmlType::Q5_0)   // ~31% of F16 VRAM
    .with_cache_type_v(GgmlType::Q5_0);  // quality ≈ F16 thanks to rotation

let ctx = model.new_context(&backend, ctx_params)?;
```

```rust
// Disable rotation for a single context (e.g. benchmarking baseline):
let ctx_params = LlamaContextParams::default()
    .with_cache_type_k(GgmlType::Q5_0)
    .with_attn_rot_disabled(true);   // ← TurboQuant OFF for this context

let ctx = model.new_context(&backend, ctx_params)?;
```

```rust
// Global process-level toggle (call before creating any context):
use llama_cpp_4::quantize::{attn_rot_disabled, set_attn_rot_disabled};

set_attn_rot_disabled(true);
assert!(attn_rot_disabled());

set_attn_rot_disabled(false); // restore
```

### Live demo

```bash
# API reference + PPL table (no model required)
cargo run -p turbo-quant -- --show-api

# Run both passes and compare outputs directly
cargo run -p turbo-quant -- \
    --model model.gguf \
    --kv-type q5_0 \
    --prompt "The capital of France is" \
    --n-predict 16
```

---

## GPU acceleration

| Feature | Hardware | Flag |
|---|---|---|
| `cuda` | NVIDIA (CUDA) | `--features cuda` |
| `metal` | Apple Silicon | `--features metal` |
| `vulkan` | AMD / Intel / cross-platform | `--features vulkan` |
| `native` | CPU with AVX2/NEON auto-detect | `--features native` |
| `openmp` | Multi-core CPU (default on) | `--features openmp` |
| `rpc` | Remote compute backend | `--features rpc` |

```bash
# Metal (macOS)
cargo run -p openai-server --features metal -- --n-gpu-layers 99 \
    local model.gguf

# CUDA (Linux/Windows)
cargo run -p openai-server --features cuda -- --n-gpu-layers 99 \
    local model.gguf

# Vulkan (cross-platform)
cargo run -p openai-server --features vulkan -- --n-gpu-layers 99 \
    hf-model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## Hugging Face model download

All examples and the server accept a `hf-model <repo> [quant]` subcommand
that downloads models from the Hub (cached in `~/.cache/huggingface/`).

```bash
# Interactive quant picker for repos with many options
cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF

# Select by quant name (downloads all shards automatically)
cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF Q4_K_M

# Exact filename
cargo run -p openai-server -- \
    hf-model TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf
```

Set `HUGGING_FACE_HUB_TOKEN` for gated models.

---

## Development

```bash
# Clone with submodules (llama.cpp is a submodule of llama-cpp-sys-4)
git clone --recursive https://github.com/eugenehp/llama-cpp-rs

# Or after cloning without --recursive
git submodule update --init --recursive

# Build everything
cargo build

# Run all unit tests (no model required)
cargo test

# Run server unit tests specifically
cargo test -p openai-server
```

### Updating llama.cpp

```bash
cd llama-cpp-sys-4/llama.cpp
git fetch origin master
git checkout origin/master  # or a specific commit/tag
cd ../..
cargo build          # build.rs regenerates bindings automatically
```

---

## Multimodal Images

### Via the OpenAI-compatible server

```shell
cargo run -p openai-server --features mtmd --release -- \
    hf-model unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_0
```

Or with an explicit mmproj path:

```shell
cargo run -p openai-server --features mtmd -- \
    --mmproj mmproj-BF16.gguf \
    hf-model unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_0
```

### Standalone multimodal example

```shell
cargo run --features mtmd -p mtmd -- \
    --model /path/to/model.gguf \
    --mmproj /path/to/mmproj.gguf \
    --image /path/to/image.jpg \
    --prompt "Describe this image."
```

---

## Credits

Originally derived from [llama-cpp-2](https://crates.io/crates/llama-cpp-2) — thanks to those contributors.  
See also [bitnet-cpp-rs](https://github.com/eugenehp/bitnet-cpp-rs) for highly-quantized BitNet model support.

## Citation

```bibtex
@software{hauptmann2025llamacpprs,
  author    = {Hauptmann, Eugene},
  title     = {{llama-cpp-4}: llama-cpp {Rust} wrapper},
  year      = {2025},
  version   = {0.2.18},
  url       = {https://github.com/eugenehp/llama-cpp-rs},
}
```

## License

This project is licensed under the [MIT License](/LICENSE).

## Copyright

© 2025-2026, Eugene Hauptmann
