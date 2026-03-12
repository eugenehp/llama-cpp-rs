# 🦙 llama-cpp-rs

[![Crates.io](https://img.shields.io/crates/v/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)
[![docs.rs](https://img.shields.io/docsrs/llama-cpp-4.svg)](https://docs.rs/llama-cpp-4)
[![License](https://img.shields.io/crates/l/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)

Safe Rust bindings to [llama.cpp](https://github.com/ggerganov/llama.cpp), tracking upstream closely.

| Crate | Description | crates.io |
|---|---|---|
| [`llama-cpp-4`](llama-cpp-4/) | Safe high-level API | [![](https://img.shields.io/crates/v/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4) |
| [`llama-cpp-sys-4`](llama-cpp-sys-4/) | Raw bindgen bindings | [![](https://img.shields.io/crates/v/llama-cpp-sys-4.svg)](https://crates.io/crates/llama-cpp-sys-4) |

**llama.cpp version:** b8249 (March 2026)

---

## Examples

| Package name | Directory | Description |
|---|---|---|
| `simple` | [`examples/simple/`](examples/simple/) | Single-turn text completion from CLI or Hugging Face |
| `chat` | [`examples/chat/`](examples/chat/) | Interactive multi-turn chat REPL |
| `embeddings` | [`examples/embeddings/`](examples/embeddings/) | Batch embedding with cosine similarity |
| `split-model-example` | [`examples/split_model/`](examples/split_model/) | Load sharded / split GGUF files |
| `openai-server` | [`examples/server/`](examples/server/) | OpenAI-compatible HTTP server with streaming and tool calling |
| `rpc-example` | [`examples/rpc/`](examples/rpc/) | Remote procedure call backend |
| `mtmd` | [`examples/mtmd/`](examples/mtmd/) | Multimodal (vision / audio) inference (requires `--features mtmd`) |

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
git fetch --tags
git checkout b8249   # or latest tag
cd ../..
cargo build          # build.rs regenerates bindings automatically
```

---

### Quick test

```shell
cargo run -p openai-server -- hf-model unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_0
```

Send request via CURL:

```shell
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"stream":false,"max_tokens":500}'
```

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

```shell
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"messages": [{
"role": "user",
"content": [
{"type": "text", "text": "What is in this picture?"},
{"type": "image_url", "image_url": {"url":
"https://upload.wikimedia.org/wikipedia/commons/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg"}}
]
}],
"max_tokens": 256
}'
```

Working end to end:

```shell
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"messages": [{
"role": "user",
"content": [
{"type": "text", "text": "What is in this picture?"},
{"type": "image_url", "image_url": {"url":
"https://upload.wikimedia.org/wikipedia/commons/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg"}}
]
}],
"max_tokens": 256
}'
{"choices":[{"finish_reason":"length","index":0,"message":{"content":"<think>\n\n</think>\n\nThis picture feature a **ginger and white cat** sitting upright outdoor on a bed of dry, brown leaves.\n\nHere’s a detaile breakdo:\n\n- **Subject**: The main focus is the cat, which has a classic “orange tabby and white” coat pattern.\n  - Its fur is mostly ginger (orange) with distinc darker orange stripes — typical of a tabby.\n  - It has a white face mask around its nose and mouth, a white chest and belly, and white paws (“socks”).\n  - Its eyes are strikin greenish-yellow, and it’s looking directl at the camera with an alert, calm express.\n  - The ears are perked up, and whiskers are clearly visible.\n\n- **Setting**: The cat is seated on the ground covered in fallen, dried leaves — suggest an autumn or early winter scene, possibl in a garden or wooded area.\n  - The backgro is softly blurred (shallow depth of field), drawing attenti to the cat while hinting at more foliage or grass behind it.\n\n- **Photography Style**: The image is well-composed, with natural lightin that highlig the texture of the cat’s fur and the rich colors. The bokeh effect","role":"assistant"}}],"created":1773097066,"id":"chatcmpl-1773097066","model":"Qwen3.5-27B-Q4_0","object":"chat.completion","usage":{"completion_tokens":256,"prompt_tokens":0,"total_tokens":256}}% 
```

---

## Credits

Originally derived from [llama-cpp-2](https://crates.io/crates/llama-cpp-2) — thanks to those contributors.  
See also [bitnet-cpp-rs](https://github.com/eugenehp/bitnet-cpp-rs) for highly-quantized BitNet model support.

## Citation

If you use gpu-fft in academic work, please cite it as:

**BibTeX**
```bibtex
@software{hauptmann2025llamacpprs,
  author    = {Hauptmann, Eugene},
  title     = {{llama-cpp-4}: llama-cpp {Rust} wrapper},
  year      = {2025},
  version   = {0.2.9},
  url       = {https://github.com/eugenehp/llama-cpp-rs},
}
```

**Plain text (APA)**
> Hauptmann, E. (2025). *llama-cpp-4: llama-cpp Rust wrapper* (v0.2.9).
> https://github.com/eugenehp/llama-cpp-rs

## License

This project is licensed under the [MIT License](/LICENSE).

## Copyright

© 2025-2026, Eugene Hauptmann

