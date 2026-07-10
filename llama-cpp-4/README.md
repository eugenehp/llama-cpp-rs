# llama-cpp-4

[![Crates.io](https://img.shields.io/crates/v/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)
[![docs.rs](https://img.shields.io/docsrs/llama-cpp-4.svg)](https://docs.rs/llama-cpp-4)
[![License](https://img.shields.io/crates/l/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)

Safe Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp).
Tracks upstream closely — designed to stay current rather than provide a thick abstraction layer.

**llama.cpp version:** `4fc4ec5 (b9859)` · **Crate version:** 0.4.0

---

## Add to your project

```toml
[dependencies]
llama-cpp-4 = "0.4.0"

# GPU support (pick one or more)
# llama-cpp-4 = { version = "0.4.0", features = ["cuda"] }
# llama-cpp-4 = { version = "0.4.0", features = ["metal"] }
# llama-cpp-4 = { version = "0.4.0", features = ["vulkan"] }
```

---

## Prelude

Import the common inference types in one line:

```rust
use llama_cpp_4::prelude::*;
```

The prelude re-exports backend, model, context, batching, sampling, errors, fit/memory
helpers, speculative-decoding types, and quantization symbols. The same core types are
also at the crate root (`llama_cpp_4::LlamaModel`, etc.) if you prefer explicit paths.

| Category | Types |
|---|---|
| Inference | `LlamaBackend`, `LlamaModel`, `LlamaContext`, `LlamaBatch`, `LlamaSampler`, `LlamaSamplerParams`, `LlamaToken`, `LlamaTokenDataArray` |
| Tokenising | `AddBos`, `Special` |
| Chat | `LlamaChatMessage` |
| Model introspection | `LlamaBackendDevice`, `LlamaBackendDeviceType` |
| Context tuning | `LlamaFlashAttnType`, `LlamaContextType`, `LlamaAttentionType`, `RopeScalingType`, `ParamsCloneError` |
| KV overrides | `ParamOverrideValue` |
| Memory / fit | `get_device_memory_data`, `fit_params`, `FitParams`, `MemoryBreakdownEntry` |
| Tensor capture | `TensorCapture`, `CapturedTensor` |
| Speculative | `MtpSession`, `Eagle3Session` (+ configs) |
| Quantization | `QuantizeParams`, `TensorTypeOverride`, `GgmlType`, `LlamaFtype`, `model_quantize` |

See [`prelude`](src/prelude.rs) on docs.rs for runnable examples (generation, chat, embeddings, memory estimation).

---

## Feature flags

| Feature | Default | Description |
|---|---|---|
| `openmp` | ✅ | Multi-threaded CPU inference via OpenMP |
| `mtmd` | ✅ | Multimodal (vision / audio) via `libmtmd` |
| `dynamic-link` | ✅ | Link llama.cpp as a shared library |
| `cuda` | | NVIDIA GPU via CUDA |
| `metal` | | Apple GPU via Metal |
| `vulkan` | | Cross-platform GPU via Vulkan |
| `native` | | CPU auto-tune for current arch (AVX2, NEON, …) |
| `rpc` | | Remote compute backend |

---

## API overview

All snippets below assume `use llama_cpp_4::prelude::*;`.

### Backend

```rust
// Initialise once per process. Configures hardware backends (CUDA, Metal, …).
let backend = LlamaBackend::init()?;
```

### Loading a model

```rust
use std::pin::pin;

let mut params = LlamaModelParams::default().with_n_gpu_layers(99);
let params = pin!(params);

let model = LlamaModel::load_from_file(&backend, "model.gguf", &params)?;

println!("vocab size : {}", model.n_vocab());
println!("context len: {}", model.n_ctx_train());
println!("embed dim  : {}", model.n_embd());

// Multi-GPU / MoE introspection
println!("devices    : {}", model.n_devices());
println!("experts    : {}", model.n_expert());
for dev in model.devices() {
    let (free, total) = dev.memory();
    println!("  {} — {} / {} bytes free", dev.name()?, free, total);
}
```

### Memory estimation (before full load)

```rust
use std::path::Path;

let report = get_device_memory_data(
    Path::new("model.gguf"),
    &LlamaModelParams::default().with_n_gpu_layers(99),
    &LlamaContextParams::default(),
    llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
)?;
for entry in &report.entries {
    println!("projected: {} bytes", entry.used());
}
```

### Auto-fit parameters to device memory

```rust
use llama_cpp_4::fit::{fit_params, FitParams};

let backend = LlamaBackend::init()?;
let fitted = fit_params(
    &backend,
    Path::new("model.gguf"),
    FitParams::default().with_n_ctx_min(512),
)?;

let model = LlamaModel::load_from_file(&backend, "model.gguf", &fitted.model_params)?;
let ctx = model.new_context(&backend, fitted.context_params)?;
```

### Tokenising

```rust
let tokens = model.str_to_token("Hello, world!", AddBos::Always)?;
let text   = model.token_to_str(tokens[0], Special::Plaintext)?;
let bytes  = model.token_to_bytes(tokens[0], Special::Plaintext)?;
```

### Chat template

```rust
let messages = vec![
    LlamaChatMessage::new("system".into(), "You are helpful.".into())?,
    LlamaChatMessage::new("user".into(),   "What is 2+2?".into())?,
];
let prompt = model.apply_chat_template(None, messages, true)?;
```

### Creating a context

```rust
use std::num::NonZeroU32;

let params = LlamaContextParams::default()
    .with_n_ctx(NonZeroU32::new(4096))
    .with_n_batch(512)
    .with_n_threads(8)
    .with_flash_attn_type(LlamaFlashAttnType::Auto);

let mut ctx = model.new_context(&backend, params)?;
```

### Batched decode (prefill + generation)

```rust
let mut batch = LlamaBatch::new(512, 1);

for (i, &tok) in tokens.iter().enumerate() {
    let last = i == tokens.len() - 1;
    batch.add(tok, i as i32, &[0], last)?;
}
ctx.decode(&mut batch)?;

batch.clear();
batch.add(new_token, pos, &[0], true)?;
ctx.decode(&mut batch)?;
```

### Sampling

```rust
let sampler = LlamaSampler::chain_simple([
    LlamaSampler::top_k(40),
    LlamaSampler::top_p(0.95, 1),
    LlamaSampler::temp(0.8),
    LlamaSampler::dist(42),
]);

let token = sampler.sample(&ctx, batch.n_tokens() - 1);
if model.is_eog_token(token) { /* done */ }
let bytes = model.token_to_bytes(token, Special::Plaintext)?;
```

### KV cache

```rust
ctx.clear_kv_cache_seq(Some(0), None, None)?; // clear sequence 0
ctx.clear_kv_cache();                          // clear all sequences
```

### Embeddings

```rust
use std::num::NonZeroU32;

let params = LlamaContextParams::default()
    .with_embeddings(true)
    .with_n_ctx(NonZeroU32::new(512));
let mut ctx = model.new_context(&backend, params)?;

// ... fill batch, decode ...
let vec = ctx.embeddings_seq_ith(0)?;
```

### Runtime memory breakdown

```rust
for entry in ctx.memory_breakdown() {
    println!("{}: {} bytes", entry.buft_name, entry.total());
}
```

### Tensor capture (hidden states)

Hook `cb_eval` during decode to copy per-layer outputs (`"l_out-N"`) or other
named graph nodes:

```rust
use llama_cpp_4::prelude::*;

let mut capture = TensorCapture::for_layers(&[13, 20, 27]);
let ctx_params = LlamaContextParams::default().with_tensor_capture(&mut capture);
let mut ctx = model.new_context(&backend, ctx_params)?;

// ... fill batch, decode ...
ctx.decode(&mut batch)?;

if let Some(layer) = capture.get_layer(13) {
    println!("{} tokens × {} dims", layer.n_tokens(), layer.n_embd());
    let hidden = layer.token_embedding(0).unwrap();
}
```

See also [`context::tensor_capture`](src/context/tensor_capture.rs) and
`examples/eagle` (EAGLE-3 uses specific anchor layers).

### LoRA adapters

```rust
let adapter = model.load_lora_adapter("adapter.gguf", 1.0)?;
ctx.set_lora_adapter(&adapter, 1.0)?;
ctx.lora_adapter_remove()?;
```

### Performance counters

```rust
let perf = ctx.timings();
println!("prompt eval: {:.2} ms", perf.t_p_eval_ms());
ctx.perf_context_reset();
```

---

## Full example: text generation

```rust
use llama_cpp_4::prelude::*;
use std::num::NonZeroU32;

fn main() -> anyhow::Result<()> {
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(
        &backend,
        "model.gguf",
        &LlamaModelParams::default(),
    )?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048));
    let mut ctx = model.new_context(&backend, ctx_params)?;

    let tokens = model.str_to_token("The answer is", AddBos::Always)?;
    let n_prompt = tokens.len();

    let mut batch = LlamaBatch::new(2048, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch.add(tok, i as i32, &[0], i == n_prompt - 1)?;
    }
    ctx.decode(&mut batch)?;

    let sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.8),
        LlamaSampler::dist(0),
    ]);

    let mut pos = n_prompt as i32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    for _ in 0..256 {
        let token = sampler.sample(&ctx, 0);
        if model.is_eog_token(token) {
            break;
        }

        let bytes = model.token_to_bytes(token, Special::Plaintext)?;
        let mut piece = String::new();
        decoder.decode_to_string(&bytes, &mut piece, false);
        print!("{piece}");

        batch.clear();
        batch.add(token, pos, &[0], true)?;
        ctx.decode(&mut batch)?;
        pos += 1;
    }
    Ok(())
}
```

---

## Safety

This crate wraps a C++ library via FFI. The safe API prevents most misuse, but
some patterns (e.g. using a context after its model is dropped) can still cause
UB. File an issue if you spot any.

## Examples in this repo

| Crate | Description |
|---|---|
| [`simple`](../examples/simple/) | Single-turn completion |
| [`chat`](../examples/chat/) | Interactive multi-turn REPL |
| [`openai-server`](../examples/server/) | OpenAI-compatible HTTP API |
| [`mtp`](../examples/mtp/) | MTP speculative decoding |
| [`eagle`](../examples/eagle/) | EAGLE-3 speculative decoding |
| [`incremental-chat`](../examples/incremental-chat/) | Incremental prefill while typing |
| [`fit-params`](../examples/fit-params/) | Auto-fit `n_ctx` / GPU layers to device memory |

---

## Requirements

- Rust 1.75+
- `clang` (for bindgen at build time)
- A C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- For CUDA: CUDA toolkit 11.8+
- For Metal: Xcode 14+

---

## Testing

Unit tests run without a model (vocab-only fixtures from the build tree when available):

```bash
cargo test -p llama-cpp-4
```

End-to-end integration tests load a real GGUF and exercise decode, generation, embeddings,
memory breakdown, fit helpers, and tensor capture:

```bash
./scripts/fetch-test-model.sh
cargo test -p llama-cpp-4 --test test_integration -- --test-threads=1
```

Or point at any local checkpoint:

```bash
LLAMA_TEST_MODEL=/path/to/model.gguf \
  cargo test -p llama-cpp-4 --test test_integration -- --test-threads=1
```

Use `--test-threads=1` because `llama_decode` is not safe to exercise in parallel across
contexts in the same process.
