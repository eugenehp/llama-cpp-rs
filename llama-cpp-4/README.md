# llama-cpp-4

[![Crates.io](https://img.shields.io/crates/v/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)
[![docs.rs](https://img.shields.io/docsrs/llama-cpp-4.svg)](https://docs.rs/llama-cpp-4)
[![License](https://img.shields.io/crates/l/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)

Safe Rust bindings to [llama.cpp](https://github.com/ggerganov/llama.cpp).  
Tracks upstream closely — designed to stay current rather than provide a thick abstraction layer.

**llama.cpp version:** b8249 · **Crate version:** 0.2.0

---

## Add to your project

```toml
[dependencies]
llama-cpp-4 = "0.2.0"

# GPU support (pick one or more)
# llama-cpp-4 = { version = "0.2.0", features = ["cuda"] }
# llama-cpp-4 = { version = "0.2.0", features = ["metal"] }
# llama-cpp-4 = { version = "0.2.0", features = ["vulkan"] }
```

---

## Feature flags

| Feature | Default | Description |
|---|---|---|
| `openmp` | ✅ | Multi-threaded CPU inference via OpenMP |
| `cuda` | | NVIDIA GPU via CUDA |
| `metal` | | Apple GPU via Metal |
| `vulkan` | | Cross-platform GPU via Vulkan |
| `native` | | CPU auto-tune for current arch (AVX2, NEON, …) |
| `rpc` | | Remote compute backend |
| `dynamic-link` | | Link llama.cpp as a shared library instead of static |

---

## API overview

### Backend

```rust
use llama_cpp_4::llama_backend::LlamaBackend;

// Initialise once per process. Configures hardware backends (CUDA, Metal, …).
let backend = LlamaBackend::init()?;
// Optional: suppress llama.cpp's stderr log spam
let backend = LlamaBackend::init_numa(NumaStrategy::Disabled)?;
```

### Loading a model

```rust
use llama_cpp_4::model::{params::LlamaModelParams, LlamaModel};

let mut params = LlamaModelParams::default();
params = params.with_n_gpu_layers(99); // offload all layers to GPU

let model = LlamaModel::load_from_file(&backend, "model.gguf", &params)?;

// Metadata
println!("vocab size : {}", model.n_vocab());
println!("context len: {}", model.n_ctx_train());
println!("embed dim  : {}", model.n_embd());

// Chat template (Jinja, if the model includes one)
if let Ok(tmpl) = model.get_chat_template(4096) {
    println!("template   : {tmpl}");
}
```

### Tokenising

```rust
use llama_cpp_4::model::{AddBos, Special};

let tokens = model.str_to_token("Hello, world!", AddBos::Always)?;
let text   = model.token_to_str(tokens[0], Special::Plaintext)?;
let bytes  = model.token_to_bytes(tokens[0], Special::Plaintext)?;

// Batch: token_to_piece is available on the context too
```

### Chat template

```rust
use llama_cpp_4::model::LlamaChatMessage;

let messages = vec![
    LlamaChatMessage::new("system".into(),    "You are helpful.".into())?,
    LlamaChatMessage::new("user".into(),      "What is 2+2?".into())?,
];
// Pass None to use the model's built-in template
let prompt = model.apply_chat_template(None, messages, true)?;
```

### Creating a context

```rust
use llama_cpp_4::context::params::LlamaContextParams;
use std::num::NonZeroU32;

let params = LlamaContextParams::default()
    .with_n_ctx(NonZeroU32::new(4096))
    .with_n_batch(512)
    .with_n_threads(8)
    .with_flash_attn(true);   // Flash Attention 2

let mut ctx = model.new_context(&backend, params)?;
```

### Batched decode (prefill + generation)

```rust
use llama_cpp_4::llama_batch::LlamaBatch;

let mut batch = LlamaBatch::new(512, 1);

// Add prompt tokens; only the last token needs logits
for (i, &tok) in tokens.iter().enumerate() {
    let last = i == tokens.len() - 1;
    batch.add(tok, i as i32, &[0], last)?;
}
ctx.decode(&mut batch)?;

// Generate one token at a time
batch.clear();
batch.add(new_token, pos, &[0], true)?;
ctx.decode(&mut batch)?;
```

### Sampling

```rust
use llama_cpp_4::sampling::LlamaSampler;

// Simple chain
let sampler = LlamaSampler::chain_simple([
    LlamaSampler::top_k(40),
    LlamaSampler::top_p(0.95, 1),
    LlamaSampler::temp(0.8),
    LlamaSampler::dist(/* seed */ 42),
]);

// Or greedy
let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

// Or GBNF grammar (constrained decoding)
let grammar = r#"root ::= "yes" | "no""#;
let sampler = LlamaSampler::chain_simple([
    LlamaSampler::grammar(&model, grammar, "root"),
    LlamaSampler::greedy(),
]);

// Sample a token
let token = sampler.sample(&ctx, batch.n_tokens() - 1);

// Check for end-of-generation
if model.is_eog_token(token) { break; }

// Decode to text
let bytes = model.token_to_bytes(token, Special::Plaintext)?;
```

### KV cache

```rust
// The KV cache is managed through the memory handle
use llama_cpp_4::context::kv_cache::SeqRm;

ctx.clear_kv_cache_seq(0, None, None)?; // clear sequence 0
```

### Embeddings

```rust
use llama_cpp_4::context::params::LlamaContextParams;

let params = LlamaContextParams::default()
    .with_embeddings(true)
    .with_n_ctx(NonZeroU32::new(512));
let mut ctx = model.new_context(&backend, params)?;

// ... fill batch, decode ...

// Pooled (sequence-level) embedding
let vec = ctx.embeddings_seq_ith(0)?;

// Per-token embedding
let vec = ctx.embeddings_ith(last_token_pos)?;
```

### LoRA adapters

```rust
let adapter = model.load_lora_adapter("adapter.gguf", 1.0)?;
ctx.set_lora_adapter(&adapter, 1.0)?;
// Remove all adapters
ctx.lora_adapter_remove()?;
```

### Performance counters

```rust
let perf = ctx.perf_context();
println!("prompt eval: {:.2} ms", perf.t_p_eval_ms);
println!("generation : {:.2} ms  ({:.1} t/s)", perf.t_eval_ms, ...);
ctx.perf_context_reset();
```

---

## Full example: text generation

```rust
use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
};
use std::num::NonZeroU32;

fn main() -> anyhow::Result<()> {
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(
        &backend, "model.gguf", &LlamaModelParams::default()
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
        if model.is_eog_token(token) { break; }

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

## Requirements

- Rust 1.75+
- `clang` (for bindgen at build time)
- A C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- For CUDA: CUDA toolkit 11.8+
- For Metal: Xcode 14+
