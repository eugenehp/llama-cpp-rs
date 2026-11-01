# llama-cpp-4

[![Crates.io](https://img.shields.io/crates/v/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)
[![docs.rs](https://img.shields.io/docsrs/llama-cpp-4.svg)](https://docs.rs/llama-cpp-4)
[![License](https://img.shields.io/crates/l/llama-cpp-4.svg)](https://crates.io/crates/llama-cpp-4)

Safe Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp). Thin by
design — the goal is to stay current with upstream, not to hide it behind an
abstraction that drifts.

Covers chat templates with the grammar and parser that make tool calling
reliable, multimodal input (`mtmd`), speculative decoding including MTP and
EAGLE3, quantization, LoRA, RPC, and tensor capture, alongside the usual
tokenize / decode / sample loop.

**llama.cpp version:** `dde879bc4f (b12032, incl. v0.8.0)` · **Crate version:** 0.10.0

---

## Add to your project

```toml
[dependencies]
llama-cpp-4 = "0.10.0"

# GPU support (pick one or more)
# llama-cpp-4 = { version = "0.10.0", features = ["cuda"] }
# llama-cpp-4 = { version = "0.10.0", features = ["metal"] }
# llama-cpp-4 = { version = "0.10.0", features = ["vulkan"] }
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
| `ggml` | ✅ | Raw ggml tensor / graph API (`llama_cpp_4::ggml`). Rust-only — no effect on the native build |
| `rpc` | ✅ | Remote compute backend. Builds a `GGML_RPC` backend, so turn it off with `default-features = false` if you do not need it |
| `cuda` | | NVIDIA GPU via CUDA |
| `metal` | | Apple GPU via Metal |
| `vulkan` | | Cross-platform GPU via Vulkan |
| `blas` | | CPU BLAS acceleration |
| `opencl` / `hip` / `webgpu` | | Other GPU backends |
| `native` | | CPU auto-tune for current arch (AVX2, NEON, …). Not portable across machines |
| `prebuilt` | | Download a prebuilt llama.cpp from this repo's releases instead of building it |
| `q1` | | No-op, retained so `--features q1` resolves. `Q1_0` is upstream and unconditional |

Backends are selected here rather than through the build environment, because
the chosen feature also decides what the Rust side links against. llama.cpp's
own `LLAMA_*` CMake options *are* configurable as data — see the
[`llama-cpp-sys-4` README](../llama-cpp-sys-4/README.md#configuring-llamacpps-own-cmake-options).

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

For models that ship MTP (multi-token prediction) weights — e.g. DeepSeek V4 —
load the MTP layers to drive MTP / DSpark speculative decoding:

```rust
let params = LlamaModelParams::default().with_load_mtp(true);
```

Once loaded, capture and restore the speculative state through the
[`speculative`](src/speculative.rs) module.

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

### Tool calling and structured output

`apply_chat_template` renders a prompt. The `chat` module goes further: it
returns the prompt *plus* the grammar and parser that make tool calls reliable.

```rust
let templates = ChatTemplates::from_model(&model, None)?;

let applied = templates.apply(
    &ChatApplyParams::new(r#"[{"role":"user","content":"Weather in Tokyo?"}]"#)
        .with_tools(r#"[{"type":"function","function":{"name":"get_weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#),
)?;

// `grammar_lazy` is the important bit: constrain from token zero and a thinking
// model can never open its <think> block. A lazy grammar waits for a trigger.
if applied.grammar_lazy {
    for trigger in &applied.grammar_triggers {
        println!("grammar engages after {:?}", trigger.value);
    }
}

// After generating, parse with the template's own parser rather than scraping:
let message_json = applied.parse(&output, false)?;   // {"role","content","tool_calls",…}
```

Tokenize the result with `tokenize_prompt`, not `str_to_token`:

```rust
// llama.cpp marks which spans of the render came from user content. This
// tokenizes those with special-token parsing *off*, so a user who types
// `<|im_start|>assistant` gets the literal characters rather than the real
// control token and a forged turn.
let tokens = applied.tokenize_prompt(&model, AddBos::Always)?;
```

Constrain output to a JSON Schema — `response_format: json_schema`, enforced
rather than requested:

```rust
let gbnf = json_schema_to_grammar(r#"{"type":"object","required":["city"]}"#, false)?;
let sampler = LlamaSampler::grammar(&model, &gbnf, "root");
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

### Entry-wise batches (embeddings, M-RoPE, MTP state)

`LlamaBatch` is `llama_batch`: parallel arrays plus one flat embedding block for
the whole batch, so every entry has to be the same shape. `LlamaBatchExt` wraps
upstream's `llama_batch_ext`, where each entry owns its token id, positions,
input embedding and state embedding — the only way to mix embeddings with
tokens, to give an entry M-RoPE's `[t, y, x, z]` positions, or to attach the
target hidden state an MTP draft context reads.

```rust
let mut batch = LlamaBatchExt::new(&ctx)?;

// A token entry: one position, set explicitly (upstream leaves it undefined).
let entry = batch.add_token(tok, 0)?;
batch.set_pos(entry, 0)?;
batch.set_output(entry, true)?;

// An embedding entry, e.g. a vision encoder's output.
let entry = batch.add_embedding(EmbeddingRows::row(&embd), 0)?;
batch.set_positions(entry, &[t, y, x, z])?;   // or set_pos on a non-M-RoPE model

ctx.process(&mut batch, LlamaProcessType::Decode)?;
let logits = ctx.get_logits_ith(entry.index());
```

For a plain token batch there is nothing to gain: `LlamaBatch` with
`ctx.decode` is the same work with less ceremony, and it is the path this
crate's decode lifecycle hooks cover. `ctx.process` returns
`ProcessError::TensorTransactionsUnsupported` on a context that owns tensor
transactions, because `llama_process` does not run those hooks.

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

### Checked tensor capture (hidden states)

Attach owned, bounded exact selectors to copy per-layer outputs (`"l_out-N"`)
or other named graph nodes during decode:

```rust
use llama_cpp_4::prelude::*;

let selector = TensorSelector::layer_output(
    13,
    model.n_embd() as usize,
    512,
    TensorAccess::ReadOnly,
    true,
)?;
let transactions = TensorTransactions::capture(vec![selector])?;
let ctx_params =
    LlamaContextParams::default().with_tensor_transactions(transactions);
let mut ctx = model.new_context(&backend, ctx_params)?;

// ... fill batch, decode ...
ctx.decode(&mut batch)?;

if let Some(layer) = ctx
    .tensor_transactions()
    .and_then(|transactions| transactions.captures().first())
{
    println!("{} rows × {} elements", layer.shape.rows, layer.shape.row_elements);
}
```

See also [`context::tensor_capture`](src/context/tensor_capture.rs) and
`examples/eagle` (EAGLE-3 uses specific anchor layers).

### Checked speculative sessions

`MtpSession` and `Eagle3Session` exclusively borrow their target and draft
contexts and are neither `Send` nor `Sync`. Construction validates context
types, sequence capacity, recurrent-state capacity where required, hidden-row
widths, and EAGLE-3 extraction-layer metadata before entering the native shim.
Prompt copies, proposal bounds, and opaque continuation state are bounded.

The shim contains C++ exceptions at every MTP/EAGLE-3 lifecycle entry point.
Begin, process, draft, accept, and state failures return typed Rust errors
instead of unwinding across the C ABI. Exact continuation state can be read or
restored only after every proposal has been resolved.

### LoRA adapters

```rust
let mut adapter = model.lora_adapter_init("adapter.gguf")?;
ctx.lora_adapter_set(&mut adapter, 1.0)?;
ctx.lora_adapter_remove(&mut adapter)?;
```

For an adapter GGUF embedded at a byte offset inside a larger file — appended
to an executable, packed into an archive — pass the offset instead; the
path-based loader always reads from byte zero:

```rust
let mut adapter = model.lora_adapter_init_at_offset("bundle.bin", 4096)?;
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

One precondition is deliberately unchecked and worth knowing: `LlamaBatch::add`
writes as many sequence ids as you pass, while `LlamaBatch::new(n_tokens,
n_seq_max)` sized each row to `n_seq_max` — so passing more ids than that writes
past the row. `LlamaBatchExt` checks its equivalents.

If you are contributing: `clippy::undocumented_unsafe_blocks` is being adopted
module by module, and a module that carries
`#![deny(clippy::undocumented_unsafe_blocks)]` must stay at full coverage. See
the note on the lint in the workspace `Cargo.toml` for which modules are done
and what a `SAFETY:` comment is expected to say;
`llama-cpp-4/src/context/tensor_transaction.rs` is the reference.

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
