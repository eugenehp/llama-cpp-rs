//! Run `llama-cpp-4` inside a WebAssembly runtime.
//!
//! # Why this looks like an ordinary binary
//!
//! The browser target is `wasm32-unknown-emscripten`, not
//! `wasm32-unknown-unknown`: llama.cpp needs libc, libc++, pthreads and a
//! filesystem, and only Emscripten supplies them. That means Emscripten's own
//! JS glue rather than `wasm-bindgen`, and a plain `main()` that `emcc` wires
//! up — so the same source runs natively and in a tab.
//!
//! # Getting the model in
//!
//! There is no real filesystem. Emscripten offers a virtual one, and which
//! backend you mount decides how the GGUF arrives:
//!
//! | Flag | Where the bytes come from | Use |
//! |---|---|---|
//! | `-sNODERAWFS=1` | the host filesystem, directly | node, tests |
//! | `--preload-file m.gguf` | a `.data` file fetched next to the `.js` | browser demo |
//! | `WORKERFS` / `MEMFS` at runtime | a `fetch()` or `<input type=file>` | real browser app |
//!
//! All three end up as a path, which is why no special loader is needed here.
//! Note that `MEMFS` keeps the whole file in the *same* linear memory the
//! weights load into, so it costs roughly twice the model size; `WORKERFS`
//! does not and is the one to reach for in a browser.
//!
//! # Build
//!
//! ```bash
//! source /path/to/emsdk/emsdk_env.sh
//! rustup target add wasm32-unknown-emscripten
//!
//! RUSTFLAGS="-C link-arg=-sNODERAWFS=1 -C link-arg=-sALLOW_MEMORY_GROWTH=1 \
//!            -C link-arg=-sEXIT_RUNTIME=1 -C link-arg=-sTOTAL_STACK=8MB" \
//!   cargo build --release --target wasm32-unknown-emscripten -p wasm-example
//!
//! node target/wasm32-unknown-emscripten/release/llama-wasm.js model.gguf "Once upon"
//! ```
//!
//! For threads add `-C link-arg=-pthread -C link-arg=-sPTHREAD_POOL_SIZE=4`, and
//! serve the page with `Cross-Origin-Opener-Policy: same-origin` and
//! `Cross-Origin-Embedder-Policy: require-corp` — without those two headers the
//! browser withholds `SharedArrayBuffer` and you are single-threaded.

use std::io::Write;
use std::num::NonZeroU32;

use anyhow::{bail, Context, Result};
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaModel, Special};
use llama_cpp_4::sampling::LlamaSampler;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .context("usage: llama-wasm <model.gguf> [prompt] [n_tokens]")?;
    let prompt = args.next().unwrap_or_else(|| "Once upon a time".to_owned());
    let n_predict: u32 = args.next().as_deref().unwrap_or("24").parse()?;

    // Surfaces as a clear message instead of llama.cpp's load failure, because
    // "not found" in a virtual filesystem usually means the file was never
    // mounted rather than mistyped.
    if !std::path::Path::new(&model_path).is_file() {
        bail!(
            "{model_path} is not present in the Emscripten filesystem. \
             Mount it with -sNODERAWFS=1, --preload-file, or WORKERFS."
        );
    }

    let backend = LlamaBackend::init()?;
    report_backends();

    // mmap cannot work here: Emscripten's virtual filesystem has no pages to
    // map, so the loader must read the file instead. `Auto` would pick mmap
    // when the backend claims support, so say `None` explicitly.
    let model_params =
        std::pin::pin!(LlamaModelParams::default()
            .with_load_mode(llama_cpp_4::model::params::LlamaLoadMode::None));
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("failed to load {model_path}"))?;

    eprintln!(
        "model: {} params, n_embd={}, n_ctx_train={}",
        model.n_params(),
        model.n_embd(),
        model.n_ctx_train()
    );

    let tokens = model.str_to_token(&prompt, AddBos::Always)?;

    // Saturating rather than wrapping: a prompt long enough to overflow this
    // would fail the context allocation anyway, and should not quietly come
    // back around to a tiny n_ctx.
    let n_prompt = u32::try_from(tokens.len()).context("prompt too long for wasm32")?;
    let n_ctx = n_prompt.saturating_add(n_predict).max(64);

    let mut ctx = model
        .new_context(
            &backend,
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)),
        )
        .context("failed to create context")?;

    let mut batch = LlamaBatch::new(tokens.len().max(64), 1);
    let last = tokens.len() - 1;
    for (i, token) in tokens.iter().copied().enumerate() {
        // `i` is bounded by `n_prompt`, which already fit in a u32.
        let pos = i32::try_from(i).context("prompt too long for wasm32")?;
        batch.add(token, pos, &[0], i == last)?;
    }
    ctx.decode(&mut batch).context("prompt decode failed")?;

    print!("{prompt}");
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut n_cur = batch.n_tokens();
    let mut produced: u32 = 0;

    while produced < n_predict {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        print!("{}", model.token_to_str(token, Special::Plaintext)?);
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        produced += 1;
        ctx.decode(&mut batch).context("decode failed")?;
    }
    println!();

    let perf = ctx.timings();
    eprintln!(
        "decoded {produced} tokens; eval={} prompt_eval={}",
        perf.n_eval(),
        perf.n_p_eval()
    );
    Ok(())
}

/// Print what ggml found, which is the quickest way to tell whether the wasm
/// SIMD kernels and WebGPU are actually in this build.
fn report_backends() {
    eprintln!("system: {}", llama_cpp_4::print_system_info());
}
