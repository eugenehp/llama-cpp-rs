//! Minimal text-generation core for the Android example, plus the JNI bridge.
//!
//! [`generate`] is a self-contained "load a GGUF, run greedy decoding, return
//! the text" helper built on the safe [`llama_cpp_4`] API. It is deliberately
//! platform-agnostic so the exact same code path is exercised three ways:
//!
//! * on Android (`aarch64-linux-android`) through the JNI wrapper below,
//! * on `aarch64-unknown-linux-gnu` under QEMU by the `smoke` binary in CI,
//! * natively on the host when you `cargo run --bin smoke`.

use std::num::NonZeroU32;
use std::path::Path;
use std::pin::pin;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use llama_cpp_4::prelude::*;

/// llama.cpp permits exactly one backend init per process; a JNI library may be
/// called many times over an app's lifetime, so we memoise it.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend> {
    // `get_or_try_init` is unstable, so init eagerly and cache on success.
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let backend = LlamaBackend::init().context("failed to init llama backend")?;
    Ok(BACKEND.get_or_init(|| backend))
}

/// Load `model_path`, greedily generate up to `max_new_tokens` from `prompt`,
/// and return the generated text (prompt excluded).
///
/// Errors are returned rather than panicking so callers (JNI, CLI) can surface
/// them cleanly.
pub fn generate(model_path: &Path, prompt: &str, max_new_tokens: i32) -> Result<String> {
    let backend = backend()?;

    let model_params = pin!(LlamaModelParams::default());
    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .with_context(|| format!("failed to load model: {}", model_path.display()))?;

    // A small context is plenty for the demo and keeps memory low on-device.
    let n_ctx = NonZeroU32::new(512);
    let ctx_params = LlamaContextParams::default().with_n_ctx(n_ctx);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("failed to create llama context")?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize prompt: {prompt:?}"))?;
    anyhow::ensure!(!tokens.is_empty(), "prompt tokenized to zero tokens");

    // Prefill the prompt; only the last token needs logits.
    let mut batch = LlamaBatch::new(512, 1);
    let last = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch.add(token, i, &[0], i == last)?;
    }
    ctx.decode(&mut batch).context("prompt decode failed")?;

    // Greedy decode loop. Accumulate raw bytes and lossily decode once at the
    // end so multi-byte tokens split across steps still render.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut out = Vec::<u8>::new();
    let mut n_cur = batch.n_tokens();

    for _ in 0..max_new_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        out.extend_from_slice(&model.token_to_bytes(token, Special::Tokenize)?);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch).context("token decode failed")?;
    }

    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// JNI bridge for `com.example.llama.LlamaBridge.generate(...)`.
///
/// Compiled only for Android targets so the host and `aarch64-linux-gnu`
/// (QEMU) builds never depend on the `jni` crate.
#[cfg(target_os = "android")]
mod jni_bridge {
    use super::generate;
    use jni::objects::{JClass, JString};
    use jni::sys::{jint, jstring};
    use jni::JNIEnv;
    use std::path::Path;

    /// `external fun generate(modelPath: String, prompt: String, maxNewTokens: Int): String`
    #[no_mangle]
    pub extern "system" fn Java_com_example_llama_LlamaBridge_generate<'local>(
        mut env: JNIEnv<'local>,
        _class: JClass<'local>,
        model_path: JString<'local>,
        prompt: JString<'local>,
        max_new_tokens: jint,
    ) -> jstring {
        let model_path: String = match env.get_string(&model_path) {
            Ok(s) => s.into(),
            Err(e) => return throw_back(&mut env, format!("bad modelPath: {e}")),
        };
        let prompt: String = match env.get_string(&prompt) {
            Ok(s) => s.into(),
            Err(e) => return throw_back(&mut env, format!("bad prompt: {e}")),
        };

        let text = match generate(Path::new(&model_path), &prompt, max_new_tokens) {
            Ok(text) => text,
            // Surface the error to the app as the returned string rather than
            // unwinding across the FFI boundary.
            Err(e) => format!("error: {e:#}"),
        };

        env.new_string(text)
            .map(|s| s.into_raw())
            .unwrap_or(std::ptr::null_mut())
    }

    fn throw_back(env: &mut JNIEnv, msg: String) -> jstring {
        env.new_string(format!("error: {msg}"))
            .map(|s| s.into_raw())
            .unwrap_or(std::ptr::null_mut())
    }
}
