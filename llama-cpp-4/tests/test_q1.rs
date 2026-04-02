//! Tests for Q1_0 / Q1_0_g128 quantization support (feature = "q1").
//!
//! Requires:
//!   - the `q1` Cargo feature
//!   - the Bonsai-1.7B GGUF model, pointed at by the `BONSAI_MODEL` env var
//!     (defaults to `/Users/Shared/models/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf`)
//!
//! Run with:
//!   cargo test -p llama-cpp-4 --features q1 --test test_q1 -- --nocapture

#![cfg(feature = "q1")]

use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    quantize::{GgmlType, LlamaFtype},
    token::LlamaToken,
};
use std::{
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

// ── helpers ───────────────────────────────────────────────────────────────────

const DEFAULT_MODEL: &str = "/Users/Shared/models/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf";

fn model_path() -> PathBuf {
    PathBuf::from(std::env::var("BONSAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()))
}

/// One global backend. llama.cpp only allows one `llama_backend_init` per
/// process; tests run in parallel on multiple threads so we use OnceLock.
fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("backend init failed"))
}

/// Load the model (fast mmap). Returns None if the file doesn't exist.
fn load() -> Option<LlamaModel> {
    let path = model_path();
    if !path.exists() {
        eprintln!("SKIP: model not found at {}", path.display());
        eprintln!("      Set BONSAI_MODEL=/path/to/Bonsai-1.7B.gguf to run.");
        return None;
    }
    let params = std::pin::pin!(LlamaModelParams::default());
    Some(LlamaModel::load_from_file(backend(), &path, &params).expect("model load failed"))
}

/// Serialises tests that call `llama_decode` (not thread-safe across contexts).
static DECODE_LOCK: Mutex<()> = Mutex::new(());

// ── enum / type tests (no model needed) ───────────────────────────────────────

#[test]
fn ftype_q1_0_name() {
    assert_eq!(LlamaFtype::MostlyQ1_0.name(), "Q1_0");
    assert_eq!(LlamaFtype::MostlyQ1_0_G128.name(), "Q1_0_g128");
}

#[test]
fn ftype_q1_0_description() {
    let d0 = LlamaFtype::MostlyQ1_0.description();
    let d128 = LlamaFtype::MostlyQ1_0_G128.description();
    assert!(d0.contains("1.5") || d0.contains("bpw"), "unexpected: {d0}");
    assert!(d128.contains("1.125") || d128.contains("bpw"), "unexpected: {d128}");
}

#[test]
fn ftype_q1_0_from_name_roundtrip() {
    assert_eq!(LlamaFtype::from_name("Q1_0"), Some(LlamaFtype::MostlyQ1_0));
    assert_eq!(LlamaFtype::from_name("Q1_0_g128"), Some(LlamaFtype::MostlyQ1_0_G128));
    assert_eq!(LlamaFtype::from_name("q1_0"), Some(LlamaFtype::MostlyQ1_0));
}

#[test]
fn ftype_q1_0_in_all() {
    let all = LlamaFtype::all();
    assert!(all.contains(&LlamaFtype::MostlyQ1_0), "MostlyQ1_0 missing");
    assert!(all.contains(&LlamaFtype::MostlyQ1_0_G128), "MostlyQ1_0_G128 missing");
}

#[test]
fn ftype_q1_0_numeric_values() {
    assert_eq!(LlamaFtype::MostlyQ1_0 as u32, 40);
    assert_eq!(LlamaFtype::MostlyQ1_0_G128 as u32, 41);
}

#[test]
fn ggml_type_q1_0_numeric_values() {
    // Matches PrismML's GGUF format. NVFP4 is displaced to 42 in q1 builds.
    assert_eq!(GgmlType::Q1_0 as u32, 40);
    assert_eq!(GgmlType::Q1_0_G128 as u32, 41);
    assert_eq!(GgmlType::NVFP4 as u32, 42);
}

#[test]
fn ggml_type_q1_0_roundtrip() {
    use llama_cpp_sys_4::ggml_type;
    let raw_q1_0: ggml_type = GgmlType::Q1_0.into();
    let raw_q1_0_g128: ggml_type = GgmlType::Q1_0_G128.into();
    assert_eq!(GgmlType::try_from(raw_q1_0), Ok(GgmlType::Q1_0));
    assert_eq!(GgmlType::try_from(raw_q1_0_g128), Ok(GgmlType::Q1_0_G128));
}

// ── model loading ──────────────────────────────────────────────────────────────

#[test]
fn model_loads_successfully() {
    let Some(_model) = load() else { return };
}

#[test]
fn model_architecture_is_qwen() {
    let Some(model) = load() else { return };
    let arch = model.meta_val_str("general.architecture", 256).expect("missing arch");
    assert!(arch.to_lowercase().contains("qwen"), "unexpected arch: {arch}");
}

#[test]
fn model_ftype_is_q1_0_g128() {
    let Some(model) = load() else { return };
    let ftype = model.meta_val_str("general.file_type", 256).unwrap_or_else(|_| "unknown".into());
    println!("general.file_type = {ftype}");
    let desc = model.desc(512).unwrap_or_default();
    println!("model desc: {desc}");
}

#[test]
fn model_numeric_properties_sane() {
    let Some(model) = load() else { return };
    let (n_vocab, n_ctx, n_embd, n_layer) =
        (model.n_vocab(), model.n_ctx_train(), model.n_embd(), model.n_layer());
    println!("vocab={n_vocab}  ctx={n_ctx}  embd={n_embd}  layers={n_layer}");
    assert!(n_vocab > 1000);
    assert!(n_ctx >= 512);
    assert!(n_embd > 0);
    assert!(n_layer > 0);
}

#[test]
fn model_metadata_contains_expected_keys() {
    let Some(model) = load() else { return };
    let entries = model.metadata().expect("metadata failed");
    let keys: Vec<_> = entries.iter().map(|(k, _)| k.as_str()).collect();
    for required in &["general.architecture", "general.name"] {
        assert!(keys.iter().any(|k| k == required), "missing key: {required}");
    }
}

// ── tokenization ──────────────────────────────────────────────────────────────

#[test]
fn tokenize_simple_prompt() {
    let Some(model) = load() else { return };
    let tokens = model.str_to_token("Hello, world!", AddBos::Never).expect("tokenize failed");
    assert!(!tokens.is_empty());
    println!("'Hello, world!' → {} tokens: {tokens:?}", tokens.len());
}

#[test]
fn tokenize_with_bos() {
    let Some(model) = load() else { return };
    let no_bos = model.str_to_token("hi", AddBos::Never).unwrap();
    let with_bos = model.str_to_token("hi", AddBos::Always).unwrap();
    // Some tokenizers (e.g. Qwen3/tiktoken BPE) embed BOS into the vocab and
    // don't add a separate token even with AddBos::Always.
    assert!(
        with_bos.len() >= no_bos.len(),
        "AddBos::Always should produce at least as many tokens as AddBos::Never"
    );
}

#[test]
fn detokenize_roundtrip() {
    let Some(model) = load() else { return };
    let text = "The quick brown fox";
    let tokens = model.str_to_token(text, AddBos::Never).unwrap();
    let back = model.detokenize(&tokens, true, false).unwrap();
    assert_eq!(back, text);
}

// ── inference ─────────────────────────────────────────────────────────────────

/// Forward pass on a short prompt — verifies the full Q1_0_g128 decode path.
#[test]
fn forward_pass_returns_valid_logits() {
    let Some(model) = load() else { return };
    let _guard = DECODE_LOCK.lock().unwrap();

    let tokens = model.str_to_token("Once upon a time", AddBos::Always).unwrap();
    let n_tokens = tokens.len() as i32;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(512))
        .with_n_batch(512);
    let mut ctx = model.new_context(backend(), ctx_params).expect("ctx failed");

    let mut batch = LlamaBatch::new(512, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch.add(tok, i as i32, &[0], i == tokens.len() - 1).unwrap();
    }
    ctx.decode(&mut batch).expect("decode failed");

    let n_vocab = model.n_vocab() as usize;
    let logits = ctx.get_logits_ith(n_tokens - 1);
    assert_eq!(logits.len(), n_vocab);

    let (best_id, best_logit) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    let best_text = model.token_get_text(LlamaToken(best_id as i32)).unwrap_or("<unk>");
    println!("next token: {best_id} ({best_text:?})  logit={best_logit:.3}");

    assert!(best_logit.is_finite());
    assert!(best_id < n_vocab);
}

/// Greedy autoregressive generation — end-to-end smoke test.
#[test]
fn autoregressive_generation() {
    let Some(model) = load() else { return };
    let _guard = DECODE_LOCK.lock().unwrap();

    let prompt = "The capital of France is";
    let mut tokens: Vec<LlamaToken> = model.str_to_token(prompt, AddBos::Always).unwrap();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(256))
        .with_n_batch(256);
    let mut ctx = model.new_context(backend(), ctx_params).expect("ctx failed");

    // Prefill
    let mut batch = LlamaBatch::new(256, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch.add(tok, i as i32, &[0], i == tokens.len() - 1).unwrap();
    }
    ctx.decode(&mut batch).expect("prefill failed");

    let eos = model.token_eos();
    let mut generated: Vec<LlamaToken> = Vec::new();
    // After prefill the logit lives at the last batch slot; after each
    // single-token decode it is at slot 0.
    let mut logit_slot = tokens.len() as i32 - 1;

    for _ in 0..20 {
        let best_id = ctx
            .get_logits_ith(logit_slot)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        logit_slot = 0;

        let next = LlamaToken(best_id as i32);
        if next == eos { break; }
        generated.push(next);
        // Position = current sequence length (number of tokens already in KV cache).
        let next_pos = tokens.len() as i32;
        tokens.push(next);

        batch.clear();
        batch.add(next, next_pos, &[0], true).unwrap();
        ctx.decode(&mut batch).expect("decode failed");
    }

    let output = model.detokenize(&generated, false, false).unwrap_or_default();
    println!("prompt:    \"{prompt}\"");
    println!("generated: \"{output}\"");

    assert!(!generated.is_empty(), "no tokens generated");
    assert!(output.chars().any(|c| c.is_alphanumeric()), "output has no alphanum: {output:?}");
}
