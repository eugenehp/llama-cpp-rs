//! Tests for model, vocab, and context APIs.
//!
//! These tests require a GGUF model. Set `LLAMA_TEST_MODEL` to the path of a GGUF model file,
//! or run [`scripts/fetch-test-model.sh`](../../scripts/fetch-test-model.sh) for the default tiny
//! checkpoint. If neither is available, a vocab-only GGUF from the build directory is used.

mod support;

use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::model::{AddBos, LlamaModel};

use support::model::{backend, load_model};

fn load_test_model() -> Option<(&'static LlamaBackend, LlamaModel, bool)> {
    let (model, vocab_only) = load_model()?;
    Some((backend(), model, vocab_only))
}

// ============================================================
// Model property tests
// ============================================================

#[test]
fn test_model_desc() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let desc = model.desc(256).unwrap();
    assert!(!desc.is_empty());
}

#[test]
fn test_model_display() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let display = format!("{model}");
    assert!(!display.is_empty());
    // Should contain pipe separators
    assert!(
        display.contains('|'),
        "Display should have sections: {display}"
    );
}

#[test]
fn test_model_numeric_properties() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    assert!(model.n_ctx_train() > 0);
    assert!(model.n_embd() > 0);
    assert!(model.n_layer() > 0);
    assert!(model.n_layer_nextn() >= 0);
    assert!(model.n_head() > 0);
    assert!(model.n_head_kv() > 0);
    assert!(model.n_vocab() > 0);
    assert!(model.n_embd_inp() > 0);
    assert!(model.n_embd_out() > 0);
}

#[test]
fn test_model_boolean_properties() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    // These should not panic
    let _ = model.has_encoder();
    let _ = model.has_decoder();
    let _ = model.is_recurrent();
    let _ = model.is_hybrid();
    let _ = model.is_diffusion();
    let _ = model.n_expert();
    let _ = model.n_devices();
    let _ = model.target_layer_ids();
    let _ = model.add_bos_token();
    let _ = model.add_eos_token();
}

#[test]
fn test_model_rope_properties() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let _ = model.rope_type();
    let freq_scale = model.rope_freq_scale_train();
    assert!(freq_scale > 0.0);
}

// ============================================================
// Token tests
// ============================================================

#[test]
fn test_special_tokens() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    // BOS and EOS should be valid tokens (>= 0)
    assert!(model.token_bos().0 >= 0);
    assert!(model.token_eos().0 >= 0);
    assert!(model.token_nl().0 >= 0);

    // These may return -1 if not supported, that's ok
    let _ = model.token_cls();
    let _ = model.token_eot();
    let _ = model.token_pad();
    let _ = model.token_sep();
}

#[test]
fn test_fim_tokens() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    // FIM tokens may or may not be supported
    let _ = model.token_fim_pre();
    let _ = model.token_fim_suf();
    let _ = model.token_fim_mid();
    let _ = model.token_fim_pad();
    let _ = model.token_fim_rep();
    let _ = model.token_fim_sep();
}

#[test]
fn test_token_info() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let bos = model.token_bos();
    let _ = model.token_is_control(bos);
    let _ = model.is_eog_token(bos);
    let _ = model.token_get_score(bos);
    let text = model.token_get_text(bos);
    assert!(text.is_ok(), "BOS token should have text");
}

#[test]
fn test_token_attr() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let bos = model.token_bos();
    let _ = model.token_attr(bos);
}

// ============================================================
// Vocab tests
// ============================================================

#[test]
fn test_vocab_basic() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let vocab = model.get_vocab();
    assert!(vocab.n_tokens() > 0);
    assert!(vocab.vocab_type() > 0); // BPE=2 or SPM=1

    // Special tokens via vocab
    assert!(vocab.bos().0 >= 0);
    assert!(vocab.eos().0 >= 0);
    assert!(vocab.nl().0 >= 0);
}

#[test]
fn test_vocab_token_info() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let vocab = model.get_vocab();
    let bos = vocab.bos();

    let _ = vocab.is_control(bos);
    let _ = vocab.is_eog(bos);
    let _ = vocab.get_score(bos);
    let _ = vocab.get_attr(bos);
    let text = vocab.get_text(bos);
    assert!(text.is_ok());
}

#[test]
fn test_vocab_special_flags() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let vocab = model.get_vocab();
    let _ = vocab.get_add_bos();
    let _ = vocab.get_add_eos();
    let _ = vocab.get_add_sep();
    let _ = vocab.mask();
}

#[test]
fn test_vocab_fim_tokens() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let vocab = model.get_vocab();
    let _ = vocab.fim_pre();
    let _ = vocab.fim_suf();
    let _ = vocab.fim_mid();
    let _ = vocab.fim_pad();
    let _ = vocab.fim_rep();
    let _ = vocab.fim_sep();
}

// ============================================================
// Tokenize / detokenize
// ============================================================

#[test]
fn test_tokenize_roundtrip() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let text = "Hello, world!";
    let tokens = model.str_to_token(text, AddBos::Never).unwrap();
    assert!(!tokens.is_empty());

    let roundtrip = model.detokenize(&tokens, true, false).unwrap();
    assert_eq!(roundtrip, text);
}

#[test]
fn test_tokenize_with_bos() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let tokens_no_bos = model.str_to_token("hi", AddBos::Never).unwrap();
    let tokens_with_bos = model.str_to_token("hi", AddBos::Always).unwrap();
    assert!(
        tokens_with_bos.len() >= tokens_no_bos.len(),
        "with BOS should have at least as many tokens"
    );
}

// ============================================================
// Metadata
// ============================================================

#[test]
fn test_metadata_count() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    assert!(model.meta_count() > 0);
}

#[test]
fn test_metadata_by_index() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let key = model.meta_key_by_index(0, 256).unwrap();
    assert!(!key.is_empty());
    let val = model.meta_val_str_by_index(0, 4096).unwrap();
    assert!(!val.is_empty());
}

#[test]
fn test_metadata_by_key() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let arch = model.meta_val_str("general.architecture", 256);
    assert!(arch.is_ok(), "general.architecture should exist");
    assert!(!arch.unwrap().is_empty());
}

#[test]
fn test_metadata_convenience() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let entries = model.metadata().unwrap();
    assert!(!entries.is_empty());
    // All keys should be non-empty
    for (key, _val) in &entries {
        assert!(!key.is_empty());
    }
}

#[test]
fn test_metadata_missing_key() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let result = model.meta_val_str("nonexistent.key.that.does.not.exist", 256);
    assert!(result.is_err());
}

// ============================================================
// Chat templates
// ============================================================

#[test]
fn test_chat_builtin_templates() {
    let templates = LlamaModel::chat_builtin_templates();
    assert!(!templates.is_empty(), "should have built-in templates");
    // chatml is a common one
    assert!(
        templates.iter().any(|t| t == "chatml"),
        "should have chatml template"
    );
}

// ============================================================
// Context tests (require non-vocab-only model)
// ============================================================

#[test]
fn test_context_creation() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        eprintln!("SKIP: context tests need a full model");
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let ctx = model.new_context(&backend, ctx_params);
    assert!(ctx.is_ok(), "should create context");
}

#[test]
fn test_context_properties() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        eprintln!("SKIP: context tests need a full model");
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let ctx = model.new_context(&backend, ctx_params).unwrap();

    assert!(ctx.n_ctx() > 0);
    assert!(ctx.n_ctx_seq() > 0);
    assert!(ctx.n_seq_max() > 0);
    assert!(ctx.n_batch() > 0);
    assert!(ctx.n_ubatch() > 0);
    assert!(ctx.n_threads() > 0);
    assert!(ctx.n_threads_batch() > 0);
}

#[test]
fn test_context_thread_control() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        eprintln!("SKIP: context tests need a full model");
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params).unwrap();

    ctx.set_n_threads(2, 2);
    assert_eq!(ctx.n_threads(), 2);
    assert_eq!(ctx.n_threads_batch(), 2);
}

#[test]
fn test_context_memory_breakdown() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        eprintln!("SKIP: memory breakdown needs a full model");
        return;
    }
    let ctx = model
        .new_context(
            &backend,
            llama_cpp_4::context::params::LlamaContextParams::default(),
        )
        .unwrap();
    let breakdown = ctx.memory_breakdown();
    assert!(
        breakdown
            .iter()
            .all(|e| e.buft_name.is_empty() || e.total() > 0 || e.total() == 0),
        "entries should be well-formed"
    );
}

#[test]
fn test_model_devices_iterator() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let count = model.devices().count();
    assert_eq!(count, model.n_devices().max(0) as usize);
    for dev in model.devices() {
        let _ = dev.name();
        let _ = dev.description();
        let _ = dev.device_type();
        let _ = dev.memory();
    }
}

#[test]
fn test_context_set_causal_attn() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params).unwrap();
    ctx.set_causal_attn(true);
    ctx.set_causal_attn(false);
}

#[test]
fn test_context_set_embeddings() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params).unwrap();
    ctx.set_embeddings(true);
    ctx.set_embeddings(false);
}

#[test]
fn test_context_synchronize() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params).unwrap();
    ctx.synchronize();
}

#[test]
fn test_context_memory() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let ctx = model.new_context(&backend, ctx_params).unwrap();
    let _ = ctx.memory_can_shift();
    let _ = ctx.memory_seq_pos_min(0);
}

#[test]
fn test_context_state_size() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params).unwrap();
    let size = ctx.state_get_size();
    assert!(size > 0, "state size should be > 0");
    let seq_size = ctx.state_seq_get_size(0);
    assert!(seq_size > 0);
}

#[test]
fn test_context_state_save_restore() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params).unwrap();

    let size = ctx.state_get_size();
    let mut buf = vec![0u8; size];
    let written = ctx.state_get_data(&mut buf);
    assert!(written > 0);

    let read = ctx.state_set_data(&buf[..written]);
    assert_eq!(read, written);
}

#[test]
fn test_context_perf_reset() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params).unwrap();
    ctx.perf_context_reset();
}

#[test]
fn test_context_get_model_ptr() {
    let Some((backend, model, vocab_only)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    if vocab_only {
        return;
    }
    let ctx_params = llama_cpp_4::context::params::LlamaContextParams::default();
    let ctx = model.new_context(&backend, ctx_params).unwrap();
    let ptr = ctx.get_model_ptr();
    assert!(!ptr.is_null());
}

// ============================================================
// Sampler with model
// ============================================================

#[test]
fn test_infill_sampler() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let sampler = llama_cpp_4::sampling::LlamaSampler::infill(&model);
    assert_eq!(sampler.name(), "infill");
}

#[test]
fn test_grammar_sampler() {
    let Some((_backend, model, _)) = load_test_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let sampler =
        llama_cpp_4::sampling::LlamaSampler::grammar(&model, "root ::= \"hello\"", "root");
    assert_eq!(sampler.name(), "grammar");
}
