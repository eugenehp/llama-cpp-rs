//! End-to-end integration tests that load a real GGUF checkpoint.
//!
//! # Running
//!
//! ```bash
//! # Download the default tiny model (~1 MB), then run:
//! ./scripts/fetch-test-model.sh
//! cargo test -p llama-cpp-4 --test test_integration -- --test-threads=1
//!
//! # Or point at any local GGUF:
//! LLAMA_TEST_MODEL=/path/to/model.gguf \
//!     cargo test -p llama-cpp-4 --test test_integration -- --test-threads=1
//! ```
//!
//! Tests skip (pass) when no full model is available. Use `--test-threads=1`
//! because `llama_decode` is not exercised safely in parallel across contexts.

mod support;

use std::num::NonZeroU32;

use llama_cpp_4::fit::{fit_params, get_device_memory_data, FitParams};
use llama_cpp_4::prelude::*;

use support::model::{backend, decode_guard, load_full_model, skip_no_model, test_model_path};

#[test]
fn integration_model_loads_and_has_weights() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    assert!(model.n_layer() > 0);
    assert!(model.n_embd() > 0);
    assert!(model.n_vocab() > 0);
    assert!(model.model_size() > 0);
}

#[test]
fn integration_devices_iterator() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    assert_eq!(model.devices().count(), model.n_devices().max(0) as usize);
    for dev in model.devices() {
        let name = dev.name().expect("device name");
        assert!(!name.is_empty());
        let (_free, _total) = dev.memory();
    }
}

#[test]
fn integration_get_device_memory_data() {
    let Some(path) = test_model_path() else {
        skip_no_model();
        return;
    };
    if support::model::find_test_model().is_some_and(|f| f.vocab_only) {
        eprintln!("SKIP: get_device_memory_data needs a full model");
        return;
    }

    let report = get_device_memory_data(
        &path,
        &LlamaModelParams::default(),
        &LlamaContextParams::default().with_n_ctx(None),
        llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
    )
    .expect("device memory estimate");

    assert!(!report.entries.is_empty());
    assert!(report.hyperparams.n_ctx_train > 0);
}

#[test]
fn integration_fit_params() {
    let Some(path) = test_model_path() else {
        skip_no_model();
        return;
    };
    if support::model::find_test_model().is_some_and(|f| f.vocab_only) {
        eprintln!("SKIP: fit_params needs a full model");
        return;
    }

    let result = fit_params(backend(), &path, FitParams::default().with_n_ctx_min(32))
        .expect("fit_params should succeed on tiny model");

    let model_params = std::pin::pin!(result.model_params);
    let model =
        LlamaModel::load_from_file(backend(), &path, &model_params).expect("load fitted model");
    let ctx = model
        .new_context(backend(), result.context_params)
        .expect("context from fitted params");
    assert!(ctx.n_ctx() > 0);
}

#[test]
fn integration_decode_prefill() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let _guard = decode_guard();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(128))
        .with_n_batch(128);
    let mut ctx = model
        .new_context(backend(), ctx_params)
        .expect("create context");

    let tokens = model
        .str_to_token("Once upon a time", AddBos::Always)
        .expect("tokenize");
    assert!(!tokens.is_empty());

    let mut batch = LlamaBatch::new(128, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == tokens.len() - 1)
            .expect("batch add");
    }
    ctx.decode(&mut batch).expect("decode");

    let logits = ctx.get_logits_ith(batch.n_tokens() - 1);
    assert_eq!(logits.len(), model.n_vocab() as usize);
    assert!(logits.iter().any(|x| x.is_finite()));
}

#[test]
fn integration_greedy_generation() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let _guard = decode_guard();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(128))
        .with_n_batch(128);
    let mut ctx = model
        .new_context(backend(), ctx_params)
        .expect("create context");

    let prompt = "The capital of France is";
    let tokens = model.str_to_token(prompt, AddBos::Always).unwrap();

    let mut batch = LlamaBatch::new(128, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == tokens.len() - 1)
            .unwrap();
    }
    ctx.decode(&mut batch).unwrap();

    let eos = model.token_eos();
    let mut generated = Vec::new();
    let mut pos = tokens.len() as i32;
    let mut logit_idx = batch.n_tokens() - 1;

    for _ in 0..8 {
        let logits = ctx.get_logits_ith(logit_idx);
        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let token = LlamaToken(best as i32);
        if token == eos {
            break;
        }
        generated.push(token);

        batch.clear();
        batch.add(token, pos, &[0], true).unwrap();
        ctx.decode(&mut batch).unwrap();
        pos += 1;
        logit_idx = 0;
    }

    assert!(
        !generated.is_empty(),
        "expected at least one generated token"
    );
    let text = model
        .detokenize(&generated, false, false)
        .unwrap_or_default();
    assert!(
        text.chars().any(|c| c.is_alphanumeric()),
        "generated text should contain alphanumerics: {text:?}"
    );
}

#[test]
fn integration_embeddings() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let _guard = decode_guard();

    let ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_n_ctx(NonZeroU32::new(64))
        .with_n_batch(64);
    let mut ctx = model.new_context(backend(), ctx_params).unwrap();

    let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
    let mut batch = LlamaBatch::new(64, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch.add(tok, i as i32, &[0], true).unwrap();
    }
    ctx.decode(&mut batch).unwrap();

    let last = batch.n_tokens() - 1;
    let emb = ctx.embeddings_ith(last).expect("token embedding");
    assert_eq!(emb.len(), model.n_embd() as usize);
    assert!(emb.iter().any(|x| *x != 0.0));
}

#[test]
fn integration_memory_breakdown_after_decode() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let _guard = decode_guard();

    let mut ctx = model
        .new_context(
            backend(),
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(64)),
        )
        .unwrap();

    let tokens = model.str_to_token("hi", AddBos::Always).unwrap();
    let mut batch = LlamaBatch::new(64, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == tokens.len() - 1)
            .unwrap();
    }
    ctx.decode(&mut batch).unwrap();

    let breakdown = ctx.memory_breakdown();
    assert!(
        breakdown
            .iter()
            .all(|e| !e.buft_name.is_empty() || e.total() == 0),
        "breakdown entries should have buffer names when non-empty"
    );
}

#[test]
fn integration_apply_chat_template_if_supported() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let messages = vec![LlamaChatMessage::new("user".into(), "Hello".into()).unwrap()];
    match model.apply_chat_template(None, &messages, true) {
        Ok(prompt) => {
            assert!(!prompt.is_empty());
            let tokens = model.str_to_token(&prompt, AddBos::Always);
            assert!(tokens.is_ok(), "templated prompt should tokenize");
        }
        Err(e) => {
            eprintln!("SKIP: model has no chat template: {e}");
        }
    }
}

#[test]
fn integration_tensor_capture_last_layer() {
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let _guard = decode_guard();

    let last_layer = (model.n_layer() - 1) as usize;
    let mut capture = TensorCapture::for_layers(&[last_layer]);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(64))
        .with_n_batch(64)
        .with_tensor_capture(&mut capture);
    let mut ctx = model.new_context(backend(), ctx_params).unwrap();

    let tokens = model.str_to_token("test", AddBos::Always).unwrap();
    let mut batch = LlamaBatch::new(64, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == tokens.len() - 1)
            .unwrap();
    }
    ctx.decode(&mut batch).unwrap();

    let layer = capture
        .get_layer(last_layer)
        .expect("last layer hidden state");
    assert!(layer.n_embd() > 0);
    assert!(layer.n_tokens() > 0);
    assert_eq!(layer.data.len(), layer.n_embd() * layer.n_tokens());
}
