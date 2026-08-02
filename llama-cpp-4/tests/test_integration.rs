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
//! Tests skip (pass) when no full model is available. Each test holds a
//! process-wide lock ([`support::model::llama_guard`]) across its entire
//! llama.cpp interaction, because model loading, context creation, decode, and
//! the `fit` / `get_device_memory_data` helpers are not thread-safe (the `fit`
//! helpers install a process-global log callback capturing stack locals, so a
//! concurrent load on another thread would invoke a stale callback and crash).
//! That makes the suite safe under the default parallel runner; CI still passes
//! `--test-threads=1` as belt-and-suspenders.

mod support;

use std::num::NonZeroU32;

use llama_cpp_4::fit::{fit_params, get_device_memory_data, FitParams};
use llama_cpp_4::prelude::*;

use support::model::{backend, llama_guard, load_full_model, skip_no_model, test_model_path};

#[test]
fn integration_model_loads_and_has_weights() {
    let _guard = llama_guard();
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
    let _guard = llama_guard();
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
    let _guard = llama_guard();
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
    let _guard = llama_guard();
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
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

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
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

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
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

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
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

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
    let _guard = llama_guard();
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
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let last_layer = (model.n_layer() - 1) as usize;
    let mut capture = TensorCapture::for_layers(&[last_layer]);

    let ctx_params = unsafe {
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(64))
            .with_n_batch(64)
            .with_tensor_capture(&mut capture)
    };
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

/// Owned tensor-transaction capture over a real decode: exercises the refactored
/// FFI hot path (`read_tensor` uninitialized fill, byte-name matching,
/// index-keyed row bookkeeping, retained capture).
#[test]
fn integration_tensor_transactions_capture() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let n_embd = model.n_embd() as usize;
    // Layer 0 is always computed for every submitted token (the last layer can
    // be pruned to output positions, which would fail the row-coverage check).
    let selector =
        TensorSelector::layer_output(0, n_embd, 64, TensorAccess::ReadOnly, true).unwrap();
    let transactions = TensorTransactions::capture(vec![selector]).unwrap();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(64))
        .with_n_batch(64)
        .with_tensor_transactions(transactions);
    let mut ctx = model.new_context(backend(), ctx_params).unwrap();

    let tokens = model
        .str_to_token("Once upon a time", AddBos::Always)
        .unwrap();
    let mut batch = LlamaBatch::new(64, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == tokens.len() - 1)
            .unwrap();
    }
    ctx.decode(&mut batch).unwrap();

    let transactions = ctx.tensor_transactions().expect("owned transactions");
    assert!(
        transactions.failure().is_none(),
        "callback failure: {:?}",
        transactions.failure()
    );
    let captures = transactions.captures();
    assert_eq!(captures.len(), 1, "one retained tensor for one decode");
    let capture = &captures[0];
    assert_eq!(capture.name, "l_out-0");
    assert_eq!(capture.shape.row_elements, n_embd);
    assert_eq!(capture.shape.rows, tokens.len());
    assert_eq!(capture.rows.len(), tokens.len());
    match &capture.data {
        CapturedTensorData::F32(values) => {
            assert_eq!(values.len(), n_embd * tokens.len());
            assert!(values.iter().all(|value| value.is_finite()));
        }
        other => panic!("expected f32 capture, got {other:?}"),
    }
}

/// Owned transactional write-back driven by a *closure* handler: exercises the
/// blanket `FnMut` impl, both finiteness scans, and the commit (`copy_tensor_set`)
/// path end-to-end.
#[test]
fn integration_tensor_transactions_readwrite_commits() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let n_embd = model.n_embd() as usize;
    let selector =
        TensorSelector::layer_output(0, n_embd, 64, TensorAccess::ReadWriteF32, false).unwrap();

    // A closure is a handler (blanket impl). Clamp to a huge finite range: a
    // no-op for real residual values that still drives the write-back path.
    let transactions = TensorTransactions::new(vec![selector], |mut txn: TensorTransaction<'_>| {
        if let TensorDataMut::F32(values) = &mut txn.data {
            for value in values.iter_mut() {
                *value = value.clamp(-1.0e30, 1.0e30);
            }
        }
        Ok(TensorWriteback::Commit)
    })
    .unwrap();

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(64))
        .with_n_batch(64)
        .with_tensor_transactions(transactions);
    let mut ctx = model.new_context(backend(), ctx_params).unwrap();

    let tokens = model.str_to_token("hello world", AddBos::Always).unwrap();
    let mut batch = LlamaBatch::new(64, 1);
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == tokens.len() - 1)
            .unwrap();
    }
    ctx.decode(&mut batch).unwrap();

    let transactions = ctx.tensor_transactions().expect("owned transactions");
    assert!(
        transactions.failure().is_none(),
        "callback failure: {:?}",
        transactions.failure()
    );
    // Non-retaining selector: the commit path ran, nothing is captured.
    assert!(transactions.captures().is_empty());
}
