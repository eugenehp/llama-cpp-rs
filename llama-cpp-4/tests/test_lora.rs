//! Tests for the `LoRA` adapter loaders, including the `FILE *`-based one.
//!
//! # Running
//!
//! ```bash
//! ./scripts/fetch-test-model.sh
//! cargo test -p llama-cpp-4 --test test_lora -- --test-threads=1
//! ```
//!
//! There is no `LoRA` fixture in the tree, so these cover the loaders' failure
//! paths and the plumbing around them — which is where the interesting
//! behaviour is: the offset variant has to open, seek and close a C `FILE *`
//! that Rust's `std::fs` cannot produce.

mod support;

use std::path::PathBuf;

use llama_cpp_4::prelude::*;

use support::model::{llama_guard, load_full_model, skip_no_model, test_model_path};

#[test]
fn a_missing_adapter_file_is_reported_with_its_path() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let missing = PathBuf::from("definitely/not/a/real/adapter.gguf");
    match model.lora_adapter_init_at_offset(&missing, 0) {
        Err(LlamaLoraAdapterInitError::FileOpen { path, reason }) => {
            assert_eq!(path, missing);
            assert!(!reason.is_empty(), "the OS reason should be carried along");
        }
        other => panic!("expected FileOpen, got {other:?}"),
    }
}

/// A GGUF that is not a `LoRA` adapter has to be rejected by llama.cpp rather
/// than half-loaded. Reaching this at all proves the `FILE *` was opened,
/// positioned and handed over.
#[test]
fn a_non_lora_gguf_is_rejected() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let Some(path) = test_model_path() else {
        skip_no_model();
        return;
    };

    assert_eq!(
        model.lora_adapter_init_at_offset(&path, 0).err(),
        Some(LlamaLoraAdapterInitError::NullResult)
    );
    // The path-based loader agrees, which is the behaviour being matched.
    assert_eq!(
        model.lora_adapter_init(&path).err(),
        Some(LlamaLoraAdapterInitError::NullResult)
    );
}

/// Seeking past the end of a file succeeds for a read handle, so the failure
/// surfaces from llama.cpp's reader rather than from the seek.
#[test]
fn an_offset_past_the_end_fails_in_the_reader() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let Some(path) = test_model_path() else {
        skip_no_model();
        return;
    };

    assert_eq!(
        model.lora_adapter_init_at_offset(&path, 1 << 30).err(),
        Some(LlamaLoraAdapterInitError::NullResult)
    );
}

/// Upstream explicitly handles a NULL handle, and so must the wrapper.
#[test]
fn a_null_file_pointer_is_rejected() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    // SAFETY: NULL is the one value `llama_adapter_lora_init_from_file_ptr`
    // documents as handled; it logs and returns null.
    let result = unsafe { model.lora_adapter_init_from_file_ptr(std::ptr::null_mut()) };
    assert_eq!(result.err(), Some(LlamaLoraAdapterInitError::NullResult));
}
