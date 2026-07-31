//! Model-backed benchmarks: the per-`llama_decode` overhead the owned
//! tensor-transaction FFI hooks add on a *real* graph — the part the model-free
//! `tensor_transactions` bench explicitly could not measure (the native
//! `ggml_backend_tensor_get`/`_set` copies and per-node matching over the actual
//! decode graph).
//!
//! Requires the tiny `stories260K.gguf` (or `LLAMA_TEST_MODEL`); the decode
//! benches skip with a note when no model is present. The ABI-guard bench is
//! model-free.
//!
//! Run: `cargo bench -p llama-cpp-4 --bench decode_hooks`

use std::hint::black_box;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::OnceLock;

use criterion::{criterion_group, criterion_main, Criterion};
use llama_cpp_4::prelude::*;

static BACKEND: OnceLock<Option<LlamaBackend>> = OnceLock::new();

fn backend() -> Option<&'static LlamaBackend> {
    BACKEND
        .get_or_init(|| LlamaBackend::init().ok())
        .as_ref()
}

fn model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LLAMA_TEST_MODEL") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    for candidate in [
        "../target/test-models/stories260K.gguf",
        "target/test-models/stories260K.gguf",
    ] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn load_model() -> Option<(&'static LlamaBackend, LlamaModel)> {
    let backend = backend()?;
    let params = std::pin::pin!(LlamaModelParams::default());
    let model = LlamaModel::load_from_file(backend, &model_path()?, &params).ok()?;
    Some((backend, model))
}

fn prompt_batch(model: &LlamaModel) -> LlamaBatch {
    let tokens = model
        .str_to_token("Once upon a time", AddBos::Always)
        .expect("tokenize");
    let mut batch = LlamaBatch::new(64, 1);
    for (i, &token) in tokens.iter().enumerate() {
        batch
            .add(token, i as i32, &[0], i == tokens.len() - 1)
            .expect("batch add");
    }
    batch
}

fn ctx_params() -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(64))
        .with_n_batch(64)
}

/// The fail-closed ABI guard is a single `extern "C"` int-returning call made
/// once per context setup. Confirm it is effectively free.
fn bench_abi_guard(c: &mut Criterion) {
    c.bench_function("abi_guard_call", |b| {
        b.iter(|| black_box(unsafe { llama_cpp_sys_4::llama_cpp_rs_decode_hooks_abi_v1() }));
    });
}

/// Per-decode overhead of the owned tensor-transaction hooks on a real graph.
/// Each iteration clears the KV cache and re-decodes a fixed prompt, so the
/// only difference between arms is the installed callback work.
fn bench_decode(c: &mut Criterion) {
    let Some((backend, model)) = load_model() else {
        eprintln!(
            "SKIP decode_hooks/decode benches: no model \
             (set LLAMA_TEST_MODEL or run scripts/fetch-test-model.sh)"
        );
        return;
    };
    let n_embd = model.n_embd() as usize;

    // Warm up process-wide one-time costs (Metal pipeline compilation, graph
    // reservation) on a throwaway context so no single arm below is charged for
    // them — otherwise whichever arm decodes first looks artificially slow.
    {
        let mut ctx = model.new_context(backend, ctx_params()).unwrap();
        let mut batch = prompt_batch(&model);
        for _ in 0..25 {
            ctx.clear_kv_cache();
            ctx.decode(&mut batch).unwrap();
        }
    }

    let mut group = c.benchmark_group("decode");

    // Baseline: plain decode, no callbacks installed.
    {
        let mut ctx = model.new_context(backend, ctx_params()).unwrap();
        let mut batch = prompt_batch(&model);
        group.bench_function("baseline", |b| {
            b.iter(|| {
                ctx.clear_kv_cache();
                ctx.decode(&mut batch).unwrap();
                black_box(&ctx);
            });
        });
    }

    // ReadOnly capture of `l_out-0`: decode-begin/end hooks + per-node name
    // matching over the whole graph + one bounded tensor copy, drained each iter.
    {
        let selector =
            TensorSelector::layer_output(0, n_embd, 64, TensorAccess::ReadOnly, true).unwrap();
        let txns = TensorTransactions::capture(vec![selector]).unwrap();
        let mut ctx = model
            .new_context(backend, ctx_params().with_tensor_transactions(txns))
            .unwrap();
        let mut batch = prompt_batch(&model);
        group.bench_function("with_capture", |b| {
            b.iter(|| {
                ctx.clear_kv_cache();
                ctx.decode(&mut batch).unwrap();
                if let Some(transactions) = ctx.tensor_transactions_mut() {
                    let _ = transactions.take_captures();
                }
                black_box(&ctx);
            });
        });
    }

    // ReadWriteF32 no-op commit of `l_out-0`: exercises the read + finiteness
    // scans + write-back path each decode (non-retaining, nothing to drain).
    {
        let selector =
            TensorSelector::layer_output(0, n_embd, 64, TensorAccess::ReadWriteF32, false).unwrap();
        let txns = TensorTransactions::new(vec![selector], |mut txn: TensorTransaction<'_>| {
            if let TensorDataMut::F32(values) = &mut txn.data {
                for value in values.iter_mut() {
                    *value = value.clamp(-1.0e30, 1.0e30);
                }
            }
            Ok(TensorWriteback::Commit)
        })
        .unwrap();
        let mut ctx = model
            .new_context(backend, ctx_params().with_tensor_transactions(txns))
            .unwrap();
        let mut batch = prompt_batch(&model);
        group.bench_function("with_readwrite_commit", |b| {
            b.iter(|| {
                ctx.clear_kv_cache();
                ctx.decode(&mut batch).unwrap();
                black_box(&ctx);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_abi_guard, bench_decode);
criterion_main!(benches);
