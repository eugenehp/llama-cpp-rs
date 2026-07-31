//! Benchmarks for the owned tensor-transaction API
//! ([`llama_cpp_4::context::tensor_transaction`]).
//!
//! These are **model-free**: they exercise the pure-Rust per-callback machinery
//! that runs on every selected graph node during decode — selector-program
//! validation, the handler-dispatch write path, the finiteness validation, and
//! the scratch-buffer allocation strategy. The two native memcpys
//! (`ggml_backend_tensor_get` / `_set`) require a live backend/model and are
//! therefore excluded; everything else here is what the binding adds on top of
//! that copy for each callback invocation.
//!
//! Run with: `cargo bench -p llama-cpp-4 --bench tensor_transactions`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use llama_cpp_4::{
    TensorAccess, TensorBatchRow, TensorDataMut, TensorElementType, TensorRowMapping,
    TensorSelector, TensorShape, TensorTransaction, TensorTransactionError,
    TensorTransactionHandler, TensorTransactions, TensorWriteback,
};

/// Representative element counts for one selected tensor: a small residual
/// row (`n_embd`), a modest multi-token batch, and a large prefill.
const ELEMENT_COUNTS: [usize; 3] = [4_096, 65_536, 1_048_576];

/// A minimal finite-`f32` transform, matching the shape of a real handler.
struct AddOne;

impl TensorTransactionHandler for AddOne {
    fn apply(
        &mut self,
        mut transaction: TensorTransaction<'_>,
    ) -> Result<TensorWriteback, TensorTransactionError> {
        if let TensorDataMut::F32(values) = &mut transaction.data {
            for value in values.iter_mut() {
                *value += 1.0;
            }
        }
        Ok(TensorWriteback::Commit)
    }
}

/// Builds `count` uniquely, canonically-named read-only selectors that satisfy
/// the collective element bound.
fn make_selectors(count: usize) -> Vec<TensorSelector> {
    (0..count)
        .map(|index| {
            TensorSelector::new(
                // zero-padded so lexical order matches numeric order
                format!("l_out-{index:04}"),
                TensorElementType::F32,
                1_024,
                8,
                TensorAccess::ReadOnly,
                TensorRowMapping::BatchTokens,
                false,
            )
            .expect("valid selector")
        })
        .collect()
}

/// Cost of constructing + validating a selector program (name ordering,
/// per-selector and collective element-bound checks).
fn bench_selector_program(c: &mut Criterion) {
    let mut group = c.benchmark_group("selector_program_build");
    for count in [1_usize, 16, 128] {
        let selectors = make_selectors(count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &selectors, |b, sel| {
            b.iter(|| TensorTransactions::capture(black_box(sel.clone())).expect("valid"));
        });
    }
    group.finish();
}

/// The model-free portion of `process()` for a retained `ReadWriteF32`
/// selector, faithful to the current implementation: allocate a fresh scratch
/// buffer, scan for finiteness, clone for rollback (retain), dispatch the
/// handler, then re-scan the committed output. Only the two `ggml` memcpys are
/// omitted (they need a live tensor).
fn bench_callback_write_path(c: &mut Criterion) {
    let rows = vec![TensorBatchRow {
        batch_index: 0,
        position: 0,
        sequence_ids: vec![0],
    }];
    let mut handler = AddOne;

    let mut group = c.benchmark_group("callback_write_path");
    for elements in ELEMENT_COUNTS {
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(elements),
            &elements,
            |b, &n| {
                b.iter(|| {
                    // (1) fresh scratch allocation — current per-callback behavior
                    let mut values = vec![0.0_f32; n];
                    // (2) pre-handler finiteness validation
                    let finite = values.iter().all(|v| v.is_finite());
                    debug_assert!(finite);
                    // (3) rollback copy taken because the selector retains
                    let _original = values.clone();
                    // (4) handler dispatch (real trait-object call)
                    let shape = TensorShape {
                        row_elements: n,
                        rows: 1,
                        elements: n,
                    };
                    let writeback = handler
                        .apply(TensorTransaction {
                            name: "l_out-0000",
                            shape,
                            rows: &rows,
                            access: TensorAccess::ReadWriteF32,
                            data: TensorDataMut::F32(&mut values),
                        })
                        .expect("handler ok");
                    // (5) post-commit finiteness validation
                    if matches!(writeback, TensorWriteback::Commit) {
                        let ok = values.iter().all(|v| v.is_finite());
                        black_box(ok);
                    }
                    black_box(&values);
                });
            },
        );
    }
    group.finish();
}

/// Scratch-buffer strategy: allocating a new `Vec` every callback (current)
/// versus reusing one persistent buffer (proposed). Quantifies the allocation
/// the binding could avoid across a long generation loop.
fn bench_buffer_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("scratch_buffer");
    for elements in ELEMENT_COUNTS {
        group.throughput(Throughput::Elements(elements as u64));

        group.bench_with_input(
            BenchmarkId::new("alloc_per_call", elements),
            &elements,
            |b, &n| {
                b.iter(|| {
                    let mut values = vec![0.0_f32; n];
                    for value in &mut values {
                        *value += 1.0;
                    }
                    black_box(&values);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("reused_buffer", elements),
            &elements,
            |b, &n| {
                let mut values = vec![0.0_f32; n];
                b.iter(|| {
                    values.iter_mut().for_each(|value| *value = 0.0);
                    for value in &mut values {
                        *value += 1.0;
                    }
                    black_box(&values);
                });
            },
        );
    }
    group.finish();
}

/// Isolated cost of the finiteness validation over one tensor copy — `process()`
/// performs this up to twice per `ReadWriteF32` callback.
fn bench_finiteness_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("finiteness_scan");
    for elements in ELEMENT_COUNTS {
        let values = vec![1.0_f32; elements];
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(elements),
            &values,
            |b, v| {
                b.iter(|| black_box(v.iter().all(|value| value.is_finite())));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_selector_program,
    bench_callback_write_path,
    bench_buffer_strategy,
    bench_finiteness_scan
);
criterion_main!(benches);
