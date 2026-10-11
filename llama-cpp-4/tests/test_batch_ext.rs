//! End-to-end tests for [`LlamaBatchExt`] and [`LlamaContext::process`].
//!
//! # Running
//!
//! ```bash
//! ./scripts/fetch-test-model.sh
//! cargo test -p llama-cpp-4 --test test_batch_ext -- --test-threads=1
//! ```
//!
//! Tests skip (pass) when no full model is available. Each holds the
//! process-wide llama.cpp lock for its whole body — see
//! [`support::model::llama_guard`].

// llama.cpp indexes batch positions with `i32` while Rust collections use
// `usize`. The checkpoint under test has hundreds of tokens, so none of these
// conversions can overflow.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

mod support;

use std::num::NonZeroU32;

use llama_cpp_4::prelude::*;

use support::model::{backend, llama_guard, load_full_model, skip_no_model};

const N_CTX: u32 = 128;

fn context(model: &LlamaModel) -> LlamaContext<'_> {
    model
        .new_context(
            backend(),
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(N_CTX))
                .with_n_batch(N_CTX),
        )
        .expect("create context")
}

fn prompt(model: &LlamaModel) -> Vec<LlamaToken> {
    model
        .str_to_token("Once upon a time", AddBos::Always)
        .expect("tokenize")
}

/// Fill `batch` with `tokens` at positions `0..n`, output on the last entry.
fn fill(batch: &mut LlamaBatchExt, tokens: &[LlamaToken]) -> BatchEntry {
    let mut last = None;
    for (i, &token) in tokens.iter().enumerate() {
        let entry = batch.add_token(token, 0).expect("add token");
        batch.set_pos(entry, i as i32).expect("set pos");
        batch
            .set_output(entry, i == tokens.len() - 1)
            .expect("set output");
        last = Some(entry);
    }
    last.expect("prompt must not be empty")
}

/// The headline claim: decoding through `llama_process` with an entry-wise
/// batch gives the same logits as the `llama_decode` path for the same tokens.
/// Anything mis-wired in the wrapper — positions, sequence ids, output flags —
/// shows up as a divergence here.
#[test]
fn process_decode_matches_the_llama_batch_path() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let tokens = prompt(&model);

    let reference = {
        let mut ctx = context(&model);
        let mut batch = LlamaBatch::new(N_CTX as usize, 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i == tokens.len() - 1)
                .expect("batch add");
        }
        ctx.decode(&mut batch).expect("decode");
        ctx.get_logits_ith(batch.n_tokens() - 1).to_vec()
    };

    let mut ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let last = fill(&mut batch, &tokens);
    ctx.process(&mut batch, LlamaProcessType::Decode)
        .expect("process");

    let logits = ctx.get_logits_ith(last.index());
    assert_eq!(logits.len(), model.n_vocab() as usize);
    assert!(logits.iter().any(|x| x.is_finite()));
    assert_eq!(
        logits, reference,
        "llama_process and llama_decode must agree on the same tokens"
    );
}

/// `clear` keeps the allocation, so the batch is reusable across decodes — and
/// the second round must land on the same logits as a fresh batch would.
#[test]
fn a_cleared_batch_decodes_again() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let tokens = prompt(&model);

    let mut ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");

    let last = fill(&mut batch, &tokens);
    ctx.process(&mut batch, LlamaProcessType::Decode)
        .expect("first process");
    let first = ctx.get_logits_ith(last.index()).to_vec();

    ctx.clear_kv_cache();
    batch.clear();
    assert!(batch.is_empty());
    assert!(batch.outputs().is_empty());

    let last = fill(&mut batch, &tokens);
    ctx.process(&mut batch, LlamaProcessType::Decode)
        .expect("second process");
    assert_eq!(ctx.get_logits_ith(last.index()), first);
}

/// An entry handle from before a `clear` names a slot that no longer exists.
/// Upstream would write into whatever got recycled into that index; this must
/// reject it.
#[test]
fn a_stale_entry_handle_is_rejected() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let entry = batch.add_token(LlamaToken(0), 0).expect("add token");
    batch.clear();

    assert_eq!(
        batch.set_pos(entry, 0),
        Err(BatchExtError::NoSuchEntry {
            entry: entry.index(),
            len: 0
        })
    );
    assert!(matches!(
        batch.set_output(entry, true),
        Err(BatchExtError::NoSuchEntry { .. })
    ));
    assert!(matches!(
        batch.add_seq(entry, 0),
        Err(BatchExtError::NoSuchEntry { .. })
    ));
}

/// `llama_batch_ext_set_pos` reads the entry's position arity straight off the
/// caller's pointer, so a slice of the wrong length must never reach it.
#[test]
fn position_arity_is_enforced() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let entry = batch.add_token(LlamaToken(0), 0).expect("add token");

    // A token entry takes exactly one position, whatever the model's RoPE type.
    batch.set_positions(entry, &[7]).expect("one position");
    assert_eq!(
        batch.set_positions(entry, &[7, 8]),
        Err(BatchExtError::PositionArity {
            entry: entry.index(),
            expected: 1,
            got: 2
        })
    );
    assert!(matches!(
        batch.set_positions(entry, &[]),
        Err(BatchExtError::PositionArity { .. })
    ));

    // The test checkpoint is not M-RoPE, so an embedding entry also takes one.
    assert_eq!(batch.n_pos_per_embd(), 1);
    let embd = vec![0.0_f32; model.n_embd_inp() as usize];
    let entry = batch
        .add_embedding(EmbeddingRows::row(&embd), 0)
        .expect("add embedding");
    batch.set_pos(entry, 1).expect("one position");
}

/// A batch is bound to the context that built it: it caches that context's
/// batch size and sequence count, and holds a pointer to its memory module.
#[test]
fn a_batch_from_another_context_is_refused() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let tokens = prompt(&model);

    let other = context(&model);
    let mut batch = LlamaBatchExt::new(&other).expect("init batch_ext");
    fill(&mut batch, &tokens);

    let mut ctx = context(&model);
    assert_eq!(
        ctx.process(&mut batch, LlamaProcessType::Decode),
        Err(ProcessError::ContextMismatch)
    );
}

/// Tensor transactions ride on hooks patched into `llama_decode`.
/// `llama_process` never reaches them, so running the batch would leave the
/// transactions inert — fail instead of doing that quietly.
#[test]
fn process_refuses_a_context_with_tensor_transactions() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };
    let tokens = prompt(&model);

    let n_embd = model.n_embd() as usize;
    let selector =
        TensorSelector::layer_output(0, n_embd, N_CTX as usize, TensorAccess::ReadOnly, true)
            .unwrap();
    let transactions = TensorTransactions::capture(vec![selector]).unwrap();

    let mut ctx = model
        .new_context(
            backend(),
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(N_CTX))
                .with_n_batch(N_CTX)
                .with_tensor_transactions(transactions),
        )
        .expect("create context");

    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    fill(&mut batch, &tokens);

    assert_eq!(
        ctx.process(&mut batch, LlamaProcessType::Decode),
        Err(ProcessError::TensorTransactionsUnsupported)
    );
}

/// Capacity comes from the context's `n_batch`, and overflowing it is an error
/// rather than a silently dropped entry.
#[test]
fn the_batch_fills_up_at_n_batch() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = model
        .new_context(
            backend(),
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(N_CTX))
                .with_n_batch(4)
                .with_n_ubatch(4),
        )
        .expect("create context");

    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    assert_eq!(batch.capacity(), 4);
    for i in 0..4 {
        let entry = batch.add_token(LlamaToken(0), 0).expect("add token");
        batch.set_pos(entry, i).expect("set pos");
    }
    assert_eq!(batch.len(), 4);
    assert_eq!(
        batch.add_token(LlamaToken(0), 0),
        Err(BatchExtError::Full(4))
    );
}

#[test]
fn out_of_range_sequence_and_token_ids_are_errors() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");

    assert_eq!(
        batch.add_token(LlamaToken(0), 9),
        Err(BatchExtError::InvalidSeqId {
            seq_id: 9,
            n_seq_max: ctx.n_seq_max()
        })
    );
    let beyond_vocab = model.n_vocab();
    assert_eq!(
        batch.add_token(LlamaToken(beyond_vocab), 0),
        Err(BatchExtError::InvalidToken(beyond_vocab))
    );
    assert!(batch.is_empty(), "a rejected add must not leave an entry");
}

/// `outputs()` mirrors what the context will let you read afterwards, and
/// toggling an entry off removes it.
#[test]
fn outputs_track_the_entries_with_output_enabled() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let first = batch.add_token(LlamaToken(0), 0).expect("add token");
    let second = batch.add_token(LlamaToken(0), 0).expect("add token");

    batch.set_output(first, true).unwrap();
    batch.set_output(second, true).unwrap();
    assert_eq!(batch.outputs(), [first.index(), second.index()]);

    // Idempotent, and removable.
    batch.set_output(second, true).unwrap();
    assert_eq!(batch.outputs(), [first.index(), second.index()]);
    batch.set_output(first, false).unwrap();
    assert_eq!(batch.outputs(), [second.index()]);
}

/// An embedding entry whose row width matches the model's decoder input width
/// decodes, which is the capability `llama_batch` cannot offer alongside tokens.
#[test]
fn an_embedding_entry_decodes() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let mut ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");

    let n_embd_inp = model.n_embd_inp() as usize;
    let embd: Vec<f32> = (0..n_embd_inp)
        .map(|i| f32::from(u8::try_from(i % 7).unwrap()) * 0.01)
        .collect();
    let entry = batch
        .add_embedding(EmbeddingRows::row(&embd), 0)
        .expect("add embedding");
    batch.set_pos(entry, 0).expect("set pos");
    batch.set_output(entry, true).expect("set output");

    ctx.process(&mut batch, LlamaProcessType::Decode)
        .expect("process an embedding batch");

    let logits = ctx.get_logits_ith(entry.index());
    assert_eq!(logits.len(), model.n_vocab() as usize);
    assert!(logits.iter().any(|x| x.is_finite()));
}

/// `llama_batch_ext_add_embd` appends the entry and *only then* validates the
/// block, with no API to remove one. The wrapper must mirror that entry anyway
/// — otherwise every later index is off by one against upstream's — and must
/// refuse to submit the batch while the hole is there.
#[test]
fn a_rejected_embedding_leaves_a_tracked_incomplete_entry() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let mut ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");

    // A width that is neither the decoder nor the encoder input width.
    let wrong = vec![0.0_f32; model.n_embd_inp() as usize + 1];
    assert_eq!(
        batch.add_embedding(EmbeddingRows::row(&wrong), 0),
        Err(BatchExtError::EmbeddingRejected)
    );

    // Upstream kept the entry, so the mirror has to as well.
    assert_eq!(batch.len(), 1);
    assert!(batch.has_incomplete_entry());

    // And the next entry must land at index 1, matching upstream.
    let next = batch.add_token(LlamaToken(0), 0).expect("add token");
    assert_eq!(next.index(), 1);

    assert_eq!(
        ctx.process(&mut batch, LlamaProcessType::Decode),
        Err(ProcessError::IncompleteEntry(0))
    );

    // `clear` is the way out.
    batch.clear();
    assert!(!batch.has_incomplete_entry());
}

/// `add_empty` deliberately makes an entry with no input, to be filled with
/// `set_token_embedding`. Submitting it unfilled must fail rather than decode a
/// hole.
#[test]
fn an_unfilled_empty_entry_blocks_processing() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let mut ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let entry = batch.add_empty(0).expect("add empty");
    batch.set_pos(entry, 0).expect("set pos");
    batch.set_output(entry, true).expect("set output");

    assert!(batch.has_incomplete_entry());
    assert_eq!(
        ctx.process(&mut batch, LlamaProcessType::Decode),
        Err(ProcessError::IncompleteEntry(entry.index()))
    );

    // Filling it in completes the entry and the batch decodes.
    let embd = vec![0.0_f32; model.n_embd_inp() as usize];
    batch
        .set_token_embedding(entry, EmbeddingRows::row(&embd))
        .expect("set embedding");
    assert!(!batch.has_incomplete_entry());
    ctx.process(&mut batch, LlamaProcessType::Decode)
        .expect("process");
}

/// Every embedding in one batch shares a width, fixed by the first. The second
/// one is caught in Rust, so it never reaches the entry-appending C call.
#[test]
fn a_second_embedding_of_a_different_width_is_rejected_without_appending() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");

    let n_embd_inp = model.n_embd_inp() as usize;
    let good = vec![0.0_f32; n_embd_inp];
    batch
        .add_embedding(EmbeddingRows::row(&good), 0)
        .expect("first embedding");

    let narrow = vec![0.0_f32; n_embd_inp - 1];
    assert_eq!(
        batch.add_embedding(EmbeddingRows::row(&narrow), 0),
        Err(BatchExtError::EmbeddingWidthMismatch {
            expected: n_embd_inp,
            got: n_embd_inp - 1
        })
    );
    assert_eq!(batch.len(), 1, "a Rust-side reject must not append");
    assert!(!batch.has_incomplete_entry());
}

/// llama.cpp allows one input embedding per entry; a second is a mistake worth
/// naming rather than an opaque `false`.
#[test]
fn an_entry_takes_only_one_input_embedding() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let embd = vec![0.0_f32; model.n_embd_inp() as usize];
    let entry = batch
        .add_embedding(EmbeddingRows::row(&embd), 0)
        .expect("add embedding");

    assert_eq!(
        batch.set_token_embedding(entry, EmbeddingRows::row(&embd)),
        Err(BatchExtError::EmbeddingAlreadySet(entry.index()))
    );
}

/// A state embedding only exists on an MTP context; a default context must say
/// no rather than accept and ignore it.
#[test]
fn a_state_embedding_is_refused_by_a_default_context() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    assert_eq!(ctx.context_type(), LlamaContextType::Default);

    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let entry = batch.add_token(LlamaToken(0), 0).expect("add token");
    let state = vec![0.0_f32; model.n_embd() as usize];
    assert_eq!(
        batch.set_state_embedding(entry, EmbeddingRows::row(&state)),
        Err(BatchExtError::EmbeddingRejected)
    );
}

/// The decision-head marking is a no-op for a normal model, but it must round
/// trip through the mangled `llama-ext.h` entry point without failing.
#[test]
fn decision_order_can_be_set() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = context(&model);
    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let entry = batch.add_token(LlamaToken(0), 0).expect("add token");
    batch
        .set_decision_order(entry, LlamaDecisionOrder::QuestionChoice)
        .expect("set decision order");
    batch
        .set_decision_order(entry, LlamaDecisionOrder::None)
        .expect("reset decision order");
}

/// A second sequence id on an entry has to be within the context's `n_seq_max`.
#[test]
fn add_seq_shares_an_entry_between_sequences() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let ctx = model
        .new_context(
            backend(),
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(N_CTX))
                .with_n_batch(N_CTX)
                .with_n_seq_max(2),
        )
        .expect("create context");

    let mut batch = LlamaBatchExt::new(&ctx).expect("init batch_ext");
    let entry = batch.add_token(LlamaToken(0), 0).expect("add token");
    batch.add_seq(entry, 1).expect("share with seq 1");
    assert_eq!(
        batch.add_seq(entry, 2),
        Err(BatchExtError::InvalidSeqId {
            seq_id: 2,
            n_seq_max: 2
        })
    );
}

/// The read side of `set_causal_attn`: a default decoder context is causal, and
/// toggling it is visible through the getter.
#[test]
fn causal_attn_reads_back_what_was_set() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let mut ctx = context(&model);
    assert!(ctx.causal_attn(), "a decoder context defaults to causal");
    ctx.set_causal_attn(false);
    assert!(!ctx.causal_attn());
    ctx.set_causal_attn(true);
    assert!(ctx.causal_attn());
}

#[test]
fn moe_cache_size_round_trips() {
    let params = LlamaContextParams::default().with_moe_cache_size(16 * 1024 * 1024);
    assert_eq!(params.moe_cache_size(), 16 * 1024 * 1024);
    assert_eq!(LlamaContextParams::default().moe_cache_size(), 0);
}

/// A non-zero `moe_cache_size` is a requirement rather than a hint: llama.cpp
/// refuses the context when it cannot honour it, which for the CPU-only,
/// non-MoE test checkpoint it never can. Pinning that here so the docs on
/// [`LlamaContextParams::with_moe_cache_size`] stay honest — a caller who sets
/// it speculatively gets a failed context, not a slower one.
#[test]
fn moe_cache_size_is_refused_by_a_model_that_cannot_honour_it() {
    let _guard = llama_guard();
    let Some(model) = load_full_model() else {
        skip_no_model();
        return;
    };

    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_moe_cache_size(16 * 1024 * 1024);

    assert_eq!(
        model.new_context(backend(), params).err(),
        Some(LlamaContextLoadError::NullReturn),
        "a MoE cache budget on a non-MoE CPU model must fail loudly"
    );

    // Zero is the default and always fine.
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX)
        .with_moe_cache_size(0);
    let ctx = model
        .new_context(backend(), params)
        .expect("create context");
    // llama.cpp may round n_ctx up when padding the KV cache, so only the
    // request being honoured is interesting here.
    assert!(ctx.n_ctx() >= N_CTX);
}
