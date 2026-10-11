//! Safe wrapper around `llama_batch_ext`, llama.cpp's entry-wise batch.
//!
//! [`LlamaBatch`](crate::llama_batch::LlamaBatch) wraps the original
//! `llama_batch`: six parallel arrays, one entry per token, and a single flat
//! `embd` block for the whole batch. That shape cannot describe a batch where
//! different entries carry differently-shaped inputs, which is what multimodal
//! and speculative decoding need. `llama_batch_ext`
//! ([llama.cpp #24669](https://github.com/ggml-org/llama.cpp/pull/24669))
//! replaces the arrays with a list of entries that each own their token id,
//! positions, input embedding and "state" embedding.
//!
//! Reach for this instead of [`LlamaBatch`](crate::llama_batch::LlamaBatch)
//! when you need one of the things only an entry can express:
//!
//! - **an input embedding on some entries but not others** — a vision encoder's
//!   output interleaved with text tokens, where `llama_batch` forces the whole
//!   batch to be one or the other;
//! - **M-RoPE positions**, which are `[t, y, x, z]` per embedding entry rather
//!   than one scalar — see [`LlamaBatchExt::set_positions`];
//! - **a state embedding** (`llama_batch_ext_set_embd_state`), the hidden state
//!   an MTP draft context reads from the target model.
//!
//! For plain token decoding there is nothing to gain:
//! [`LlamaBatch`](crate::llama_batch::LlamaBatch) is the same work with less
//! ceremony, and `llama_decode` is where this crate's decode lifecycle hooks
//! live — see [`LlamaContext::process`](crate::context::LlamaContext::process).
//!
//! # Example
//!
//! ```no_run
//! use llama_cpp_4::context::LlamaContext;
//! use llama_cpp_4::llama_batch_ext::{LlamaBatchExt, LlamaProcessType};
//! use llama_cpp_4::token::LlamaToken;
//!
//! # fn f(ctx: &mut LlamaContext<'_>, tokens: &[LlamaToken])
//! #     -> Result<(), Box<dyn std::error::Error>> {
//! let mut batch = LlamaBatchExt::new(ctx)?;
//!
//! let mut pos = 0;
//! let mut last = None;
//! for &token in tokens {
//!     let entry = batch.add_token(token, 0)?;
//!     batch.set_pos(entry, pos)?;
//!     pos += 1;
//!     last = Some(entry);
//! }
//! if let Some(entry) = last {
//!     batch.set_output(entry, true)?;
//! }
//!
//! ctx.process(&mut batch, LlamaProcessType::Decode)?;
//! # Ok(())
//! # }
//! ```

use std::ptr::NonNull;

use llama_cpp_sys_4::{llama_pos, llama_seq_id};

use crate::context::LlamaContext;
use crate::token::LlamaToken;

/// Positions an embedding entry carries on an M-RoPE or interleaved-M-RoPE
/// model: `[t, y, x, z]`, where `t` doubles as the KV-cache position.
///
/// Mirrors `GGML_MROPE_SECTIONS`. A model with any other `RoPE` type uses a
/// single position per entry; [`LlamaBatchExt::n_pos_per_embd`] reports which
/// applies.
pub const MROPE_SECTIONS: usize = llama_cpp_sys_4::GGML_MROPE_SECTIONS as usize;

/// A handle to one entry of a [`LlamaBatchExt`], returned by the `add_*`
/// methods and accepted by the `set_*` methods.
///
/// This is the entry's index in the batch. It is a distinct type because every
/// other `i32` in this module — token ids, sequence ids, positions — is also an
/// `i32`, and mixing them up is otherwise a silent error rather than a
/// compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchEntry(i32);

impl BatchEntry {
    /// The entry's index in the batch, which is also the index to pass to
    /// [`LlamaContext::get_logits_ith`](crate::context::LlamaContext::get_logits_ith)
    /// after processing.
    #[must_use]
    pub fn index(self) -> i32 {
        self.0
    }
}

impl From<BatchEntry> for i32 {
    fn from(entry: BatchEntry) -> Self {
        entry.0
    }
}

/// A borrowed, row-major block of embedding values, mirroring `llama_embd`.
///
/// llama.cpp copies the values out when the view is handed to the batch, so the
/// borrow only has to outlive the call.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingRows<'a> {
    data: &'a [f32],
    n_rows: usize,
    n_embd: usize,
}

impl<'a> EmbeddingRows<'a> {
    /// A `n_rows` × `n_embd` block.
    ///
    /// # Errors
    ///
    /// Returns [`BatchExtError::EmbeddingShape`] unless
    /// `n_rows * n_embd == data.len()`. llama.cpp takes the shape on trust and
    /// reads `n_rows * n_embd` floats, so a wrong shape would read out of
    /// bounds.
    pub fn new(data: &'a [f32], n_rows: usize, n_embd: usize) -> Result<Self, BatchExtError> {
        if n_rows.checked_mul(n_embd) != Some(data.len()) {
            return Err(BatchExtError::EmbeddingShape {
                n_rows,
                n_embd,
                len: data.len(),
            });
        }
        Ok(Self {
            data,
            n_rows,
            n_embd,
        })
    }

    /// A single row of `data.len()` values — the common case.
    #[must_use]
    pub fn row(data: &'a [f32]) -> Self {
        Self {
            data,
            n_rows: 1,
            n_embd: data.len(),
        }
    }

    /// Number of rows in the block.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Width of one row.
    #[must_use]
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Total values in the block — the width llama.cpp fixes per batch.
    fn total(self) -> usize {
        self.data.len()
    }

    fn as_raw(self) -> llama_cpp_sys_4::llama_embd {
        llama_cpp_sys_4::llama_embd {
            data: self.data.as_ptr(),
            n_rows: self.n_rows,
            n_embd: self.n_embd,
        }
    }
}

/// Which graph to run a batch through.
///
/// Mirrors `llama_process_type`.
//
// Cast the sys constants to `u32` so the discriminants compile on both clang
// (where bindgen emits `c_uint`, making the cast a no-op) and MSVC (where it
// emits `c_int` and the cast is required).
#[allow(clippy::unnecessary_cast)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaProcessType {
    /// Run the encoder — the equivalent of `llama_encode`.
    Encode = llama_cpp_sys_4::LLAMA_PROCESS_TYPE_ENCODE as u32,
    /// Run the decoder — the equivalent of `llama_decode`.
    Decode = llama_cpp_sys_4::LLAMA_PROCESS_TYPE_DECODE as u32,
}

impl From<LlamaProcessType> for llama_cpp_sys_4::llama_process_type {
    fn from(value: LlamaProcessType) -> Self {
        value as u32 as Self
    }
}

/// What a joint decision head (`clef` and friends) reads an entry as.
///
/// Mirrors `llama_decision_order`
/// ([llama.cpp #29831](https://github.com/ggml-org/llama.cpp/pull/29831)). A
/// run of entries sharing a value is one span, spans are separated by entries
/// marked [`Self::None`], and an option belongs to the last question before it.
/// Only decision models read this; everything else leaves it at
/// [`Self::None`].
#[allow(clippy::unnecessary_cast)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaDecisionOrder {
    /// Not read by the decision head. The default.
    #[default]
    None = llama_cpp_sys_4::LLAMA_DECISION_ORDER_NONE as u32,
    /// Text of a question.
    QuestionNoul = llama_cpp_sys_4::LLAMA_DECISION_ORDER_QUESTION_NOUL as u32,
    /// Question asking for a choice between the options that follow.
    QuestionChoice = llama_cpp_sys_4::LLAMA_DECISION_ORDER_QUESTION_CHOICE as u32,
    /// Question asking for a score.
    QuestionScore = llama_cpp_sys_4::LLAMA_DECISION_ORDER_QUESTION_SCORE as u32,
    /// Text of an option belonging to the preceding question.
    Option = llama_cpp_sys_4::LLAMA_DECISION_ORDER_OPTION as u32,
}

impl From<LlamaDecisionOrder> for llama_cpp_sys_4::llama_decision_order {
    fn from(value: LlamaDecisionOrder) -> Self {
        value as u32 as Self
    }
}

/// Errors from building a [`LlamaBatchExt`].
///
/// llama.cpp reports most of these as a bare `false` with the detail in its
/// log, so the variants carry whatever the Rust side can establish on its own.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BatchExtError {
    /// `llama_batch_ext_init` returned null.
    #[error("llama.cpp returned null from llama_batch_ext_init")]
    NullReturn,
    /// The batch already holds `n_batch` entries.
    #[error("batch is full at {0} entries (the context's n_batch)")]
    Full(u32),
    /// The token id is negative or not below the model's vocabulary size.
    #[error("token id {0} is out of range for this model's vocabulary")]
    InvalidToken(i32),
    /// The sequence id is negative or not below the context's `n_seq_max`.
    #[error("sequence id {seq_id} is out of range (n_seq_max = {n_seq_max})")]
    InvalidSeqId {
        /// The rejected sequence id.
        seq_id: llama_seq_id,
        /// The context's `n_seq_max`.
        n_seq_max: u32,
    },
    /// `n_rows * n_embd` did not match the length of the value buffer.
    #[error("embedding shape {n_rows} x {n_embd} does not match a buffer of {len} values")]
    EmbeddingShape {
        /// Rows the caller declared.
        n_rows: usize,
        /// Row width the caller declared.
        n_embd: usize,
        /// Values actually supplied.
        len: usize,
    },
    /// The entry handle does not belong to this batch — it came from another
    /// batch, or from this one before [`LlamaBatchExt::clear`].
    #[error("entry {entry} is not in this batch ({len} entries)")]
    NoSuchEntry {
        /// The rejected entry index.
        entry: i32,
        /// Entries currently in the batch.
        len: usize,
    },
    /// The wrong number of positions was supplied for the entry. A token entry
    /// takes one; an embedding entry takes
    /// [`LlamaBatchExt::n_pos_per_embd`].
    #[error("entry {entry} needs exactly {expected} position(s), got {got}")]
    PositionArity {
        /// The entry being positioned.
        entry: i32,
        /// Positions the entry requires.
        expected: usize,
        /// Positions supplied.
        got: usize,
    },
    /// Every embedding in one batch must have the same total width, fixed by
    /// the first one accepted.
    #[error("this batch's embeddings are {expected} values wide, got {got}")]
    EmbeddingWidthMismatch {
        /// Width fixed by the first embedding in the batch.
        expected: usize,
        /// Width supplied.
        got: usize,
    },
    /// The entry already carries an input embedding; llama.cpp allows one.
    #[error("entry {0} already has an input embedding")]
    EmbeddingAlreadySet(i32),
    /// The entry already carries a state embedding; llama.cpp allows one.
    #[error("entry {0} already has a state embedding")]
    StateAlreadySet(i32),
    /// llama.cpp rejected an embedding: the row width matches neither the
    /// decoder nor the encoder input width, or (for a state embedding) this
    /// context takes none — only an MTP context does. The specific reason is in
    /// the llama.cpp log.
    ///
    /// When this comes from [`LlamaBatchExt::add_embedding`], llama.cpp has
    /// already appended the entry by the time it validates the block, and
    /// offers no way to remove one — so the batch is left holding an entry with
    /// no input, and [`LlamaBatchExt::clear`] is the only way forward. See
    /// [`LlamaBatchExt::has_incomplete_entry`].
    #[error("llama.cpp rejected the embedding (see the llama.cpp log for the reason)")]
    EmbeddingRejected,
}

/// What a batch entry carries so far.
///
/// This mirrors the state `llama_batch_ext` keeps per entry, because two things
/// the C API takes on trust depend on it: the number of positions
/// `llama_batch_ext_set_pos` reads off the caller's pointer, and whether the
/// entry has an input at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    /// Carries a token id, so it takes one position rather than
    /// `n_pos_per_embd`.
    has_token: bool,
    /// Carries an input embedding.
    has_embd: bool,
    /// Carries a state embedding.
    has_state: bool,
}

impl Entry {
    fn token() -> Self {
        Self {
            has_token: true,
            has_embd: false,
            has_state: false,
        }
    }

    fn embedding() -> Self {
        Self {
            has_token: false,
            has_embd: true,
            has_state: false,
        }
    }

    /// An entry with no input at all: `add_empty` that was never filled in, or
    /// an `add_embedding` whose block llama.cpp rejected after appending the
    /// entry. Processing one is not meaningful.
    fn empty() -> Self {
        Self {
            has_token: false,
            has_embd: false,
            has_state: false,
        }
    }

    fn is_incomplete(self) -> bool {
        !self.has_token && !self.has_embd
    }

    /// Positions `llama_batch_ext_set_pos` will read for this entry.
    fn n_pos(self, n_pos_per_embd: usize) -> usize {
        if self.has_token {
            1
        } else {
            n_pos_per_embd
        }
    }
}

/// A safe wrapper around `llama_batch_ext`.
///
/// Build it with [`Self::new`], add entries, configure each entry through the
/// [`BatchEntry`] handle its `add_*` returned, then hand it to
/// [`LlamaContext::process`](crate::context::LlamaContext::process).
///
/// # Positions are not implied
///
/// Unlike [`LlamaBatch::add`](crate::llama_batch::LlamaBatch::add), adding an
/// entry does *not* take its position — upstream leaves it undefined and
/// expects an explicit [`Self::set_pos`]. The underlying entry starts at
/// position `0`, so a forgotten call is not a type error. Set the position for
/// every entry.
///
/// # Context binding
///
/// `llama_batch_ext_init` reads the context's batch size, sequence count and
/// model geometry, and keeps a pointer to its memory module for position
/// validation. The batch is therefore only usable with the context that built
/// it: [`LlamaContext::process`](crate::context::LlamaContext::process) checks
/// that and refuses anything else, so a batch outliving its context cannot be
/// processed by a later one.
#[derive(Debug)]
pub struct LlamaBatchExt {
    raw: NonNull<llama_cpp_sys_4::llama_batch_ext>,
    /// The context that built this batch, checked by `process`. See
    /// `LlamaContext::id`.
    pub(crate) ctx_id: u64,
    /// One per entry, in upstream's order. See [`Entry`].
    entries: Vec<Entry>,
    /// Entries with output enabled. Mirrors
    /// [`LlamaBatch::initialized_logits`](crate::llama_batch::LlamaBatch) so
    /// the context can validate `get_logits_ith` after processing.
    pub(crate) initialized_logits: Vec<i32>,
    /// Total width of this batch's embeddings, fixed by the first one accepted.
    embd_width: Option<usize>,
    n_pos_per_embd: usize,
    n_batch: u32,
    n_seq_max: u32,
    n_vocab: i32,
}

// SAFETY: the batch is a plain owned buffer of entries. Nothing inside it is
// shared with another thread, and every method that mutates it takes `&mut
// self`. The context pointer it keeps internally is only dereferenced by
// `llama_process`, which this crate reaches only through `&mut LlamaContext`.
unsafe impl Send for LlamaBatchExt {}

impl LlamaBatchExt {
    /// Build a batch for `ctx`.
    ///
    /// Capacity is the context's `n_batch`; the sequence ids it will accept are
    /// `0..n_seq_max`. Both are read from the context here, so they cannot
    /// drift from it later.
    ///
    /// # Errors
    ///
    /// Returns [`BatchExtError::NullReturn`] if llama.cpp could not allocate.
    pub fn new(ctx: &LlamaContext<'_>) -> Result<Self, BatchExtError> {
        let raw = unsafe { llama_cpp_sys_4::llama_batch_ext_init(ctx.context.as_ptr()) };
        let raw = NonNull::new(raw).ok_or(BatchExtError::NullReturn)?;

        // Mirrors `llama_hparams::n_pos_per_embd()`, which the batch uses to
        // decide how many positions to read for an embedding entry.
        let rope_type = ctx.model.rope_type();
        let n_pos_per_embd = if rope_type == llama_cpp_sys_4::LLAMA_ROPE_TYPE_MROPE
            || rope_type == llama_cpp_sys_4::LLAMA_ROPE_TYPE_IMROPE
        {
            MROPE_SECTIONS
        } else {
            1
        };

        Ok(Self {
            raw,
            ctx_id: ctx.id(),
            entries: Vec::new(),
            initialized_logits: Vec::new(),
            embd_width: None,
            n_pos_per_embd,
            n_batch: ctx.n_batch(),
            n_seq_max: ctx.n_seq_max(),
            n_vocab: ctx.model.n_vocab(),
        })
    }

    /// Drop every entry, keeping the allocation for the next round.
    ///
    /// [`BatchEntry`] handles from before the call are no longer valid; the
    /// `set_*` methods reject them rather than writing to a recycled entry.
    pub fn clear(&mut self) {
        unsafe { llama_cpp_sys_4::llama_batch_ext_clear(self.raw.as_ptr()) }
        self.entries.clear();
        self.initialized_logits.clear();
        self.embd_width = None;
    }

    /// Number of entries in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the batch has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether any entry carries no input — neither a token id nor an input
    /// embedding — which makes the batch unprocessable.
    ///
    /// Two things leave one behind: [`Self::add_empty`] that was never filled
    /// in with [`Self::set_token_embedding`], and an [`Self::add_embedding`]
    /// whose block llama.cpp rejected *after* appending the entry. There is no
    /// upstream call to remove an entry, so the way out is [`Self::clear`].
    /// [`LlamaContext::process`](crate::context::LlamaContext::process) checks
    /// this rather than submitting a batch with a hole in it.
    #[must_use]
    pub fn has_incomplete_entry(&self) -> bool {
        self.entries.iter().any(|entry| entry.is_incomplete())
    }

    /// Index of the first entry carrying no input, if any.
    pub(crate) fn first_incomplete_entry(&self) -> Option<i32> {
        let idx = self
            .entries
            .iter()
            .position(|entry| entry.is_incomplete())?;
        i32::try_from(idx).ok()
    }

    /// Maximum entries this batch can hold — the context's `n_batch`.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.n_batch
    }

    /// Positions an embedding entry in this batch requires:
    /// [`MROPE_SECTIONS`] on an M-RoPE model, `1` otherwise. A token entry
    /// always takes one.
    #[must_use]
    pub fn n_pos_per_embd(&self) -> usize {
        self.n_pos_per_embd
    }

    /// Entries with output enabled, in the order they were enabled.
    #[must_use]
    pub fn outputs(&self) -> &[i32] {
        &self.initialized_logits
    }

    /// Add a token entry for sequence `seq_id`.
    ///
    /// Set its position with [`Self::set_pos`]; upstream leaves it undefined.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::Full`] when the batch is at capacity,
    /// [`BatchExtError::InvalidSeqId`] for an out-of-range sequence, or
    /// [`BatchExtError::InvalidToken`] for a token id outside the vocabulary.
    pub fn add_token(
        &mut self,
        LlamaToken(id): LlamaToken,
        seq_id: llama_seq_id,
    ) -> Result<BatchEntry, BatchExtError> {
        // Checked here rather than left to upstream: `llama_batch_ext_add_token`
        // appends the entry and *then* validates the id, so a bad id would
        // leave an entry with no input behind.
        if id < 0 || id >= self.n_vocab {
            return Err(BatchExtError::InvalidToken(id));
        }
        let idx =
            unsafe { llama_cpp_sys_4::llama_batch_ext_add_token(self.raw.as_ptr(), seq_id, id) };
        self.finish_add(idx, seq_id, Entry::token(), BatchExtError::InvalidToken(id))
    }

    /// Add an embedding-only entry for sequence `seq_id`.
    ///
    /// The row width must match the model's decoder input width
    /// (`n_embd_inp`) or its encoder input width, and every embedding in one
    /// batch must have the same width.
    ///
    /// Positions come from [`Self::set_positions`]: an embedding entry takes
    /// [`Self::n_pos_per_embd`] of them, not one.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::Full`], [`BatchExtError::InvalidSeqId`],
    /// [`BatchExtError::EmbeddingWidthMismatch`] if it disagrees with the
    /// batch's established width, or [`BatchExtError::EmbeddingRejected`] if
    /// llama.cpp would not take the block.
    ///
    /// `EmbeddingRejected` here leaves an entry with no input in the batch —
    /// upstream appends before it validates, and offers no way to remove one.
    /// See [`Self::has_incomplete_entry`].
    pub fn add_embedding(
        &mut self,
        embd: EmbeddingRows<'_>,
        seq_id: llama_seq_id,
    ) -> Result<BatchEntry, BatchExtError> {
        self.check_embd_width(embd)?;
        let idx = unsafe {
            llama_cpp_sys_4::llama_batch_ext_add_embd(self.raw.as_ptr(), seq_id, embd.as_raw())
        };
        let entry = self.finish_add(
            idx,
            seq_id,
            Entry::embedding(),
            BatchExtError::EmbeddingRejected,
        );
        if entry.is_ok() {
            self.embd_width = Some(embd.total());
        }
        entry
    }

    /// Add an entry with neither a token id nor an embedding, to be filled in
    /// with [`Self::set_token_embedding`].
    ///
    /// Treated as an embedding entry for positioning: it takes
    /// [`Self::n_pos_per_embd`] positions.
    ///
    /// Until it is filled in, the entry counts as incomplete and
    /// [`LlamaContext::process`](crate::context::LlamaContext::process) will
    /// refuse the batch — see [`Self::has_incomplete_entry`].
    ///
    /// # Errors
    ///
    /// [`BatchExtError::Full`] or [`BatchExtError::InvalidSeqId`].
    pub fn add_empty(&mut self, seq_id: llama_seq_id) -> Result<BatchEntry, BatchExtError> {
        let idx = unsafe { llama_cpp_sys_4::llama_batch_ext_add(self.raw.as_ptr(), seq_id) };
        // `llama_batch_ext_add` has no -2 case; any negative is -1 or -3.
        self.finish_add(
            idx,
            seq_id,
            Entry::empty(),
            BatchExtError::Full(self.n_batch),
        )
    }

    /// Map the `i32` an `add_*` returned onto a [`BatchEntry`] or an error, and
    /// record the entry's kind.
    ///
    /// Upstream's codes are `-1` full, `-2` the input was rejected (what that
    /// means depends on which `add_*` was called, hence `rejected`), `-3`
    /// invalid sequence id.
    fn finish_add(
        &mut self,
        idx: i32,
        seq_id: llama_seq_id,
        entry: Entry,
        rejected: BatchExtError,
    ) -> Result<BatchEntry, BatchExtError> {
        match idx {
            // `-1` (full) and `-3` (bad sequence id) are raised before the
            // entry is appended, so there is nothing to mirror.
            -1 => Err(BatchExtError::Full(self.n_batch)),
            -3 => Err(BatchExtError::InvalidSeqId {
                seq_id,
                n_seq_max: self.n_seq_max,
            }),
            // `-2` is raised *after* it was appended — the input was rejected,
            // not the entry. Mirror it anyway, or every later index would be
            // off by one against upstream's.
            -2 => {
                self.entries.push(Entry::empty());
                Err(rejected)
            }
            idx if idx < 0 => Err(rejected),
            idx => {
                // `add_*` appends, so the new entry is always at `len()`. Keep
                // the mirror in step even if upstream ever renumbers.
                debug_assert_eq!(usize::try_from(idx).ok(), Some(self.entries.len()));
                self.entries.push(entry);
                Ok(BatchEntry(idx))
            }
        }
    }

    /// Also feed this entry to `seq_id`. The entry keeps the sequence it was
    /// added with.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::NoSuchEntry`] for a stale handle, or
    /// [`BatchExtError::InvalidSeqId`] for an out-of-range sequence.
    pub fn add_seq(
        &mut self,
        entry: BatchEntry,
        seq_id: llama_seq_id,
    ) -> Result<(), BatchExtError> {
        self.check_entry(entry)?;
        let ok =
            unsafe { llama_cpp_sys_4::llama_batch_ext_add_seq(self.raw.as_ptr(), entry.0, seq_id) };
        if ok {
            Ok(())
        } else {
            Err(BatchExtError::InvalidSeqId {
                seq_id,
                n_seq_max: self.n_seq_max,
            })
        }
    }

    /// Set the position of a token entry.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::NoSuchEntry`] for a stale handle, or
    /// [`BatchExtError::PositionArity`] if the entry is an embedding entry on
    /// an M-RoPE model and needs [`Self::n_pos_per_embd`] positions — use
    /// [`Self::set_positions`] for those.
    pub fn set_pos(&mut self, entry: BatchEntry, pos: llama_pos) -> Result<(), BatchExtError> {
        self.set_positions(entry, &[pos])
    }

    /// Set all positions of an entry.
    ///
    /// `pos` must hold exactly the number of positions the entry takes: one for
    /// a token entry, [`Self::n_pos_per_embd`] for an embedding entry. On an
    /// M-RoPE model that is `[t, y, x, z]`, where `t` is also the KV-cache
    /// position.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::NoSuchEntry`] for a stale handle, or
    /// [`BatchExtError::PositionArity`] for the wrong number of positions —
    /// llama.cpp reads the arity it expects straight off the pointer, so a
    /// short slice would read out of bounds.
    pub fn set_positions(
        &mut self,
        entry: BatchEntry,
        pos: &[llama_pos],
    ) -> Result<(), BatchExtError> {
        let expected = self.check_entry(entry)?.n_pos(self.n_pos_per_embd);
        if pos.len() != expected {
            return Err(BatchExtError::PositionArity {
                entry: entry.0,
                expected,
                got: pos.len(),
            });
        }
        let ok = unsafe {
            llama_cpp_sys_4::llama_batch_ext_set_pos(self.raw.as_ptr(), entry.0, pos.as_ptr())
        };
        debug_assert!(ok, "set_pos rejected an entry that check_entry accepted");
        Ok(())
    }

    /// Attach an input embedding to an entry that already has a token id, so
    /// the entry carries both.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::NoSuchEntry`] for a stale handle,
    /// [`BatchExtError::EmbeddingAlreadySet`] if the entry has one,
    /// [`BatchExtError::EmbeddingWidthMismatch`] if it disagrees with the
    /// batch's established width, or [`BatchExtError::EmbeddingRejected`] if
    /// llama.cpp would not take the block.
    pub fn set_token_embedding(
        &mut self,
        entry: BatchEntry,
        embd: EmbeddingRows<'_>,
    ) -> Result<(), BatchExtError> {
        if self.check_entry(entry)?.has_embd {
            return Err(BatchExtError::EmbeddingAlreadySet(entry.0));
        }
        self.check_embd_width(embd)?;
        let ok = unsafe {
            llama_cpp_sys_4::llama_batch_ext_set_embd_token(
                self.raw.as_ptr(),
                entry.0,
                embd.as_raw(),
            )
        };
        if !ok {
            return Err(BatchExtError::EmbeddingRejected);
        }
        self.embd_width = Some(embd.total());
        self.mark(entry, |slot| slot.has_embd = true);
        Ok(())
    }

    /// Attach a *state* embedding to an entry — hidden state carried over from
    /// an earlier stage, such as the target model's layers for MTP or a vision
    /// encoder's deepstack output.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::NoSuchEntry`] for a stale handle,
    /// [`BatchExtError::StateAlreadySet`] if the entry has one, or
    /// [`BatchExtError::EmbeddingRejected`] when the width is wrong or this
    /// context takes no state embedding at all — only an MTP context does, so
    /// check
    /// [`LlamaContext::context_type`](crate::context::LlamaContext::context_type).
    pub fn set_state_embedding(
        &mut self,
        entry: BatchEntry,
        state: EmbeddingRows<'_>,
    ) -> Result<(), BatchExtError> {
        if self.check_entry(entry)?.has_state {
            return Err(BatchExtError::StateAlreadySet(entry.0));
        }
        let ok = unsafe {
            llama_cpp_sys_4::llama_batch_ext_set_embd_state(
                self.raw.as_ptr(),
                entry.0,
                state.as_raw(),
            )
        };
        if !ok {
            return Err(BatchExtError::EmbeddingRejected);
        }
        self.mark(entry, |slot| slot.has_state = true);
        Ok(())
    }

    /// Enable or disable output for an entry, making its logits and embeddings
    /// readable after processing.
    ///
    /// Upstream exposes this twice — `llama_batch_ext_set_output_logits` and
    /// `llama_batch_ext_set_output_embd` — and the two call the same setter, so
    /// there is one method here. Should they ever diverge, this one keeps the
    /// logits meaning.
    ///
    /// # Errors
    ///
    /// [`BatchExtError::NoSuchEntry`] for a stale handle.
    pub fn set_output(&mut self, entry: BatchEntry, value: bool) -> Result<(), BatchExtError> {
        self.check_entry(entry)?;
        let ok = unsafe {
            llama_cpp_sys_4::llama_batch_ext_set_output_logits(self.raw.as_ptr(), entry.0, value)
        };
        debug_assert!(ok, "set_output rejected an entry that check_entry accepted");

        if value {
            if !self.initialized_logits.contains(&entry.0) {
                self.initialized_logits.push(entry.0);
            }
        } else {
            self.initialized_logits.retain(|i| *i != entry.0);
        }
        Ok(())
    }

    /// Mark what a joint decision head should read this entry as.
    ///
    /// Only decision models (`clef` and friends) look at this; see
    /// [`LlamaDecisionOrder`].
    ///
    /// # Errors
    ///
    /// [`BatchExtError::NoSuchEntry`] for a stale handle.
    pub fn set_decision_order(
        &mut self,
        entry: BatchEntry,
        order: LlamaDecisionOrder,
    ) -> Result<(), BatchExtError> {
        self.check_entry(entry)?;
        let ok = unsafe {
            llama_cpp_sys_4::llama_batch_ext_set_decision_order(
                self.raw.as_ptr(),
                entry.0,
                order.into(),
            )
        };
        debug_assert!(
            ok,
            "set_decision_order rejected an entry that check_entry accepted"
        );
        Ok(())
    }

    /// Validate an entry handle against this batch, returning its state.
    fn check_entry(&self, entry: BatchEntry) -> Result<Entry, BatchExtError> {
        usize::try_from(entry.0)
            .ok()
            .and_then(|idx| self.entries.get(idx))
            .copied()
            .ok_or(BatchExtError::NoSuchEntry {
                entry: entry.0,
                len: self.entries.len(),
            })
    }

    /// Reject an embedding that disagrees with the width the batch settled on,
    /// which upstream would otherwise reject with the detail only in its log.
    fn check_embd_width(&self, embd: EmbeddingRows<'_>) -> Result<(), BatchExtError> {
        match self.embd_width {
            Some(expected) if expected != embd.total() => {
                Err(BatchExtError::EmbeddingWidthMismatch {
                    expected,
                    got: embd.total(),
                })
            }
            _ => Ok(()),
        }
    }

    /// Record that `entry` gained an input or state embedding.
    fn mark(&mut self, entry: BatchEntry, update: impl FnOnce(&mut Entry)) {
        if let Ok(idx) = usize::try_from(entry.0) {
            if let Some(slot) = self.entries.get_mut(idx) {
                update(slot);
            }
        }
    }

    pub(crate) fn as_ptr(&mut self) -> *mut llama_cpp_sys_4::llama_batch_ext {
        self.raw.as_ptr()
    }
}

impl Drop for LlamaBatchExt {
    fn drop(&mut self) {
        // `llama_batch_ext` has no destructor beyond freeing its own vectors —
        // it never dereferences the context pointer it holds — so this is safe
        // even once the originating context is gone.
        unsafe { llama_cpp_sys_4::llama_batch_ext_free(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_rows_rejects_a_shape_that_does_not_match_the_buffer() {
        let data = [0.0_f32; 6];
        assert_eq!(
            EmbeddingRows::new(&data, 2, 4).unwrap_err(),
            BatchExtError::EmbeddingShape {
                n_rows: 2,
                n_embd: 4,
                len: 6
            }
        );
        let rows = EmbeddingRows::new(&data, 2, 3).unwrap();
        assert_eq!((rows.n_rows(), rows.n_embd()), (2, 3));
    }

    #[test]
    fn embedding_rows_row_is_one_row_of_everything() {
        let data = [1.0_f32, 2.0, 3.0];
        let rows = EmbeddingRows::row(&data);
        assert_eq!((rows.n_rows(), rows.n_embd()), (1, 3));
    }

    /// `n_rows * n_embd` must not be allowed to wrap into a length that happens
    /// to match — that would hand llama.cpp a pointer with a bogus extent.
    #[test]
    fn embedding_rows_shape_does_not_wrap() {
        let data = [0.0_f32; 1];
        assert!(EmbeddingRows::new(&data, usize::MAX, 2).is_err());
    }

    #[test]
    fn mrope_sections_matches_upstream() {
        assert_eq!(MROPE_SECTIONS, 4);
    }

    #[test]
    fn process_type_maps_onto_the_sys_constants() {
        assert_eq!(
            llama_cpp_sys_4::llama_process_type::from(LlamaProcessType::Encode),
            llama_cpp_sys_4::LLAMA_PROCESS_TYPE_ENCODE
        );
        assert_eq!(
            llama_cpp_sys_4::llama_process_type::from(LlamaProcessType::Decode),
            llama_cpp_sys_4::LLAMA_PROCESS_TYPE_DECODE
        );
    }

    #[test]
    fn decision_order_maps_onto_the_sys_constants() {
        assert_eq!(
            llama_cpp_sys_4::llama_decision_order::from(LlamaDecisionOrder::None),
            llama_cpp_sys_4::LLAMA_DECISION_ORDER_NONE
        );
        assert_eq!(
            llama_cpp_sys_4::llama_decision_order::from(LlamaDecisionOrder::Option),
            llama_cpp_sys_4::LLAMA_DECISION_ORDER_OPTION
        );
    }
}
