//! Safe wrapper around `llama_context`.
//!
//! Submodules:
//!
//! - [`tensor_capture`] — hook `cb_eval` during [`LlamaContext::decode`] to copy
//!   intermediate tensors (per-layer hidden states, norms, …).
//! - [`memory_breakdown`] — per-buffer memory usage after load/decode.
//! - [`kv_cache`] — sequence copy, shift, and clear helpers.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::pin::Pin;
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};

use llama_cpp_sys_4::llama_pooling_type;
use params::{LlamaContextType, LlamaPoolingType};
use perf::PerfContextData;

use crate::llama_batch::LlamaBatch;
use crate::llama_batch_ext::{LlamaBatchExt, LlamaProcessType};
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError, ProcessError,
};

/// Source of [`LlamaContext::id`].
///
/// A [`LlamaBatchExt`] is bound to the context that built it, and comparing raw
/// `llama_context` pointers is not enough to detect a stale binding: the
/// allocator can hand a new context the address a dropped one had. A counter
/// that only ever goes up cannot be reused that way.
static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(0);

pub mod kv_cache;
pub mod memory_breakdown;
pub mod params;
pub mod perf;
pub mod session;
pub mod tensor_capture;
pub mod tensor_transaction;

pub use memory_breakdown::MemoryBreakdownEntry;
pub use tensor_capture::{CapturedTensor, TensorCapture};
pub use tensor_transaction::{
    CapturedTensorData, TensorAccess, TensorBatchRow, TensorCallbackFailure, TensorDataMut,
    TensorElementType, TensorFiniteValidation, TensorRowMapping, TensorSelector, TensorShape,
    TensorTransaction, TensorTransactionError, TensorTransactionHandler, TensorTransactions,
    TensorWriteback, TransactionalTensorCapture,
};

/// The size of a graph reserved by
/// [`LlamaContext::reserve_graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphReservation {
    /// Compute nodes in the graph.
    ///
    /// ggml exposes no public leaf count, so this is the whole measurement —
    /// enough to compare one batch shape against another, which is what it is
    /// for.
    pub n_nodes: i32,
}

/// [`LlamaContext::reserve_graph`] could not build a graph for that shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("llama.cpp could not reserve a graph for n_tokens={n_tokens}, n_seqs={n_seqs}, n_outputs={n_outputs}")]
pub struct ReserveGraphError {
    /// Tokens requested.
    pub n_tokens: u32,
    /// Sequences requested.
    pub n_seqs: u32,
    /// Outputs requested.
    pub n_outputs: u32,
}

/// A safe wrapper around the `llama_context` C++ context.
///
/// This struct provides a safe interface to interact with the `llama_context` used by the `LlamaModel`.
/// It encapsulates the raw C++ context pointer and provides additional fields for managing the model and
/// context-specific settings like embeddings and logits.
///
/// The `LlamaContext` struct ensures that the C++ context is always valid by using the `NonNull` type for
/// the context pointer, preventing it from being null. The struct also holds a reference to the model
/// (`LlamaModel`) that the context is tied to, along with some internal state like whether embeddings are enabled
/// and the initialized logits for the context.
///
/// # Fields
///
/// - `context`: A non-null pointer to the raw C++ `llama_context`. This is the main context used for interacting with the model.
/// - `model`: A reference to the `LlamaModel` associated with this context. This model provides the data and parameters
///   that the context interacts with.
/// - `initialized_logits`: A vector used to store the initialized logits. These are used in the model's processing and
///   are kept separate from the context data.
/// - `embeddings_enabled`: A boolean flag indicating whether embeddings are enabled in the context. This is useful for
///   controlling whether embedding data is generated during the interaction with the model.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaContext<'a> {
    pub(crate) context: NonNull<llama_cpp_sys_4::llama_context>,
    /// a reference to the contexts model.
    pub model: &'a LlamaModel,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
    context_type: LlamaContextType,
    tensor_transactions: Option<Pin<Box<TensorTransactions>>>,
    /// Identity of this context, unique for the life of the process. See
    /// [`NEXT_CONTEXT_ID`].
    id: u64,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    /// Creates a new instance of `LlamaContext` with the provided model, context, and embeddings flag.
    ///
    /// This function initializes a new `LlamaContext` object, which is used to interact with the
    /// `LlamaModel`. The context is created from a pointer to a C++ context and the embeddings flag
    /// determines whether embeddings are enabled in the context.
    ///
    /// # Parameters
    ///
    /// - `llama_model`: A reference to an existing `LlamaModel` that will be used with the new context.
    /// - `llama_context`: A non-null pointer to an existing `llama_cpp_sys_4::llama_context` representing
    ///   the context created in previous steps. This context is necessary for interacting with the model.
    /// - `embeddings_enabled`: A boolean flag indicating whether embeddings are enabled in this context.
    /// - `context_type`: The exact graph/context mode selected at allocation.
    ///
    /// # Returns
    ///
    /// This function returns a new instance of `LlamaContext` initialized with the given parameters:
    /// - The model reference (`llama_model`) is stored in the context.
    /// - The raw context pointer (`llama_context`) is wrapped in a `NonNull` to ensure safety.
    /// - The `embeddings_enabled` flag is used to determine if embeddings are enabled for the context.
    ///
    /// # Example
    /// ```ignore
    /// let llama_model = LlamaModel::load_from_file(&backend, "path/to/model", &params).unwrap();
    /// let context_ptr = NonNull::new(some_llama_context_ptr).unwrap();
    /// let context = LlamaContext::new(
    ///     &llama_model,
    ///     context_ptr,
    ///     true,
    ///     LlamaContextType::Default,
    ///     None,
    /// );
    /// // Now you can use the context
    /// ```
    pub(crate) fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_sys_4::llama_context>,
        embeddings_enabled: bool,
        context_type: LlamaContextType,
        tensor_transactions: Option<Pin<Box<TensorTransactions>>>,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
            embeddings_enabled,
            context_type,
            tensor_transactions,
            id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Identity of this context, unique among every context this process has
    /// created. Used to tie a [`LlamaBatchExt`] to the context that built it.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Returns the context type selected at native allocation.
    #[must_use]
    pub fn context_type(&self) -> LlamaContextType {
        self.context_type
    }

    /// Gets the max number of logical tokens that can be submitted to decode. Must be greater than or equal to `n_ubatch`.
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch. Must be less than or equal to `n_batch`.
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_ubatch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result =
            unsafe { llama_cpp_sys_4::llama_decode(self.context.as_ptr(), batch.llama_batch) };
        if let Some(failure) = self
            .tensor_transactions
            .as_ref()
            .and_then(|transactions| transactions.failure())
            .cloned()
        {
            return Err(DecodeError::TensorCallback(failure));
        }

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Run an entry-wise [`LlamaBatchExt`] through this context's encoder or
    /// decoder.
    ///
    /// This is `llama_process`, upstream's entry point for the
    /// `llama_batch_ext` API. Use it when the batch needs something
    /// [`LlamaBatch`] cannot express — per-entry embeddings, M-RoPE positions,
    /// or an MTP state embedding. For a plain token batch,
    /// [`Self::decode`] and [`Self::encode`] are the same work with less
    /// ceremony.
    ///
    /// # Decode lifecycle hooks do not apply
    ///
    /// This crate's tensor transactions
    /// ([`LlamaContextParams::with_tensor_transactions`](crate::context::params::LlamaContextParams))
    /// are driven by hooks this repo patches into `llama_decode`.
    /// `llama_process` reaches the graph without passing through it, and the
    /// hook signature is built around `llama_batch`, so it cannot describe an
    /// entry-wise batch at all. Rather than run the batch with the transactions
    /// silently inert, this returns
    /// [`ProcessError::TensorTransactionsUnsupported`].
    ///
    /// # Errors
    ///
    /// - [`ProcessError::ContextMismatch`] if `batch` was built for a
    ///   different context — it borrows its originating context's batch size,
    ///   sequence count and memory module, so it is not transferable.
    /// - [`ProcessError::TensorTransactionsUnsupported`] as described above.
    /// - [`ProcessError::IncompleteEntry`] if an entry carries no input; see
    ///   [`LlamaBatchExt::has_incomplete_entry`].
    /// - Otherwise whatever `llama_decode` would report; the return codes are
    ///   the same.
    pub fn process(
        &mut self,
        batch: &mut LlamaBatchExt,
        kind: LlamaProcessType,
    ) -> Result<(), ProcessError> {
        if batch.ctx_id != self.id {
            return Err(ProcessError::ContextMismatch);
        }
        if self.tensor_transactions.is_some() {
            return Err(ProcessError::TensorTransactionsUnsupported);
        }
        if let Some(entry) = batch.first_incomplete_entry() {
            return Err(ProcessError::IncompleteEntry(entry));
        }

        let result = unsafe {
            llama_cpp_sys_4::llama_process(self.context.as_ptr(), kind.into(), batch.as_ptr())
        };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(ProcessError::from(error)),
        }
    }

    /// Reserve a compute graph for a batch of the given shape, returning how
    /// big it is.
    ///
    /// Two uses. As **warmup**, it forces the graph build and buffer
    /// allocation that the first real decode would otherwise pay for, so
    /// first-token latency stops being an outlier. As **measurement**, the
    /// returned node count is the graph's size for that shape, which
    /// is what
    /// [`PerfContextData::n_reallocs_diff`](crate::context::perf::PerfContextData::n_reallocs_diff)
    /// counts changes in.
    ///
    /// Reserve the shapes you actually decode — typically one for prefill
    /// (`n_tokens = n_batch`) and one for single-token generation.
    ///
    /// # Why this does not hand back the graph
    ///
    /// `llama_graph_reserve` returns a `ggml_cgraph *` that upstream documents
    /// as valid only *until the next call to `llama_graph_reserve`* — it is
    /// owned by the context and reused. Returning it would be a borrow this
    /// type cannot enforce, so the counts come back instead; they are the part
    /// a caller can act on.
    ///
    /// # Errors
    ///
    /// Returns [`ReserveGraphError`] if llama.cpp could not build a graph for
    /// that shape, which is what an over-large `n_tokens` or `n_outputs` looks
    /// like.
    pub fn reserve_graph(
        &mut self,
        n_tokens: u32,
        n_seqs: u32,
        n_outputs: u32,
    ) -> Result<GraphReservation, ReserveGraphError> {
        // SAFETY: a live `&mut self` gives an exclusive, non-null context, and
        // the three arguments are scalars llama.cpp range-checks itself. The
        // returned pointer is only read here and never escapes — see the note
        // above on its lifetime.
        let graph = unsafe {
            llama_cpp_sys_4::llama_graph_reserve(self.context.as_ptr(), n_tokens, n_seqs, n_outputs)
        };
        if graph.is_null() {
            return Err(ReserveGraphError {
                n_tokens,
                n_seqs,
                n_outputs,
            });
        }
        // SAFETY: non-null and valid until the next `llama_graph_reserve`,
        // which cannot happen between here and this read.
        let n_nodes = unsafe { llama_cpp_sys_4::ggml_graph_n_nodes(graph) };
        Ok(GraphReservation { n_nodes })
    }

    /// Returns the owned tensor transaction program attached to this context.
    #[must_use]
    pub fn tensor_transactions(&self) -> Option<&TensorTransactions> {
        self.tensor_transactions.as_deref()
    }

    /// Returns the owned tensor transaction program attached to this context.
    ///
    /// Retained captures can be removed with [`TensorTransactions::take_captures`].
    pub fn tensor_transactions_mut(&mut self) -> Option<&mut TensorTransactions> {
        self.tensor_transactions
            .as_mut()
            .map(|transactions| transactions.as_mut().get_mut())
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let result =
            unsafe { llama_cpp_sys_4::llama_encode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(EncodeError::from(error)),
        }
    }

    /// Return Pooling type for Llama's Context
    #[must_use]
    pub fn pooling_type(&self) -> LlamaPoolingType {
        let pooling_type = unsafe { llama_pooling_type(self.context.as_ptr()) };

        LlamaPoolingType::from(pooling_type)
    }

    /// Get the embeddings for the `i`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_sys_4::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_4::llama_get_embeddings_seq(self.context.as_ptr(), i);

            // Technically also possible whenever `i >= max(batch.n_seq)`, but can't check that here.
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_4::llama_get_embeddings_ith(self.context.as_ptr(), i);
            // Technically also possible whenever `i >= batch.n_tokens`, but no good way of checking `n_tokens` here.
            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the logits for the last token in the context.
    ///
    /// # Returns
    /// An iterator over unsorted `LlamaTokenData` containing the
    /// logits for the last token in the context.
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    pub fn candidates(&self) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits()).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the last token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates(), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    #[must_use]
    pub fn token_data_array(&self) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates(), false)
    }

    /// Token logits obtained from the last call to `decode()`.
    /// The logits for which `batch.logits[i] != 0` are stored contiguously
    /// in the order they have appeared in the batch.
    /// Rows: number of tokens for which `batch.logits[i] != 0`
    /// Cols: `n_vocab`
    ///
    /// # Returns
    ///
    /// A slice containing the logits for the last decoded token.
    /// The size corresponds to the `n_vocab` parameter of the context's model.
    ///
    /// # Panics
    ///
    /// - `n_vocab` does not fit into a usize
    /// - token data returned is null
    #[must_use]
    pub fn get_logits(&self) -> &[f32] {
        let data = unsafe { llama_cpp_sys_4::llama_get_logits(self.context.as_ptr()) };
        assert!(!data.is_null(), "logits data for last token is null");
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `i` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        assert!(
            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
            "n_ctx ({}) must be greater than i ({})",
            self.n_ctx(),
            i
        );

        let data = unsafe { llama_cpp_sys_4::llama_get_logits_ith(self.context.as_ptr(), i) };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Get the number of context tokens per sequence.
    #[must_use]
    pub fn n_ctx_seq(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_ctx_seq(self.context.as_ptr()) }
    }

    /// Get the maximum number of sequences.
    #[must_use]
    pub fn n_seq_max(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_seq_max(self.context.as_ptr()) }
    }

    /// Get the number of recurrent-state snapshots per sequence.
    #[must_use]
    pub fn n_rs_seq(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_rs_seq(self.context.as_ptr()) }
    }

    /// Get the number of threads used for generation.
    #[must_use]
    pub fn n_threads(&self) -> i32 {
        unsafe { llama_cpp_sys_4::llama_n_threads(self.context.as_ptr()) }
    }

    /// Get the number of threads used for batch processing.
    #[must_use]
    pub fn n_threads_batch(&self) -> i32 {
        unsafe { llama_cpp_sys_4::llama_n_threads_batch(self.context.as_ptr()) }
    }

    /// Set the number of threads used for generation and batch processing.
    pub fn set_n_threads(&mut self, n_threads: i32, n_threads_batch: i32) {
        unsafe {
            llama_cpp_sys_4::llama_set_n_threads(self.context.as_ptr(), n_threads, n_threads_batch);
        }
    }

    /// Set whether to use causal attention.
    ///
    /// If set to `false`, the model will use non-causal attention, which is
    /// needed for embedding models.
    pub fn set_causal_attn(&mut self, causal_attn: bool) {
        unsafe {
            llama_cpp_sys_4::llama_set_causal_attn(self.context.as_ptr(), causal_attn);
        }
    }

    /// Whether the context is currently using causal attention.
    ///
    /// The counterpart to [`Self::set_causal_attn`]. Worth reading back rather
    /// than tracking yourself: the initial value comes from
    /// [`LlamaContextParams::with_attention_type`](crate::context::params::LlamaContextParams::with_attention_type)
    /// via the model's own default, not from anything the caller set.
    #[must_use]
    pub fn causal_attn(&self) -> bool {
        unsafe { llama_cpp_sys_4::llama_get_causal_attn(self.context.as_ptr()) }
    }

    /// Set whether to compute embeddings.
    ///
    /// This allows toggling embedding mode at runtime (as opposed to only at
    /// context creation time).
    pub fn set_embeddings(&mut self, embeddings: bool) {
        self.embeddings_enabled = embeddings;
        unsafe {
            llama_cpp_sys_4::llama_set_embeddings(self.context.as_ptr(), embeddings);
        }
    }

    /// Mark the next computation as a warmup run.
    ///
    /// Warmup runs are useful for GPU backends to compile kernels before
    /// actual inference begins.
    pub fn set_warmup(&mut self, warmup: bool) {
        unsafe {
            llama_cpp_sys_4::llama_set_warmup(self.context.as_ptr(), warmup);
        }
    }

    /// Wait for all pending async computations to finish.
    pub fn synchronize(&mut self) {
        unsafe {
            llama_cpp_sys_4::llama_synchronize(self.context.as_ptr());
        }
    }

    /// Get all embeddings for the current context.
    ///
    /// Returns a slice of all embeddings from the last decoded batch.
    /// For pooled embeddings use [`embeddings_seq_ith`](Self::embeddings_seq_ith) instead.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the embeddings pointer is null.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn get_embeddings(&self) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_4::llama_get_embeddings(self.context.as_ptr());
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Toggle extraction of next-n embeddings (Rust name: pre-norm) — hidden
    /// states used by MTP draft heads. Upstream C API: `llama_set_embeddings_nextn`
    /// (llama.cpp PR #23198 and later renames).
    ///
    /// If `masked` is `true`, pre-norm rows are extracted only for tokens
    /// whose `batch.logits[i]` is non-zero. If `masked` is `false`, rows are
    /// extracted for every token in the batch regardless of `batch.logits` —
    /// callers can then leave `batch.logits[i] = false` on prompt-fill
    /// positions and avoid copying the full logits row for each one.
    ///
    /// Upstream's MTP session init configures pre-norm extraction on the target
    /// and draft contexts automatically. Call this manually only for custom
    /// speculative setups.
    pub fn set_embeddings_pre_norm(&mut self, value: bool, masked: bool) {
        unsafe {
            llama_cpp_sys_4::llama_set_embeddings_nextn(self.context.as_ptr(), value, masked);
        }
    }

    /// Get the full pre-norm embeddings buffer for the last decoded batch.
    ///
    /// Returns `None` when pre-norm embeddings are disabled or the buffer
    /// hasn't been populated. The length of the returned slice is
    /// `n_embd * <number of pre-norm rows>` — interpretation of the row
    /// count depends on whether the setter was called with `masked=true`
    /// (one row per sampled token) or `masked=false` (one row per batch
    /// token). Use [`get_embeddings_pre_norm_ith`](Self::get_embeddings_pre_norm_ith)
    /// when you only need a single row.
    ///
    /// # Panics
    ///
    /// Panics if `n_embd` does not fit in `usize`.
    #[must_use]
    pub fn get_embeddings_pre_norm(&self) -> Option<&[f32]> {
        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");
        unsafe {
            let p = llama_cpp_sys_4::llama_get_embeddings_nextn(self.context.as_ptr());
            if p.is_null() {
                None
            } else {
                Some(slice::from_raw_parts(p, n_embd))
            }
        }
    }

    /// Get the pre-norm embedding row for the `i`th output position of the
    /// last decoded batch. Returns `None` if upstream rejects the index
    /// (e.g. masked mode with `batch.logits[i] == 0`, or out of range).
    ///
    /// # Panics
    ///
    /// Panics if `n_embd` does not fit in `usize`.
    #[must_use]
    pub fn get_embeddings_pre_norm_ith(&self, i: i32) -> Option<&[f32]> {
        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");
        unsafe {
            let p = llama_cpp_sys_4::llama_get_embeddings_nextn_ith(self.context.as_ptr(), i);
            if p.is_null() {
                None
            } else {
                Some(slice::from_raw_parts(p, n_embd))
            }
        }
    }

    /// Select which `NextN` block the MTP draft graph runs.
    ///
    /// `offset` indexes past the trunk transformer layers (`0` = first `NextN`
    /// head). Required for multi-head MTP models such as Step3.5; restore to
    /// `0` after drafting. See [`crate::mtp`] for the full speculative loop.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use llama_cpp_4::model::LlamaModel;
    /// # use llama_cpp_4::mtp::MtpSession;
    /// # use llama_cpp_4::token::LlamaToken;
    /// # fn f(model: &LlamaModel, session: &mut MtpSession<'_, '_>, n_past: i32,
    /// #      last_token: LlamaToken) -> Result<(), Box<dyn std::error::Error>> {
    /// for head in 0..model.n_layer_nextn() {
    ///     // Via the session: it borrows the draft context exclusively.
    ///     session.draft_context_mut().set_nextn_layer_offset(head);
    ///     let drafts = session.draft(0, n_past, last_token)?;
    /// }
    /// session.draft_context_mut().set_nextn_layer_offset(0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_nextn_layer_offset(&mut self, offset: i32) {
        unsafe {
            llama_cpp_sys_4::llama_set_nextn_layer_offset(self.context.as_ptr(), offset);
        }
    }

    /// Return the paired context set via
    /// [`crate::context::params::LlamaContextParams::with_ctx_other`].
    ///
    /// The pointer refers to the other live context created during
    /// [`crate::model::LlamaModel::new_context`]; it is `None` when no pairing
    /// was configured.
    #[must_use]
    pub fn ctx_other(&self) -> Option<NonNull<llama_cpp_sys_4::llama_context>> {
        NonNull::new(unsafe { llama_cpp_sys_4::llama_get_ctx_other(self.context.as_ptr()) })
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_sys_4::ggml_time_init() }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> PerfContextData {
        let perf_context_data =
            unsafe { llama_cpp_sys_4::llama_perf_context(self.context.as_ptr()) };
        PerfContextData { perf_context_data }
    }

    /// Reset the performance counters for the context.
    pub fn perf_context_reset(&mut self) {
        unsafe { llama_cpp_sys_4::llama_perf_context_reset(self.context.as_ptr()) }
    }

    /// Check if the KV cache memory supports shifting.
    #[must_use]
    pub fn memory_can_shift(&self) -> bool {
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_can_shift(mem)
        }
    }

    /// Get the minimum position in a sequence's KV cache.
    #[must_use]
    pub fn memory_seq_pos_min(&self, seq_id: i32) -> i32 {
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_pos_min(mem, seq_id)
        }
    }

    /// Print a human-readable memory breakdown to stderr via llama.cpp.
    ///
    /// For structured access use [`Self::memory_breakdown`].
    pub fn memory_breakdown_print(&self) {
        unsafe {
            llama_cpp_sys_4::common_memory_breakdown_print(self.context.as_ptr());
        }
    }

    /// Return structured per-buffer memory usage for this context.
    ///
    /// Each [`memory_breakdown::MemoryBreakdownEntry`] reports model weights,
    /// KV / recurrent cache, and compute scratch bytes for one ggml buffer
    /// type. Returns an empty vector when no buffers are registered.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llama_cpp_4::prelude::*;
    ///
    /// fn main() {
    ///     let backend = LlamaBackend::init().unwrap();
    ///     let model = LlamaModel::load_from_file(&backend, "model.gguf", &LlamaModelParams::default()).unwrap();
    ///     let ctx = model.new_context(&backend, LlamaContextParams::default()).unwrap();
    ///     let total: usize = ctx.memory_breakdown().iter().map(|e| e.total()).sum();
    ///     println!("context uses {total} bytes across all buffer types");
    /// }
    /// ```
    #[must_use]
    pub fn memory_breakdown(&self) -> Vec<memory_breakdown::MemoryBreakdownEntry> {
        memory_breakdown::collect_memory_breakdown(self.context.as_ptr())
    }

    /// Enable or disable extraction of input embeddings for a transformer layer.
    ///
    /// Maps to `llama_set_embeddings_layer_inp`. After a successful
    /// [`Self::decode`], read the vector with [`Self::get_embeddings_layer_inp`].
    pub fn set_embeddings_layer_inp(&mut self, layer_id: u32, value: bool) {
        unsafe {
            llama_cpp_sys_4::llama_set_embeddings_layer_inp(self.context.as_ptr(), layer_id, value);
        }
    }

    /// Get input embeddings for `layer_id` from the last decoded batch.
    ///
    /// Returns `None` when the layer was not enabled via
    /// [`Self::set_embeddings_layer_inp`] or when upstream has no data for
    /// `layer_id`. The slice length is [`LlamaModel::n_embd`].
    ///
    /// # Panics
    ///
    /// Panics if `n_embd` does not fit in `usize`.
    #[must_use]
    pub fn get_embeddings_layer_inp(&self, layer_id: u32) -> Option<&[f32]> {
        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");
        unsafe {
            let p =
                llama_cpp_sys_4::llama_get_embeddings_layer_inp(self.context.as_ptr(), layer_id);
            if p.is_null() {
                None
            } else {
                Some(slice::from_raw_parts(p, n_embd))
            }
        }
    }

    /// Get the size of the full context state in bytes.
    ///
    /// This is the size needed for [`state_get_data`](Self::state_get_data) and
    /// [`state_set_data`](Self::state_set_data).
    #[must_use]
    pub fn state_get_size(&mut self) -> usize {
        unsafe { llama_cpp_sys_4::llama_state_get_size(self.context.as_ptr()) }
    }

    /// Copy the full context state into a byte buffer.
    ///
    /// The buffer must be at least [`state_get_size`](Self::state_get_size) bytes.
    ///
    /// Returns the number of bytes written.
    pub fn state_get_data(&mut self, dst: &mut [u8]) -> usize {
        unsafe {
            llama_cpp_sys_4::llama_state_get_data(
                self.context.as_ptr(),
                dst.as_mut_ptr(),
                dst.len(),
            )
        }
    }

    /// Restore the full context state from a byte buffer.
    ///
    /// Returns the number of bytes read.
    pub fn state_set_data(&mut self, src: &[u8]) -> usize {
        unsafe {
            llama_cpp_sys_4::llama_state_set_data(self.context.as_ptr(), src.as_ptr(), src.len())
        }
    }

    /// Save the context state to a file along with the given tokens.
    ///
    /// Returns `true` on success.
    ///
    /// # Panics
    ///
    /// Panics if the path contains null bytes.
    pub fn state_save_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        tokens: &[LlamaToken],
    ) -> bool {
        let path_str = path.as_ref().to_str().expect("path is not valid UTF-8");
        let c_path = std::ffi::CString::new(path_str).expect("path contains null bytes");
        unsafe {
            llama_cpp_sys_4::llama_state_save_file(
                self.context.as_ptr(),
                c_path.as_ptr(),
                tokens.as_ptr().cast(),
                tokens.len(),
            )
        }
    }

    /// Load a context state from a file.
    ///
    /// Returns `true` on success and fills `tokens_out` with the saved tokens.
    ///
    /// # Panics
    ///
    /// Panics if the path contains null bytes.
    pub fn state_load_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        tokens_out: &mut Vec<LlamaToken>,
        n_token_capacity: usize,
    ) -> bool {
        tokens_out.resize(n_token_capacity, LlamaToken(0));
        let mut n_token_count: usize = 0;
        let path_str = path.as_ref().to_str().expect("path is not valid UTF-8");
        let c_path = std::ffi::CString::new(path_str).expect("path contains null bytes");
        let ok = unsafe {
            llama_cpp_sys_4::llama_state_load_file(
                self.context.as_ptr(),
                c_path.as_ptr(),
                tokens_out.as_mut_ptr().cast(),
                n_token_capacity,
                std::ptr::addr_of_mut!(n_token_count),
            )
        };
        if ok {
            tokens_out.truncate(n_token_count);
        }
        ok
    }

    /// Get the size of a single sequence's state in bytes.
    #[must_use]
    pub fn state_seq_get_size(&mut self, seq_id: i32) -> usize {
        unsafe { llama_cpp_sys_4::llama_state_seq_get_size(self.context.as_ptr(), seq_id) }
    }

    /// Copy a single sequence's state into a byte buffer.
    ///
    /// Returns the number of bytes written.
    pub fn state_seq_get_data(&mut self, dst: &mut [u8], seq_id: i32) -> usize {
        unsafe {
            llama_cpp_sys_4::llama_state_seq_get_data(
                self.context.as_ptr(),
                dst.as_mut_ptr(),
                dst.len(),
                seq_id,
            )
        }
    }

    /// Restore a single sequence's state from a byte buffer.
    ///
    /// Returns the number of bytes read.
    pub fn state_seq_set_data(&mut self, src: &[u8], dest_seq_id: i32) -> usize {
        unsafe {
            llama_cpp_sys_4::llama_state_seq_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                dest_seq_id,
            )
        }
    }

    /// Save a single sequence's state to a file.
    ///
    /// Returns the number of bytes written (0 on failure).
    ///
    /// # Panics
    ///
    /// Panics if the path contains null bytes.
    pub fn state_seq_save_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        seq_id: i32,
        tokens: &[LlamaToken],
    ) -> usize {
        let path_str = path.as_ref().to_str().expect("path is not valid UTF-8");
        let c_path = std::ffi::CString::new(path_str).expect("path contains null bytes");
        unsafe {
            llama_cpp_sys_4::llama_state_seq_save_file(
                self.context.as_ptr(),
                c_path.as_ptr(),
                seq_id,
                tokens.as_ptr().cast(),
                tokens.len(),
            )
        }
    }

    /// Load a single sequence's state from a file.
    ///
    /// Returns the number of bytes read (0 on failure).
    ///
    /// # Panics
    ///
    /// Panics if the path contains null bytes.
    pub fn state_seq_load_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        dest_seq_id: i32,
        tokens_out: &mut Vec<LlamaToken>,
        n_token_capacity: usize,
    ) -> usize {
        tokens_out.resize(n_token_capacity, LlamaToken(0));
        let mut n_token_count: usize = 0;
        let path_str = path.as_ref().to_str().expect("path is not valid UTF-8");
        let c_path = std::ffi::CString::new(path_str).expect("path contains null bytes");
        let ret = unsafe {
            llama_cpp_sys_4::llama_state_seq_load_file(
                self.context.as_ptr(),
                c_path.as_ptr(),
                dest_seq_id,
                tokens_out.as_mut_ptr().cast(),
                n_token_capacity,
                std::ptr::addr_of_mut!(n_token_count),
            )
        };
        if ret > 0 {
            tokens_out.truncate(n_token_count);
        }
        ret
    }

    /// Set a control vector on the context.
    ///
    /// # Parameters
    ///
    /// - `data`: The control vector data (embedding values). Pass an empty slice to clear.
    /// - `n_embd`: The embedding dimension.
    /// - `il_start`: The starting layer index (inclusive).
    /// - `il_end`: The ending layer index (exclusive).
    ///
    /// # Errors
    ///
    /// Returns `Err` with the error code if the operation fails.
    pub fn set_adapter_cvec(
        &mut self,
        data: &[f32],
        n_embd: i32,
        il_start: i32,
        il_end: i32,
    ) -> Result<(), i32> {
        let ret = unsafe {
            llama_cpp_sys_4::llama_set_adapter_cvec(
                self.context.as_ptr(),
                data.as_ptr(),
                data.len(),
                n_embd,
                il_start,
                il_end,
            )
        };
        if ret != 0 {
            Err(ret)
        } else {
            Ok(())
        }
    }

    /// Get sampled token debug info for the `i`th position.
    ///
    /// Returns the sampled token at position `i` from the last decode call.
    #[must_use]
    pub fn get_sampled_token_ith(&self, i: i32) -> LlamaToken {
        let token =
            unsafe { llama_cpp_sys_4::llama_get_sampled_token_ith(self.context.as_ptr(), i) };
        LlamaToken(token)
    }

    /// Get sampled candidate tokens for the `i`th position.
    ///
    /// Returns a slice of candidate tokens from the last decode call.
    #[must_use]
    pub fn get_sampled_candidates_ith(&self, i: i32) -> &[LlamaToken] {
        let count = unsafe {
            llama_cpp_sys_4::llama_get_sampled_candidates_count_ith(self.context.as_ptr(), i)
        } as usize;
        if count == 0 {
            return &[];
        }
        let ptr =
            unsafe { llama_cpp_sys_4::llama_get_sampled_candidates_ith(self.context.as_ptr(), i) };
        if ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(ptr.cast::<LlamaToken>(), count) }
    }

    /// Get the number of sampled logits for the `i`th position.
    #[must_use]
    pub fn get_sampled_logits_count_ith(&self, i: i32) -> u32 {
        unsafe { llama_cpp_sys_4::llama_get_sampled_logits_count_ith(self.context.as_ptr(), i) }
    }

    /// Get sampled logits for the `i`th position.
    ///
    /// Returns a slice of logit values from the last decode call.
    #[must_use]
    pub fn get_sampled_logits_ith(&self, i: i32) -> &[f32] {
        let count = self.get_sampled_logits_count_ith(i) as usize;
        if count == 0 {
            return &[];
        }
        let ptr =
            unsafe { llama_cpp_sys_4::llama_get_sampled_logits_ith(self.context.as_ptr(), i) };
        if ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(ptr, count) }
    }

    /// Get the number of sampled probabilities for the `i`th position.
    #[must_use]
    pub fn get_sampled_probs_count_ith(&self, i: i32) -> u32 {
        unsafe { llama_cpp_sys_4::llama_get_sampled_probs_count_ith(self.context.as_ptr(), i) }
    }

    /// Get sampled probabilities for the `i`th position.
    ///
    /// Returns a slice of probability values from the last decode call.
    #[must_use]
    pub fn get_sampled_probs_ith(&self, i: i32) -> &[f32] {
        let count = self.get_sampled_probs_count_ith(i) as usize;
        if count == 0 {
            return &[];
        }
        let ptr = unsafe { llama_cpp_sys_4::llama_get_sampled_probs_ith(self.context.as_ptr(), i) };
        if ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(ptr, count) }
    }

    /// Get the size of a single sequence's state with flags.
    #[must_use]
    pub fn state_seq_get_size_ext(&mut self, seq_id: i32, flags: u32) -> usize {
        unsafe {
            llama_cpp_sys_4::llama_state_seq_get_size_ext(self.context.as_ptr(), seq_id, flags)
        }
    }

    /// Copy a single sequence's state into a byte buffer with flags.
    ///
    /// Returns the number of bytes written.
    pub fn state_seq_get_data_ext(&mut self, dst: &mut [u8], seq_id: i32, flags: u32) -> usize {
        unsafe {
            llama_cpp_sys_4::llama_state_seq_get_data_ext(
                self.context.as_ptr(),
                dst.as_mut_ptr(),
                dst.len(),
                seq_id,
                flags,
            )
        }
    }

    /// Restore a single sequence's state from a byte buffer with flags.
    ///
    /// Returns the number of bytes read.
    pub fn state_seq_set_data_ext(&mut self, src: &[u8], dest_seq_id: i32, flags: u32) -> usize {
        unsafe {
            llama_cpp_sys_4::llama_state_seq_set_data_ext(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                dest_seq_id,
                flags,
            )
        }
    }

    /// Set an abort callback for the context.
    ///
    /// The callback is called periodically during computation. If it returns `true`,
    /// the computation is aborted.
    ///
    /// # Safety
    ///
    /// The callback data must remain valid for the lifetime of the context or until
    /// the callback is replaced.
    pub unsafe fn set_abort_callback(
        &mut self,
        callback: llama_cpp_sys_4::ggml_abort_callback,
        data: *mut std::ffi::c_void,
    ) {
        llama_cpp_sys_4::llama_set_abort_callback(self.context.as_ptr(), callback, data);
    }

    /// Attach a thread pool to the context.
    ///
    /// # Safety
    ///
    /// The thread pools must remain valid for the lifetime of the context or until
    /// they are detached.
    pub unsafe fn attach_threadpool(
        &mut self,
        threadpool: llama_cpp_sys_4::ggml_threadpool_t,
        threadpool_batch: llama_cpp_sys_4::ggml_threadpool_t,
    ) {
        llama_cpp_sys_4::llama_attach_threadpool(
            self.context.as_ptr(),
            threadpool,
            threadpool_batch,
        );
    }

    /// Detach the thread pool from the context.
    pub fn detach_threadpool(&mut self) {
        unsafe {
            llama_cpp_sys_4::llama_detach_threadpool(self.context.as_ptr());
        }
    }

    /// Set a sampler for a specific sequence.
    ///
    /// Returns `true` on success.
    pub fn set_sampler(
        &mut self,
        seq_id: i32,
        sampler: &mut crate::sampling::LlamaSampler,
    ) -> bool {
        unsafe {
            llama_cpp_sys_4::llama_set_sampler(
                self.context.as_ptr(),
                seq_id,
                sampler.sampler.as_ptr(),
            )
        }
    }

    /// Get the raw model pointer from this context.
    ///
    /// This is mainly useful for FFI interop. In normal usage, access
    /// the model via the `model` field instead.
    #[must_use]
    pub fn get_model_ptr(&self) -> *const llama_cpp_sys_4::llama_model {
        unsafe { llama_cpp_sys_4::llama_get_model(self.context.as_ptr()) }
    }

    /// Sets a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        let err_code = unsafe {
            // llama_set_adapter_lora / llama_rm_adapter_lora were replaced by llama_set_adapters_lora
            // which takes a full list of adapters + scales at once (b8249+)
            let mut adapter_ptr = adapter.lora_adapter.as_ptr();
            let mut scale_val = scale;
            llama_cpp_sys_4::llama_set_adapters_lora(
                self.context.as_ptr(),
                &raw mut adapter_ptr,
                1,
                &raw mut scale_val,
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
        }

        tracing::debug!("Set lora adapter");
        Ok(())
    }

    /// Remove all lora adapters from the context.
    ///
    /// Note: as of llama.cpp b8249 the per-adapter remove API was replaced by
    /// `llama_set_adapters_lora` which operates on the full adapter list at once.
    /// Calling this function clears **all** adapters currently set on the context.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        _adapter: &mut LlamaLoraAdapter,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            llama_cpp_sys_4::llama_set_adapters_lora(
                self.context.as_ptr(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
        }

        tracing::debug!("Remove lora adapter");
        Ok(())
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::llama_free(self.context.as_ptr()) }
    }
}
