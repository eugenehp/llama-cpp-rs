//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_sys_4::llama_pooling_type;
use params::LlamaPoolingType;
use perf::PerfContextData;

use crate::llama_batch::LlamaBatch;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError,
};

pub mod kv_cache;
pub mod params;
pub mod perf;
pub mod session;

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
    ///
    /// # Returns
    ///
    /// This function returns a new instance of `LlamaContext` initialized with the given parameters:
    /// - The model reference (`llama_model`) is stored in the context.
    /// - The raw context pointer (`llama_context`) is wrapped in a `NonNull` to ensure safety.
    /// - The `embeddings_enabled` flag is used to determine if embeddings are enabled for the context.
    ///
    /// # Example
    /// ```
    /// let llama_model = LlamaModel::load("path/to/model").unwrap();
    /// let context_ptr = NonNull::new(some_llama_context_ptr).unwrap();
    /// let context = LlamaContext::new(&llama_model, context_ptr, true);
    /// // Now you can use the context
    /// ```
    pub(crate) fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_sys_4::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
            embeddings_enabled,
        }
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

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
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
            llama_cpp_sys_4::llama_set_n_threads(
                self.context.as_ptr(),
                n_threads,
                n_threads_batch,
            );
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

    /// Print a breakdown of the memory usage.
    pub fn memory_breakdown_print(&self) {
        unsafe {
            llama_cpp_sys_4::llama_memory_breakdown_print(self.context.as_ptr());
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
            llama_cpp_sys_4::llama_state_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
            )
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
        let ptr = unsafe {
            llama_cpp_sys_4::llama_get_sampled_candidates_ith(self.context.as_ptr(), i)
        };
        if ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(ptr.cast::<LlamaToken>(), count) }
    }

    /// Get the number of sampled logits for the `i`th position.
    #[must_use]
    pub fn get_sampled_logits_count_ith(&self, i: i32) -> u32 {
        unsafe {
            llama_cpp_sys_4::llama_get_sampled_logits_count_ith(self.context.as_ptr(), i)
        }
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
        let ptr = unsafe {
            llama_cpp_sys_4::llama_get_sampled_logits_ith(self.context.as_ptr(), i)
        };
        if ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(ptr, count) }
    }

    /// Get the number of sampled probabilities for the `i`th position.
    #[must_use]
    pub fn get_sampled_probs_count_ith(&self, i: i32) -> u32 {
        unsafe {
            llama_cpp_sys_4::llama_get_sampled_probs_count_ith(self.context.as_ptr(), i)
        }
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
        let ptr = unsafe {
            llama_cpp_sys_4::llama_get_sampled_probs_ith(self.context.as_ptr(), i)
        };
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
    pub fn state_seq_set_data_ext(
        &mut self,
        src: &[u8],
        dest_seq_id: i32,
        flags: u32,
    ) -> usize {
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
