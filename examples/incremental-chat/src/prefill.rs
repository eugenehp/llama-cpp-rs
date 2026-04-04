//! Incremental prefill tracker — shared between the interactive chat and the
//! benchmark.
//!
//! Keeps track of which tokens have been decoded into the KV cache and
//! minimises redundant work when the user edits their input.

use anyhow::{anyhow, Context, Result};
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::token::LlamaToken;

/// How many trailing tokens to withhold from speculative prefill.
///
/// BPE tokenization is non-incremental: appending a single character can
/// change the last 1-2 tokens retroactively (e.g. `["hel"]` → `["hell"]` →
/// `["hello"]`).  By withholding the last `STABLE_MARGIN` tokens we avoid
/// repeatedly decoding and invalidating the tail.  The withheld tokens are
/// decoded in the final flush when the user presses Enter.
const STABLE_MARGIN: usize = 2;

/// Tracks which tokens have been decoded into the KV cache so that only the
/// delta needs to be processed when the user edits their input.
pub struct IncrementalPrefill {
    /// Tokens whose KV projections are already in the cache.
    cached_tokens: Vec<LlamaToken>,
    /// Position offset — tokens before this are "frozen" history that we never
    /// touch (system prompt + prior conversation turns).
    history_len: usize,
    /// Maximum batch size for micro-batches.
    batch_size: usize,
}

impl IncrementalPrefill {
    /// Create a new tracker.
    ///
    /// - `history_len`: number of tokens already in the KV cache (system prompt
    ///   + prior turns). The prefill tracker will never touch positions below
    ///   this offset.
    /// - `batch_size`: max tokens per micro-batch sent to `decode()`.
    pub fn new(history_len: usize, batch_size: usize) -> Self {
        Self {
            cached_tokens: Vec::new(),
            history_len,
            batch_size,
        }
    }

    /// Length of the longest common prefix between cached and new tokens.
    fn common_prefix_len(&self, new_tokens: &[LlamaToken]) -> usize {
        self.cached_tokens
            .iter()
            .zip(new_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Speculatively prefill tokens while the user is still typing.
    ///
    /// Holds back the last [`STABLE_MARGIN`] tokens to avoid BPE churn.
    /// Use [`flush`] when the user submits to decode the remainder.
    ///
    /// Returns the number of tokens decoded in this call.
    pub fn prefill_speculative(
        &mut self,
        ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
        batch: &mut LlamaBatch,
        new_tokens: &[LlamaToken],
    ) -> Result<usize> {
        let stable_end = new_tokens.len().saturating_sub(STABLE_MARGIN);
        let stable = &new_tokens[..stable_end];
        self.sync_inner(ctx, batch, stable)
    }

    /// Final flush — decode all remaining tokens (including the unstable tail).
    ///
    /// Call this when the user presses Enter. Returns the number of tokens
    /// decoded in this call.
    pub fn flush(
        &mut self,
        ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
        batch: &mut LlamaBatch,
        final_tokens: &[LlamaToken],
    ) -> Result<usize> {
        self.sync_inner(ctx, batch, final_tokens)
    }

    /// Core sync logic shared by speculative prefill and final flush.
    fn sync_inner(
        &mut self,
        ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
        batch: &mut LlamaBatch,
        new_tokens: &[LlamaToken],
    ) -> Result<usize> {
        let common = self.common_prefix_len(new_tokens);

        // Trim KV cache from the divergence point onward.
        if common < self.cached_tokens.len() {
            let trim_from = (self.history_len + common) as u32;
            ctx.clear_kv_cache_seq(Some(0), Some(trim_from), None)
                .map_err(|e| anyhow!("failed to trim KV cache: {e}"))?;
            self.cached_tokens.truncate(common);
        }

        let to_decode = &new_tokens[common..];
        if to_decode.is_empty() {
            return Ok(0);
        }

        let mut decoded = 0;
        for chunk in to_decode.chunks(self.batch_size) {
            batch.clear();
            for (i, &token) in chunk.iter().enumerate() {
                let pos = (self.history_len + common + decoded + i) as i32;
                let is_last = (decoded + i) == to_decode.len() - 1;
                batch.add(token, pos, &[0], is_last)?;
            }
            ctx.decode(batch)
                .with_context(|| "llama_decode() failed during incremental prefill")?;
            decoded += chunk.len();
        }

        self.cached_tokens = new_tokens.to_vec();
        Ok(decoded)
    }

    /// How many tokens are currently decoded in the KV cache (for this turn only).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.cached_tokens.len()
    }

    /// Whether anything has been decoded yet.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.cached_tokens.is_empty()
    }
}
