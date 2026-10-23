//! utilities for working with the kv cache

use crate::context::LlamaContext;
use std::num::{NonZeroU8, TryFromIntError};

/// Errors that can occur when attempting to prepare values for the kv cache
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum KvCacheConversionError {
    /// Sequence id conversion to i32 failed
    #[error("Provided sequence id is too large for a i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    /// Position 0 conversion to i32 failed
    #[error("Provided start position is too large for a i32")]
    P0TooLarge(#[source] TryFromIntError),
    /// Position 1 conversion to i32 failed
    #[error("Provided end position is too large for a i32")]
    P1TooLarge(#[source] TryFromIntError),
}

impl LlamaContext<'_> {
    /// How many backend buffers a state write of this context's memory would
    /// touch.
    ///
    /// `0` for an empty cache. Useful before
    /// [`state_seq_save_file`](crate::context::LlamaContext::state_seq_save_file)
    /// and friends: a KV cache spread across several backends is written buffer
    /// by buffer, so the count is a proxy for how much work a save will be, and
    /// a cache reporting zero has nothing to save yet.
    ///
    /// Wraps `llama_memory_get_buffers` through the `ext_shim`, because
    /// upstream returns a `std::vector` by value and bindgen cannot express
    /// that — see the note in `build.rs`.
    #[must_use]
    pub fn memory_buffer_count(&self) -> usize {
        // SAFETY: the memory handle is borrowed from a live context, and a null
        // `out` asks the shim for the count without writing anything.
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_buffers_collect(mem, std::ptr::null_mut(), 0)
        }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `size` - The size of the cache to copy.
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_cp(mem, src, dest, 0, size);
        }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is copied up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is copied starting from `p0`.
    ///
    /// # Errors
    ///
    /// Returns [`KvCacheConversionError`] if either position exceeds the maximum `i32` value.
    pub fn copy_kv_cache_seq(
        &mut self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_cp(mem, src, dest, p0, p1);
        }
        Ok(())
    }

    /// Clear the kv cache for the given sequence within the specified range `[p0, p1)`.
    ///
    /// Returns `false` only when partial sequence removals fail. Full sequence removals always succeed.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to clear the cache for. If `None`, matches all sequences
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is cleared up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is cleared from `p0`.
    ///
    /// # Errors
    ///
    /// Returns [`KvCacheConversionError`] if the sequence id or either position exceeds the maximum `i32` value.
    pub fn clear_kv_cache_seq(
        &mut self,
        src: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, KvCacheConversionError> {
        let src = src
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::SeqIdTooLarge)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        Ok(unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_rm(mem, src, p0, p1)
        })
    }

    /// Clear the KV cache
    pub fn clear_kv_cache(&mut self) {
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_clear(mem, true);
        }
    }

    /// Removes all tokens that do not belong to the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to keep
    pub fn llama_kv_cache_seq_keep(&mut self, seq_id: i32) {
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_keep(mem, seq_id);
        }
    }

    #[allow(clippy::doc_markdown)]
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in `[p0, p1)`.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `delta` - The relative position to add to the tokens
    ///
    /// # Errors
    ///
    /// Returns [`KvCacheConversionError`] if either position exceeds the maximum `i32` value.
    pub fn kv_cache_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_add(mem, seq_id, p0, p1, delta);
        }
        Ok(())
    }

    /// Integer division of the positions by factor of `d > 1`.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `d` - The factor to divide the positions by
    ///
    /// # Errors
    ///
    /// Returns [`KvCacheConversionError`] if either position exceeds the maximum `i32` value.
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        d: NonZeroU8,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let d = i32::from(d.get());
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_div(mem, seq_id, p0, p1, d);
        }
        Ok(())
    }

    /// Returns the largest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the max position for
    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.context.as_ptr());
            llama_cpp_sys_4::llama_memory_seq_pos_max(mem, seq_id)
        }
    }
}
