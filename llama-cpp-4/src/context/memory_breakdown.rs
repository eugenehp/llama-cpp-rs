//! Per-buffer-type memory usage reported by llama.cpp.
//!
//! [`MemoryBreakdownEntry`] values are produced by
//! [`crate::context::LlamaContext::memory_breakdown`] and classify bytes into
//! model weights, KV / recurrent cache, and temporary compute buffers for each
//! ggml backend buffer type (e.g. `CUDA0`, `Metal`, `Host`).
//!
//! # Examples
//!
//! ```no_run
//! use llama_cpp_4::prelude::*;
//!
//! fn main() {
//!     let backend = LlamaBackend::init().unwrap();
//!     let model = LlamaModel::load_from_file(&backend, "model.gguf", &LlamaModelParams::default()).unwrap();
//!     let ctx = model.new_context(&backend, LlamaContextParams::default()).unwrap();
//!     for entry in ctx.memory_breakdown() {
//!         println!("{}: {} bytes total", entry.buft_name, entry.total());
//!     }
//! }
//! ```

use std::ffi::CStr;

/// Memory attributed to a single backend buffer type (e.g. CUDA0, Host).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBreakdownEntry {
    /// Human-readable buffer-type name from ggml.
    pub buft_name: String,
    /// Bytes used by model weights on this buffer type.
    pub model: usize,
    /// Bytes used by the KV / recurrent context cache.
    pub context: usize,
    /// Bytes used by temporary compute buffers.
    pub compute: usize,
}

impl MemoryBreakdownEntry {
    /// Sum of model, context, and compute bytes.
    #[must_use]
    pub fn total(&self) -> usize {
        self.model + self.context + self.compute
    }
}

fn raw_entry_to_rust(
    entry: &llama_cpp_sys_4::llama_memory_breakdown_entry,
) -> MemoryBreakdownEntry {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(entry.buft_name.as_ptr().cast(), entry.buft_name.len())
    };
    let name = CStr::from_bytes_until_nul(bytes)
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default();
    MemoryBreakdownEntry {
        buft_name: name,
        model: entry.model,
        context: entry.context,
        compute: entry.compute,
    }
}

/// Collect memory breakdown entries for a live context.
///
/// Wraps the `ext_shim` helper around `llama_get_memory_breakdown`. Grows the
/// output buffer until every entry fits. Returns an empty vector when the
/// context pointer is invalid or no buffers are registered yet.
///
/// Prefer [`crate::context::LlamaContext::memory_breakdown`] in application code.
#[must_use]
pub(crate) fn collect_memory_breakdown(
    ctx: *const llama_cpp_sys_4::llama_context,
) -> Vec<MemoryBreakdownEntry> {
    if ctx.is_null() {
        return Vec::new();
    }

    let mut capacity = 16usize;
    loop {
        let mut raw = vec![
            llama_cpp_sys_4::llama_memory_breakdown_entry {
                buft_name: [0; 128],
                model: 0,
                context: 0,
                compute: 0,
            };
            capacity
        ];

        let n = unsafe {
            llama_cpp_sys_4::llama_memory_breakdown_collect(ctx, raw.as_mut_ptr(), capacity)
        };

        if n < capacity {
            return raw
                .into_iter()
                .take(n)
                .map(|e| raw_entry_to_rust(&e))
                .collect();
        }

        capacity = capacity.saturating_mul(2);
        if capacity > 4096 {
            return raw.into_iter().map(|e| raw_entry_to_rust(&e)).collect();
        }
    }
}
