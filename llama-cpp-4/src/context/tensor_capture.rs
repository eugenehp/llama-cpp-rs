//! Capture intermediate tensor outputs during [`crate::LlamaContext::decode`].
//!
//! llama.cpp builds a computation graph for each forward pass. Every node has a
//! string name — for transformer blocks the layer output is typically
//! `"l_out-{N}"` (e.g. `"l_out-13"`), attention norms are `"attn_norm-{N}"`, and
//! the final norm is `"result_norm"`.
//!
//! The graph evaluation callback (`cb_eval`) runs in two phases for each node:
//!
//! | Phase | `ask` | Behaviour |
//! |---|---|---|
//! | Ask | `true` | Return `true` to request a copy of this tensor's data. |
//! | Data | `false` | Tensor is computed; data is copied via `ggml_backend_tensor_get`. |
//!
//! [`TensorCapture`] implements that callback and stores matching tensors in a
//! [`HashMap`] you can read after `decode()` finishes.
//!
//! # Typical use cases
//!
//! - **Layer probing** — inspect hidden states at specific depths.
//! - **EAGLE / distillation** — read draft-model anchor layers (see `examples/eagle`).
//! - **Debugging** — dump norms or attention outputs with [`TensorCapture::for_prefix`].
//!
//! # Setup
//!
//! 1. Build a [`TensorCapture`] with the filter you need ([`TensorCapture::for_layers`]
//!    is the common case).
//! 2. Pass it to [`LlamaContextParams::with_tensor_capture`](crate::LlamaContextParams::with_tensor_capture). The capture must
//!    **outlive** the [`LlamaContext`](crate::LlamaContext).
//! 3. Run [`LlamaContext::decode`](crate::LlamaContext::decode) as usual.
//! 4. Read [`CapturedTensor`] values via [`TensorCapture::get_layer`],
//!    [`TensorCapture::get`], or [`TensorCapture::iter`].
//!
//! Call [`TensorCapture::clear`](crate::TensorCapture::clear) before reusing the same capture on another batch.
//!
//! # Example
//!
//! ```no_run
//! use llama_cpp_4::prelude::*;
//! use std::num::NonZeroU32;
//!
//! fn main() {
//!     let backend = LlamaBackend::init().unwrap();
//!     let model = LlamaModel::load_from_file(
//!         &backend,
//!         "model.gguf",
//!         &LlamaModelParams::default(),
//!     )
//!     .unwrap();
//!
//!     let mut capture = TensorCapture::for_layers(&[13, 20, 27]);
//!     let ctx_params = LlamaContextParams::default()
//!         .with_n_ctx(NonZeroU32::new(512))
//!         .with_tensor_capture(&mut capture);
//!     let mut ctx = model.new_context(&backend, ctx_params).unwrap();
//!
//!     let tokens = model.str_to_token("Hello", AddBos::Always).unwrap();
//!     let mut batch = LlamaBatch::new(512, 1);
//!     for (i, &tok) in tokens.iter().enumerate() {
//!         batch
//!             .add(tok, i as i32, &[0], i == tokens.len() - 1)
//!             .unwrap();
//!     }
//!     ctx.decode(&mut batch).unwrap();
//!
//!     for &layer in &[13, 20, 27] {
//!         if let Some(t) = capture.get_layer(layer) {
//!             println!(
//!                 "l_out-{layer}: {} tokens × {} dims",
//!                 t.n_tokens(),
//!                 t.n_embd()
//!             );
//!             if let Some(vec) = t.token_embedding(0) {
//!                 println!("  first token, first 3 dims: {:?}", &vec[..3.min(vec.len())]);
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! # Tensor layout
//!
//! Each [`CapturedTensor`] stores a flat `f32` buffer with
//! `data[token_idx * n_embd + dim_idx]` (ggml row-major: `ne0` = embedding dim,
//! `ne1` = token count). Use [`CapturedTensor::token_embedding`] to slice one row.

use std::collections::HashMap;

/// A single tensor copied out of the decode graph.
///
/// Produced by [`TensorCapture`] after a successful [`crate::LlamaContext::decode`].
/// For layer outputs (`"l_out-N"`), [`Self::layer`] is set to `N`.
#[derive(Debug, Clone)]
pub struct CapturedTensor {
    /// Graph node name (e.g. `"l_out-13"`, `"result_norm"`).
    pub name: String,
    /// Layer index when `name` is `"l_out-{N}"`, otherwise `None`.
    pub layer: Option<usize>,
    /// First dimension (typically `n_embd` / hidden size).
    pub ne0: usize,
    /// Second dimension (typically number of tokens in the batch position).
    pub ne1: usize,
    /// Flattened `ne0 * ne1` values in ggml row-major order.
    ///
    /// Index as `data[token_idx * ne0 + dim_idx]`.
    pub data: Vec<f32>,
}

impl CapturedTensor {
    /// Number of embedding dimensions (alias for [`Self::ne0`]).
    #[inline]
    #[must_use]
    pub fn n_embd(&self) -> usize {
        self.ne0
    }

    /// Number of token positions (alias for [`Self::ne1`]).
    #[inline]
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        self.ne1
    }

    /// Hidden-state vector for one token index.
    ///
    /// Returns `None` when `token_idx >= n_tokens()`.
    #[must_use]
    pub fn token_embedding(&self, token_idx: usize) -> Option<&[f32]> {
        if token_idx >= self.ne1 {
            return None;
        }
        let start = token_idx * self.ne0;
        Some(&self.data[start..start + self.ne0])
    }
}

/// Strategy for selecting which tensors to capture.
#[derive(Debug, Clone)]
enum CaptureFilter {
    /// `"l_out-{N}"` for each listed layer index `N`.
    Layers(Vec<usize>),
    /// Exact graph node names.
    Names(Vec<String>),
    /// Names starting with a prefix (e.g. `"attn_out"`).
    Prefix(String),
    /// Every node (can be very large — debug only).
    All,
}

/// Captures intermediate tensors during [`crate::LlamaContext::decode`].
///
/// Attach with [`LlamaContextParams::with_tensor_capture`](crate::LlamaContextParams::with_tensor_capture) before creating the
/// context. The same instance can be reused across decodes if you call
/// [`Self::clear`] between passes.
///
/// # Lifetime
///
/// The capture must outlive the [`crate::LlamaContext`] it is wired into;
/// [`LlamaContextParams::with_tensor_capture`](crate::LlamaContextParams::with_tensor_capture) takes `&mut TensorCapture` to
/// enforce this at compile time.
pub struct TensorCapture {
    filter: CaptureFilter,
    captured: HashMap<String, CapturedTensor>,
}

impl std::fmt::Debug for TensorCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorCapture")
            .field("filter", &self.filter)
            .field("captured_count", &self.captured.len())
            .field("captured_keys", &self.captured.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl TensorCapture {
    /// Capture transformer layer outputs `"l_out-{N}"` for the given indices.
    ///
    /// This is the usual choice for hidden-state extraction. EAGLE-3 draft models
    /// often use three layers at ~50%, 75%, and 100% depth — e.g. `[13, 20, 27]`
    /// on a 28-layer model.
    #[must_use]
    pub fn for_layers(layer_indices: &[usize]) -> Self {
        Self {
            filter: CaptureFilter::Layers(layer_indices.to_vec()),
            captured: HashMap::new(),
        }
    }

    /// Capture tensors whose graph names match exactly.
    ///
    /// Example names: `"result_norm"`, `"l_out-27"`.
    #[must_use]
    pub fn for_names(names: &[&str]) -> Self {
        Self {
            filter: CaptureFilter::Names(
                names.iter().map(std::string::ToString::to_string).collect(),
            ),
            captured: HashMap::new(),
        }
    }

    /// Capture every tensor whose name starts with `prefix`.
    ///
    /// Useful for families like `"attn_out-*"` or `"attn_norm-*"`.
    #[must_use]
    pub fn for_prefix(prefix: &str) -> Self {
        Self {
            filter: CaptureFilter::Prefix(prefix.to_string()),
            captured: HashMap::new(),
        }
    }

    /// Capture **all** graph nodes.
    ///
    /// Warning: memory use scales with model size and sequence length. Prefer
    /// [`Self::for_layers`] or [`Self::for_names`] in production code.
    #[must_use]
    pub fn all() -> Self {
        Self {
            filter: CaptureFilter::All,
            captured: HashMap::new(),
        }
    }

    /// Drop captured tensors but keep the filter (safe to call before another decode).
    pub fn clear(&mut self) {
        self.captured.clear();
    }

    /// Lookup by full graph name (e.g. `"l_out-13"`).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&CapturedTensor> {
        self.captured.get(name)
    }

    /// Lookup a layer output (`"l_out-{layer_idx}"`).
    #[must_use]
    pub fn get_layer(&self, layer_idx: usize) -> Option<&CapturedTensor> {
        self.captured.get(&format!("l_out-{layer_idx}"))
    }

    /// Whether `"l_out-{layer_idx}"` was captured in the last decode.
    #[must_use]
    pub fn has_layer(&self, layer_idx: usize) -> bool {
        self.captured.contains_key(&format!("l_out-{layer_idx}"))
    }

    /// Number of tensors stored from the most recent decode.
    #[must_use]
    pub fn len(&self) -> usize {
        self.captured.len()
    }

    /// `true` when [`Self::len`] is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.captured.is_empty()
    }

    /// Iterate `(name, tensor)` pairs from the last decode.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &CapturedTensor)> {
        self.captured.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Sorted layer indices present among captured `"l_out-*"` tensors.
    #[must_use]
    pub fn captured_layers(&self) -> Vec<usize> {
        let mut layers: Vec<usize> = self.captured.values().filter_map(|ct| ct.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    fn matches(&self, name: &str) -> bool {
        match &self.filter {
            CaptureFilter::Layers(indices) => {
                if let Some(suffix) = name.strip_prefix("l_out-") {
                    if let Ok(idx) = suffix.parse::<usize>() {
                        return indices.contains(&idx);
                    }
                }
                false
            }
            CaptureFilter::Names(names) => names.iter().any(|n| n == name),
            CaptureFilter::Prefix(prefix) => name.starts_with(prefix.as_str()),
            CaptureFilter::All => true,
        }
    }

    fn store(&mut self, name: String, ne0: usize, ne1: usize, data: Vec<f32>) {
        let layer = name
            .strip_prefix("l_out-")
            .and_then(|s| s.parse::<usize>().ok());

        self.captured.insert(
            name.clone(),
            CapturedTensor {
                name,
                layer,
                ne0,
                ne1,
                data,
            },
        );
    }
}

/// `cb_eval` callback installed by [`LlamaContextParams::with_tensor_capture`](crate::LlamaContextParams::with_tensor_capture).
///
/// # Safety
///
/// `user_data` must point to a live [`TensorCapture`] for the context lifetime.
pub(crate) unsafe extern "C" fn tensor_capture_callback(
    t: *mut llama_cpp_sys_4::ggml_tensor,
    ask: bool,
    user_data: *mut std::ffi::c_void,
) -> bool {
    if t.is_null() || user_data.is_null() {
        return false;
    }

    let name_bytes = &(*t).name;
    let len = name_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_bytes.len());
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
        name_bytes.as_ptr().cast::<u8>(),
        len,
    ));

    let state = &mut *user_data.cast::<TensorCapture>();

    if !state.matches(name) {
        return false;
    }

    if ask {
        return true;
    }

    let ne0 = usize::try_from((*t).ne[0]).expect("tensor ne[0] must be non-negative");
    let ne1 = usize::try_from((*t).ne[1]).expect("tensor ne[1] must be non-negative");
    let n_elements = ne0 * ne1;

    let mut buf = vec![0f32; n_elements];
    llama_cpp_sys_4::ggml_backend_tensor_get(
        t,
        buf.as_mut_ptr().cast::<std::ffi::c_void>(),
        0,
        n_elements * std::mem::size_of::<f32>(),
    );

    state.store(name.to_string(), ne0, ne1, buf);

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_for_layers_matching() {
        let capture = TensorCapture::for_layers(&[13, 20, 27]);
        assert!(capture.matches("l_out-13"));
        assert!(capture.matches("l_out-20"));
        assert!(capture.matches("l_out-27"));
        assert!(!capture.matches("l_out-0"));
        assert!(!capture.matches("l_out-14"));
        assert!(!capture.matches("attn_norm-13"));
        assert!(!capture.matches("result_norm"));
    }

    #[test]
    fn test_for_names_matching() {
        let capture = TensorCapture::for_names(&["result_norm", "l_out-27"]);
        assert!(capture.matches("result_norm"));
        assert!(capture.matches("l_out-27"));
        assert!(!capture.matches("l_out-13"));
        assert!(!capture.matches("result_output"));
    }

    #[test]
    fn test_for_prefix_matching() {
        let capture = TensorCapture::for_prefix("attn_out");
        assert!(capture.matches("attn_out-0"));
        assert!(capture.matches("attn_out-27"));
        assert!(!capture.matches("attn_norm-0"));
        assert!(!capture.matches("l_out-0"));
    }

    #[test]
    fn test_all_matching() {
        let capture = TensorCapture::all();
        assert!(capture.matches("l_out-13"));
        assert!(capture.matches("result_norm"));
        assert!(capture.matches("anything"));
    }

    #[test]
    fn test_store_and_get() {
        let mut capture = TensorCapture::for_layers(&[13]);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        capture.store("l_out-13".to_string(), 3, 2, data.clone());

        assert_eq!(capture.len(), 1);
        assert!(!capture.is_empty());

        let ct = capture.get("l_out-13").unwrap();
        assert_eq!(ct.name, "l_out-13");
        assert_eq!(ct.layer, Some(13));
        assert_eq!(ct.n_embd(), 3);
        assert_eq!(ct.n_tokens(), 2);
        assert_eq!(ct.data, data);

        let ct2 = capture.get_layer(13).unwrap();
        assert_eq!(ct2.name, ct.name);
        assert!(capture.has_layer(13));
        assert!(!capture.has_layer(14));
    }

    #[test]
    fn test_token_embedding() {
        let mut capture = TensorCapture::for_layers(&[5]);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        capture.store("l_out-5".to_string(), 3, 2, data);

        let ct = capture.get_layer(5).unwrap();
        assert_eq!(ct.token_embedding(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(ct.token_embedding(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(ct.token_embedding(2), None);
    }

    #[test]
    fn test_captured_layers() {
        let mut capture = TensorCapture::for_layers(&[5, 10, 20]);
        capture.store("l_out-10".to_string(), 2, 1, vec![0.0, 0.0]);
        capture.store("l_out-5".to_string(), 2, 1, vec![0.0, 0.0]);
        assert_eq!(capture.captured_layers(), vec![5, 10]);
    }

    #[test]
    fn test_clear() {
        let mut capture = TensorCapture::for_layers(&[5]);
        capture.store("l_out-5".to_string(), 2, 1, vec![0.0, 0.0]);
        assert_eq!(capture.len(), 1);
        capture.clear();
        assert_eq!(capture.len(), 0);
        assert!(capture.is_empty());
    }

    #[test]
    fn test_non_layer_tensor() {
        let mut capture = TensorCapture::for_names(&["result_norm"]);
        capture.store("result_norm".to_string(), 4, 3, vec![0.0; 12]);
        let ct = capture.get("result_norm").unwrap();
        assert_eq!(ct.layer, None);
        assert_eq!(ct.n_embd(), 4);
        assert_eq!(ct.n_tokens(), 3);
    }
}
