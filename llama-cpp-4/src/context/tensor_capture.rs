//! Capture intermediate tensor outputs during decode via the `cb_eval` callback.
//!
//! During `llama_decode`, llama.cpp evaluates a computation graph where each
//! tensor node has a name (e.g. `"l_out-13"` for layer 13's output,
//! `"attn_norm-5"` for layer 5's attention norm, `"result_norm"` for the
//! final norm output).
//!
//! The `cb_eval` callback is invoked for every tensor node:
//! - **Ask phase** (`ask = true`):  return `true` to request this tensor's data.
//! - **Data phase** (`ask = false`): the tensor data is computed and available
//!   to copy out via `ggml_backend_tensor_get()`.
//!
//! [`TensorCapture`] provides a safe, reusable wrapper around this mechanism.
//!
//! # Example
//!
//! ```rust,ignore
//! use llama_cpp_4::context::params::LlamaContextParams;
//! use llama_cpp_4::context::tensor_capture::TensorCapture;
//!
//! // Capture layers 13, 20, 27
//! let mut capture = TensorCapture::for_layers(&[13, 20, 27]);
//!
//! let ctx_params = LlamaContextParams::default()
//!     .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
//!     .with_embeddings(true)
//!     .with_tensor_capture(&mut capture);
//!
//! let mut ctx = model.new_context(&backend, ctx_params)?;
//! // ... add tokens to batch ...
//! ctx.decode(&mut batch)?;
//!
//! // Read captured hidden states
//! for &layer in &[13, 20, 27] {
//!     if let Some(info) = capture.get(layer) {
//!         println!("Layer {}: shape [{}, {}]", layer, info.n_embd, info.n_tokens);
//!         // info.data contains [n_tokens * n_embd] f32 values
//!         // Layout: data[token_idx * n_embd + dim_idx]
//!     }
//! }
//! ```

use std::collections::HashMap;

/// Information about a single captured tensor.
#[derive(Debug, Clone)]
pub struct CapturedTensor {
    /// The tensor name (e.g. `"l_out-13"`).
    pub name: String,
    /// The layer index extracted from the name, or `None` if the name
    /// doesn't follow the `"prefix-N"` pattern.
    pub layer: Option<usize>,
    /// First dimension (typically `n_embd` / hidden dimension).
    pub ne0: usize,
    /// Second dimension (typically `n_tokens`).
    pub ne1: usize,
    /// Flattened f32 data with `ne0 * ne1` elements.
    ///
    /// Layout (row-major from ggml's perspective):
    /// `data[token_idx * ne0 + dim_idx]`
    ///
    /// This matches the ggml tensor layout where `ne[0]` is the
    /// innermost (contiguous) dimension.
    pub data: Vec<f32>,
}

impl CapturedTensor {
    /// Number of embedding dimensions (alias for `ne0`).
    #[inline]
    pub fn n_embd(&self) -> usize {
        self.ne0
    }

    /// Number of tokens (alias for `ne1`).
    #[inline]
    pub fn n_tokens(&self) -> usize {
        self.ne1
    }

    /// Get the hidden state for a specific token.
    ///
    /// Returns a slice of `n_embd` floats, or `None` if `token_idx` is
    /// out of range.
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
    /// Capture tensors named `"l_out-{N}"` for specific layer indices.
    Layers(Vec<usize>),
    /// Capture tensors whose names exactly match the given strings.
    Names(Vec<String>),
    /// Capture tensors whose names start with the given prefix.
    Prefix(String),
    /// Capture all tensors (warning: can be very large).
    All,
}

/// Captures intermediate tensor outputs during `llama_decode`.
///
/// Create a `TensorCapture`, attach it to `LlamaContextParams` via
/// [`with_tensor_capture`](super::params::LlamaContextParams::with_tensor_capture),
/// then call `decode()`. After decode completes, read captured data
/// via [`get`], [`get_layer`], or [`iter`].
///
/// # Lifetime & Safety
///
/// The `TensorCapture` must outlive the `LlamaContext` it is attached to.
/// The borrow is enforced by [`with_tensor_capture`](super::params::LlamaContextParams::with_tensor_capture)
/// taking `&mut self`.
pub struct TensorCapture {
    filter: CaptureFilter,
    /// Captured tensors keyed by name.
    captured: HashMap<String, CapturedTensor>,
}

impl std::fmt::Debug for TensorCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorCapture")
            .field("filter", &self.filter)
            .field("captured_count", &self.captured.len())
            .field(
                "captured_keys",
                &self.captured.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl TensorCapture {
    /// Create a capture that intercepts layer outputs `"l_out-{N}"` for
    /// the specified layer indices.
    ///
    /// This is the most common use case for extracting per-layer hidden
    /// states from a language model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Capture layers 13, 20, 27 (typical for LLaMA-3.2-3B with positions [0.5, 0.75, 1.0])
    /// let mut capture = TensorCapture::for_layers(&[13, 20, 27]);
    /// ```
    pub fn for_layers(layer_indices: &[usize]) -> Self {
        Self {
            filter: CaptureFilter::Layers(layer_indices.to_vec()),
            captured: HashMap::new(),
        }
    }

    /// Create a capture that intercepts tensors with exact matching names.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut capture = TensorCapture::for_names(&["result_norm", "l_out-27"]);
    /// ```
    pub fn for_names(names: &[&str]) -> Self {
        Self {
            filter: CaptureFilter::Names(names.iter().map(|s| s.to_string()).collect()),
            captured: HashMap::new(),
        }
    }

    /// Create a capture that intercepts all tensors whose name starts with
    /// the given prefix.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Capture all attention outputs
    /// let mut capture = TensorCapture::for_prefix("attn_out");
    /// ```
    pub fn for_prefix(prefix: &str) -> Self {
        Self {
            filter: CaptureFilter::Prefix(prefix.to_string()),
            captured: HashMap::new(),
        }
    }

    /// Create a capture that intercepts **all** tensors.
    ///
    /// ⚠️ Warning: this can produce very large amounts of data.
    /// Use only for debugging or inspection.
    pub fn all() -> Self {
        Self {
            filter: CaptureFilter::All,
            captured: HashMap::new(),
        }
    }

    /// Clear all previously captured data, keeping the filter configuration.
    ///
    /// Call this before a new `decode()` if reusing the capture across
    /// multiple batches.
    pub fn clear(&mut self) {
        self.captured.clear();
    }

    /// Get a captured tensor by its full name (e.g. `"l_out-13"`).
    pub fn get(&self, name: &str) -> Option<&CapturedTensor> {
        self.captured.get(name)
    }

    /// Get a captured layer output by layer index.
    ///
    /// Looks up `"l_out-{layer_idx}"`.
    pub fn get_layer(&self, layer_idx: usize) -> Option<&CapturedTensor> {
        self.captured.get(&format!("l_out-{layer_idx}"))
    }

    /// Returns `true` if the specified layer was captured.
    pub fn has_layer(&self, layer_idx: usize) -> bool {
        self.captured.contains_key(&format!("l_out-{layer_idx}"))
    }

    /// Number of tensors captured so far.
    pub fn len(&self) -> usize {
        self.captured.len()
    }

    /// Returns `true` if no tensors have been captured.
    pub fn is_empty(&self) -> bool {
        self.captured.is_empty()
    }

    /// Iterate over all captured tensors.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &CapturedTensor)> {
        self.captured.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get all captured layer indices (sorted).
    pub fn captured_layers(&self) -> Vec<usize> {
        let mut layers: Vec<usize> = self
            .captured
            .values()
            .filter_map(|ct| ct.layer)
            .collect();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    // ── Internal: callback matching ──────────────────────────────────

    /// Check if a tensor name matches the capture filter.
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

    /// Store a captured tensor.
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

// ── The extern "C" callback ──────────────────────────────────────────────

/// The `cb_eval` callback function passed to llama.cpp.
///
/// # Safety
///
/// This function is called from C code during graph evaluation.
/// `user_data` must point to a valid `TensorCapture` instance.
pub(crate) unsafe extern "C" fn tensor_capture_callback(
    t: *mut llama_cpp_sys_4::ggml_tensor,
    ask: bool,
    user_data: *mut std::ffi::c_void,
) -> bool {
    if t.is_null() || user_data.is_null() {
        return false;
    }

    // Read tensor name from the fixed-size C array
    let name_bytes = &(*t).name;
    let len = name_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_bytes.len());
    let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
        name_bytes.as_ptr() as *const u8,
        len,
    ));

    let state = &mut *(user_data as *mut TensorCapture);

    if !state.matches(name) {
        return false;
    }

    if ask {
        return true;
    }

    // Data phase: copy tensor data out
    let ne0 = (*t).ne[0] as usize;
    let ne1 = (*t).ne[1] as usize;
    let n_elements = ne0 * ne1;

    let mut buf = vec![0f32; n_elements];
    llama_cpp_sys_4::ggml_backend_tensor_get(
        t,
        buf.as_mut_ptr() as *mut std::ffi::c_void,
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

        // Also accessible by layer index
        let ct2 = capture.get_layer(13).unwrap();
        assert_eq!(ct2.name, ct.name);
        assert!(capture.has_layer(13));
        assert!(!capture.has_layer(14));
    }

    #[test]
    fn test_token_embedding() {
        let mut capture = TensorCapture::for_layers(&[5]);
        // 2 tokens, 3 dims: token0=[1,2,3], token1=[4,5,6]
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
