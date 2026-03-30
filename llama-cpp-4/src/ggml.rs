//! Safe wrappers around core ggml graph computation APIs.
//!
//! This module provides the building blocks for creating and executing tensor computation
//! graphs using ggml backends. It's used for operations like LoRA merging, importance
//! matrix computation, and control vector generation.
//!
//! # Example
//!
//! ```rust,ignore
//! use llama_cpp_4::ggml::*;
//!
//! // Create a backend and context
//! let backend = GgmlBackend::cpu()?;
//! let mut ctx = GgmlContext::new(1024 * 1024, true)?;
//!
//! // Create tensors
//! let a = ctx.new_tensor_2d(GgmlType::F32, 4, 4);
//! let b = ctx.new_tensor_2d(GgmlType::F32, 4, 4);
//!
//! // Build computation graph
//! let sum = ctx.add(&a, &b);
//! let mut graph = ctx.new_graph();
//! graph.build_forward(&sum);
//!
//! // Allocate and compute
//! let mut alloc = GgmlAllocr::new(&backend);
//! alloc.alloc_graph(&mut graph);
//! // ... set tensor data ...
//! backend.graph_compute(&mut graph);
//! // ... get results ...
//! ```

use std::ffi::CStr;
use std::ptr::NonNull;

/// Re-export the raw ggml types for advanced usage.
pub use llama_cpp_sys_4::ggml_type;

/// A safe wrapper around `ggml_context`.
#[derive(Debug)]
pub struct GgmlContext {
    ctx: NonNull<llama_cpp_sys_4::ggml_context>,
}

impl GgmlContext {
    /// Create a new ggml context.
    ///
    /// # Parameters
    /// - `mem_size`: Memory pool size in bytes for tensor metadata.
    /// - `no_alloc`: If true, tensor data is not allocated (use with backend allocation).
    ///
    /// # Panics
    ///
    /// Panics if ggml returns a null pointer.
    #[must_use]
    pub fn new(mem_size: usize, no_alloc: bool) -> Self {
        let params = llama_cpp_sys_4::ggml_init_params {
            mem_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc,
        };
        let ctx = unsafe { llama_cpp_sys_4::ggml_init(params) };
        Self {
            ctx: NonNull::new(ctx).expect("ggml_init returned null"),
        }
    }

    /// Get the raw context pointer.
    pub fn as_ptr(&self) -> *mut llama_cpp_sys_4::ggml_context {
        self.ctx.as_ptr()
    }

    // ── Tensor creation ──────────────────────────────────────

    /// Create a 1D tensor.
    #[must_use]
    pub fn new_tensor_1d(&self, typ: ggml_type, ne0: i64) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_new_tensor_1d(self.ctx.as_ptr(), typ, ne0) };
        GgmlTensor(NonNull::new(t).expect("ggml_new_tensor_1d returned null"))
    }

    /// Create a 2D tensor.
    #[must_use]
    pub fn new_tensor_2d(&self, typ: ggml_type, ne0: i64, ne1: i64) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_new_tensor_2d(self.ctx.as_ptr(), typ, ne0, ne1) };
        GgmlTensor(NonNull::new(t).expect("ggml_new_tensor_2d returned null"))
    }

    /// Create a 3D tensor.
    #[must_use]
    pub fn new_tensor_3d(&self, typ: ggml_type, ne0: i64, ne1: i64, ne2: i64) -> GgmlTensor {
        let t =
            unsafe { llama_cpp_sys_4::ggml_new_tensor_3d(self.ctx.as_ptr(), typ, ne0, ne1, ne2) };
        GgmlTensor(NonNull::new(t).expect("ggml_new_tensor_3d returned null"))
    }

    /// Create a 4D tensor.
    #[must_use]
    pub fn new_tensor_4d(
        &self,
        typ: ggml_type,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensor {
        let t = unsafe {
            llama_cpp_sys_4::ggml_new_tensor_4d(self.ctx.as_ptr(), typ, ne0, ne1, ne2, ne3)
        };
        GgmlTensor(NonNull::new(t).expect("ggml_new_tensor_4d returned null"))
    }

    /// Create a tensor with the same shape and type as another.
    #[must_use]
    pub fn dup_tensor(&self, src: &GgmlTensor) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_dup_tensor(self.ctx.as_ptr(), src.0.as_ptr()) };
        GgmlTensor(NonNull::new(t).expect("ggml_dup_tensor returned null"))
    }

    /// Create a new tensor with arbitrary dimensions.
    #[must_use]
    pub fn new_tensor(&self, typ: ggml_type, ne: &[i64]) -> GgmlTensor {
        let t = unsafe {
            llama_cpp_sys_4::ggml_new_tensor(
                self.ctx.as_ptr(),
                typ,
                ne.len() as i32,
                ne.as_ptr(),
            )
        };
        GgmlTensor(NonNull::new(t).expect("ggml_new_tensor returned null"))
    }

    // ── Tensor operations (build graph nodes) ────────────────

    /// Element-wise addition: `a + b`
    #[must_use]
    pub fn add(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let t =
            unsafe { llama_cpp_sys_4::ggml_add(self.ctx.as_ptr(), a.0.as_ptr(), b.0.as_ptr()) };
        GgmlTensor(NonNull::new(t).expect("ggml_add returned null"))
    }

    /// Matrix multiplication: `a @ b`
    #[must_use]
    pub fn mul_mat(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let t = unsafe {
            llama_cpp_sys_4::ggml_mul_mat(self.ctx.as_ptr(), a.0.as_ptr(), b.0.as_ptr())
        };
        GgmlTensor(NonNull::new(t).expect("ggml_mul_mat returned null"))
    }

    /// Scale tensor: `a * s`
    #[must_use]
    pub fn scale(&self, a: &GgmlTensor, s: f32) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_scale(self.ctx.as_ptr(), a.0.as_ptr(), s) };
        GgmlTensor(NonNull::new(t).expect("ggml_scale returned null"))
    }

    /// Cast tensor to a different type.
    #[must_use]
    pub fn cast(&self, a: &GgmlTensor, typ: ggml_type) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_cast(self.ctx.as_ptr(), a.0.as_ptr(), typ) };
        GgmlTensor(NonNull::new(t).expect("ggml_cast returned null"))
    }

    /// Make tensor contiguous in memory.
    #[must_use]
    pub fn cont(&self, a: &GgmlTensor) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_cont(self.ctx.as_ptr(), a.0.as_ptr()) };
        GgmlTensor(NonNull::new(t).expect("ggml_cont returned null"))
    }

    /// Transpose a tensor.
    #[must_use]
    pub fn transpose(&self, a: &GgmlTensor) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_transpose(self.ctx.as_ptr(), a.0.as_ptr()) };
        GgmlTensor(NonNull::new(t).expect("ggml_transpose returned null"))
    }

    /// Reshape to 1D.
    #[must_use]
    pub fn reshape_1d(&self, a: &GgmlTensor, ne0: i64) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_reshape_1d(self.ctx.as_ptr(), a.0.as_ptr(), ne0) };
        GgmlTensor(NonNull::new(t).expect("ggml_reshape_1d returned null"))
    }

    /// Reshape to 2D.
    #[must_use]
    pub fn reshape_2d(&self, a: &GgmlTensor, ne0: i64, ne1: i64) -> GgmlTensor {
        let t =
            unsafe { llama_cpp_sys_4::ggml_reshape_2d(self.ctx.as_ptr(), a.0.as_ptr(), ne0, ne1) };
        GgmlTensor(NonNull::new(t).expect("ggml_reshape_2d returned null"))
    }

    /// Create a 1D view of a tensor.
    #[must_use]
    pub fn view_1d(&self, a: &GgmlTensor, ne0: i64, offset: usize) -> GgmlTensor {
        let t = unsafe {
            llama_cpp_sys_4::ggml_view_1d(self.ctx.as_ptr(), a.0.as_ptr(), ne0, offset)
        };
        GgmlTensor(NonNull::new(t).expect("ggml_view_1d returned null"))
    }

    // ── Graph creation ───────────────────────────────────────

    /// Create a new computation graph.
    #[must_use]
    pub fn new_graph(&self) -> GgmlGraph {
        let g = unsafe { llama_cpp_sys_4::ggml_new_graph(self.ctx.as_ptr()) };
        GgmlGraph(NonNull::new(g).expect("ggml_new_graph returned null"))
    }

    // ── Tensor iteration ─────────────────────────────────────

    /// Get the first tensor in this context.
    #[must_use]
    pub fn first_tensor(&self) -> Option<GgmlTensor> {
        let t = unsafe { llama_cpp_sys_4::ggml_get_first_tensor(self.ctx.as_ptr()) };
        NonNull::new(t).map(GgmlTensor)
    }

    /// Get the next tensor after `tensor` in this context.
    #[must_use]
    pub fn next_tensor(&self, tensor: &GgmlTensor) -> Option<GgmlTensor> {
        let t =
            unsafe { llama_cpp_sys_4::ggml_get_next_tensor(self.ctx.as_ptr(), tensor.0.as_ptr()) };
        NonNull::new(t).map(GgmlTensor)
    }
}

impl Drop for GgmlContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::ggml_free(self.ctx.as_ptr()) }
    }
}

// ── Tensor ──────────────────────────────────────────────────

/// A wrapper around a `ggml_tensor` pointer.
///
/// Tensors are owned by their `GgmlContext` and must not outlive it.
/// This wrapper does NOT free the tensor on drop.
#[derive(Clone, Copy)]
pub struct GgmlTensor(pub(crate) NonNull<llama_cpp_sys_4::ggml_tensor>);

impl GgmlTensor {
    /// Get the raw tensor pointer.
    pub fn as_ptr(&self) -> *mut llama_cpp_sys_4::ggml_tensor {
        self.0.as_ptr()
    }

    /// Set the tensor's name.
    pub fn set_name(&self, name: &str) {
        let c_name = std::ffi::CString::new(name).expect("name contains null bytes");
        unsafe { llama_cpp_sys_4::ggml_set_name(self.0.as_ptr(), c_name.as_ptr()) };
    }

    /// Get the number of elements.
    #[must_use]
    pub fn nelements(&self) -> i64 {
        unsafe { llama_cpp_sys_4::ggml_nelements(self.0.as_ptr()) }
    }

    /// Get the total size in bytes.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        unsafe { llama_cpp_sys_4::ggml_nbytes(self.0.as_ptr()) }
    }

    /// Get the element size in bytes.
    #[must_use]
    pub fn element_size(&self) -> usize {
        unsafe { llama_cpp_sys_4::ggml_element_size(self.0.as_ptr()) }
    }

    /// Get the tensor type.
    #[must_use]
    pub fn typ(&self) -> ggml_type {
        unsafe { (*self.0.as_ptr()).type_ }
    }

    /// Get the tensor dimensions (ne).
    #[must_use]
    pub fn ne(&self) -> [i64; 4] {
        unsafe { (*self.0.as_ptr()).ne }
    }

    /// Get the tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        unsafe {
            let ptr = (*self.0.as_ptr()).name.as_ptr();
            CStr::from_ptr(ptr).to_str().unwrap_or("")
        }
    }
}

impl std::fmt::Debug for GgmlTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ne = self.ne();
        write!(
            f,
            "GgmlTensor({:?}, [{}, {}, {}, {}], {} bytes)",
            self.name(),
            ne[0],
            ne[1],
            ne[2],
            ne[3],
            self.nbytes()
        )
    }
}

// ── Graph ───────────────────────────────────────────────────

/// A wrapper around `ggml_cgraph`.
///
/// Graphs are owned by their `GgmlContext` and must not outlive it.
pub struct GgmlGraph(NonNull<llama_cpp_sys_4::ggml_cgraph>);

impl std::fmt::Debug for GgmlGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgmlGraph").finish()
    }
}

impl GgmlGraph {
    /// Get the raw graph pointer.
    pub fn as_ptr(&mut self) -> *mut llama_cpp_sys_4::ggml_cgraph {
        self.0.as_ptr()
    }

    /// Add a tensor and its dependencies to the forward computation graph.
    pub fn build_forward(&mut self, tensor: &GgmlTensor) {
        unsafe { llama_cpp_sys_4::ggml_build_forward_expand(self.0.as_ptr(), tensor.0.as_ptr()) }
    }

    /// Get a node (output tensor) by index. Use -1 for the last node.
    #[must_use]
    pub fn node(&mut self, i: i32) -> GgmlTensor {
        let t = unsafe { llama_cpp_sys_4::ggml_graph_node(self.0.as_ptr(), i) };
        GgmlTensor(NonNull::new(t).expect("graph_node returned null"))
    }
}

// ── Backend ─────────────────────────────────────────────────

/// A safe wrapper around `ggml_backend`.
#[derive(Debug)]
pub struct GgmlBackend {
    backend: llama_cpp_sys_4::ggml_backend_t,
}

impl GgmlBackend {
    /// Create a CPU backend.
    #[must_use]
    pub fn cpu() -> Self {
        let backend = unsafe { llama_cpp_sys_4::ggml_backend_cpu_init() };
        assert!(!backend.is_null(), "ggml_backend_cpu_init returned null");
        Self { backend }
    }

    /// Set the number of threads for the CPU backend.
    pub fn set_n_threads(&self, n_threads: i32) {
        unsafe { llama_cpp_sys_4::ggml_backend_cpu_set_n_threads(self.backend, n_threads) }
    }

    /// Allocate all tensors in a context on this backend.
    ///
    /// Returns the buffer handle which must be kept alive.
    pub fn alloc_ctx_tensors(
        &self,
        ctx: &GgmlContext,
    ) -> *mut llama_cpp_sys_4::ggml_backend_buffer {
        unsafe { llama_cpp_sys_4::ggml_backend_alloc_ctx_tensors(ctx.as_ptr(), self.backend) }
    }

    /// Compute a graph.
    pub fn graph_compute(&self, graph: &mut GgmlGraph) {
        unsafe { llama_cpp_sys_4::ggml_backend_graph_compute(self.backend, graph.as_ptr()) };
    }

    /// Get the default buffer type for this backend.
    pub fn default_buffer_type(&self) -> llama_cpp_sys_4::ggml_backend_buffer_type_t {
        unsafe { llama_cpp_sys_4::ggml_backend_get_default_buffer_type(self.backend) }
    }

    /// Get the raw backend pointer.
    pub fn as_ptr(&self) -> llama_cpp_sys_4::ggml_backend_t {
        self.backend
    }
}

impl Drop for GgmlBackend {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::ggml_backend_free(self.backend) }
    }
}

// ── Graph allocator ─────────────────────────────────────────

/// A safe wrapper around `ggml_gallocr`.
#[derive(Debug)]
pub struct GgmlAllocr {
    alloc: llama_cpp_sys_4::ggml_gallocr_t,
}

impl GgmlAllocr {
    /// Create a new graph allocator for the given backend.
    #[must_use]
    pub fn new(backend: &GgmlBackend) -> Self {
        let alloc = unsafe { llama_cpp_sys_4::ggml_gallocr_new(backend.default_buffer_type()) };
        assert!(!alloc.is_null(), "ggml_gallocr_new returned null");
        Self { alloc }
    }

    /// Allocate all tensors in a graph.
    pub fn alloc_graph(&self, graph: &mut GgmlGraph) -> bool {
        unsafe { llama_cpp_sys_4::ggml_gallocr_alloc_graph(self.alloc, graph.as_ptr()) }
    }
}

impl Drop for GgmlAllocr {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::ggml_gallocr_free(self.alloc) }
    }
}

// ── Utility functions ───────────────────────────────────────

/// Set tensor data from a byte slice.
///
/// # Safety
///
/// The tensor must be allocated and the data must be the correct size.
pub unsafe fn tensor_set(tensor: &GgmlTensor, data: &[u8]) {
    llama_cpp_sys_4::ggml_backend_tensor_set(
        tensor.0.as_ptr(),
        data.as_ptr().cast(),
        0,
        data.len(),
    );
}

/// Get tensor data into a byte slice.
///
/// # Safety
///
/// The tensor must be allocated and the buffer must be large enough.
pub unsafe fn tensor_get(tensor: &GgmlTensor, data: &mut [u8]) {
    llama_cpp_sys_4::ggml_backend_tensor_get(
        tensor.0.as_ptr(),
        data.as_mut_ptr().cast(),
        0,
        data.len(),
    );
}

/// Free a backend buffer.
///
/// # Safety
///
/// The buffer must be valid and not already freed.
pub unsafe fn buffer_free(buffer: *mut llama_cpp_sys_4::ggml_backend_buffer) {
    llama_cpp_sys_4::ggml_backend_buffer_free(buffer);
}

/// Get the overhead in bytes for tensor metadata.
#[must_use]
pub fn tensor_overhead() -> usize {
    unsafe { llama_cpp_sys_4::ggml_tensor_overhead() }
}

/// Get the overhead in bytes for a computation graph.
#[must_use]
pub fn graph_overhead() -> usize {
    unsafe { llama_cpp_sys_4::ggml_graph_overhead() }
}

/// Check if a type is quantized.
#[must_use]
pub fn is_quantized(typ: ggml_type) -> bool {
    unsafe { llama_cpp_sys_4::ggml_is_quantized(typ) }
}

/// Get the name of a ggml type.
#[must_use]
pub fn type_name(typ: ggml_type) -> &'static str {
    unsafe {
        let ptr = llama_cpp_sys_4::ggml_type_name(typ);
        if ptr.is_null() {
            "unknown"
        } else {
            CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        }
    }
}
