//! RPC backend for distributed inference

use crate::rpc::error::RpcError;
use llama_cpp_sys_4 as sys;
use std::ffi::CString;
use std::ptr::NonNull;

/// RPC backend for distributed inference across multiple machines
pub struct RpcBackend {
    backend: NonNull<sys::ggml_backend>,
    endpoint: String,
    device: u32,
}

impl RpcBackend {
    /// Initialize a new RPC backend for the given endpoint and remote device.
    ///
    /// # Arguments
    /// * `endpoint` - The RPC server endpoint (e.g., "127.0.0.1:50052")
    /// * `device` - Index of the device to use on the remote server. A single
    ///   endpoint can expose several devices; pass `0` for the first.
    ///
    /// # Errors
    ///
    /// Returns [`RpcError::StringConversion`] if `endpoint` contains an interior
    /// NUL, or [`RpcError::InitializationFailed`] if llama.cpp could not reach
    /// the endpoint or the device index is out of range.
    ///
    /// # Example
    /// ```no_run
    /// use llama_cpp_4::rpc::RpcBackend;
    ///
    /// let backend = RpcBackend::init("127.0.0.1:50052", 0)?;
    /// # Ok::<(), llama_cpp_4::rpc::RpcError>(())
    /// ```
    pub fn init(endpoint: &str, device: u32) -> Result<Self, RpcError> {
        let c_endpoint = CString::new(endpoint)?;

        let backend = unsafe { sys::ggml_backend_rpc_init(c_endpoint.as_ptr(), device) };

        NonNull::new(backend)
            .map(|ptr| Self {
                backend: ptr,
                endpoint: endpoint.to_string(),
                device,
            })
            .ok_or_else(|| RpcError::InitializationFailed {
                endpoint: endpoint.to_string(),
            })
    }

    /// Check if a backend is an RPC backend
    #[must_use]
    pub fn is_rpc(&self) -> bool {
        unsafe { sys::ggml_backend_is_rpc(self.backend.as_ptr()) }
    }

    /// Get the buffer type for this RPC backend
    #[must_use]
    pub fn buffer_type(&self) -> Option<NonNull<sys::ggml_backend_buffer_type>> {
        let c_endpoint = CString::new(self.endpoint.as_str()).ok()?;
        let buffer_type =
            unsafe { sys::ggml_backend_rpc_buffer_type(c_endpoint.as_ptr(), self.device) };
        NonNull::new(buffer_type)
    }

    /// Query the available memory on the remote device
    ///
    /// Returns (`free_memory`, `total_memory`) in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RpcError::MemoryQueryFailed`] when the server reports a total
    /// of zero, which is how an unreachable endpoint surfaces here.
    pub fn get_device_memory(&self) -> Result<(usize, usize), RpcError> {
        let c_endpoint = CString::new(self.endpoint.as_str())?;

        let mut free: usize = 0;
        let mut total: usize = 0;

        unsafe {
            sys::ggml_backend_rpc_get_device_memory(
                c_endpoint.as_ptr(),
                self.device,
                std::ptr::from_mut(&mut free),
                std::ptr::from_mut(&mut total),
            );
        }

        if total == 0 {
            Err(RpcError::MemoryQueryFailed)
        } else {
            Ok((free, total))
        }
    }

    /// Get the endpoint this backend is connected to
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Index of the remote device this backend is bound to.
    #[must_use]
    pub fn device(&self) -> u32 {
        self.device
    }

    /// Get the raw backend pointer for FFI calls.
    ///
    /// The pointer is owned by this `RpcBackend` and is freed on drop; do not
    /// free it, and do not use it after this value goes out of scope.
    #[must_use]
    pub fn as_ptr(&self) -> NonNull<sys::ggml_backend> {
        self.backend
    }
}

impl Drop for RpcBackend {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_free(self.backend.as_ptr());
        }
    }
}

// Safety: RpcBackend can be sent between threads
unsafe impl Send for RpcBackend {}
// Safety: RpcBackend can be shared between threads (the C API is thread-safe)
unsafe impl Sync for RpcBackend {}

impl std::fmt::Debug for RpcBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcBackend")
            .field("endpoint", &self.endpoint)
            .field("device", &self.device)
            .field("is_rpc", &self.is_rpc())
            // `backend` is an opaque llama.cpp pointer with nothing useful to show.
            .finish_non_exhaustive()
    }
}
