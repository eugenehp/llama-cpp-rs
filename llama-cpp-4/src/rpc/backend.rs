//! RPC backend for distributed inference

use crate::rpc::error::RpcError;
use llama_cpp_sys_4 as sys;
use std::ffi::CString;
use std::ptr::NonNull;

/// RPC backend for distributed inference across multiple machines
pub struct RpcBackend {
    backend: NonNull<sys::ggml_backend>,
    endpoint: String,
}

impl RpcBackend {
    /// Initialize a new RPC backend for the given endpoint
    /// 
    /// # Arguments
    /// * `endpoint` - The RPC server endpoint (e.g., "127.0.0.1:50052")
    /// 
    /// # Example
    /// ```no_run
    /// use llama_cpp_4::rpc::RpcBackend;
    /// 
    /// let backend = RpcBackend::init("127.0.0.1:50052")?;
    /// ```
    pub fn init(endpoint: &str) -> Result<Self, RpcError> {
        let c_endpoint = CString::new(endpoint)
            .map_err(|e| RpcError::StringConversion(e))?;
        
        let backend = unsafe {
            sys::ggml_backend_rpc_init(c_endpoint.as_ptr())
        };
        
        NonNull::new(backend)
            .map(|ptr| Self {
                backend: ptr,
                endpoint: endpoint.to_string(),
            })
            .ok_or_else(|| RpcError::InitializationFailed {
                endpoint: endpoint.to_string(),
            })
    }
    
    /// Check if a backend is an RPC backend
    pub fn is_rpc(&self) -> bool {
        unsafe { sys::ggml_backend_is_rpc(self.backend.as_ptr()) }
    }
    
    /// Get the buffer type for this RPC backend
    pub fn buffer_type(&self) -> Option<NonNull<sys::ggml_backend_buffer_type>> {
        let c_endpoint = CString::new(self.endpoint.as_str()).ok()?;
        let buffer_type = unsafe {
            sys::ggml_backend_rpc_buffer_type(c_endpoint.as_ptr())
        };
        NonNull::new(buffer_type)
    }
    
    /// Query the available memory on the remote device
    /// 
    /// Returns (free_memory, total_memory) in bytes
    pub fn get_device_memory(&self) -> Result<(usize, usize), RpcError> {
        let c_endpoint = CString::new(self.endpoint.as_str())
            .map_err(|e| RpcError::StringConversion(e))?;
        
        let mut free: usize = 0;
        let mut total: usize = 0;
        
        unsafe {
            sys::ggml_backend_rpc_get_device_memory(
                c_endpoint.as_ptr(),
                &mut free,
                &mut total,
            );
        }
        
        if total == 0 {
            Err(RpcError::MemoryQueryFailed)
        } else {
            Ok((free, total))
        }
    }
    
    /// Get the endpoint this backend is connected to
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
    
    /// Get the raw backend pointer for FFI calls
    pub(crate) fn as_ptr(&self) -> NonNull<sys::ggml_backend> {
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
            .field("is_rpc", &self.is_rpc())
            .finish()
    }
}