//! RPC server for hosting backends

use crate::rpc::error::RpcError;
use llama_cpp_sys_4 as sys;
use std::ffi::CString;
use std::ptr::NonNull;

/// RPC server for hosting a backend that can be accessed remotely
pub struct RpcServer {
    backend: NonNull<sys::ggml_backend>,
    endpoint: String,
}

impl RpcServer {
    /// Start an RPC server for the given backend
    /// 
    /// # Arguments
    /// * `backend` - The backend to expose via RPC
    /// * `endpoint` - The endpoint to listen on (e.g., "0.0.0.0:50052")
    /// * `free_mem` - Amount of free memory to advertise (0 for auto)
    /// * `total_mem` - Total memory to advertise (0 for auto)
    /// 
    /// # Example
    /// ```no_run
    /// use llama_cpp_4::rpc::RpcServer;
    /// 
    /// // Assuming you have a backend initialized
    /// let server = RpcServer::start(
    ///     backend,
    ///     "0.0.0.0:50052",
    ///     0,
    ///     0,
    /// )?;
    /// ```
    pub fn start(
        backend: NonNull<sys::ggml_backend>,
        endpoint: &str,
        free_mem: usize,
        total_mem: usize,
    ) -> Result<Self, RpcError> {
        let c_endpoint = CString::new(endpoint)
            .map_err(|e| RpcError::StringConversion(e))?;
        
        unsafe {
            sys::ggml_backend_rpc_start_server(
                backend.as_ptr(),
                c_endpoint.as_ptr(),
                free_mem,
                total_mem,
            );
        }
        
        Ok(Self {
            backend,
            endpoint: endpoint.to_string(),
        })
    }
    
    /// Get the endpoint this server is listening on
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
    
    /// Get the backend this server is hosting
    pub fn backend(&self) -> NonNull<sys::ggml_backend> {
        self.backend
    }
}

impl std::fmt::Debug for RpcServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcServer")
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

// Safety: RpcServer can be sent between threads
unsafe impl Send for RpcServer {}
// Safety: RpcServer can be shared between threads
unsafe impl Sync for RpcServer {}

/// Add a new RPC device
/// 
/// This function registers a new RPC device that can be used for inference.
/// 
/// # Arguments
/// * `endpoint` - The RPC server endpoint to connect to
/// 
/// # Returns
/// The device handle if successful
pub fn add_rpc_device(endpoint: &str) -> Result<NonNull<sys::ggml_backend_device>, RpcError> {
    let c_endpoint = CString::new(endpoint)
        .map_err(|e| RpcError::StringConversion(e))?;
    
    let device = unsafe {
        sys::ggml_backend_rpc_add_device(c_endpoint.as_ptr())
    };
    
    NonNull::new(device)
        .ok_or_else(|| RpcError::InitializationFailed {
            endpoint: endpoint.to_string(),
        })
}