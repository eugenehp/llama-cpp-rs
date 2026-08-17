//! RPC server for hosting backends

use crate::rpc::error::RpcError;
use llama_cpp_sys_4 as sys;
use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;

/// Serve one or more local devices over RPC.
///
/// **This call blocks.** `ggml_backend_rpc_start_server` runs the accept loop on
/// the calling thread and does not return while the server is live, so run it on
/// a dedicated thread if the caller needs to stay responsive.
///
/// # Arguments
/// * `endpoint` - Address to listen on (e.g. `"0.0.0.0:50052"`)
/// * `cache_dir` - Directory for the server-side tensor cache, or `None` to
///   disable caching
/// * `n_threads` - Worker threads used to service requests
/// * `devices` - Local devices to expose; clients address them by index, in the
///   order given here
///
/// # Errors
///
/// Returns [`RpcError::StringConversion`] if `endpoint` or `cache_dir` contains
/// an interior NUL, [`RpcError::InvalidEndpoint`] if `cache_dir` is not valid
/// UTF-8, and [`RpcError::ServerError`] if `devices` is empty or exceeds
/// llama.cpp's server limit.
///
/// # Example
/// ```no_run
/// use llama_cpp_4::rpc::serve;
///
/// // `devices` comes from the ggml backend registry.
/// # let devices: Vec<std::ptr::NonNull<llama_cpp_sys_4::ggml_backend_device>> = vec![];
/// serve("0.0.0.0:50052", None, 4, &devices)?;
/// # Ok::<(), llama_cpp_4::rpc::RpcError>(())
/// ```
pub fn serve(
    endpoint: &str,
    cache_dir: Option<&Path>,
    n_threads: usize,
    devices: &[NonNull<sys::ggml_backend_device>],
) -> Result<(), RpcError> {
    if devices.is_empty() {
        return Err(RpcError::ServerError {
            message: "at least one device must be exposed".to_owned(),
        });
    }
    if devices.len() > sys::GGML_RPC_MAX_SERVERS as usize {
        return Err(RpcError::ServerError {
            message: format!(
                "{} devices requested but llama.cpp serves at most {}",
                devices.len(),
                sys::GGML_RPC_MAX_SERVERS
            ),
        });
    }

    let c_endpoint = CString::new(endpoint)?;
    let c_cache_dir = cache_dir
        .map(|dir| {
            let dir = dir.to_str().ok_or_else(|| RpcError::InvalidEndpoint {
                endpoint: dir.display().to_string(),
            })?;
            CString::new(dir).map_err(RpcError::from)
        })
        .transpose()?;

    // `ggml_backend_dev_t` is a raw pointer, so a `NonNull` slice has the same
    // layout; copy into an owned Vec rather than casting to keep that implicit.
    let mut device_ptrs: Vec<sys::ggml_backend_dev_t> =
        devices.iter().map(|d| d.as_ptr()).collect();

    unsafe {
        sys::ggml_backend_rpc_start_server(
            c_endpoint.as_ptr(),
            c_cache_dir
                .as_ref()
                .map_or(std::ptr::null(), |d| d.as_ptr()),
            n_threads,
            device_ptrs.len(),
            device_ptrs.as_mut_ptr(),
        );
    }

    Ok(())
}

/// Register a remote RPC server with the ggml backend registry.
///
/// Returns the backend *registration* covering every device the endpoint
/// exposes; use [`RpcBackend::init`](crate::rpc::RpcBackend::init) to bind one
/// of those devices.
///
/// # Errors
///
/// Returns [`RpcError::StringConversion`] if `endpoint` contains an interior
/// NUL, or [`RpcError::InitializationFailed`] if the endpoint could not be
/// registered.
pub fn add_rpc_server(endpoint: &str) -> Result<NonNull<sys::ggml_backend_reg>, RpcError> {
    let c_endpoint = CString::new(endpoint)?;

    let reg = unsafe { sys::ggml_backend_rpc_add_server(c_endpoint.as_ptr()) };

    NonNull::new(reg).ok_or_else(|| RpcError::InitializationFailed {
        endpoint: endpoint.to_string(),
    })
}
