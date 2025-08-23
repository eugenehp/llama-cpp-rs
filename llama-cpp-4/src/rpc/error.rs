//! Error types for RPC functionality

use std::ffi::NulError;
use thiserror::Error;

/// Errors that can occur in RPC operations
#[derive(Debug, Error)]
pub enum RpcError {
    /// Failed to initialize RPC backend
    #[error("Failed to initialize RPC backend for endpoint: {endpoint}")]
    InitializationFailed { endpoint: String },
    
    /// Invalid endpoint format
    #[error("Invalid endpoint format: {endpoint}")]
    InvalidEndpoint { endpoint: String },
    
    /// Connection failed
    #[error("Failed to connect to RPC server at {endpoint}: {reason}")]
    ConnectionFailed { endpoint: String, reason: String },
    
    /// Server error
    #[error("RPC server error: {message}")]
    ServerError { message: String },
    
    /// Memory query failed
    #[error("Failed to query device memory")]
    MemoryQueryFailed,
    
    /// String conversion error
    #[error("Failed to convert C string: {0}")]
    StringConversion(#[from] NulError),
    
    /// UTF-8 conversion error
    #[error("Invalid UTF-8: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    
    /// Feature not available
    #[error("RPC feature not compiled in. Build with --features rpc")]
    NotAvailable,
}