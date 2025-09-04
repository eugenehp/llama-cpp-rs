//! RPC backend support for distributed inference
//! 
//! This module provides support for running inference across multiple machines
//! using the RPC (Remote Procedure Call) backend.

#[cfg(feature = "rpc")]
pub mod backend;

#[cfg(feature = "rpc")]
pub mod server;

#[cfg(feature = "rpc")]
pub mod error;

#[cfg(feature = "rpc")]
pub use backend::RpcBackend;

#[cfg(feature = "rpc")]
pub use server::RpcServer;

#[cfg(feature = "rpc")]
pub use error::RpcError;

#[cfg(feature = "rpc")]
pub use server::add_rpc_device;