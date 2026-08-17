//! RPC backend support for distributed inference
//!
//! This module provides support for running inference across multiple machines
//! using the RPC (Remote Procedure Call) backend.
//!
//! A single endpoint can expose several devices, so both connecting
//! ([`RpcBackend::init`]) and serving ([`serve`]) are device-aware: clients
//! address a remote device by index, in the order the server listed them.

#[cfg(feature = "rpc")]
pub mod backend;

#[cfg(feature = "rpc")]
pub mod server;

#[cfg(feature = "rpc")]
pub mod error;

#[cfg(feature = "rpc")]
pub use backend::RpcBackend;

#[cfg(feature = "rpc")]
pub use error::RpcError;

#[cfg(feature = "rpc")]
pub use server::{add_rpc_server, serve};
