//! See [llama-cpp-4](https://crates.io/crates/llama-cpp-4) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// `bare_urls` fires on doxygen comments carried over from llama.cpp's headers,
// which bindgen copies verbatim — upstream's prose is not this crate's to
// reformat, and the lint is what kept `cargo doc -D warnings` from ever
// passing for this crate.
#[allow(
    unnecessary_transmutes,
    rustdoc::bare_urls,
    clippy::missing_safety_doc,
    clippy::ptr_offset_with_cast
)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;

pub mod common;
