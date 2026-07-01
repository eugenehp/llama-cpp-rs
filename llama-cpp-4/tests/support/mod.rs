//! Shared helpers for integration tests that load a GGUF checkpoint.
//!
//! Model resolution order:
//! 1. `LLAMA_TEST_MODEL` — explicit path to a `.gguf` file
//! 2. `../target/test-models/stories260K.gguf` (from [`scripts/fetch-test-model.sh`])
//! 3. Vocab-only `ggml-vocab-llama-bpe.gguf` from the `llama-cpp-sys-4` build tree

pub mod model;
