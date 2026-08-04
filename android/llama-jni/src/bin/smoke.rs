//! arm64 smoke test for `llama-cpp-4`.
//!
//! Loads a GGUF and greedily generates a few tokens via [`llama_jni::generate`].
//! CI cross-builds this for `aarch64-unknown-linux-gnu` and runs it under
//! `qemu-aarch64` to validate the arm64 native libraries end-to-end. It also
//! runs natively: `cargo run -p llama-jni --bin smoke -- path/to/model.gguf`.
//!
//! Model: `LLAMA_TEST_MODEL` env var, else the first CLI argument.
//! Prompt: the next CLI argument, else a default.

use std::path::PathBuf;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let (model, prompt) = match std::env::var("LLAMA_TEST_MODEL") {
        Ok(model) => (PathBuf::from(model), args.first().cloned()),
        Err(_) => {
            let model = args
                .first()
                .context("set LLAMA_TEST_MODEL or pass a GGUF path as the first argument")?;
            (PathBuf::from(model), args.get(1).cloned())
        }
    };
    let prompt = prompt.unwrap_or_else(|| "Once upon a time".to_string());

    println!("model : {}", model.display());
    println!("prompt: {prompt}");

    let output = llama_jni::generate(&model, &prompt, 24).context("generation failed")?;

    println!("output: {output}");
    anyhow::ensure!(!output.trim().is_empty(), "generation produced no output");
    println!("SMOKE OK ({} bytes generated)", output.len());
    Ok(())
}
