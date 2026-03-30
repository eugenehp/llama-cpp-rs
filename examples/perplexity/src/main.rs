//! # Perplexity
//!
//! Calculate the perplexity of a model on a text dataset.
//! Perplexity measures how well a model predicts a text — lower is better.
//!
//! This is the Rust equivalent of llama.cpp's `llama-perplexity` tool.
//!
//! ## Usage
//!
//! ```console
//! # Download wikitext-2 test set:
//! # wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
//! # unzip wikitext-2-raw-v1.zip
//!
//! cargo run -p perplexity -- -m model.gguf -f wiki.test.raw
//! cargo run -p perplexity -- -m model.gguf -f wiki.test.raw --chunks 10
//! ```
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{bail, Context, Result};
use clap::Parser;
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaModel};
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command(about = "Calculate perplexity of a model on a text dataset")]
struct Args {
    /// Path to the GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Path to the text file to evaluate
    #[arg(short = 'f', long)]
    file: PathBuf,

    /// Context size (default: model's training context)
    #[arg(short = 'c', long)]
    ctx_size: Option<u32>,

    /// Number of chunks to evaluate (-1 = all)
    #[arg(long, default_value_t = -1)]
    chunks: i32,
}

/// Compute log-softmax for a single token given logits over the full vocabulary.
fn log_softmax(logits: &[f32], token: i32) -> (f64, f32) {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max_logit) as f64).exp()).sum();
    let log_sum_exp = sum_exp.ln();
    let log_prob = (logits[token as usize] - max_logit) as f64 - log_sum_exp;
    let prob = ((logits[token as usize] - max_logit) as f64 - log_sum_exp).exp() as f32;
    (log_prob, prob)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read the input text
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("failed to read: {}", args.file.display()))?;

    // Load model
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| "failed to load model")?;

    eprintln!("Model: {model}");

    let n_ctx = args.ctx_size.unwrap_or(model.n_ctx_train());
    let _n_vocab = model.n_vocab();

    // Set batch size to context size so we can process each chunk in one decode call
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "failed to create context")?;

    let add_bos = if model.add_bos_token() {
        AddBos::Always
    } else {
        AddBos::Never
    };

    // Tokenize the entire text
    eprintln!("Tokenizing input...");
    let tokens = model
        .str_to_token(&text, add_bos)
        .with_context(|| "tokenization failed")?;

    let n_ctx = n_ctx as i32;

    if (tokens.len() as i32) < 2 * n_ctx {
        bail!(
            "Need at least {} tokens for context size {}, but input has only {} tokens",
            2 * n_ctx,
            n_ctx,
            tokens.len()
        );
    }

    let n_chunk_max = tokens.len() as i32 / n_ctx;
    let n_chunk = if args.chunks < 0 {
        n_chunk_max
    } else {
        args.chunks.min(n_chunk_max)
    };

    eprintln!(
        "Calculating perplexity over {} chunks, n_ctx={}, tokens={}",
        n_chunk,
        n_ctx,
        tokens.len()
    );

    // Use the second half of each context window for perplexity calculation.
    // The first half provides context for the model to condition on.
    let first = n_ctx / 2;

    let mut count = 0_i64;
    let mut nll = 0.0_f64; // negative log-likelihood accumulator

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);

    for i in 0..n_chunk {
        let start = (i * n_ctx) as usize;

        let t_start = llama_cpp_4::ggml_time_us();

        // Clear KV cache for each chunk
        ctx.clear_kv_cache();

        // Process entire chunk at once (n_ctx must fit in one batch for simplicity)
        // Enable logits for the second half of the context window
        batch.clear();
        for k in 0..n_ctx as usize {
            let pos = k as i32;
            let logits_enabled = pos >= first;
            batch.add(tokens[start + k], pos, &[0], logits_enabled)?;
        }

        ctx.decode(&mut batch)
            .with_context(|| format!("decode failed at chunk {i}"))?;
        ctx.synchronize();

        // Calculate perplexity for the second half of the window.
        // For each position p in [first..n_ctx-1), the logits at position p
        // predict the token at position (p + 1).
        for pos in first..(n_ctx - 1) {
            let token_idx = start + pos as usize + 1;
            let target_token = tokens[token_idx].0;
            let logits = ctx.get_logits_ith(pos);

            let (log_prob, _prob) = log_softmax(logits, target_token);
            nll -= log_prob;
            count += 1;
        }

        let t_end = llama_cpp_4::ggml_time_us();
        let t_chunk = (t_end - t_start) as f64 / 1_000_000.0;

        let ppl = (nll / count as f64).exp();
        eprint!("[{}]{:.4},", i + 1, ppl);
        std::io::stderr().flush()?;

        // Print ETA after first chunk
        if i == 0 && n_chunk > 1 {
            let eta_seconds = t_chunk * n_chunk as f64;
            if eta_seconds >= 3600.0 {
                eprintln!(
                    " {:.1} seconds per pass - ETA {:.0} hours {:.1} minutes",
                    t_chunk,
                    eta_seconds / 3600.0,
                    (eta_seconds % 3600.0) / 60.0
                );
            } else {
                eprintln!(
                    " {:.1} seconds per pass - ETA {:.1} minutes",
                    t_chunk,
                    eta_seconds / 60.0
                );
            }
        }
    }

    eprintln!();

    let ppl = (nll / count as f64).exp();
    let avg_nll = nll / count as f64;

    println!();
    println!("Final perplexity: {:.4} (avg NLL: {:.6}, {} tokens evaluated)", ppl, avg_nll, count);

    Ok(())
}
