//! # Importance Matrix (simplified)
//!
//! Compute per-token probability statistics that indicate which parts of the model
//! are most important for a given calibration dataset. This information is used
//! to improve quantization quality.
//!
//! The C++ `llama-imatrix` tool collects per-tensor activation statistics via
//! an eval callback and saves them in GGUF format. This simplified Rust version
//! computes per-token log-probabilities and perplexity statistics that serve as
//! a proxy for importance — tokens with low probability indicate areas where the
//! model struggles and where quantization precision matters most.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p imatrix -- -m model.gguf -f calibration.txt
//! cargo run -p imatrix -- -m model.gguf -f calibration.txt -o stats.dat --chunks 5
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
#[command(about = "Compute importance statistics from a calibration dataset")]
struct Args {
    /// Path to the GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Path to calibration text file
    #[arg(short = 'f', long)]
    file: PathBuf,

    /// Output file for statistics (binary format)
    #[arg(short = 'o', long, default_value = "imatrix-stats.dat")]
    output: PathBuf,

    /// Context size
    #[arg(short = 'c', long, default_value_t = 512)]
    ctx_size: u32,

    /// Number of chunks to process (-1 = all)
    #[arg(long, default_value_t = -1)]
    chunks: i32,
}

/// Per-position statistics
struct TokenStat {
    token_id: i32,
    position: i32,
    log_prob: f64,
    prob: f32,
    rank: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("failed to read: {}", args.file.display()))?;

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| "failed to load model")?;

    eprintln!("Model: {model}");

    let n_ctx = args.ctx_size;
    let n_vocab = model.n_vocab();

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

    eprintln!("Tokenizing...");
    let tokens = model.str_to_token(&text, add_bos)?;

    let n_ctx_i = n_ctx as i32;
    if (tokens.len() as i32) < 2 * n_ctx_i {
        bail!("Need at least {} tokens, input has {}", 2 * n_ctx_i, tokens.len());
    }

    let n_chunk_max = tokens.len() as i32 / n_ctx_i;
    let n_chunk = if args.chunks < 0 {
        n_chunk_max
    } else {
        args.chunks.min(n_chunk_max)
    };

    eprintln!("Processing {} chunks (ctx={}, tokens={})...", n_chunk, n_ctx, tokens.len());

    let first = n_ctx_i / 2;
    let mut all_stats: Vec<TokenStat> = Vec::new();
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);

    for i in 0..n_chunk {
        let start = (i * n_ctx_i) as usize;

        ctx.clear_kv_cache();
        batch.clear();

        for k in 0..n_ctx as usize {
            let pos = k as i32;
            batch.add(tokens[start + k], pos, &[0], pos >= first)?;
        }

        ctx.decode(&mut batch)?;
        ctx.synchronize();

        // Collect stats for second half
        for pos in first..(n_ctx_i - 1) {
            let token_idx = start + pos as usize + 1;
            let target = tokens[token_idx].0;
            let logits = ctx.get_logits_ith(pos);

            // Log-softmax
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = logits.iter().map(|&l| ((l - max_logit) as f64).exp()).sum();
            let log_prob = (logits[target as usize] - max_logit) as f64 - sum_exp.ln();
            let prob = log_prob.exp() as f32;

            // Rank: how many tokens have higher logit
            let target_logit = logits[target as usize];
            let rank = logits.iter().filter(|&&l| l > target_logit).count() as u32;

            all_stats.push(TokenStat {
                token_id: target,
                position: pos,
                log_prob,
                prob,
                rank,
            });
        }

        let count = all_stats.len();
        let avg_nll: f64 = -all_stats.iter().map(|s| s.log_prob).sum::<f64>() / count as f64;
        let ppl = avg_nll.exp();
        let avg_rank: f64 = all_stats.iter().map(|s| s.rank as f64).sum::<f64>() / count as f64;
        eprint!("[{}] ppl={:.4} avg_rank={:.1}  ", i + 1, ppl, avg_rank);
        std::io::stderr().flush()?;
    }
    eprintln!();

    // Summary
    let count = all_stats.len();
    let total_nll: f64 = -all_stats.iter().map(|s| s.log_prob).sum::<f64>();
    let ppl = (total_nll / count as f64).exp();
    let avg_rank: f64 = all_stats.iter().map(|s| s.rank as f64).sum::<f64>() / count as f64;
    let low_prob: usize = all_stats.iter().filter(|s| s.prob < 0.01).count();

    println!();
    println!("=== Importance Statistics ===");
    println!("Tokens evaluated : {count}");
    println!("Perplexity       : {ppl:.4}");
    println!("Avg token rank   : {avg_rank:.1}");
    println!("Low-prob tokens  : {low_prob} ({:.1}%)", 100.0 * low_prob as f64 / count as f64);

    // Save binary stats
    let mut out = std::fs::File::create(&args.output)?;
    // Header
    out.write_all(b"IMAT")?;
    out.write_all(&(count as u32).to_le_bytes())?;
    out.write_all(&(n_vocab as u32).to_le_bytes())?;
    // Per-token data
    for stat in &all_stats {
        out.write_all(&stat.token_id.to_le_bytes())?;
        out.write_all(&stat.position.to_le_bytes())?;
        out.write_all(&stat.log_prob.to_le_bytes())?;
        out.write_all(&stat.prob.to_le_bytes())?;
        out.write_all(&stat.rank.to_le_bytes())?;
    }

    eprintln!("Statistics saved to {}", args.output.display());

    Ok(())
}
