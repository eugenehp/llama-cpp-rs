//! # Control Vector Generator (simplified)
//!
//! Generate control vectors by comparing model embeddings between positive and
//! negative prompt pairs. Control vectors can steer model behavior (e.g. more/less
//! creative, formal, etc.) when applied via `set_adapter_cvec`.
//!
//! The C++ `cvector-generator` tool extracts hidden states and performs PCA.
//! This simplified version computes mean embedding differences across prompt pairs,
//! which captures the primary direction of behavioral change.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p cvector-generator -- -m model.gguf \
//!     --positive "Write a creative story" \
//!     --negative "Write a boring story" \
//!     -o cvector.bin
//! ```
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{Context, Result};
use clap::Parser;
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaModel};
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command(about = "Generate control vectors from positive/negative prompt pairs")]
struct Args {
    /// Path to the GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Positive prompts (can specify multiple)
    #[arg(long, required = true)]
    positive: Vec<String>,

    /// Negative prompts (must match count of positive)
    #[arg(long, required = true)]
    negative: Vec<String>,

    /// Output file for control vector
    #[arg(short = 'o', long, default_value = "control-vector.bin")]
    output: PathBuf,

    /// Starting layer for control vector application
    #[arg(long, default_value_t = 1)]
    layer_start: i32,

    /// Ending layer for control vector application (0 = n_layer)
    #[arg(long, default_value_t = 0)]
    layer_end: i32,
}

/// Get embeddings for a prompt by running it through the model with embeddings enabled.
fn get_embeddings(
    ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
    model: &LlamaModel,
    prompt: &str,
) -> Result<Vec<f32>> {
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(tokens.len(), 1);

    let last = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        // Enable logits for the last token to get its hidden state
        batch.add(token, i, &[0], i == last)?;
    }

    ctx.clear_kv_cache();
    ctx.decode(&mut batch)?;

    // Get embeddings for the last token
    let emb = ctx.embeddings_ith(last)?;
    Ok(emb.to_vec())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.positive.len() != args.negative.len() {
        anyhow::bail!(
            "Number of positive ({}) and negative ({}) prompts must match",
            args.positive.len(),
            args.negative.len()
        );
    }

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| "failed to load model")?;

    let n_embd = model.n_embd() as usize;
    let n_layer = model.n_layer();
    let layer_end = if args.layer_end == 0 {
        n_layer
    } else {
        args.layer_end
    };

    eprintln!("Model: {model}");
    eprintln!("Embedding dim: {n_embd}");
    eprintln!("Layers: {n_layer} (applying to {}..{})", args.layer_start, layer_end);
    eprintln!("Prompt pairs: {}", args.positive.len());

    // Create context with embeddings enabled
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
        .with_embeddings(true);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "failed to create context")?;

    // Compute mean difference vector across prompt pairs
    let mut diff = vec![0.0_f64; n_embd];
    let n_pairs = args.positive.len();

    for (i, (pos, neg)) in args.positive.iter().zip(args.negative.iter()).enumerate() {
        eprint!("  Pair {}/{}: ", i + 1, n_pairs);

        let emb_pos = get_embeddings(&mut ctx, &model, pos)
            .with_context(|| format!("failed on positive prompt: {pos}"))?;
        let emb_neg = get_embeddings(&mut ctx, &model, neg)
            .with_context(|| format!("failed on negative prompt: {neg}"))?;

        for j in 0..n_embd {
            diff[j] += (emb_pos[j] - emb_neg[j]) as f64;
        }

        // Compute cosine similarity between positive and negative
        let dot: f64 = emb_pos
            .iter()
            .zip(emb_neg.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let mag_p: f64 = emb_pos.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let mag_n: f64 = emb_neg.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cossim = dot / (mag_p * mag_n);
        eprintln!("cosine_sim={cossim:.4}");
    }

    // Normalize
    for v in &mut diff {
        *v /= n_pairs as f64;
    }

    // Convert to f32
    let cvector: Vec<f32> = diff.iter().map(|&v| v as f32).collect();

    // Compute magnitude
    let magnitude: f64 = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!("Control vector magnitude: {magnitude:.6}");

    // Save: binary format with header
    let mut out = std::fs::File::create(&args.output)?;
    use std::io::Write;
    // Header: magic, n_embd, layer_start, layer_end
    out.write_all(b"CVEC")?;
    out.write_all(&(n_embd as u32).to_le_bytes())?;
    out.write_all(&(args.layer_start as u32).to_le_bytes())?;
    out.write_all(&(layer_end as u32).to_le_bytes())?;
    // Data: f32 values
    for &v in &cvector {
        out.write_all(&v.to_le_bytes())?;
    }

    eprintln!("Control vector saved to {}", args.output.display());
    eprintln!(
        "Apply with: ctx.set_adapter_cvec(&cvector, {}, {}, {})",
        n_embd, args.layer_start, layer_end
    );

    Ok(())
}
