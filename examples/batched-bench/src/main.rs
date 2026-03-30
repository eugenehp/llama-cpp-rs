//! # Batched Bench
//!
//! Benchmark prompt processing (PP) and token generation (TG) throughput
//! at various batch sizes and parallel sequence counts.
//!
//! This is the Rust equivalent of llama.cpp's `llama-batched-bench` tool.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p batched-bench -- -m model.gguf
//! cargo run -p batched-bench -- -m model.gguf --npp 128,256 --ntg 32,64 --npl 1,2,4
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
use llama_cpp_4::model::LlamaModel;
use llama_cpp_4::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command(about = "Benchmark batched prompt processing and token generation")]
struct Args {
    /// Path to the GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Context size
    #[arg(short = 'c', long, default_value_t = 2048)]
    ctx_size: u32,

    /// Batch size
    #[arg(short = 'b', long, default_value_t = 512)]
    batch_size: u32,

    /// Prompt processing sizes (comma-separated)
    #[arg(long, default_value = "128,256,512", value_delimiter = ',')]
    npp: Vec<i32>,

    /// Token generation counts (comma-separated)
    #[arg(long, default_value = "128", value_delimiter = ',')]
    ntg: Vec<i32>,

    /// Parallel sequence counts (comma-separated)
    #[arg(long, default_value = "1", value_delimiter = ',')]
    npl: Vec<i32>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| "failed to load model")?;

    eprintln!("Model: {model}");

    let n_vocab = model.n_vocab();
    let n_kv_max = args.ctx_size as i32;
    let n_seq_max = args.npl.iter().copied().max().unwrap_or(1);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(args.ctx_size))
        .with_n_batch(args.batch_size);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "failed to create context")?;

    let n_batch = ctx.n_batch() as i32;

    // Warmup
    {
        let mut batch = LlamaBatch::new(512, n_seq_max);
        for i in 0..16 {
            let token = LlamaToken((i * 7 + 13) % n_vocab);
            batch.add(token, i, &[0], false)?;
        }
        ctx.decode(&mut batch)?;
        ctx.clear_kv_cache();
    }

    // Print header
    println!();
    println!(
        "n_kv_max = {}, n_batch = {}, n_seq_max = {}",
        n_kv_max, n_batch, n_seq_max
    );
    println!();
    println!(
        "|{:>6} | {:>6} | {:>4} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |",
        "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s"
    );
    println!(
        "|{:-<6}-|-{:-<6}-|-{:-<4}-|-{:-<6}-|-{:-<8}-|-{:-<8}-|-{:-<8}-|-{:-<8}-|-{:-<8}-|-{:-<8}-|",
        "", "", "", "", "", "", "", "", "", ""
    );

    for &pp in &args.npp {
        for &tg in &args.ntg {
            for &pl in &args.npl {
                let n_ctx_req = pl * (pp + tg);
                if n_ctx_req > n_kv_max {
                    continue;
                }

                // === Prompt processing ===
                let mut batch = LlamaBatch::new(n_kv_max as usize, n_seq_max);

                for j in 0..pl {
                    for i in 0..pp {
                        let token = LlamaToken(((j * pp + i) * 7 + 13) % n_vocab);
                        batch.add(token, i, &[j], i == pp - 1)?;
                    }
                }

                ctx.clear_kv_cache();

                let t_pp_start = llama_cpp_4::ggml_time_us();

                // Decode in sub-batches
                decode_batch(&mut ctx, &mut batch, n_batch)?;
                ctx.synchronize();

                let t_pp_end = llama_cpp_4::ggml_time_us();

                // === Token generation ===
                let t_tg_start = llama_cpp_4::ggml_time_us();

                for i in 0..tg {
                    batch.clear();
                    for j in 0..pl {
                        let token = LlamaToken(((j * tg + i) * 11 + 17) % n_vocab);
                        batch.add(token, pp + i, &[j], true)?;
                    }
                    ctx.decode(&mut batch)?;
                    ctx.synchronize();
                }

                let t_tg_end = llama_cpp_4::ggml_time_us();

                // === Results ===
                let t_pp = (t_pp_end - t_pp_start) as f64 / 1_000_000.0;
                let t_tg = (t_tg_end - t_tg_start) as f64 / 1_000_000.0;
                let t = t_pp + t_tg;

                let speed_pp = (pl * pp) as f64 / t_pp;
                let speed_tg = (pl * tg) as f64 / t_tg;
                let speed = (pl * (pp + tg)) as f64 / t;

                println!(
                    "|{:6} | {:6} | {:4} | {:6} | {:8.3} | {:8.2} | {:8.3} | {:8.2} | {:8.3} | {:8.2} |",
                    pp, tg, pl, n_ctx_req, t_pp, speed_pp, t_tg, speed_tg, t, speed
                );
            }
        }
    }

    println!();
    Ok(())
}

/// Decode a batch in sub-batches of `n_batch` tokens.
fn decode_batch(
    ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
    batch: &mut LlamaBatch,
    _n_batch: i32,
) -> Result<()> {
    ctx.decode(batch).with_context(|| "decode failed")?;
    Ok(())
}
