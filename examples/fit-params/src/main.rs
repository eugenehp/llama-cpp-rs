//! # Fit Params
//!
//! Auto-fit model and context parameters (`n_gpu_layers`, `n_ctx`) to available memory.
//! This is the Rust equivalent of llama.cpp's `llama-fit-params` tool.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p fit-params -- -m model.gguf
//! cargo run -p fit-params -- -m model.gguf --min-ctx 1024
//! ```

use anyhow::Result;
use clap::Parser;
use llama_cpp_4::prelude::*;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command(about = "Auto-fit model parameters to available memory")]
struct Args {
    /// Path to the GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Minimum context size
    #[arg(long, default_value_t = 512)]
    min_ctx: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let backend = LlamaBackend::init()?;

    let result = fit_params(
        &backend,
        &args.model,
        FitParams::default().with_n_ctx_min(args.min_ctx),
    )
    .map_err(|e| match e {
        FitParamsError::InvalidPath => anyhow::anyhow!("invalid model path"),
        FitParamsError::CouldNotFit => anyhow::anyhow!("could not fit parameters to device memory"),
        FitParamsError::Failed => anyhow::anyhow!("parameter fitting failed"),
    })?;

    let n_ctx = result.context_params.n_ctx().map_or(0, NonZeroU32::get);
    print!("-c {n_ctx} -ngl {}", result.model_params.n_gpu_layers());

    let splits = result.active_tensor_split();
    if splits.len() > 1 {
        print!(" -ts ");
        for (i, &split) in splits.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{:.0}", split.max(0.0));
        }
    }

    println!();

    Ok(())
}
