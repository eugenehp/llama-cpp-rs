//! # Fit Params
//!
//! Auto-fit model and context parameters (n_gpu_layers, n_ctx) to available memory.
//! This is the Rust equivalent of llama.cpp's `llama-fit-params` tool.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p fit-params -- -m model.gguf
//! cargo run -p fit-params -- -m model.gguf --min-ctx 1024
//! ```

use anyhow::{bail, Result};
use clap::Parser;
use llama_cpp_4::llama_backend::LlamaBackend;
use std::ffi::CString;
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

    let _backend = LlamaBackend::init()?;

    let c_path = CString::new(
        args.model
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid path"))?,
    )?;

    let mut mparams = unsafe { llama_cpp_sys_4::llama_model_default_params() };
    let mut cparams = unsafe { llama_cpp_sys_4::llama_context_default_params() };

    let nd = llama_cpp_4::max_devices();
    let mut tensor_split = vec![0.0_f32; nd];

    let ntbo = llama_cpp_4::max_tensor_buft_overrides();
    let mut tensor_buft_overrides =
        vec![
            llama_cpp_sys_4::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            };
            ntbo + 1
        ];

    let mut margins = vec![0_usize; nd + 1];

    let status = unsafe {
        llama_cpp_4::params_fit(
            c_path.as_ptr(),
            &mut mparams,
            &mut cparams,
            tensor_split.as_mut_ptr(),
            tensor_buft_overrides.as_mut_ptr(),
            margins.as_mut_ptr(),
            args.min_ctx,
            llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
        )
    };

    if status != llama_cpp_sys_4::LLAMA_PARAMS_FIT_STATUS_SUCCESS {
        bail!("params_fit failed (status={status})");
    }

    // Print fitted parameters
    print!("-c {} -ngl {}", cparams.n_ctx, mparams.n_gpu_layers);

    // Print tensor split if multi-device
    let mut nd_active = nd;
    while nd_active > 1 && tensor_split[nd_active - 1] == 0.0 {
        nd_active -= 1;
    }
    if nd_active > 1 {
        print!(" -ts ");
        for (i, &split) in tensor_split[..nd_active].iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{}", split as u32);
        }
    }

    println!();

    Ok(())
}
