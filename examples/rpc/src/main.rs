//! Example of using RPC backend for distributed inference
//!
//! This example demonstrates how to use the RPC backend to distribute
//! inference across multiple machines.
//!
//! To run this example:
//! 1. Start an RPC server on a remote machine (or locally for testing)
//! 2. Run this client with the server's endpoint

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{split_id, HFClientSync};
use llama_cpp_4::prelude::*;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::ptr::NonNull;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run as server mode
    #[arg(long)]
    server: bool,

    /// Endpoint for RPC (e.g., "127.0.0.1:50052")
    #[arg(short, long, default_value = "127.0.0.1:50052")]
    endpoint: String,

    /// Index of the remote device to use (client mode). A single endpoint can
    /// expose several devices, in the order the server listed them.
    #[arg(long, default_value_t = 0)]
    device: u32,

    /// Directory for the server-side tensor cache (server mode)
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Worker threads used to service RPC requests (server mode)
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Model to use (HuggingFace repo ID or local path)
    #[arg(short, long)]
    model: Option<String>,

    /// Prompt to generate text from (client mode only)
    #[arg(short, long, default_value = "Once upon a time")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "128")]
    max_tokens: usize,

    /// Use CPU backend for server mode
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize the llama backend
    let backend = LlamaBackend::init()?;

    if args.server {
        run_server(&args)
    } else {
        run_client(&args, &backend)
    }
}

/// Collect the local devices to expose, from ggml's global backend registry.
///
/// `LlamaBackend::init()` in `main` is what populates the registry, so this must
/// run after it.
fn collect_devices(cpu_only: bool) -> Result<Vec<NonNull<llama_cpp_sys_4::ggml_backend_device>>> {
    let mut devices = Vec::new();

    // SAFETY: the registry is process-global and populated by backend init; the
    // device pointers it hands out are owned by ggml and live for the life of
    // the process, so they outlive the `serve` call below.
    unsafe {
        for i in 0..llama_cpp_sys_4::ggml_backend_dev_count() {
            let Some(dev) = NonNull::new(llama_cpp_sys_4::ggml_backend_dev_get(i)) else {
                continue;
            };
            let is_cpu = llama_cpp_sys_4::ggml_backend_dev_type(dev.as_ptr())
                == llama_cpp_sys_4::GGML_BACKEND_DEVICE_TYPE_CPU;
            if cpu_only && !is_cpu {
                continue;
            }
            let name =
                std::ffi::CStr::from_ptr(llama_cpp_sys_4::ggml_backend_dev_name(dev.as_ptr()))
                    .to_string_lossy()
                    .into_owned();
            println!("  device {}: {name}", devices.len());
            devices.push(dev);
        }
    }

    anyhow::ensure!(
        !devices.is_empty(),
        "no backend devices available to serve (try without --cpu)"
    );
    Ok(devices)
}

fn run_server(args: &Args) -> Result<()> {
    println!("Starting RPC server on {}", args.endpoint);

    let devices = collect_devices(args.cpu)?;

    println!(
        "Serving {} device(s) with {} thread(s)",
        devices.len(),
        args.threads
    );
    println!("Press Ctrl+C to stop the server");

    // Blocks: llama.cpp runs the accept loop on this thread and does not return
    // while the server is live.
    serve(
        &args.endpoint,
        args.cache_dir.as_deref(),
        args.threads,
        &devices,
    )?;

    Ok(())
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn run_client(args: &Args, backend: &LlamaBackend) -> Result<()> {
    println!("Connecting to RPC server at {}", args.endpoint);

    // Initialize RPC backend against the requested remote device
    let rpc_backend = RpcBackend::init(&args.endpoint, args.device)?;
    println!("Connected to RPC backend: {rpc_backend:?}");

    // Query device memory
    match rpc_backend.get_device_memory() {
        Ok((free, total)) => {
            println!(
                "Remote device memory: {:.2} GB free / {:.2} GB total",
                free as f64 / 1_073_741_824.0,
                total as f64 / 1_073_741_824.0
            );
        }
        Err(e) => {
            println!("Could not query device memory: {}", e);
        }
    }

    // Load model
    let model_path = if let Some(model) = &args.model {
        if model.contains('/') && !model.contains('.') {
            // Looks like a HuggingFace repo ID
            download_model(model)?
        } else {
            // Local path
            PathBuf::from(model)
        }
    } else {
        // Default model
        download_model("TheBloke/Llama-2-7B-Chat-GGUF")?
    };

    println!("Loading model from {:?}", model_path);

    // Load the model
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
        .context("Failed to load model")?;

    // Create context
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048));

    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("Failed to create context")?;

    // Tokenize prompt
    let tokens_list = model
        .str_to_token(&args.prompt, AddBos::Always)
        .context("Failed to tokenize prompt")?;

    println!("Prompt: {}", args.prompt);
    println!("Generating {} tokens...\n", args.max_tokens);

    // Create batch
    let mut batch = LlamaBatch::new(512, 1);

    // Add tokens to batch
    let last_index = tokens_list.len() - 1;
    for (i, token) in tokens_list.iter().enumerate() {
        let is_last = i == last_index;
        batch.add(*token, i as i32, &[0], is_last)?;
    }

    // Decode the batch
    ctx.decode(&mut batch).context("Failed to decode batch")?;

    // Set up sampler
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    // Generate tokens
    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    while n_decode < args.max_tokens {
        // Sample next token
        let new_token_id = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(new_token_id);

        // Check for EOS
        if model.is_eog_token(new_token_id) {
            println!();
            break;
        }

        // Convert token to string and print
        let token_str = model
            .token_to_str(new_token_id, Special::Tokenize)
            .context("Failed to convert token to string")?;

        print!("{}", token_str);
        std::io::stdout().flush()?;

        // Prepare next batch
        batch.clear();
        batch.add(new_token_id, n_cur, &[0], true)?;

        n_cur += 1;
        n_decode += 1;

        // Decode the batch
        ctx.decode(&mut batch).context("Failed to decode batch")?;
    }

    println!("\n\nGenerated {} tokens", n_decode);

    Ok(())
}

fn download_model(repo: &str) -> Result<PathBuf> {
    println!("Downloading model from HuggingFace: {}", repo);

    let api = HFClientSync::new().context("unable to create huggingface api")?;
    let (owner, name) = split_id(repo);
    let repo = api.model(owner, name);

    // Try to find a GGUF file
    let files = repo.info().send()?;
    let siblings = files.siblings.as_deref().unwrap_or_default();

    // Look for Q4_K_M quantization first, then any GGUF
    let gguf_file = siblings
        .iter()
        .find(|f| f.rfilename.contains("Q4_K_M") && f.rfilename.ends_with(".gguf"))
        .or_else(|| siblings.iter().find(|f| f.rfilename.ends_with(".gguf")))
        .context("No GGUF file found in repository")?;

    println!("Downloading {}", gguf_file.rfilename);
    let path = repo
        .download_file()
        .filename(gguf_file.rfilename.clone())
        .send()?;

    Ok(path)
}
