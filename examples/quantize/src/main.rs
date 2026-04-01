//! # Quantize
//!
//! Quantize a GGUF model to a smaller precision using the typed Rust API.
//! This is the Rust equivalent of llama.cpp's `llama-quantize` tool.
//!
//! ## Usage
//!
//! ```console
//! # basic
//! cargo run -p quantize -- model-f16.gguf model-q4_k_m.gguf Q4_K_M
//! cargo run -p quantize -- model-f16.gguf Q4_K_M              # auto output name
//!
//! # dry-run: show size without writing
//! cargo run -p quantize -- --dry-run model.gguf Q4_K_M
//!
//! # re-quantize an already-quantized model
//! cargo run -p quantize -- --allow-requantize model-q8.gguf Q4_K_M
//!
//! # keep output tensor in F16, everything else Q4_K_M
//! cargo run -p quantize -- --tensor-type output=F16 model-f16.gguf Q4_K_M
//!
//! # prune layers 0 and 1
//! cargo run -p quantize -- --prune-layer 0 --prune-layer 1 model-f16.gguf Q4_K_M
//!
//! # list all available quant types
//! cargo run -p quantize -- --list-types
//! ```

#![allow(clippy::cast_precision_loss)]

use anyhow::{bail, Result};
use clap::Parser;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::quantize::{
    GgmlType, LlamaFtype, QuantizeParams, TensorTypeOverride,
    attn_rot_disabled, set_attn_rot_disabled,
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(clap::Parser, Debug)]
#[command(about = "Quantize a GGUF model to a smaller precision")]
struct Args {
    /// Input model file (F16 or F32 GGUF)
    #[arg(required_unless_present = "list_types")]
    input: Option<String>,

    /// Output file or quantization type.
    /// If this looks like a quant type (e.g. Q4_K_M), the output filename is
    /// auto-generated.  Otherwise treated as the output path, and the next
    /// argument must be the quant type.
    #[arg(required_unless_present = "list_types")]
    output_or_type: Option<String>,

    /// Quantization type (when output path was given as the second arg)
    quant_type: Option<String>,

    /// Number of threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    nthreads: i32,

    /// Allow re-quantizing already-quantized tensors
    #[arg(long)]
    allow_requantize: bool,

    /// Do NOT quantize the output (lm-head) tensor
    #[arg(long)]
    leave_output_tensor: bool,

    /// Quantize every tensor to the same type (no k-quant mixing)
    #[arg(long)]
    pure: bool,

    /// Only calculate output size, do not write the model
    #[arg(long)]
    dry_run: bool,

    /// Keep the same number of shards as the input split model
    #[arg(long)]
    keep_split: bool,

    /// Force storage type for a specific tensor pattern, e.g. `output=F16`
    /// or `blk.0.*=Q8_0`.  Can be repeated.
    #[arg(long = "tensor-type", value_name = "PATTERN=TYPE")]
    tensor_type: Vec<String>,

    /// Layer index to prune from the output model.  Can be repeated.
    #[arg(long = "prune-layer", value_name = "N")]
    prune_layer: Vec<i32>,

    /// Disable TurboQuant attention rotation (enabled by default for
    /// compatible models, see llama.cpp PR #21038).
    #[arg(long)]
    disable_attn_rot: bool,

    /// List all available quantization types and exit.
    #[arg(long)]
    list_types: bool,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn print_quant_types() {
    eprintln!("Available quantization types:");
    eprintln!();
    for ftype in LlamaFtype::all() {
        eprintln!("  {:<12} — {}", ftype.name(), ftype.description());
    }
    eprintln!();
    eprintln!("GgmlType values for --tensor-type:");
    eprintln!("  F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,");
    eprintln!("  Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, IQ1S, IQ1M,");
    eprintln!("  IQ2XXS, IQ2XS, IQ2S, IQ3XXS, IQ3S,");
    eprintln!("  IQ4NL, IQ4XS, TQ1_0, TQ2_0, MXFP4, NVFP4");
}

fn parse_tensor_type_override(spec: &str) -> Result<TensorTypeOverride> {
    let (pattern, type_str) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("--tensor-type must be PATTERN=TYPE, got: {spec}"))?;

    let ty = parse_ggml_type(type_str).ok_or_else(|| {
        anyhow::anyhow!("unknown GgmlType '{type_str}' in --tensor-type {spec}")
    })?;

    TensorTypeOverride::new(pattern, ty)
        .map_err(|e| anyhow::anyhow!("invalid pattern in --tensor-type: {e}"))
}

fn parse_ggml_type(s: &str) -> Option<GgmlType> {
    match s.to_uppercase().as_str() {
        "F32" => Some(GgmlType::F32),
        "F16" => Some(GgmlType::F16),
        "BF16" => Some(GgmlType::BF16),
        "Q4_0" => Some(GgmlType::Q4_0),
        "Q4_1" => Some(GgmlType::Q4_1),
        "Q5_0" => Some(GgmlType::Q5_0),
        "Q5_1" => Some(GgmlType::Q5_1),
        "Q8_0" => Some(GgmlType::Q8_0),
        "Q8_1" => Some(GgmlType::Q8_1),
        "Q2K" | "Q2_K" => Some(GgmlType::Q2K),
        "Q3K" | "Q3_K" => Some(GgmlType::Q3K),
        "Q4K" | "Q4_K" => Some(GgmlType::Q4K),
        "Q5K" | "Q5_K" => Some(GgmlType::Q5K),
        "Q6K" | "Q6_K" => Some(GgmlType::Q6K),
        "Q8K" | "Q8_K" => Some(GgmlType::Q8K),
        "IQ1S" | "IQ1_S" => Some(GgmlType::IQ1S),
        "IQ1M" | "IQ1_M" => Some(GgmlType::IQ1M),
        "IQ2XXS" | "IQ2_XXS" => Some(GgmlType::IQ2XXS),
        "IQ2XS" | "IQ2_XS" => Some(GgmlType::IQ2XS),
        "IQ2S" | "IQ2_S" => Some(GgmlType::IQ2S),
        "IQ2M" | "IQ2_M" => None, // llama_ftype only, not a raw ggml tensor type
        "IQ3XXS" | "IQ3_XXS" => Some(GgmlType::IQ3XXS),
        "IQ3XS" | "IQ3_XS" => None, // llama_ftype only
        "IQ3S" | "IQ3_S" => Some(GgmlType::IQ3S),
        "IQ3M" | "IQ3_M" => None, // llama_ftype only
        "IQ4NL" | "IQ4_NL" => Some(GgmlType::IQ4NL),
        "IQ4XS" | "IQ4_XS" => Some(GgmlType::IQ4XS),
        "TQ1_0" | "TQ1" => Some(GgmlType::TQ1_0),
        "TQ2_0" | "TQ2" => Some(GgmlType::TQ2_0),
        "MXFP4" => Some(GgmlType::MXFP4),
        "NVFP4" => Some(GgmlType::NVFP4),
        _ => None,
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_types {
        print_quant_types();
        return Ok(());
    }

    // Unwrap required args (guaranteed by required_unless_present).
    let input = args.input.unwrap();
    let output_or_type = args.output_or_type.unwrap();

    // Resolve (output_path, ftype, ftype_name).
    let (fname_out, ftype) =
        if let Some(ftype) = LlamaFtype::from_name(&output_or_type) {
            // <input> <type>  — auto-derive output filename
            let stem = input.strip_suffix(".gguf").unwrap_or(&input);
            let out = format!("{stem}-{}.gguf", ftype.name().to_lowercase());
            (out, ftype)
        } else if let Some(ref qt) = args.quant_type {
            // <input> <output> <type>
            let ftype = LlamaFtype::from_name(qt).ok_or_else(|| {
                print_quant_types();
                anyhow::anyhow!("unknown quantization type: {qt}")
            })?;
            (output_or_type, ftype)
        } else {
            print_quant_types();
            bail!(
                "'{output_or_type}' is not a recognized quantization type.\n\
                 Usage: quantize [options] <input> [output] <type>"
            );
        };

    if !args.dry_run && input == fname_out {
        bail!("input and output files are the same: {input}");
    }

    // ── Build QuantizeParams ──────────────────────────────────────────────

    let mut params = QuantizeParams::new(ftype)
        .with_nthread(args.nthreads)
        .with_allow_requantize(args.allow_requantize)
        .with_quantize_output_tensor(!args.leave_output_tensor)
        .with_pure(args.pure)
        .with_dry_run(args.dry_run)
        .with_keep_split(args.keep_split)
        .with_pruned_layers(args.prune_layer);

    for spec in &args.tensor_type {
        params = params.with_tensor_type_override(parse_tensor_type_override(spec)?);
    }

    // ── TurboQuant ────────────────────────────────────────────────────────
    if args.disable_attn_rot {
        set_attn_rot_disabled(true);
    }
    eprintln!(
        "TurboQuant attention rotation: {}",
        if attn_rot_disabled() { "DISABLED" } else { "enabled (default)" }
    );

    // ── Run ───────────────────────────────────────────────────────────────

    let _backend = LlamaBackend::init()?;

    if args.dry_run {
        eprintln!("Calculating quantization size for '{input}' as {ftype}");
    } else {
        eprintln!("Quantizing '{input}' → '{fname_out}' as {ftype}  ({})", ftype.description());
    }
    if args.nthreads > 0 {
        eprintln!("Using {} threads", args.nthreads);
    }

    let t_start = llama_cpp_4::llama_time_us();
    llama_cpp_4::model_quantize(&input, &fname_out, &params)
        .map_err(|code| anyhow::anyhow!("quantization failed with error code {code}"))?;
    let t_ms = (llama_cpp_4::llama_time_us() - t_start) as f64 / 1000.0;

    println!("\nQuantize time: {t_ms:.2} ms");
    if !args.dry_run {
        println!("Output: {fname_out}");
    }

    Ok(())
}
