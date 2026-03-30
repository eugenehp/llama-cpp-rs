//! # Quantize
//!
//! Quantize a GGUF model to a smaller precision.
//! This is the Rust equivalent of llama.cpp's `llama-quantize` tool.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p quantize -- model-f16.gguf model-q4_k_m.gguf Q4_K_M
//! cargo run -p quantize -- model-f16.gguf Q4_K_M              # auto output name
//! cargo run -p quantize -- --dry-run model.gguf Q4_K_M        # show size without quantizing
//! cargo run -p quantize -- --allow-requantize model-q8.gguf Q4_K_M
//! ```
#![allow(clippy::cast_precision_loss)]

use anyhow::{bail, Result};
use clap::Parser;
use llama_cpp_4::llama_backend::LlamaBackend;

/// Known quantization types with human-friendly descriptions.
const QUANT_TYPES: &[(&str, u32, &str)] = &[
    ("Q4_0",      2,  " 4.34G, +0.4685 ppl @ Llama-3-8B"),
    ("Q4_1",      3,  " 4.78G, +0.4511 ppl @ Llama-3-8B"),
    ("Q5_0",      8,  " 5.21G, +0.1316 ppl @ Llama-3-8B"),
    ("Q5_1",      9,  " 5.65G, +0.1062 ppl @ Llama-3-8B"),
    ("Q8_0",      7,  " 7.96G, +0.0026 ppl @ Llama-3-8B"),
    ("Q2_K",      10, " 2.96G, +3.5199 ppl @ Llama-3-8B"),
    ("Q2_K_S",    21, " 2.96G, +3.1836 ppl @ Llama-3-8B"),
    ("Q3_K_S",    11, " 3.41G, +1.6321 ppl @ Llama-3-8B"),
    ("Q3_K_M",    12, " 3.74G, +0.6569 ppl @ Llama-3-8B"),
    ("Q3_K_L",    13, " 4.03G, +0.5562 ppl @ Llama-3-8B"),
    ("Q4_K_S",    14, " 4.37G, +0.2689 ppl @ Llama-3-8B"),
    ("Q4_K_M",    15, " 4.58G, +0.1754 ppl @ Llama-3-8B"),
    ("Q5_K_S",    16, " 5.21G, +0.1049 ppl @ Llama-3-8B"),
    ("Q5_K_M",    17, " 5.33G, +0.0569 ppl @ Llama-3-8B"),
    ("Q6_K",      18, " 6.14G, +0.0217 ppl @ Llama-3-8B"),
    ("IQ1_S",     24, " 1.56 bpw quantization"),
    ("IQ1_M",     31, " 1.75 bpw quantization"),
    ("IQ2_XXS",   19, " 2.06 bpw quantization"),
    ("IQ2_XS",    20, " 2.31 bpw quantization"),
    ("IQ2_S",     28, " 2.5  bpw quantization"),
    ("IQ2_M",     29, " 2.7  bpw quantization"),
    ("IQ3_XXS",   23, " 3.06 bpw quantization"),
    ("IQ3_XS",    22, " 3.3  bpw quantization"),
    ("IQ3_S",     26, " 3.44 bpw quantization"),
    ("IQ3_M",     27, " 3.66 bpw quantization"),
    ("IQ4_NL",    25, " 4.50 bpw non-linear quantization"),
    ("IQ4_XS",    30, " 4.25 bpw non-linear quantization"),
    ("TQ1_0",     36, " 1.69 bpw ternarization"),
    ("TQ2_0",     37, " 2.06 bpw ternarization"),
    ("F16",       1,  "14.00G, +0.0020 ppl @ Mistral-7B"),
    ("BF16",      32, "14.00G, -0.0050 ppl @ Mistral-7B"),
    ("F32",       0,  "26.00G              @ 7B"),
    ("COPY",      0,  "only copy tensors, no quantizing"),
];

fn parse_ftype(name: &str) -> Option<(u32, &str)> {
    let upper = name.to_uppercase();
    QUANT_TYPES
        .iter()
        .find(|(n, _, _)| *n == upper)
        .map(|(n, ftype, _)| (*ftype, *n))
}

fn print_quant_types() {
    eprintln!("Available quantization types:");
    eprintln!();
    for (name, ftype, desc) in QUANT_TYPES {
        if *name != "COPY" {
            eprintln!("  {:2}  or  {:<7} :{}", ftype, name, desc);
        } else {
            eprintln!("        {:<7}  :{}", name, desc);
        }
    }
}

#[derive(clap::Parser, Debug)]
#[command(about = "Quantize a GGUF model to a smaller precision")]
struct Args {
    /// Input model file (F16 or F32 GGUF)
    input: String,

    /// Output file or quantization type.
    /// If this looks like a quant type (e.g. Q4_K_M), the output filename is auto-generated.
    /// Otherwise treated as the output path, and the next argument must be the quant type.
    output_or_type: String,

    /// Quantization type (if output path was given as second arg)
    quant_type: Option<String>,

    /// Number of threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    nthreads: i32,

    /// Allow requantizing already-quantized models
    #[arg(long)]
    allow_requantize: bool,

    /// Do not quantize the output tensor
    #[arg(long)]
    leave_output_tensor: bool,

    /// Quantize all tensors to the same type (disable k-quant mixtures)
    #[arg(long)]
    pure: bool,

    /// Only calculate size, do not write output
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Parse quantization type and output path
    let (fname_out, ftype, ftype_name) = if let Some((ftype, name)) = parse_ftype(&args.output_or_type) {
        // Second arg is the quant type: <input> <type>
        let stem = args.input.trim_end_matches(".gguf");
        let out = format!("{stem}-{}.gguf", name.to_lowercase());
        (out, ftype, name.to_string())
    } else if let Some(ref qt) = args.quant_type {
        // Second arg is output path: <input> <output> <type>
        let (ftype, name) = parse_ftype(qt).ok_or_else(|| {
            print_quant_types();
            anyhow::anyhow!("unknown quantization type: {qt}")
        })?;
        (args.output_or_type.clone(), ftype, name.to_string())
    } else {
        print_quant_types();
        bail!(
            "'{}' is not a recognized quantization type. Specify: <input> [output] <type>",
            args.output_or_type
        );
    };

    if !args.dry_run && args.input == fname_out {
        bail!("input and output files are the same: {}", args.input);
    }

    let _backend = LlamaBackend::init()?;

    let mut params = llama_cpp_4::model_quantize_default_params();
    params.ftype = ftype;
    params.nthread = args.nthreads;
    params.allow_requantize = args.allow_requantize;
    params.quantize_output_tensor = !args.leave_output_tensor;
    params.pure_ = args.pure;
    params.dry_run = args.dry_run;

    if args.dry_run {
        eprintln!("Calculating quantization size for '{}' as {}", args.input, ftype_name);
    } else {
        eprintln!(
            "Quantizing '{}' to '{}' as {}",
            args.input, fname_out, ftype_name
        );
    }
    if args.nthreads > 0 {
        eprintln!("Using {} threads", args.nthreads);
    }

    let t_start = llama_cpp_4::llama_time_us();

    let result = llama_cpp_4::model_quantize(&args.input, &fname_out, Some(&params));

    let t_end = llama_cpp_4::llama_time_us();
    let t_ms = (t_end - t_start) as f64 / 1000.0;

    if result != 0 {
        bail!("quantization failed with error code {result}");
    }

    println!();
    println!("Quantize time: {:.2} ms", t_ms);
    println!("Total time:    {:.2} ms", t_ms);

    if !args.dry_run {
        println!("Output: {fname_out}");
    }

    Ok(())
}
