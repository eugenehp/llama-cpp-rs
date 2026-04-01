//! # TurboQuant demo
//!
//! Demonstrates the **attention-rotation** feature introduced in llama.cpp PR #21038,
//! informally called *TurboQuant*.
//!
//! ## What is TurboQuant?
//!
//! When a model's attention head size is a power of two, llama.cpp applies a
//! Hadamard rotation to the Q, K, and V tensors **before** storing them in the
//! KV cache.  The rotation:
//!
//! * Spreads outlier values evenly across all dimensions.
//! * Dramatically reduces the quantization error of the cached K/V tensors.
//! * Costs almost nothing at runtime (a small matrix multiply per token).
//! * Is automatically reversed when reading back from the cache, so the
//!   computed attention is mathematically identical.
//!
//! The result is that aggressive KV-cache types like `q5_0` become viable —
//! in some cases cutting PPL delta by 10-100× (see table in `--show-api`).
//!
//! The feature is **enabled by default** for all compatible models and requires
//! no changes to the GGUF file.  This example lets you compare outputs with and
//! without the rotation so the difference is visible.
//!
//! ## Usage
//!
//! ```console
//! # Compare rotation-on vs rotation-off with a quantized KV cache:
//! cargo run -p turbo-quant -- \
//!     --model model.gguf \
//!     --prompt "The capital of France is" \
//!     --kv-type q5_0 \
//!     --n-predict 16
//!
//! # Show API reference without loading a model:
//! cargo run -p turbo-quant -- --show-api
//!
//! # Quantize a model first:
//! cargo run -p quantize -- model-f16.gguf Q4_K_M
//! cargo run -p turbo-quant -- --model model-Q4_K_M.gguf --prompt "Hello"
//! ```

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use anyhow::{Context, Result};
use clap::Parser;

use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    quantize::{attn_rot_disabled, GgmlType, LlamaFtype},
    sampling::LlamaSampler,
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    about = "TurboQuant demo – compare attention-rotation on vs off",
    long_about = None
)]
struct Args {
    /// Path to a GGUF model file.
    #[arg(long, short)]
    model: Option<String>,

    /// Prompt to run through both contexts.
    #[arg(long, short, default_value = "The quick brown fox jumps over the lazy")]
    prompt: String,

    /// KV-cache quantization type for the comparison run.
    /// Use "f16" for the baseline, "q4_0" / "q5_0" to stress-test TurboQuant.
    #[arg(long, default_value = "f16")]
    kv_type: String,

    /// Number of tokens to predict.
    #[arg(long, default_value_t = 16)]
    n_predict: usize,

    /// Only print API reference, skip model loading.
    #[arg(long)]
    show_api: bool,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn parse_kv_type(s: &str) -> Option<GgmlType> {
    Some(match s.to_lowercase().as_str() {
        "f32"            => GgmlType::F32,
        "f16"            => GgmlType::F16,
        "bf16"           => GgmlType::BF16,
        "q4_0" | "q4_k" => GgmlType::Q4_0,
        "q4_1"           => GgmlType::Q4_1,
        "q5_0" | "q5_k" => GgmlType::Q5_0,
        "q5_1"           => GgmlType::Q5_1,
        "q8_0" | "q8_k" => GgmlType::Q8_0,
        _ => return None,
    })
}

// ─── inference pass ──────────────────────────────────────────────────────────

fn run_inference(
    model: &LlamaModel,
    backend: &LlamaBackend,
    ctx_params: LlamaContextParams,
    prompt: &str,
    n_predict: usize,
    label: &str,
) -> Result<String> {
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("failed to create context")?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .context("tokenize failed")?;

    let mut batch = LlamaBatch::new(512, 1);
    let last = tokens.len() - 1;
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == last)
            .context("batch add failed")?;
    }
    ctx.decode(&mut batch).context("decode failed")?;

    let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut output = String::new();
    let mut pos = tokens.len() as i32;

    for _ in 0..n_predict {
        let token = sampler.sample(&ctx, -1);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_str(token, Special::Tokenize)
            .unwrap_or_default();
        output.push_str(&piece);

        batch.clear();
        batch
            .add(token, pos, &[0], true)
            .context("batch add failed")?;
        ctx.decode(&mut batch).context("decode failed")?;
        pos += 1;
    }

    println!("[{label}] → {output}");
    Ok(output)
}

// ─── API reference printout ───────────────────────────────────────────────────

fn show_api_examples() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  TurboQuant API reference  (llama-cpp-4 / llama.cpp PR #21038)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    println!("── 1. Quantize a model ─────────────────────────────────────────────");
    println!();
    println!("  use llama_cpp_4::quantize::{{LlamaFtype, QuantizeParams,");
    println!("                               TensorTypeOverride, GgmlType}};");
    println!();
    println!("  // Quantize to Q4_K_M, keep output/lm-head tensor in F16:");
    println!("  let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM)");
    println!("      .with_nthread(8)");
    println!("      .with_quantize_output_tensor(true)");
    println!("      .with_tensor_type_override(");
    println!("          TensorTypeOverride::new(\"output\", GgmlType::F16).unwrap()");
    println!("      );");
    println!();
    println!("  llama_cpp_4::model_quantize(\"model-f16.gguf\", \"model-q4km.gguf\", &params).unwrap();");
    println!();

    println!("── 2. Inference with TurboQuant (ON by default) ────────────────────");
    println!();
    println!("  use llama_cpp_4::context::params::LlamaContextParams;");
    println!("  use llama_cpp_4::quantize::GgmlType;");
    println!();
    println!("  // TurboQuant is enabled automatically for compatible models.");
    println!("  // Pair it with a quantized KV cache for maximum memory savings:");
    println!("  let ctx_params = LlamaContextParams::default()");
    println!("      .with_cache_type_k(GgmlType::Q5_0)   // saves ~37% VRAM vs F16");
    println!("      .with_cache_type_v(GgmlType::Q5_0);  // quality ~= F16 with rotation");
    println!();
    println!("  let ctx = model.new_context(&backend, ctx_params)?;");
    println!();

    println!("── 3. Disable TurboQuant for one context ───────────────────────────");
    println!();
    println!("  // E.g. to establish a baseline for benchmarking:");
    println!("  let ctx_params = LlamaContextParams::default()");
    println!("      .with_cache_type_k(GgmlType::Q5_0)");
    println!("      .with_attn_rot_disabled(true);   // ← TurboQuant OFF");
    println!();
    println!("  let ctx = model.new_context(&backend, ctx_params)?;");
    println!();

    println!("── 4. Global process-level toggle ──────────────────────────────────");
    println!();
    println!("  use llama_cpp_4::quantize::{{set_attn_rot_disabled, attn_rot_disabled}};");
    println!();
    println!("  set_attn_rot_disabled(true);");
    println!("  assert!(attn_rot_disabled());  // true");
    println!();
    println!("  set_attn_rot_disabled(false);  // re-enable (default)");
    println!("  assert!(!attn_rot_disabled()); // false");
    println!();
    println!("  // NOTE: call this BEFORE creating any LlamaContext.");
    println!("  // Mutating environment variables is not thread-safe.");
    println!();

    println!("── 5. KV quality table (Qwen3 0.6B BF16, PPL Δ vs F16 baseline) ───");
    println!();
    println!("  KV type │ without TurboQuant │ with TurboQuant");
    println!("  ────────┼────────────────────┼────────────────");
    println!("  q5_1    │ +61.70 PPL         │ +0.44 PPL");
    println!("  q5_0    │ +17.28 PPL         │ +0.55 PPL");
    println!("  q4_1    │ +212.5 PPL         │ +8.65 PPL");
    println!("  q4_0    │ +62.02 PPL         │ +32.6 PPL");
    println!();
    println!("  Source: llama.cpp PR #21038  https://github.com/ggml-org/llama.cpp/pull/21038");
    println!();

    println!("── 6. Available model quantization types ───────────────────────────");
    println!();
    for ftype in LlamaFtype::all() {
        println!("  {:<12}  {}", ftype.name(), ftype.description());
    }
    println!();
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    if args.show_api || args.model.is_none() {
        show_api_examples();
        if args.model.is_none() && !args.show_api {
            eprintln!(
                "\nNo --model provided.\n\
                 Run with --show-api to see the API reference, or\n\
                 --model <path.gguf> to run the rotation comparison."
            );
        }
        return Ok(());
    }

    let model_path = args.model.unwrap();

    // ── Report current TurboQuant state ──────────────────────────────────
    println!(
        "TurboQuant attention rotation : {}",
        if attn_rot_disabled() {
            "DISABLED (LLAMA_ATTN_ROT_DISABLE is set in env)"
        } else {
            "enabled (default — auto for power-of-two head dims)"
        }
    );

    let kv_type = parse_kv_type(&args.kv_type).unwrap_or_else(|| {
        eprintln!(
            "Unknown KV type '{}', falling back to F16",
            args.kv_type
        );
        GgmlType::F16
    });

    println!("KV cache type                 : {}", args.kv_type.to_uppercase());
    println!("Prompt                        : {:?}", args.prompt);
    println!("Predict tokens                : {}", args.n_predict);
    println!();
    println!("(Use --kv-type q5_0 or q4_0 to make the quality difference most visible)");
    println!();

    // ── Load backend + model ──────────────────────────────────────────────
    let backend = LlamaBackend::init().context("failed to init backend")?;
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default())
        .context("failed to load model")?;

    // Shared context param builder (n_ctx + kv types).
    let base_params = || {
        LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_cache_type_k(kv_type)
            .with_cache_type_v(kv_type)
    };

    // ── Pass 1 — TurboQuant ON (default) ─────────────────────────────────
    println!("══ Pass 1 : TurboQuant ON (attn rotation enabled) ══════════════");
    let out_on = run_inference(
        &model,
        &backend,
        base_params().with_attn_rot_disabled(false),
        &args.prompt,
        args.n_predict,
        "rotation ON ",
    )?;

    println!();

    // ── Pass 2 — TurboQuant OFF ───────────────────────────────────────────
    println!("══ Pass 2 : TurboQuant OFF (attn rotation disabled) ════════════");
    let out_off = run_inference(
        &model,
        &backend,
        base_params().with_attn_rot_disabled(true),
        &args.prompt,
        args.n_predict,
        "rotation OFF",
    )?;

    println!();

    // ── Summary ───────────────────────────────────────────────────────────
    if out_on == out_off {
        println!("✓  Outputs are identical (expected for lossless KV types like F16/BF16).");
    } else {
        println!("⚠  Outputs differ — this shows the quality impact of KV quantization.");
        println!("   With TurboQuant ON the rotation-enabled output is typically more coherent.");
    }
    println!();
    println!("💡 Tip: re-run with --kv-type q4_0 for the most dramatic difference.");

    Ok(())
}
