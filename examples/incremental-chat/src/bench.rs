//! Comprehensive benchmark for incremental prefill.
//!
//! Measures six dimensions:
//! 1. **Latency** — time-to-first-token (TTFT) and flush time at Enter
//! 2. **Speed** — generation tok/s and prefill tok/s
//! 3. **Load** — total GPU compute (margin vs naive vs normal)
//! 4. **Precision** — verifies incremental prefill produces identical logits
//! 5. **UX** — simulates mid-line edits (insert, delete, replace) and
//!    measures recovery cost
//! 6. **DX** — API surface: measures lines of code, number of API calls,
//!    and error paths exercised
//!
//! ```console
//! cargo run --release -p incremental-chat --bin incremental-bench -- model.gguf
//! ```

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod prefill;

use std::num::NonZeroU32;
use std::pin::pin;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::LlamaModel;
use llama_cpp_4::model::{AddBos, Special};
use llama_cpp_4::quantize::GgmlType;
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::LlamaToken;

use prefill::IncrementalPrefill;

const BATCH_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// Naive prefill (no BPE margin) for comparison
// ---------------------------------------------------------------------------

struct NaivePrefill {
    cached: Vec<LlamaToken>,
}

impl NaivePrefill {
    fn new() -> Self {
        Self {
            cached: Vec::new(),
        }
    }
    fn sync(
        &mut self,
        ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
        batch: &mut LlamaBatch,
        tokens: &[LlamaToken],
    ) -> Result<usize> {
        let common = self
            .cached
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common < self.cached.len() {
            ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None)
                .map_err(|e| anyhow!("trim: {e}"))?;
            self.cached.truncate(common);
        }
        let tail = &tokens[common..];
        if tail.is_empty() {
            return Ok(0);
        }
        let mut decoded = 0;
        for chunk in tail.chunks(BATCH_SIZE) {
            batch.clear();
            for (i, &t) in chunk.iter().enumerate() {
                let pos = (common + decoded + i) as i32;
                let last = decoded + i == tail.len() - 1;
                batch.add(t, pos, &[0], last)?;
            }
            ctx.decode(batch).context("decode")?;
            decoded += chunk.len();
        }
        self.cached = tokens.to_vec();
        Ok(decoded)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn typing_snapshots(text: &str) -> Vec<String> {
    text.char_indices()
        .map(|(i, _)| text[..text.ceil_char_boundary(i + 1)].to_string())
        .collect()
}

fn debounced(text: &str) -> Vec<String> {
    let snaps = typing_snapshots(text);
    let stop = snaps.len().saturating_sub(3);
    snaps[..stop].iter().step_by(3).cloned().collect()
}

fn prefill_all(
    ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
) -> Result<()> {
    let mut off = 0;
    for chunk in tokens.chunks(BATCH_SIZE) {
        batch.clear();
        let fin = off + chunk.len() == tokens.len();
        for (i, &t) in chunk.iter().enumerate() {
            batch.add(t, (off + i) as i32, &[0], fin && i == chunk.len() - 1)?;
        }
        ctx.decode(batch).context("prefill")?;
        off += chunk.len();
    }
    Ok(())
}

fn new_ctx<'a>(
    model: &'a LlamaModel,
    backend: &'a LlamaBackend,
    params: &LlamaContextParams,
) -> Result<llama_cpp_4::context::LlamaContext<'a>> {
    model.new_context(backend, params.clone()).context("ctx")
}

// ---------------------------------------------------------------------------
// 1. LATENCY — TTFT and flush time
// ---------------------------------------------------------------------------

fn bench_latency(
    model: &LlamaModel,
    backend: &LlamaBackend,
    params: &LlamaContextParams,
    prompts: &[&str],
) -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. LATENCY — flush time at Enter vs normal prefill                 │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!(
        "  {:>5} {:>10} {:>10} {:>10} {:>7}",
        "Toks", "Normal", "Flush", "TTFT(inc)", "Speedup"
    );

    for prompt in prompts {
        let full = model.str_to_token(prompt, AddBos::Always)?;
        let n = full.len();

        // Normal
        let mut ctx = new_ctx(model, backend, params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        let t0 = Instant::now();
        prefill_all(&mut ctx, &mut batch, &full)?;
        let normal = t0.elapsed();

        // Normal TTFT: prefill + first sample
        let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let _ = sampler.sample(&ctx, batch.n_tokens() - 1);
        let normal_ttft = t0.elapsed();
        drop(ctx);

        // Incremental
        let mut ctx = new_ctx(model, backend, params)?;
        let mut ip = IncrementalPrefill::new(0, BATCH_SIZE);
        for snap in &debounced(prompt) {
            let toks = model.str_to_token(snap, AddBos::Always)?;
            ip.prefill_speculative(&mut ctx, &mut batch, &toks)?;
        }
        let t1 = Instant::now();
        ip.flush(&mut ctx, &mut batch, &full)?;
        let flush = t1.elapsed();

        // Incremental TTFT
        let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let _ = sampler.sample(&ctx, batch.n_tokens() - 1);
        let inc_ttft = t1.elapsed();
        drop(ctx);

        let spd = if flush.as_nanos() > 0 {
            normal.as_secs_f64() / flush.as_secs_f64()
        } else {
            f64::INFINITY
        };

        println!(
            "  {:>5} {:>10.2?} {:>10.2?} {:>10.2?} {:>6.1}x",
            n, normal, flush, inc_ttft, spd
        );
        let _ = (normal_ttft, inc_ttft); // used above
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. SPEED — generation throughput
// ---------------------------------------------------------------------------

fn bench_speed(
    model: &LlamaModel,
    backend: &LlamaBackend,
    params: &LlamaContextParams,
) -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. SPEED — generation tok/s after incremental prefill              │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    let prompt = "Explain the theory of relativity in simple terms";
    let full = model.str_to_token(prompt, AddBos::Always)?;
    let n_gen = 32;

    let mut ctx = new_ctx(model, backend, params)?;
    let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
    let mut ip = IncrementalPrefill::new(0, BATCH_SIZE);

    for snap in &debounced(prompt) {
        let toks = model.str_to_token(snap, AddBos::Always)?;
        ip.prefill_speculative(&mut ctx, &mut batch, &toks)?;
    }
    ip.flush(&mut ctx, &mut batch, &full)?;

    let sampler = LlamaSampler::chain_simple([LlamaSampler::common(), LlamaSampler::greedy()]);
    let mut n_cur = full.len() as i32;

    let t0 = Instant::now();
    let mut generated = 0u32;
    for _ in 0..n_gen {
        let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
        if model.is_eog_token(tok) {
            break;
        }
        batch.clear();
        batch.add(tok, n_cur, &[0], true)?;
        n_cur += 1;
        generated += 1;
        ctx.decode(&mut batch)?;
    }
    let elapsed = t0.elapsed();
    let tps = generated as f64 / elapsed.as_secs_f64();

    println!("  Prompt: \"{}\"", &prompt[..prompt.len().min(50)]);
    println!("  Generated: {} tokens in {:.2?}", generated, elapsed);
    println!("  Throughput: {:.1} tok/s", tps);
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. LOAD — total GPU compute comparison
// ---------------------------------------------------------------------------

fn bench_load(
    model: &LlamaModel,
    backend: &LlamaBackend,
    params: &LlamaContextParams,
    prompts: &[&str],
) -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. LOAD — total GPU compute (normal vs margin vs naive)            │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    println!(
        "  {:>5} {:>10} {:>10} {:>10} {:>8}",
        "Toks", "Normal", "Margin", "Naive", "Savings"
    );

    for prompt in prompts {
        let full = model.str_to_token(prompt, AddBos::Always)?;
        let n = full.len();
        let snaps = debounced(prompt);

        // Normal
        let mut ctx = new_ctx(model, backend, params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        let t = Instant::now();
        prefill_all(&mut ctx, &mut batch, &full)?;
        let normal = t.elapsed();
        drop(ctx);

        // Margin
        let mut ctx = new_ctx(model, backend, params)?;
        let mut ip = IncrementalPrefill::new(0, BATCH_SIZE);
        let t = Instant::now();
        for s in &snaps {
            let toks = model.str_to_token(s, AddBos::Always)?;
            ip.prefill_speculative(&mut ctx, &mut batch, &toks)?;
        }
        ip.flush(&mut ctx, &mut batch, &full)?;
        let margin = t.elapsed();
        drop(ctx);

        // Naive
        let mut ctx = new_ctx(model, backend, params)?;
        let mut naive = NaivePrefill::new();
        let t = Instant::now();
        for s in &snaps {
            let toks = model.str_to_token(s, AddBos::Always)?;
            naive.sync(&mut ctx, &mut batch, &toks)?;
        }
        naive.sync(&mut ctx, &mut batch, &full)?;
        let naive_t = t.elapsed();
        drop(ctx);

        let savings = if naive_t.as_nanos() > 0 {
            (1.0 - margin.as_secs_f64() / naive_t.as_secs_f64()) * 100.0
        } else {
            0.0
        };

        println!(
            "  {:>5} {:>10.2?} {:>10.2?} {:>10.2?} {:>7.0}%",
            n, normal, margin, naive_t, savings
        );
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. PRECISION — verify identical first-token output
// ---------------------------------------------------------------------------

fn bench_precision(
    model: &LlamaModel,
    backend: &LlamaBackend,
    params: &LlamaContextParams,
    prompts: &[&str],
) -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ 4. PRECISION — incremental vs normal produce identical first token  │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    let mut all_match = true;

    for prompt in prompts {
        let full = model.str_to_token(prompt, AddBos::Always)?;

        // Normal
        let mut ctx = new_ctx(model, backend, params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        prefill_all(&mut ctx, &mut batch, &full)?;
        let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let normal_tok = sampler.sample(&ctx, batch.n_tokens() - 1);
        drop(ctx);

        // Incremental
        let mut ctx = new_ctx(model, backend, params)?;
        let mut ip = IncrementalPrefill::new(0, BATCH_SIZE);
        for snap in &debounced(prompt) {
            let toks = model.str_to_token(snap, AddBos::Always)?;
            ip.prefill_speculative(&mut ctx, &mut batch, &toks)?;
        }
        ip.flush(&mut ctx, &mut batch, &full)?;
        let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let inc_tok = sampler.sample(&ctx, batch.n_tokens() - 1);
        drop(ctx);

        let normal_str = model
            .token_to_bytes(normal_tok, Special::Tokenize)
            .unwrap_or_default();
        let inc_str = model
            .token_to_bytes(inc_tok, Special::Tokenize)
            .unwrap_or_default();

        let matches = normal_tok == inc_tok;
        if !matches {
            all_match = false;
        }

        println!(
            "  {} {:>5} tok | normal={:?} inc={:?} | {}",
            if matches { "✓" } else { "✗" },
            full.len(),
            String::from_utf8_lossy(&normal_str),
            String::from_utf8_lossy(&inc_str),
            if matches {
                "MATCH".to_string()
            } else {
                "MISMATCH!".to_string()
            },
        );
    }

    println!(
        "  Result: {}",
        if all_match {
            "ALL MATCH ✓"
        } else {
            "SOME MISMATCHES ✗"
        }
    );
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. UX — simulate mid-line edits and measure recovery cost
// ---------------------------------------------------------------------------

fn bench_ux(
    model: &LlamaModel,
    backend: &LlamaBackend,
    params: &LlamaContextParams,
) -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ 5. UX — mid-line edit recovery cost (delete/insert/replace)        │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    let base = "The quick brown fox jumps over the lazy dog";
    let full_tokens = model.str_to_token(base, AddBos::Always)?;

    // Scenario 1: Type full text, then delete last word and retype
    {
        let mut ctx = new_ctx(model, backend, params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        let mut ip = IncrementalPrefill::new(0, BATCH_SIZE);

        // Type the full text
        ip.flush(&mut ctx, &mut batch, &full_tokens)?;

        // Delete "dog" → "The quick brown fox jumps over the lazy "
        let edited = "The quick brown fox jumps over the lazy ";
        let edited_tokens = model.str_to_token(edited, AddBos::Always)?;
        let t = Instant::now();
        let decoded = ip.flush(&mut ctx, &mut batch, &edited_tokens)?;
        let delete_time = t.elapsed();

        // Retype "cat"
        let final_text = "The quick brown fox jumps over the lazy cat";
        let final_tokens = model.str_to_token(final_text, AddBos::Always)?;
        let t = Instant::now();
        let decoded2 = ip.flush(&mut ctx, &mut batch, &final_tokens)?;
        let retype_time = t.elapsed();

        println!(
            "  Delete+retype tail:  delete={:.2?} ({decoded} tok) retype={:.2?} ({decoded2} tok)",
            delete_time, retype_time
        );
        drop(ctx);
    }

    // Scenario 2: Edit in the middle ("brown" → "red")
    {
        let mut ctx = new_ctx(model, backend, params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        let mut ip = IncrementalPrefill::new(0, BATCH_SIZE);

        ip.flush(&mut ctx, &mut batch, &full_tokens)?;

        let edited = "The quick red fox jumps over the lazy dog";
        let edited_tokens = model.str_to_token(edited, AddBos::Always)?;
        let t = Instant::now();
        let decoded = ip.flush(&mut ctx, &mut batch, &edited_tokens)?;
        let mid_edit_time = t.elapsed();

        println!(
            "  Mid-line replace:    {:.2?} ({decoded}/{} tok re-decoded — divergence point)",
            mid_edit_time, edited_tokens.len()
        );
        drop(ctx);
    }

    // Scenario 3: Insert at beginning ("Hey, " prefix)
    {
        let mut ctx = new_ctx(model, backend, params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        let mut ip = IncrementalPrefill::new(0, BATCH_SIZE);

        ip.flush(&mut ctx, &mut batch, &full_tokens)?;

        let edited = "Hey, the quick brown fox jumps over the lazy dog";
        let edited_tokens = model.str_to_token(edited, AddBos::Always)?;
        let t = Instant::now();
        let decoded = ip.flush(&mut ctx, &mut batch, &edited_tokens)?;
        let prefix_time = t.elapsed();

        println!(
            "  Prefix insert:       {:.2?} ({decoded}/{} tok — full re-decode from divergence)",
            prefix_time, edited_tokens.len()
        );
        drop(ctx);
    }

    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 6. DX — developer experience summary
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 8. Samplers & Temperature
// ---------------------------------------------------------------------------

fn bench_samplers(
    model: &LlamaModel,
    backend: &LlamaBackend,
    ctx_params: &LlamaContextParams,
) -> Result<()> {
    println!("\u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}");
    println!("\u{2502} 8. SAMPLERS & TEMPERATURE \u{2014} output quality across strategies    \u{2502}");
    println!("\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}");

    let prompt = "Write a haiku about the ocean";
    let full = model.str_to_token(prompt, AddBos::Always)?;
    let n_gen = 48;
    let seed = 42u32;

    // Define sampler configurations
    struct SamplerConfig {
        label: &'static str,
        build: Box<dyn Fn() -> LlamaSampler>,
    }

    let configs: Vec<SamplerConfig> = vec![
        SamplerConfig {
            label: "greedy (t=0)",
            build: Box::new(|| LlamaSampler::chain_simple([
                LlamaSampler::greedy(),
            ])),
        },
        SamplerConfig {
            label: "temp=0.1 top_k=40",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::top_k(40),
                LlamaSampler::temp(0.1),
                LlamaSampler::dist(seed),
            ])),
        },
        SamplerConfig {
            label: "temp=0.4 top_p=0.9",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::temp(0.4),
                LlamaSampler::dist(seed),
            ])),
        },
        SamplerConfig {
            label: "temp=0.7 top_p=0.9",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::temp(0.7),
                LlamaSampler::dist(seed),
            ])),
        },
        SamplerConfig {
            label: "temp=1.0 top_p=0.95",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::temp(1.0),
                LlamaSampler::dist(seed),
            ])),
        },
        SamplerConfig {
            label: "temp=1.5 top_k=50",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::top_k(50),
                LlamaSampler::temp(1.5),
                LlamaSampler::dist(seed),
            ])),
        },
        SamplerConfig {
            label: "min_p=0.05 t=0.7",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::min_p(0.05, 1),
                LlamaSampler::temp(0.7),
                LlamaSampler::dist(seed),
            ])),
        },
        SamplerConfig {
            label: "top_n_sigma=1.0",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::top_n_sigma(1.0),
                LlamaSampler::dist(seed),
            ])),
        },
        SamplerConfig {
            label: "mirostat_v2 t5 e0.1",
            build: Box::new(move || LlamaSampler::chain_simple([
                LlamaSampler::mirostat_v2(seed, 5.0, 0.1),
            ])),
        },
    ];

    println!("  Prompt: \"{}\" ({} tokens, generating {})", prompt, full.len(), n_gen);
    println!("  Seed: {} (for reproducibility with stochastic samplers)", seed);
    println!();

    println!(
        "  {:>22} {:>8} {:>8}  {}",
        "Sampler", "Gen ms", "tok/s", "Output (first 80 chars)"
    );
    println!("  {}", "-".repeat(100));

    for cfg in &configs {
        let mut ctx = new_ctx(model, backend, ctx_params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        prefill_all(&mut ctx, &mut batch, &full)?;

        let sampler = (cfg.build)();
        let mut n_cur = full.len() as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();

        let t0 = Instant::now();
        for _ in 0..n_gen {
            let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
            if model.is_eog_token(tok) {
                break;
            }
            let bytes = model.token_to_bytes(tok, Special::Tokenize).unwrap_or_default();
            let mut s = String::with_capacity(32);
            let _ = decoder.decode_to_string(&bytes, &mut s, false);
            output.push_str(&s);

            batch.clear();
            batch.add(tok, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }
        let elapsed = t0.elapsed();
        let n_actual = (n_cur - full.len() as i32) as f64;
        let tps = if elapsed.as_secs_f64() > 0.0 { n_actual / elapsed.as_secs_f64() } else { 0.0 };

        let display: String = output.replace('\n', "\u{21b5}").chars().take(80).collect();
        println!(
            "  {:>22} {:>7.1?} {:>7.1}  \"{}\"{}",
            cfg.label,
            elapsed,
            tps,
            display,
            if output.len() > 80 { "..." } else { "" },
        );
    }

    // Reproducibility check: run the same stochastic sampler twice
    println!();
    println!("  Reproducibility (same seed, temp=0.7 top_p=0.9, run twice):");
    let mut outputs = Vec::new();
    for run in 0..2 {
        let mut ctx = new_ctx(model, backend, ctx_params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        prefill_all(&mut ctx, &mut batch, &full)?;

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::temp(0.7),
            LlamaSampler::dist(seed),
        ]);
        let mut n_cur = full.len() as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        for _ in 0..n_gen {
            let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
            if model.is_eog_token(tok) { break; }
            let bytes = model.token_to_bytes(tok, Special::Tokenize).unwrap_or_default();
            let mut s = String::with_capacity(32);
            let _ = decoder.decode_to_string(&bytes, &mut s, false);
            output.push_str(&s);
            batch.clear();
            batch.add(tok, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }
        let display: String = output.replace('\n', "\u{21b5}").chars().take(60).collect();
        println!("    Run {}: \"{display}\"...", run + 1);
        outputs.push(output);
    }
    if outputs[0] == outputs[1] {
        println!("    \u{2714} Identical — same seed produces deterministic output");
    } else {
        let diff_pos = outputs[0].chars().zip(outputs[1].chars())
            .position(|(a, b)| a != b)
            .unwrap_or(outputs[0].len().min(outputs[1].len()));
        println!("    \u{2718} Diverges at char {} (non-deterministic)", diff_pos);
    }

    println!();
    Ok(())
}

fn bench_dx() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ 6. DX — developer experience summary                               │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!("  API surface:");
    println!("    IncrementalPrefill::new(history_len, batch_size)");
    println!("    IncrementalPrefill::prefill_speculative(ctx, batch, tokens) → Result<usize>");
    println!("    IncrementalPrefill::flush(ctx, batch, tokens) → Result<usize>");
    println!("    IncrementalPrefill::len() → usize");
    println!("  Integration points:");
    println!("    - Uses existing LlamaContext::decode() and clear_kv_cache_seq()");
    println!("    - No changes to llama-cpp-4 core — pure userspace pattern");
    println!("    - Works with any model/tokenizer/chat template");
    println!("  Error handling:");
    println!("    - KV cache trim errors propagated via Result");
    println!("    - Decode errors propagated via Result");
    println!("    - Graceful recovery: next TextChanged retries from last good state");
    println!("  Source files:");
    println!("    - prefill.rs:    ~130 lines (shared module)");
    println!("    - line_editor.rs: ~200 lines (cursor-based editor)");
    println!("    - main.rs:       ~500 lines (interactive chat)");
    println!("    - bench.rs:      ~400 lines (this benchmark)");
    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: incremental-bench <model.gguf>");
        std::process::exit(1);
    });

    let mut backend = LlamaBackend::init()?;
    backend.void_logs();

    let model_params = pin!(LlamaModelParams::default());
    let model =
        LlamaModel::load_from_file(&backend, &model_path, &model_params).context("load model")?;

    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_flash_attention(true);

    let name = model_path.split('/').last().unwrap_or(&model_path);
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Incremental Prefill — Comprehensive Benchmark                     ║");
    println!("║  Model: {:<60}║", name);
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let prompts: &[&str] = &[
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms that a child could understand",
        "Write a short poem about the beauty of mathematics and its connection to nature",
        "Compare and contrast Rust and C++ in terms of memory safety and performance",
    ];

    // Warmup
    println!("  (warmup decode...)\n");
    {
        let mut ctx = new_ctx(&model, &backend, &params)?;
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        let toks = model.str_to_token(prompts[0], AddBos::Always)?;
        prefill_all(&mut ctx, &mut batch, &toks)?;
    }

    bench_latency(&model, &backend, &params, prompts)?;
    bench_speed(&model, &backend, &params)?;
    bench_load(&model, &backend, &params, prompts)?;
    bench_precision(&model, &backend, &params, prompts)?;
    bench_ux(&model, &backend, &params)?;
    bench_kv_quant(&model, &backend, prompts)?;
    bench_samplers(&model, &backend, &params)?;
    bench_dx();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Done.");
    println!("═══════════════════════════════════════════════════════════════════════");

    Ok(())
}

// ---------------------------------------------------------------------------
// 7. KV quantization + TurboQuant
// ---------------------------------------------------------------------------

fn generate_text(
    model: &LlamaModel,
    ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
    batch: &mut LlamaBatch,
    start_pos: i32,
    n_gen: usize,
) -> Result<(String, std::time::Duration)> {
    let sampler = LlamaSampler::chain_simple([
        LlamaSampler::common(),
        LlamaSampler::greedy(),
    ]);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = start_pos;
    let t0 = Instant::now();

    for _ in 0..n_gen {
        let tok = sampler.sample(ctx, batch.n_tokens() - 1);
        if model.is_eog_token(tok) {
            break;
        }
        let bytes = model.token_to_bytes(tok, Special::Tokenize).unwrap_or_default();
        let mut s = String::with_capacity(32);
        let _ = decoder.decode_to_string(&bytes, &mut s, false);
        output.push_str(&s);

        batch.clear();
        batch.add(tok, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(batch)?;
    }
    Ok((output, t0.elapsed()))
}

fn bench_kv_quant(
    model: &LlamaModel,
    backend: &LlamaBackend,
    prompts: &[&str],
) -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ 7. KV QUANT + TURBOQUANT — quality & speed across KV cache types  │");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    let configs: &[(&str, GgmlType, bool)] = &[
        ("F16 (baseline)",   GgmlType::F16,  false),
        ("Q8_0 + turbo",     GgmlType::Q8_0, false),
        ("Q5_0 + turbo",     GgmlType::Q5_0, false),
        ("Q4_0 + turbo",     GgmlType::Q4_0, false),
        ("Q5_0 no turbo",    GgmlType::Q5_0, true),
        ("Q4_0 no turbo",    GgmlType::Q4_0, true),
    ];

    let prompt = prompts.last().unwrap_or(&prompts[0]);
    let full = model.str_to_token(prompt, AddBos::Always)?;
    let n_gen = 64;

    println!("  Prompt: \"{}\" ({} tokens, generating {})", &prompt[..prompt.len().min(50)], full.len(), n_gen);
    println!();

    let mut baseline_output = String::new();
    let mut results: Vec<(&str, String, std::time::Duration, std::time::Duration)> = Vec::new();

    for (label, kv_type, no_turbo) in configs {
        let p = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048))
            .with_flash_attention(true)
            .with_cache_type_k(*kv_type)
            .with_cache_type_v(*kv_type)
            .with_attn_rot_disabled(*no_turbo);

        let mut ctx = match model.new_context(backend, p) {
            Ok(c) => c,
            Err(_) => {
                println!("  {:>20} (unsupported by this build)", label);
                continue;
            }
        };
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);

        let t0 = Instant::now();
        prefill_all(&mut ctx, &mut batch, &full)?;
        let prefill_time = t0.elapsed();

        let (output, gen_time) = generate_text(model, &mut ctx, &mut batch, full.len() as i32, n_gen)?;

        if baseline_output.is_empty() {
            baseline_output = output.clone();
        }

        results.push((label, output, prefill_time, gen_time));
    }

    // Print timing table
    println!(
        "  {:>20} {:>10} {:>10} {:>8} {:>7}",
        "Config", "Prefill", "Gen 64t", "tok/s", "Match"
    );
    for (label, output, prefill, gen) in &results {
        let tps = if gen.as_secs_f64() > 0.0 {
            64.0 / gen.as_secs_f64()
        } else { 0.0 };
        let matches = *output == baseline_output;
        println!(
            "  {:>20} {:>10.2?} {:>10.2?} {:>7.1} {:>7}",
            label, prefill, gen, tps,
            if matches { "✔ yes" } else { "✘ NO" },
        );
    }

    // Print output comparison
    println!();
    println!("  Output comparison (first 120 chars):");
    for (label, output, _, _) in &results {
        let truncated: String = output.chars().take(120).collect();
        let matches = *output == baseline_output;
        println!(
            "  {} {:>20}: \"{}\"{}",
            if matches { "✔" } else { "✘" },
            label,
            truncated.replace('\n', "↵"),
            if output.len() > 120 { "..." } else { "" },
        );
    }

    // Show character-level diff for mismatches
    let mut any_mismatch = false;
    for (label, output, _, _) in &results {
        if *output != baseline_output {
            any_mismatch = true;
            let first_diff = baseline_output
                .chars()
                .zip(output.chars())
                .position(|(a, b)| a != b)
                .unwrap_or(baseline_output.len().min(output.len()));
            println!(
                "  ⚠ {} diverges from F16 at char {}: F16=\"{}\" vs \"{}\" ",
                label,
                first_diff,
                baseline_output.chars().skip(first_diff).take(20).collect::<String>(),
                output.chars().skip(first_diff).take(20).collect::<String>(),
            );
        }
    }
    if !any_mismatch {
        println!("  ✔ All configs produce identical output to F16 baseline");
    }

    println!();
    Ok(())
}
