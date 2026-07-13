//! # Raw-byte detokenization
//!
//! Demonstrates the raw / lossless detokenization APIs:
//!
//! - [`LlamaModel::token_to_raw_bytes`] — one token's exact piece bytes,
//!   including control/byte pieces that `token_to_bytes` filters away.
//! - [`LlamaModel::tokens_to_raw_bytes`] — the bulk counterpart for a slice.
//! - [`StreamDetokenizer`] — a stateful, UTF-8-aware decoder for
//!   token-by-token generation loops. Byte-fallback tokenizers split a single
//!   codepoint (emoji, CJK, accents) across several tokens, so decoding each
//!   token in isolation yields invalid UTF-8; the streamer buffers the partial
//!   tail until the next token completes it.
//!
//! Compare with `examples/usage.rs`, which hand-rolls the same partial-UTF-8
//! handling with an `encoding_rs` decoder over the filtering `token_to_bytes`.
//!
//! ```console
//! cargo run --example detokenize -- model.gguf
//! cargo run --example detokenize -- model.gguf "Once upon a time"
//! ```

use llama_cpp_4::prelude::*;
use std::io::Write;

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("please specify a model path");
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The quick brown fox".to_string());

    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())
        .expect("unable to load model");
    let mut ctx = model
        .new_context(&backend, LlamaContextParams::default())
        .expect("unable to create the llama_context");

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .unwrap_or_else(|_| panic!("failed to tokenize {prompt}"));

    // ── 1. Single-token raw bytes ────────────────────────────────────────────
    // `token_to_raw_bytes` forwards straight to llama.cpp, so control/byte
    // pieces that `token_to_bytes` would return empty are preserved verbatim.
    println!("prompt tokens ({}):", tokens.len());
    for &token in &tokens {
        let raw = model.token_to_raw_bytes(token, Special::Tokenize).unwrap();
        let filtered = model.token_to_bytes(token, Special::Tokenize).unwrap();
        println!(
            "  {:>6}  raw={:<20} filtered={}",
            token.0,
            format!("{raw:?}"),
            if filtered.is_empty() {
                "<dropped>".to_string()
            } else {
                format!("{filtered:?}")
            },
        );
    }

    // ── 2. Bulk raw bytes ────────────────────────────────────────────────────
    // `tokens_to_raw_bytes` concatenates every piece losslessly. For a complete
    // token sequence the bytes form valid UTF-8.
    let bulk = model
        .tokens_to_raw_bytes(&tokens, Special::Tokenize)
        .unwrap();
    println!(
        "\nbulk detokenized prompt: {:?}",
        String::from_utf8_lossy(&bulk),
    );

    // ── 3. Streaming detokenization during generation ────────────────────────
    // Feed sampled tokens one at a time; `push` returns only the text that has
    // become complete UTF-8 and buffers any partial multi-byte tail.
    let mut batch = LlamaBatch::new(512, 1);
    let last_index = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch.add(token, i, &[0], i == last_index).unwrap();
    }
    ctx.decode(&mut batch).expect("llama_decode() failed");

    let mut detok = StreamDetokenizer::new(&model, Special::Plaintext);
    let mut sampler = LlamaSampler::greedy();
    let n_len = 64;
    let mut n_cur = batch.n_tokens();

    print!("\ngenerated: ");
    while n_cur <= n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }

        // No intermediate decoder, no invalid UTF-8 across byte-fallback tokens.
        let chunk = detok.push(token).expect("detokenize failed");
        print!("{chunk}");
        std::io::stdout().flush().unwrap();

        batch.clear();
        batch.add(token, n_cur, &[0], true).unwrap();
        n_cur += 1;
        ctx.decode(&mut batch).expect("failed to eval");
    }

    // Flush any trailing complete text; errors here mean the stream ended
    // mid-character (a truncated generation).
    match detok.finish() {
        Ok(tail) => print!("{tail}"),
        Err(DetokenizeError::IncompleteUtf8(bytes)) => {
            eprintln!("\n[stream ended with {} incomplete byte(s)]", bytes.len());
        }
        Err(e) => eprintln!("\n[detokenize error: {e}]"),
    }
    println!();
}
