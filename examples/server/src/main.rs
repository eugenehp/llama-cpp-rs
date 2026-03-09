//! OpenAI-compatible chat/completion/embedding server using llama.cpp.
//!
//! # Endpoints
//!
//! | Method | Path                    | Description                     |
//! |--------|-------------------------|---------------------------------|
//! | GET    | `/health`               | Liveness check                  |
//! | GET    | `/v1/models`            | List loaded model                |
//! | POST   | `/v1/chat/completions`  | Chat (streaming + non-streaming) |
//! | POST   | `/v1/completions`       | Raw text completion (streaming)  |
//! | POST   | `/v1/embeddings`        | Dense embeddings                 |
//!
//! # Usage
//!
//! ```console
//! # Local file
//! cargo run -p openai-server -- local path/to/model.gguf
//!
//! # Hugging Face (interactive quant picker)
//! cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF
//!
//! # Hugging Face (pick quant by name, download all shards)
//! cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF Q4_K_M
//!
//! # With GPU + auth key
//! cargo run -p openai-server --features metal -- \
//!     --n-gpu-layers 99 --api-key secret \
//!     hf-model bartowski/Llama-3.2-3B-Instruct-GGUF Q4_K_M
//! ```
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod tools;

use actix_web::{http::StatusCode, web, App, HttpRequest, HttpResponse, HttpServer};
use anyhow::Context as _;
use clap::Parser;
use futures_util::stream;
use hf_hub::api::sync::{Api, ApiBuilder};
use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaModel, Special},
    sampling::LlamaSampler,
};
use serde_json::{json, Value};
use std::{
    num::NonZeroU32,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, Semaphore};
use tools::{ToolChoice, extract_tool_calls, inject_tools, normalise_messages, parse_tool_choice, parse_tools, tool_call_grammar};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "openai-server", about = "OpenAI-compatible llama.cpp server")]
struct Args {
    /// Host to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Number of layers to offload to GPU (0 = CPU only).
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,

    /// Context size override (default: use the model's trained context length).
    #[arg(short = 'c', long)]
    ctx_size: Option<NonZeroU32>,

    /// Require this bearer token on every request. Disabled when omitted.
    #[arg(long)]
    api_key: Option<String>,

    /// Maximum number of requests processed concurrently.
    /// llama.cpp contexts are not thread-safe so this effectively serialises
    /// inference while keeping HTTP connections responsive.
    #[arg(long, default_value_t = 1)]
    parallel: usize,

    #[command(subcommand)]
    model: ModelSource,
}

#[derive(clap::Subcommand, Debug)]
enum ModelSource {
    /// Load a model from a local file path.
    Local {
        /// Path to the GGUF model file.
        path: PathBuf,
    },
    /// Download a model from Hugging Face Hub (cached locally).
    ///
    /// If `<model>` is omitted the repo's GGUF files are listed and you are
    /// prompted to choose interactively (best quant auto-picked when stdin is
    /// not a terminal).  For sharded repos all shards are downloaded.
    #[clap(name = "hf-model")]
    HuggingFace {
        /// Repository id, e.g. `unsloth/Qwen3.5-397B-A17B-GGUF`.
        repo: String,
        /// Exact filename or quant directory name (e.g. `Q4_K_M`).
        /// Omit to pick interactively.
        model: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// HuggingFace model selection
// ---------------------------------------------------------------------------

const QUANT_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q5_0", "Q3_K_M", "Q3_K_S", "Q8_0",
    "Q6_K", "Q2_K", "IQ4_XS", "IQ3_M",
];

#[derive(Debug)]
struct ModelGroup {
    label: String,
    files: Vec<String>,
}

impl ModelGroup {
    fn preference_score(&self) -> usize {
        QUANT_PREFERENCE
            .iter()
            .position(|q| self.label.to_uppercase().contains(q))
            .unwrap_or(usize::MAX)
    }
}

fn collect_groups(all_ggufs: Vec<String>) -> Vec<ModelGroup> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in all_ggufs {
        let key = if let Some(slash) = path.find('/') {
            path[..slash].to_string()
        } else {
            let stem = path.trim_end_matches(".gguf");
            if let Some(of_pos) = stem.rfind("-of-") {
                let before_of = &stem[..of_pos];
                if let Some(dash) = before_of.rfind('-') {
                    let shard_num = &before_of[dash + 1..];
                    if shard_num.chars().all(|c| c.is_ascii_digit()) {
                        before_of[..dash].to_string()
                    } else {
                        stem.to_string()
                    }
                } else {
                    stem.to_string()
                }
            } else {
                stem.to_string()
            }
        };
        map.entry(key).or_default().push(path);
    }
    map.into_iter()
        .map(|(key, mut files)| {
            files.sort();
            let shard_info = if files.len() > 1 {
                format!("  [{} shards]", files.len())
            } else {
                String::new()
            };
            ModelGroup {
                label: format!("{key}{shard_info}"),
                files,
            }
        })
        .collect()
}

fn prompt_user(groups: &[ModelGroup]) -> anyhow::Result<usize> {
    use std::io::{self, IsTerminal as _, Write};
    eprintln!("\nAvailable models in repo:");
    for (i, g) in groups.iter().enumerate() {
        eprintln!("  {:>2})  {}", i + 1, g.label);
    }
    if !io::stdin().is_terminal() {
        let best = groups
            .iter()
            .enumerate()
            .min_by_key(|(_, g)| g.preference_score())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("\nNon-interactive — auto-selected: {}", groups[best].label);
        return Ok(best);
    }
    loop {
        eprint!("\nSelect a model [1–{}]: ", groups.len());
        io::stderr().flush().ok();
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        match line.trim().parse::<usize>() {
            Ok(n) if n >= 1 && n <= groups.len() => return Ok(n - 1),
            _ => eprintln!("  Enter a number between 1 and {}.", groups.len()),
        }
    }
}

fn resolve_hf(api: &Api, repo: &str, model: Option<String>) -> anyhow::Result<PathBuf> {
    let api_repo = api.model(repo.to_string());
    // Exact .gguf filename → download directly.
    if let Some(ref filename) = model {
        if filename.ends_with(".gguf") {
            return api_repo
                .get(filename)
                .with_context(|| format!("failed to download '{filename}' from '{repo}'"));
        }
    }
    let info = api_repo
        .info()
        .with_context(|| format!("failed to fetch repo info for '{repo}'"))?;
    let all_ggufs: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|n| n.ends_with(".gguf"))
        .collect();
    if all_ggufs.is_empty() {
        anyhow::bail!("no .gguf files found in repo '{repo}'");
    }
    let groups = collect_groups(all_ggufs);
    let chosen_idx = if let Some(filter) = model {
        let filter_up = filter.to_uppercase();
        groups
            .iter()
            .position(|g| {
                let label_key = g.label.split_whitespace().next().unwrap_or(&g.label);
                label_key.to_uppercase() == filter_up
                    || label_key.to_uppercase().contains(&filter_up)
            })
            .with_context(|| {
                let available: Vec<_> = groups
                    .iter()
                    .map(|g| {
                        g.label
                            .split_whitespace()
                            .next()
                            .unwrap_or(&g.label)
                            .to_string()
                    })
                    .collect();
                format!(
                    "no group matching '{filter}' in '{repo}'. Available: {}",
                    available.join(", ")
                )
            })?
    } else if groups.len() == 1 {
        eprintln!("Auto-selected: {}", groups[0].label);
        0
    } else {
        prompt_user(&groups)?
    };
    let group = &groups[chosen_idx];
    eprintln!("\nDownloading: {}", group.label);
    let mut first_path: Option<PathBuf> = None;
    for (i, file) in group.files.iter().enumerate() {
        if group.files.len() > 1 {
            eprintln!("  shard {}/{}: {file}", i + 1, group.files.len());
        }
        let path = api
            .model(repo.to_string())
            .get(file)
            .with_context(|| format!("failed to download shard '{file}'"))?;
        if first_path.is_none() {
            first_path = Some(path);
        }
    }
    first_path.ok_or_else(|| anyhow::anyhow!("no files downloaded"))
}

impl ModelSource {
    fn resolve(self) -> anyhow::Result<PathBuf> {
        match self {
            ModelSource::Local { path } => Ok(path),
            ModelSource::HuggingFace { repo, model } => {
                let api = ApiBuilder::new()
                    .with_progress(true)
                    .build()
                    .context("failed to build HF API client")?;
                resolve_hf(&api, &repo, model)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct AppState {
    backend: LlamaBackend,
    model: LlamaModel,
    chat_template: Option<String>,
    model_name: String,
    default_ctx_size: Option<NonZeroU32>,
    /// Limits the number of concurrent inference calls.
    inference_semaphore: Arc<Semaphore>,
    /// Optional bearer token that every request must present.
    api_key: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP error helpers
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct HttpError {
    status: StatusCode,
    r#type: &'static str,
    message: String,
}

fn bad_request(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::BAD_REQUEST,
        r#type: "invalid_request_error",
        message: msg.into(),
    }
}

fn unauthorized(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::UNAUTHORIZED,
        r#type: "authentication_error",
        message: msg.into(),
    }
}

fn internal_error(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        r#type: "server_error",
        message: msg.into(),
    }
}

fn error_response(err: HttpError) -> HttpResponse {
    let body = json!({
        "error": {
            "message": err.message,
            "type": err.r#type,
            "code": err.status.as_u16()
        }
    })
    .to_string();
    HttpResponse::build(err.status)
        .content_type("application/json")
        .body(body)
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

fn check_auth(req: &HttpRequest, state: &AppState) -> Option<HttpError> {
    let Some(ref expected) = state.api_key else {
        return None;
    };
    let auth = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());
    match auth {
        Some(v) if v == format!("Bearer {expected}") => None,
        _ => Some(unauthorized("invalid or missing API key")),
    }
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

fn parse_stop_sequences(req: &Value) -> Result<Vec<String>, HttpError> {
    match req.get("stop") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err(bad_request("each element of 'stop' must be a string")),
            })
            .collect(),
        _ => Err(bad_request("'stop' must be a string or array of strings")),
    }
}

/// Convert `(role, content)` pairs into the `LlamaChatMessage` vec that
/// `apply_chat_template` expects.
fn to_chat_messages(pairs: Vec<(String, String)>) -> Result<Vec<LlamaChatMessage>, HttpError> {
    pairs
        .into_iter()
        .map(|(role, content)| {
            LlamaChatMessage::new(role.clone(), content)
                .map_err(|e| bad_request(format!("invalid message (role={role}): {e}")))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Core inference engine
// ---------------------------------------------------------------------------

/// All sampling / generation parameters extracted from a request.
struct InferenceParams {
    prompt: String,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u32,
    max_tokens: u32,
    stop_seqs: Vec<String>,
    /// Optional GBNF grammar string.
    grammar: Option<String>,
}

impl InferenceParams {
    fn from_request(req: &Value, prompt: String) -> Result<Self, HttpError> {
        let temperature = req
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        if temperature < 0.0 {
            return Err(bad_request("'temperature' must be >= 0"));
        }
        let top_p = req.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        if !(0.0 < top_p && top_p <= 1.0) {
            return Err(bad_request("'top_p' must be in (0, 1]"));
        }
        let top_k = req.get("top_k").and_then(Value::as_i64).unwrap_or(0) as i32;
        if top_k < 0 {
            return Err(bad_request("'top_k' must be >= 0"));
        }
        let seed = req.get("seed").and_then(Value::as_u64).unwrap_or(0) as u32;
        let max_tokens = req
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(1024) as u32;
        if max_tokens == 0 {
            return Err(bad_request("'max_tokens' must be > 0"));
        }
        let grammar = match req.get("grammar") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            _ => return Err(bad_request("'grammar' must be a GBNF string")),
        };
        let stop_seqs = parse_stop_sequences(req)?;
        Ok(InferenceParams {
            prompt,
            temperature,
            top_p,
            top_k,
            seed,
            max_tokens,
            stop_seqs,
            grammar,
        })
    }
}

/// Why the decode loop stopped.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Run the full inference loop, calling `on_piece` for each decoded text
/// fragment.  `on_piece` returns `false` to stop early (e.g. cancelled
/// stream).  Returns `(completion_token_count, finish_reason)`.
fn run_inference<F>(
    state: &AppState,
    params: &InferenceParams,
    mut on_piece: F,
) -> Result<(u32, FinishReason), HttpError>
where
    F: FnMut(&str) -> bool,
{
    // ── Tokenise prompt ───────────────────────────────────────────────────────
    let tokens = state
        .model
        .str_to_token(&params.prompt, AddBos::Always)
        .map_err(|e| internal_error(format!("tokenisation failed: {e}")))?;

    let n_prompt = tokens.len() as u32;
    let n_ctx = state
        .default_ctx_size
        .map_or(state.model.n_ctx_train(), NonZeroU32::get)
        .max(n_prompt + params.max_tokens);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);

    let mut ctx = state
        .model
        .new_context(&state.backend, ctx_params)
        .map_err(|e| internal_error(format!("context init: {e}")))?;

    // ── Prefill ───────────────────────────────────────────────────────────────
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last = tokens.len().saturating_sub(1) as i32;
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i as i32 == last)
            .map_err(|e| internal_error(format!("batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| internal_error(format!("prefill: {e}")))?;

    // ── Sampler chain ─────────────────────────────────────────────────────────
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if let Some(gbnf) = &params.grammar {
        chain.push(LlamaSampler::grammar(&state.model, gbnf, "root"));
    }
    if params.temperature > 0.0 {
        if params.top_k > 0 {
            chain.push(LlamaSampler::top_k(params.top_k));
        }
        if params.top_p < 1.0 {
            chain.push(LlamaSampler::top_p(params.top_p, 1));
        }
        chain.push(LlamaSampler::temp(params.temperature));
        chain.push(LlamaSampler::dist(params.seed));
    } else {
        chain.push(LlamaSampler::greedy());
    }
    let sampler = LlamaSampler::chain_simple(chain);

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut n_cur = batch.n_tokens();
    let max_pos = n_cur + params.max_tokens as i32;
    let mut completion_tokens: u32 = 0;
    let mut generated = String::new(); // only used for stop-sequence matching
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut finish_reason = FinishReason::Stop;

    'decode: while n_cur < max_pos {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if state.model.is_eog_token(token) {
            break;
        }
        let bytes = state
            .model
            .token_to_bytes(token, Special::Plaintext)
            .map_err(|e| internal_error(format!("token_to_bytes: {e}")))?;
        let mut piece = String::with_capacity(8);
        let _ = decoder.decode_to_string(&bytes, &mut piece, false);
        completion_tokens += 1;

        // Check stop sequences *before* emitting (so the stop string itself
        // never appears in the output).
        generated.push_str(&piece);
        for stop in &params.stop_seqs {
            if !stop.is_empty() && generated.ends_with(stop.as_str()) {
                let trim_to = generated.len() - stop.len();
                // Emit the part before the stop token.
                let safe = &generated[..trim_to];
                // Emit only the new part (everything after what we've already
                // sent to on_piece).  To keep this simple we track how many
                // bytes we've already sent via a dedicated counter.
                // (The closure is responsible for flushing partial UTF-8.)
                let _ = on_piece(safe);
                break 'decode;
            }
        }

        if !on_piece(&piece) {
            break;
        }

        // Trim generated buffer: keep only the last max-stop-len bytes so
        // stop-sequence matching doesn't accumulate the entire response.
        let max_keep = params
            .stop_seqs
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .max(1);
        if generated.len() > max_keep * 4 {
            let drop_to = generated.len() - max_keep;
            generated.drain(..drop_to);
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| internal_error(format!("batch add: {e}")))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| internal_error(format!("decode: {e}")))?;
    }

    if n_cur >= max_pos {
        finish_reason = FinishReason::Length;
    }

    Ok((completion_tokens, finish_reason))
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

fn sse_chunk(data: &Value) -> web::Bytes {
    web::Bytes::from(format!("data: {}\n\n", data))
}

fn sse_done() -> web::Bytes {
    web::Bytes::from("data: [DONE]\n\n")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------
// Chat completions  POST /v1/chat/completions
// ---------------------------------------------------------------------------

async fn chat_completions(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Bytes,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s.to_owned(),
        Err(_) => return error_response(bad_request("body must be valid UTF-8")),
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return error_response(bad_request(format!("invalid JSON: {e}"))),
    };

    let streaming = parsed.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // ── Parse tools ──────────────────────────────────────────────────────────
    let tool_defs = match parse_tools(&parsed) {
        Ok(t) => t,
        Err(e) => return error_response(e),
    };
    let tool_choice = match parse_tool_choice(&parsed) {
        Ok(c) => c,
        Err(e) => return error_response(e),
    };

    // ── Build prompt from messages ───────────────────────────────────────────
    let prompt = {
        let mut msg_pairs = match normalise_messages(&parsed) {
            Ok(m) => m,
            Err(e) => return error_response(e),
        };

        // Inject tool definitions + usage instructions into the system message.
        inject_tools(&mut msg_pairs, &tool_defs, &tool_choice);

        let chat_msgs = match to_chat_messages(msg_pairs) {
            Ok(m) => m,
            Err(e) => return error_response(e),
        };

        let template_override = match parsed.get("chat_template") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            _ => return error_response(bad_request("'chat_template' must be a string")),
        };
        let template = template_override.or_else(|| state.chat_template.clone());
        match state.model.apply_chat_template(template, chat_msgs, true) {
            Ok(p) => p,
            Err(e) => return error_response(internal_error(format!("chat template: {e}"))),
        }
    };

    // ── Sampling params ───────────────────────────────────────────────────────
    // For forced tool calling, override grammar with the tool call grammar.
    let grammar_override = match &tool_choice {
        ToolChoice::Required => Some(tool_call_grammar(None)),
        ToolChoice::Function(name) => Some(tool_call_grammar(Some(name))),
        _ => None,
    };

    let mut params = match InferenceParams::from_request(&parsed, prompt) {
        Ok(p) => p,
        Err(e) => return error_response(e),
    };
    if let Some(g) = grammar_override {
        params.grammar = Some(g);
    }

    // When waiting for a forced tool call, suppress </tool_call> from counting
    // toward max_tokens (don't cut off mid-call).  Simply increase the budget.
    if !tool_defs.is_empty() && params.max_tokens < 512 {
        params.max_tokens = 512;
    }

    let model_name = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_name)
        .to_owned();
    let has_tools = !tool_defs.is_empty();
    let created = now_secs();
    let id = format!("chatcmpl-{created}");

    if streaming {
        run_chat_stream(state, params, id, model_name, created, has_tools).await
    } else {
        run_chat_blocking(state, params, id, model_name, created, has_tools).await
    }
}

async fn run_chat_blocking(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
    has_tools: bool,
) -> HttpResponse {
    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut raw = String::new();
        let outcome = run_inference(&state2, &params, |piece| {
            raw.push_str(piece);
            true
        });
        outcome.map(|(tokens, reason)| (raw, tokens, reason))
    })
    .await;

    match result {
        Ok(Ok((raw_output, completion_tokens, finish_reason))) => {
            let prompt_tokens = 0u32; // cheap approximation; full count needs a 2nd tokenise pass

            // Parse tool calls out of the raw output.
            let (content, tool_calls) = if has_tools {
                extract_tool_calls(&raw_output)
            } else {
                (raw_output, vec![])
            };

            let (final_finish, message) = if tool_calls.is_empty() {
                (
                    finish_reason.as_str(),
                    json!({ "role": "assistant", "content": content }),
                )
            } else {
                let calls_json: Vec<Value> = tool_calls.iter().map(|c| c.to_value()).collect();
                (
                    "tool_calls",
                    json!({
                        "role": "assistant",
                        "content": if content.is_empty() { Value::Null } else { Value::String(content) },
                        "tool_calls": calls_json
                    }),
                )
            };

            HttpResponse::Ok().content_type("application/json").body(
                json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "message": message, "finish_reason": final_finish}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                })
                .to_string(),
            )
        }
        Ok(Err(e)) => error_response(e),
        Err(e) => error_response(internal_error(format!("inference task panicked: {e}"))),
    }
}

async fn run_chat_stream(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
    has_tools: bool,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<web::Bytes>(32);
    let id2 = id.clone();
    let model2 = model_name.clone();

    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        const OBJ: &str = "chat.completion.chunk";

        // First chunk: role delta.
        let _ = tx.blocking_send(sse_chunk(&json!({
            "id": id2, "object": OBJ, "created": created, "model": model2,
            "choices": [{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]
        })));

        // Collect the whole output when tools are present so we can parse
        // tool calls before streaming; otherwise stream token-by-token.
        let mut finish_reason = FinishReason::Stop;

        if has_tools {
            // Buffered mode: collect, parse, then emit.
            let mut raw = String::new();
            if let Ok((_, fr)) = run_inference(&state2, &params, |piece| {
                raw.push_str(piece);
                true
            }) {
                finish_reason = fr;
            }

            let (content, tool_calls) = extract_tool_calls(&raw);

            if tool_calls.is_empty() {
                // No tool calls — stream content as a single delta.
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{"content":content},"finish_reason":null}]
                })));
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{},"finish_reason":finish_reason.as_str()}]
                })));
            } else {
                // Emit tool_calls delta.
                let calls_json: Vec<Value> = tool_calls.iter().map(|c| c.to_value()).collect();
                let content_val = if content.is_empty() { Value::Null } else { Value::String(content) };
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{"content":content_val,"tool_calls":calls_json},"finish_reason":null}]
                })));
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{},"finish_reason":"tool_calls"}]
                })));
            }
        } else {
            // Pure streaming: emit each token piece immediately.
            if let Ok((_, fr)) = run_inference(&state2, &params, |piece| {
                let ok = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{"content":piece},"finish_reason":null}]
                }))).is_ok();
                ok
            }) {
                finish_reason = fr;
            }
            let _ = tx.blocking_send(sse_chunk(&json!({
                "id": id2, "object": OBJ, "created": created, "model": model2,
                "choices": [{"index":0,"delta":{},"finish_reason":finish_reason.as_str()}]
            })));
        }

        let _ = tx.blocking_send(sse_done());
    });

    let body_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (Ok::<_, actix_web::Error>(chunk), rx))
    });

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(body_stream)
}

// ---------------------------------------------------------------------------
// Raw completions  POST /v1/completions
// ---------------------------------------------------------------------------

async fn completions(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Bytes,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s.to_owned(),
        Err(_) => return error_response(bad_request("body must be valid UTF-8")),
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return error_response(bad_request(format!("invalid JSON: {e}"))),
    };

    let prompt = match parsed.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            // Array of strings → join (batch not yet supported, take first)
            match arr.first() {
                Some(Value::String(s)) => s.clone(),
                _ => return error_response(bad_request("'prompt' array must contain strings")),
            }
        }
        _ => return error_response(bad_request("'prompt' must be a string")),
    };

    let streaming = parsed.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let params = match InferenceParams::from_request(&parsed, prompt) {
        Ok(p) => p,
        Err(e) => return error_response(e),
    };

    let model_name = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_name)
        .to_owned();
    let created = now_secs();
    let id = format!("cmpl-{created}");

    if streaming {
        // Reuse chat stream logic with the "text_completion" object type
        // but emit `text` delta field instead of `content`.
        run_completion_stream(state, params, id, model_name, created).await
    } else {
        run_completion_blocking(state, params, id, model_name, created).await
    }
}

async fn run_completion_blocking(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
) -> HttpResponse {
    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut text = String::new();
        run_inference(&state2, &params, |piece| {
            text.push_str(piece);
            true
        })
        .map(|(tokens, reason)| (text, tokens, reason))
    })
    .await;

    match result {
        Ok(Ok((text, completion_tokens, finish_reason))) => {
            HttpResponse::Ok().content_type("application/json").body(
                json!({
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": finish_reason.as_str()
                    }],
                    "usage": {
                        "completion_tokens": completion_tokens
                    }
                })
                .to_string(),
            )
        }
        Ok(Err(e)) => error_response(e),
        Err(e) => error_response(internal_error(format!("inference task panicked: {e}"))),
    }
}

async fn run_completion_stream(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<web::Bytes>(32);
    let id2 = id.clone();
    let model2 = model_name.clone();

    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut finish_reason = FinishReason::Stop;
        let result = run_inference(&state2, &params, |piece| {
            let chunk = sse_chunk(&json!({
                "id": id2,
                "object": "text_completion",
                "created": created,
                "model": model2,
                "choices": [{"index": 0, "text": piece, "finish_reason": null}]
            }));
            tx.blocking_send(chunk).is_ok()
        });
        if let Ok((_, fr)) = result {
            finish_reason = fr;
        }
        let last = sse_chunk(&json!({
            "id": id2,
            "object": "text_completion",
            "created": created,
            "model": model2,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason.as_str()}]
        }));
        let _ = tx.blocking_send(last);
        let _ = tx.blocking_send(sse_done());
    });

    let body_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (Ok::<_, actix_web::Error>(chunk), rx))
    });

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(body_stream)
}

// ---------------------------------------------------------------------------
// Embeddings  POST /v1/embeddings
// ---------------------------------------------------------------------------

async fn embeddings(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Bytes,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s.to_owned(),
        Err(_) => return error_response(bad_request("body must be valid UTF-8")),
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return error_response(bad_request(format!("invalid JSON: {e}"))),
    };

    // `input` may be a string or an array of strings.
    let inputs: Vec<String> = match parsed.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    Value::String(s) => out.push(s.clone()),
                    _ => return error_response(bad_request("'input' array must contain strings")),
                }
            }
            out
        }
        _ => return error_response(bad_request("'input' must be a string or array of strings")),
    };

    let model_name = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_name)
        .to_owned();

    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        // Return (vectors, total_prompt_tokens) together so `inputs` doesn't
        // need to be borrowed after the move.
        let total_tokens: u32 = inputs
            .iter()
            .filter_map(|s| state2.model.str_to_token(s, AddBos::Always).ok())
            .map(|t| t.len() as u32)
            .sum();
        embed_inputs(&state2, &inputs).map(|vecs| (vecs, total_tokens))
    })
    .await;

    match result {
        Ok(Ok((vectors, total_tokens))) => {
            let data: Vec<Value> = vectors
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    json!({
                        "object": "embedding",
                        "index": i,
                        "embedding": v
                    })
                })
                .collect();
            HttpResponse::Ok().content_type("application/json").body(
                json!({
                    "object": "list",
                    "model": model_name,
                    "data": data,
                    "usage": { "prompt_tokens": total_tokens, "total_tokens": total_tokens }
                })
                .to_string(),
            )
        }
        Ok(Err(e)) => error_response(e),
        Err(e) => error_response(internal_error(format!("embed task panicked: {e}"))),
    }
}

fn embed_inputs(state: &AppState, inputs: &[String]) -> Result<Vec<Vec<f32>>, HttpError> {
    let n_embd = state.model.n_embd() as usize;
    let mut results = Vec::with_capacity(inputs.len());

    for input in inputs {
        let tokens = state
            .model
            .str_to_token(input, AddBos::Always)
            .map_err(|e| internal_error(format!("tokenise: {e}")))?;

        let n_ctx = (tokens.len() as u32 + 16).max(64);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_embeddings(true);

        let mut ctx = state
            .model
            .new_context(&state.backend, ctx_params)
            .map_err(|e| internal_error(format!("context init: {e}")))?;

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        let last = tokens.len().saturating_sub(1) as i32;
        for (i, &tok) in tokens.iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], i as i32 == last)
                .map_err(|e| internal_error(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| internal_error(format!("decode: {e}")))?;

        // Try sequence-level pooled embedding first, fall back to last-token.
        let vec = if let Ok(emb) = ctx.embeddings_seq_ith(0) {
            emb.to_vec()
        } else if let Ok(emb) = ctx.embeddings_ith(last) {
            emb.to_vec()
        } else {
            vec![0.0f32; n_embd]
        };

        // L2-normalise.
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        results.push(vec.into_iter().map(|x| x / norm).collect());
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Simple handlers
// ---------------------------------------------------------------------------

async fn list_models(req: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let n_ctx = state
        .default_ctx_size
        .map_or(state.model.n_ctx_train(), NonZeroU32::get);
    HttpResponse::Ok()
        .content_type("application/json")
        .body(
            json!({
                "object": "list",
                "data": [{
                    "id": state.model_name,
                    "object": "model",
                    "created": now_secs(),
                    "owned_by": "llama.cpp",
                    "context_length": n_ctx,
                    "embedding_length": state.model.n_embd()
                }]
            })
            .to_string(),
        )
}

async fn health() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/json")
        .body(r#"{"status":"ok"}"#)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let model_path = args
        .model
        .resolve()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llama.cpp")
        .to_string();

    let backend = LlamaBackend::init()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let mut model_params = LlamaModelParams::default();
    if args.n_gpu_layers > 0 {
        model_params = model_params.with_n_gpu_layers(args.n_gpu_layers);
    }

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let chat_template = model.get_chat_template(4096).ok();
    if chat_template.is_some() {
        tracing::info!("Loaded built-in chat template from model");
    } else {
        tracing::warn!("No built-in chat template — supply 'chat_template' per request");
    }

    let parallel = args.parallel.max(1);
    if args.api_key.is_some() {
        tracing::info!("API key authentication enabled");
    }

    let state = web::Data::new(AppState {
        backend,
        model,
        chat_template,
        model_name,
        default_ctx_size: args.ctx_size,
        inference_semaphore: Arc::new(Semaphore::new(parallel)),
        api_key: args.api_key,
    });

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on http://{addr}  (parallel={parallel})");
    tracing::info!("Endpoints:");
    tracing::info!("  GET  /health");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  POST /v1/chat/completions  (streaming supported)");
    tracing::info!("  POST /v1/completions       (streaming supported)");
    tracing::info!("  POST /v1/embeddings");

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().error_handler(|err, _req| {
                let msg = format!("JSON parse error: {err}");
                actix_web::error::InternalError::from_response(
                    err,
                    error_response(bad_request(msg)),
                )
                .into()
            }))
            .route("/health", web::get().to(health))
            .route("/v1/models", web::get().to(list_models))
            .route("/v1/chat/completions", web::post().to(chat_completions))
            .route("/v1/completions", web::post().to(completions))
            .route("/v1/embeddings", web::post().to(embeddings))
    })
    .bind(&addr)?
    .run()
    .await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── collect_groups ───────────────────────────────────────────────────────

    #[test]
    fn single_plain_gguf() {
        let files = vec!["model.Q4_K_M.gguf".to_string()];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 1);
    }

    #[test]
    fn sharded_flat_files_grouped() {
        let files = vec![
            "model-Q4_K_M-00001-of-00003.gguf".to_string(),
            "model-Q4_K_M-00002-of-00003.gguf".to_string(),
            "model-Q4_K_M-00003-of-00003.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 3);
        assert_eq!(groups[0].files[0], "model-Q4_K_M-00001-of-00003.gguf");
    }

    #[test]
    fn subdirectory_files_grouped_by_dir() {
        let files = vec![
            "Q4_K_M/model-00001-of-00006.gguf".to_string(),
            "Q4_K_M/model-00002-of-00006.gguf".to_string(),
            "Q3_K_M/model-00001-of-00005.gguf".to_string(),
            "Q3_K_M/model-00002-of-00005.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 2);
        // BTreeMap orders alphabetically: Q3 before Q4
        assert_eq!(groups[0].label, "Q3_K_M  [2 shards]");
        assert_eq!(groups[1].label, "Q4_K_M  [2 shards]");
    }

    #[test]
    fn mixed_quants_each_get_own_group() {
        let files = vec![
            "llama-Q4_K_M.gguf".to_string(),
            "llama-Q8_0.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn preference_score_orders_correctly() {
        let files = vec![
            "Q8_0/model.gguf".to_string(),
            "Q4_K_M/model.gguf".to_string(),
            "Q3_K_S/model.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        let mut scores: Vec<_> = groups.iter().map(|g| (g.preference_score(), &g.label)).collect();
        scores.sort();
        // Q4_K_M should have the lowest (best) score
        assert!(scores[0].1.contains("Q4_K_M"), "got {scores:?}");
    }
}
