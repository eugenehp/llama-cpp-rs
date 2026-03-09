//! Minimal OpenAI-compatible chat completion server using Actix Web.
//!
//! Implements `POST /v1/chat/completions` and `GET /v1/models`.
//!
//! # Usage
//!
//! ```console
//! # Local model file
//! cargo run -p openai-server -- local path/to/model.gguf
//!
//! # Download from Hugging Face
//! cargo run -p openai-server -- hf-model <repo> <file>
//!
//! # With options
//! cargo run -p openai-server -- --host 0.0.0.0 --port 8080 local model.gguf
//! ```
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use actix_web::{http::StatusCode, web, App, HttpResponse, HttpServer};
use anyhow::Context as _;
use clap::Parser;
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
    time::{SystemTime, UNIX_EPOCH},
};

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

    /// Context size override (default: use the model's trained context size).
    #[arg(short = 'c', long)]
    ctx_size: Option<NonZeroU32>,

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
    /// Download a model from Hugging Face (or use the local cache).
    ///
    /// If `<model>` is omitted the repo's GGUF files are listed and you are
    /// prompted to choose one interactively (or the best quant is auto-picked
    /// when stdin is not a terminal).
    #[clap(name = "hf-model")]
    HuggingFace {
        /// Repository, e.g. `unsloth/Qwen3.5-397B-A17B-GGUF`.
        repo: String,
        /// Exact filename (or first-shard filename) inside the repo.
        /// Omit to select interactively.
        model: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// HuggingFace model selection
// ---------------------------------------------------------------------------

/// Preferred quantization keywords, ordered best→worst.
const QUANT_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q4_0",
    "Q5_K_M", "Q5_K_S", "Q5_0",
    "Q3_K_M", "Q3_K_S",
    "Q8_0",   "Q6_K",
    "Q2_K",   "IQ4_XS", "IQ3_M",
];

/// A logical model choice: one or more GGUF files that together make a single
/// loadable model (i.e. all shards of one quantization).
#[derive(Debug)]
struct ModelGroup {
    /// Human-readable label shown in the menu (e.g. `Q4_K_M  [5 shards]`).
    label: String,
    /// All filenames belonging to this group, sorted.
    files: Vec<String>,
}

impl ModelGroup {
    /// Quant preference score: lower = better.  `usize::MAX` = unknown.
    fn preference_score(&self) -> usize {
        QUANT_PREFERENCE
            .iter()
            .position(|q| self.label.to_uppercase().contains(q))
            .unwrap_or(usize::MAX)
    }
}

/// Collect all `.gguf` filenames from the repo and group them by quantization.
///
/// Grouping rules (applied in order):
/// 1. Files inside a sub-directory → group by directory name.
/// 2. Files matching the shard pattern `…-NNNNN-of-MMMMM.gguf` → group by
///    the common prefix.
/// 3. Everything else → each file is its own group.
fn collect_groups(all_ggufs: Vec<String>) -> Vec<ModelGroup> {
    use std::collections::BTreeMap;

    // group_key → sorted list of filenames
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for path in all_ggufs {
        // Rule 1: sub-directory present
        let key = if let Some(slash) = path.find('/') {
            path[..slash].to_string()
        } else {
            // Rule 2: shard pattern  *-NNNNN-of-MMMMM.gguf
            // Strip the shard suffix to get the common key.
            let stem = path.trim_end_matches(".gguf");
            if let Some(of_pos) = stem.rfind("-of-") {
                // walk backwards from `-of-` past the digits before it
                let before_of = &stem[..of_pos];
                if let Some(dash) = before_of.rfind('-') {
                    let shard_num = &before_of[dash + 1..];
                    if shard_num.chars().all(|c| c.is_ascii_digit()) {
                        // prefix is everything before `-NNNNN`
                        before_of[..dash].to_string()
                    } else {
                        stem.to_string()
                    }
                } else {
                    stem.to_string()
                }
            } else {
                // Rule 3: plain single file
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

/// Ask the user to pick a group from `groups`.  Returns the chosen index.
/// When stdin is not a terminal (piped / CI) the best-scoring group is
/// returned without prompting.
fn prompt_user(groups: &[ModelGroup]) -> anyhow::Result<usize> {
    use std::io::{self, IsTerminal as _, Write};

    eprintln!("\nAvailable models in repo:");
    for (i, g) in groups.iter().enumerate() {
        eprintln!("  {:>2})  {}", i + 1, g.label);
    }

    if !io::stdin().is_terminal() {
        // Non-interactive: pick the best quant automatically.
        let best = groups
            .iter()
            .enumerate()
            .min_by_key(|(_, g)| g.preference_score())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("\nNon-interactive mode — auto-selected: {}", groups[best].label);
        return Ok(best);
    }

    loop {
        eprint!("\nSelect a model [1–{}]: ", groups.len());
        io::stderr().flush().ok();

        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        let trimmed = line.trim();

        match trimmed.parse::<usize>() {
            Ok(n) if n >= 1 && n <= groups.len() => return Ok(n - 1),
            _ => eprintln!("  Please enter a number between 1 and {}.", groups.len()),
        }
    }
}

/// Resolve a HuggingFace repo to a local model path.
///
/// * If `model` ends with `.gguf` it is treated as an exact filename and
///   downloaded directly.
/// * If `model` is a bare name (no `.gguf` extension) it is matched against
///   the group labels (directory name or quant prefix) — all shards of that
///   group are downloaded.
/// * If `model` is `None` and the repo has exactly one group it is
///   auto-selected; otherwise the user is prompted interactively.
///
/// For sharded models every shard is downloaded before returning the path to
/// the first one (llama.cpp discovers the rest via the naming convention).
fn resolve_hf(api: &Api, repo: &str, model: Option<String>) -> anyhow::Result<PathBuf> {
    let api_repo = api.model(repo.to_string());

    // Exact filename (ends with .gguf): download immediately, no listing needed.
    if let Some(ref filename) = model {
        if filename.ends_with(".gguf") {
            return api_repo
                .get(filename)
                .with_context(|| format!("failed to download '{filename}' from '{repo}'"));
        }
    }

    // Fetch repo metadata.
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

    // If the caller supplied a bare name (e.g. "Q4_K_M"), match it against
    // group keys/labels and select that group without prompting.
    let chosen_idx = if let Some(filter) = model {
        let filter_up = filter.to_uppercase();
        groups
            .iter()
            .position(|g| {
                // Match against the group label (directory name or quant prefix).
                // The label has the form "Q4_K_M  [6 shards]" or just "Q4_K_M".
                let label_key = g.label.split_whitespace().next().unwrap_or(&g.label);
                label_key.to_uppercase() == filter_up
                    || label_key.to_uppercase().contains(&filter_up)
            })
            .with_context(|| {
                let available: Vec<_> = groups
                    .iter()
                    .map(|g| g.label.split_whitespace().next().unwrap_or(&g.label).to_string())
                    .collect();
                format!(
                    "no group matching '{filter}' found in '{repo}'.\nAvailable: {}",
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

    // Download all shards (hf-hub caches them; already-cached files are instant).
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
// Shared application state
// ---------------------------------------------------------------------------

struct AppState {
    backend: LlamaBackend,
    model: LlamaModel,
    /// The model's built-in Jinja chat template, if any.
    chat_template: Option<String>,
    model_name: String,
    default_ctx_size: Option<NonZeroU32>,
}

// ---------------------------------------------------------------------------
// HTTP error helpers
// ---------------------------------------------------------------------------

struct HttpError {
    status: StatusCode,
    message: String,
}

fn bad_request(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::BAD_REQUEST,
        message: msg.into(),
    }
}

fn internal_error(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: msg.into(),
    }
}

fn error_response(err: HttpError) -> HttpResponse {
    let body = json!({
        "error": {
            "message": err.message,
            "type": "invalid_request_error",
            "code": err.status.as_u16()
        }
    })
    .to_string();
    HttpResponse::build(err.status)
        .content_type("application/json")
        .body(body)
}

// ---------------------------------------------------------------------------
// Request parsing helpers
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

fn parse_messages(req: &Value) -> Result<Vec<LlamaChatMessage>, HttpError> {
    let arr = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_request("'messages' must be an array"))?;

    arr.iter()
        .map(|m| {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("each message must have a 'role' string"))?;
            // 'content' may be a string or an array of content parts; we flatten to string.
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| {
                        if p.get("type").and_then(Value::as_str) == Some("text") {
                            p.get("text").and_then(Value::as_str).map(str::to_owned)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(""),
                Some(Value::Null) | None => String::new(),
                _ => {
                    return Err(bad_request(
                        "message 'content' must be a string or array of content parts",
                    ))
                }
            };
            LlamaChatMessage::new(role.to_owned(), content)
                .map_err(|e| bad_request(format!("invalid message: {e}")))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Core inference
// ---------------------------------------------------------------------------

fn run_chat_completion(state: &AppState, body: &str) -> Result<String, HttpError> {
    // ── Parse request ────────────────────────────────────────────────────────
    let req: Value =
        serde_json::from_str(body).map_err(|e| bad_request(format!("invalid JSON: {e}")))?;

    if req.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return Err(bad_request("streaming is not yet supported"));
    }

    let messages = parse_messages(&req)?;

    // Optional per-request chat template override
    let template_override = match req.get("chat_template") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) | None => None,
        _ => return Err(bad_request("'chat_template' must be a string")),
    };
    let template = template_override.or_else(|| state.chat_template.clone());

    // Sampling parameters
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

    // Optional GBNF grammar
    let grammar = match req.get("grammar") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) | None => None,
        _ => return Err(bad_request("'grammar' must be a string")),
    };

    let stop_seqs = parse_stop_sequences(&req)?;

    // ── Apply chat template ──────────────────────────────────────────────────
    let prompt = state
        .model
        .apply_chat_template(template, messages, true)
        .map_err(|e| internal_error(format!("chat template error: {e}")))?;

    // ── Tokenize ─────────────────────────────────────────────────────────────
    let tokens = state
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| internal_error(format!("tokenisation failed: {e}")))?;

    let n_ctx = state
        .default_ctx_size
        .map_or(state.model.n_ctx_train(), NonZeroU32::get)
        .max(tokens.len() as u32 + max_tokens);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);

    let mut ctx = state
        .model
        .new_context(&state.backend, ctx_params)
        .map_err(|e| internal_error(format!("context init failed: {e}")))?;

    // ── Prefill ──────────────────────────────────────────────────────────────
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last = tokens.len().saturating_sub(1) as i32;
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i as i32 == last)
            .map_err(|e| internal_error(format!("batch add failed: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| internal_error(format!("prefill decode failed: {e}")))?;

    // ── Build sampler chain ───────────────────────────────────────────────────
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if let Some(gbnf) = &grammar {
        chain.push(LlamaSampler::grammar(&state.model, gbnf, "root"));
    }
    if temperature > 0.0 {
        chain.push(LlamaSampler::temp(temperature));
        if top_k > 0 {
            chain.push(LlamaSampler::top_k(top_k));
        }
        if top_p < 1.0 {
            chain.push(LlamaSampler::top_p(top_p, 1));
        }
        chain.push(LlamaSampler::dist(seed));
    } else {
        chain.push(LlamaSampler::greedy());
    }
    let sampler = LlamaSampler::chain_simple(chain);

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut n_cur = batch.n_tokens();
    let max_pos = n_cur + max_tokens as i32;
    let mut generated = String::new();
    let mut completion_tokens: u32 = 0;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut finish_reason = "stop";

    'decode: while n_cur < max_pos {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if state.model.is_eog_token(token) {
            break;
        }

        let bytes = state
            .model
            .token_to_bytes(token, Special::Plaintext)
            .map_err(|e| internal_error(format!("token_to_bytes failed: {e}")))?;
        let mut piece = String::with_capacity(8);
        let _ = decoder.decode_to_string(&bytes, &mut piece, false);
        generated.push_str(&piece);
        completion_tokens += 1;

        // Check stop sequences
        for stop in &stop_seqs {
            if !stop.is_empty() && generated.ends_with(stop.as_str()) {
                // Trim the stop token from the output
                let trim_to = generated.len().saturating_sub(stop.len());
                generated.truncate(trim_to);
                break 'decode;
            }
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| internal_error(format!("batch add failed: {e}")))?;
        n_cur += 1;

        ctx.decode(&mut batch)
            .map_err(|e| internal_error(format!("decode failed: {e}")))?;
    }

    if n_cur >= max_pos {
        finish_reason = "length";
    }

    // ── Build OpenAI response ────────────────────────────────────────────────
    let model_name = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(state.model_name.as_str());

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| internal_error(format!("system time error: {e}")))?
        .as_secs();

    let prompt_tokens = tokens.len() as u32;
    let total_tokens = prompt_tokens + completion_tokens;

    Ok(json!({
        "id": format!("chatcmpl-{created}"),
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    })
    .to_string())
}

// ---------------------------------------------------------------------------
// Actix handlers
// ---------------------------------------------------------------------------

async fn chat_completions(state: web::Data<AppState>, body: web::Bytes) -> HttpResponse {
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s,
        Err(_) => return error_response(bad_request("request body must be valid UTF-8")),
    };
    match run_chat_completion(&state, text) {
        Ok(body) => HttpResponse::Ok()
            .content_type("application/json")
            .body(body),
        Err(err) => error_response(err),
    }
}

async fn list_models(state: web::Data<AppState>) -> HttpResponse {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    HttpResponse::Ok()
        .content_type("application/json")
        .body(
            json!({
                "object": "list",
                "data": [{
                    "id": state.model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "llama.cpp"
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

    // Try to fetch the model's built-in chat template
    let chat_template = model.get_chat_template(4096).ok();
    if chat_template.is_some() {
        tracing::info!("Loaded built-in chat template from model");
    } else {
        tracing::warn!(
            "Model has no built-in chat template — requests must supply 'chat_template'"
        );
    }

    let state = web::Data::new(AppState {
        backend,
        model,
        chat_template,
        model_name,
        default_ctx_size: args.ctx_size,
    });

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("OpenAI-compatible server listening on http://{addr}");

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(
                web::JsonConfig::default()
                    .error_handler(|err, _req| {
                        let msg = format!("JSON parse error: {err}");
                        actix_web::error::InternalError::from_response(
                            err,
                            error_response(bad_request(msg)),
                        )
                        .into()
                    }),
            )
            .route("/health", web::get().to(health))
            .route("/v1/models", web::get().to(list_models))
            .route(
                "/v1/chat/completions",
                web::post().to(chat_completions),
            )
    })
    .bind(&addr)?
    .run()
    .await
}
