//! End-to-end integration tests for the openai-server binary.
//!
//! These tests spawn the real server process and exercise every endpoint
//! over HTTP using `reqwest`.
//!
//! # Running
//!
//! ```bash
//! # Unit tests only (no model required)
//! cargo test -p openai-server
//!
//! # Integration tests (requires a GGUF model)
//! LLAMA_TEST_MODEL=/path/to/model.gguf \
//!     cargo test -p openai-server --test integration -- --nocapture
//!
//! # Integration tests with API key auth enabled
//! LLAMA_TEST_MODEL=/path/to/model.gguf LLAMA_TEST_API_KEY=secret \
//!     cargo test -p openai-server --test integration -- --nocapture
//! ```
//!
//! Tests automatically skip (pass) when `LLAMA_TEST_MODEL` is not set.
//!
//! # Server binary
//!
//! The test looks for the binary in `target/debug/openai-server` or
//! `target/release/openai-server` (whichever is newer). If neither exists it
//! runs `cargo build -p openai-server` first.

use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

const TEST_PORT: u16 = 18_080;
const TEST_AUTH_PORT: u16 = 18_081;
const TEST_API_KEY: &str = "integration_test_key";
const STARTUP_TIMEOUT_SECS: u64 = 120;

static MODEL_PATH: OnceLock<Option<String>> = OnceLock::new();

/// Resolve the test model path from env vars, downloading from HF if needed.
///
/// Priority:
/// 1. `LLAMA_TEST_MODEL`  — explicit local path
/// 2. `LLAMA_TEST_HF_REPO` + optional `LLAMA_TEST_HF_QUANT`
///    — download via the server binary's own HF logic (`--print-path`)
/// 3. `None` — tests skip (all model-dependent tests pass trivially)
fn model_path() -> Option<String> {
    MODEL_PATH.get_or_init(resolve_model_path).clone()
}

fn resolve_model_path() -> Option<String> {
    // Fast path: explicit local file.
    if let Ok(p) = std::env::var("LLAMA_TEST_MODEL") {
        if !p.is_empty() {
            return Some(p);
        }
    }

    // HF repo: use the server binary to download and get the cache path.
    let repo = std::env::var("LLAMA_TEST_HF_REPO").ok()?;
    let quant = std::env::var("LLAMA_TEST_HF_QUANT").ok();

    eprintln!(
        "[testbench] Downloading from HF: {repo} {}",
        quant.as_deref().unwrap_or("(interactive)")
    );

    let bin = server_binary();
    let mut cmd = Command::new(&bin);
    cmd.arg("--print-path").arg("hf-model").arg(&repo);
    if let Some(ref q) = quant {
        cmd.arg(q);
    }
    // Show download progress on stderr; capture stdout for the path.
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());

    let output = cmd.output().ok()?;
    if !output.status.success() {
        eprintln!("[testbench] HF download failed");
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if path.is_empty() {
        return None;
    }
    eprintln!("[testbench] Model cached at: {path}");
    Some(path)
}

/// Locate the server binary, building it if it doesn't exist yet.
fn server_binary() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = Path::new(manifest)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let debug = root.join("target/debug/openai-server");
    let release = root.join("target/release/openai-server");

    // Prefer whichever is newer (avoids stale releases).
    let candidates = [debug.clone(), release.clone()];
    let best = candidates
        .iter()
        .filter(|p| p.exists())
        .max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok());

    if let Some(bin) = best {
        return bin.clone();
    }

    eprintln!("[testbench] Binary not found — running `cargo build -p openai-server`…");
    let status = Command::new("cargo")
        .args(["build", "-p", "openai-server"])
        .current_dir(&root)
        .status()
        .expect("cargo build failed to launch");
    assert!(status.success(), "cargo build -p openai-server failed");
    debug
}

/// Spawn the server and wait until `/health` responds.
fn start_server(model: &str, port: u16, api_key: Option<&str>) -> Child {
    let bin = server_binary();
    let mut cmd = Command::new(&bin);
    cmd.arg("--port").arg(port.to_string());
    if let Some(key) = api_key {
        cmd.arg("--api-key").arg(key);
    }
    cmd.args(["local", model]);
    cmd.stdout(Stdio::null()).stderr(Stdio::null());

    let child = cmd.spawn().expect("failed to spawn openai-server");

    let base = format!("http://127.0.0.1:{port}");
    let client = Client::new();
    let deadline = Instant::now() + Duration::from_secs(STARTUP_TIMEOUT_SECS);

    loop {
        if Instant::now() > deadline {
            panic!(
                "[testbench] Server on port {port} did not become ready within \
                 {STARTUP_TIMEOUT_SECS}s"
            );
        }
        if let Ok(r) = client.get(format!("{base}/health")).send() {
            if r.status().is_success() {
                eprintln!("[testbench] Server ready on {base}");
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    child
}

/// Global server instance (no auth).
static SERVER: OnceLock<String> = OnceLock::new();

/// Global server instance (with API key auth).
static AUTH_SERVER: OnceLock<String> = OnceLock::new();

fn server_url() -> Option<String> {
    let model = model_path()?;
    let url = SERVER.get_or_init(|| {
        // Spawn; we intentionally leak the Child — the OS cleans it up when
        // the test binary exits.
        std::mem::forget(start_server(&model, TEST_PORT, None));
        format!("http://127.0.0.1:{TEST_PORT}")
    });
    Some(url.clone())
}

fn auth_server_url() -> Option<String> {
    let model = model_path()?;
    let url = AUTH_SERVER.get_or_init(|| {
        std::mem::forget(start_server(&model, TEST_AUTH_PORT, Some(TEST_API_KEY)));
        format!("http://127.0.0.1:{TEST_AUTH_PORT}")
    });
    Some(url.clone())
}

/// Convenience: GET `path`, assert 200, return parsed JSON.
fn get_json(base: &str, path: &str) -> Value {
    let url = format!("{base}{path}");
    let resp = Client::new()
        .get(&url)
        .send()
        .unwrap_or_else(|e| panic!("GET {url} failed: {e}"));
    assert_eq!(resp.status().as_u16(), 200, "GET {url} → unexpected status");
    resp.json()
        .unwrap_or_else(|e| panic!("GET {url}: bad JSON body: {e}"))
}

/// Convenience: POST JSON body, assert expected status, return parsed JSON.
fn post_json(base: &str, path: &str, body: Value, expected_status: u16) -> Value {
    let url = format!("{base}{path}");
    let resp = Client::new()
        .post(&url)
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("POST {url} failed: {e}"));
    assert_eq!(
        resp.status().as_u16(),
        expected_status,
        "POST {url} → unexpected status (body: {})",
        resp.text().unwrap_or_default()
    );
    if expected_status == 200 {
        resp.json()
            .unwrap_or_else(|e| panic!("POST {url}: bad JSON body: {e}"))
    } else {
        Value::Null
    }
}

/// Convenience: POST with auth header.
fn post_json_auth(base: &str, path: &str, body: Value, key: &str, expected_status: u16) -> Value {
    let url = format!("{base}{path}");
    let resp = Client::new()
        .post(&url)
        .bearer_auth(key)
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("POST {url} failed: {e}"));
    assert_eq!(
        resp.status().as_u16(),
        expected_status,
        "POST {url} → unexpected status"
    );
    if expected_status == 200 {
        resp.json().unwrap_or_else(|e| panic!("bad JSON: {e}"))
    } else {
        Value::Null
    }
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

#[test]
fn health_check() {
    let Some(base) = server_url() else { return };
    let body = get_json(&base, "/health");
    assert_eq!(body["status"], "ok");
    eprintln!("[✓] GET /health");
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

#[test]
fn list_models() {
    let Some(base) = server_url() else { return };
    let body = get_json(&base, "/v1/models");
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data must be array");
    assert!(!data.is_empty(), "at least one model must be listed");
    let m = &data[0];
    assert!(m["id"].is_string(), "model.id must be string");
    assert!(m["context_length"].as_u64().unwrap_or(0) > 0);
    eprintln!("[✓] GET /v1/models → {}", m["id"]);
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — non-streaming
// ---------------------------------------------------------------------------

#[test]
fn chat_completion_non_streaming() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [{"role":"user","content":"Reply with a single word: hello"}],
            "max_tokens": 16,
            "temperature": 0
        }),
        200,
    );
    assert_eq!(body["object"], "chat.completion");
    let choices = body["choices"].as_array().expect("choices array");
    assert!(!choices.is_empty());
    let msg = &choices[0]["message"];
    assert_eq!(msg["role"], "assistant");
    assert!(msg["content"].is_string());
    let finish = choices[0]["finish_reason"].as_str().unwrap_or("");
    assert!(
        finish == "stop" || finish == "length",
        "unexpected finish_reason: {finish}"
    );
    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].as_u64().is_some());
    assert!(usage["completion_tokens"].as_u64().unwrap_or(0) > 0);
    eprintln!("[✓] POST /v1/chat/completions (non-streaming)");
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — streaming (SSE)
// ---------------------------------------------------------------------------

#[test]
fn chat_completion_streaming() {
    let Some(base) = server_url() else { return };

    let url = format!("{base}/v1/chat/completions");
    let resp = Client::new()
        .post(&url)
        .json(&json!({
            "messages": [{"role":"user","content":"Say: hi"}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": true
        }))
        .send()
        .expect("request failed");

    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "text/event-stream"
    );

    let text = resp.text().expect("body read failed");
    assert!(text.contains("data: "), "SSE must contain data: lines");
    assert!(text.contains("data: [DONE]"), "SSE must end with [DONE]");

    // Every data line (except [DONE]) must be valid JSON with the right shape.
    let mut saw_content = false;
    let mut saw_finish = false;
    for line in text.lines() {
        let Some(raw) = line.strip_prefix("data: ") else {
            continue;
        };
        if raw == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(raw)
            .unwrap_or_else(|e| panic!("invalid SSE JSON: {e}\nline: {raw}"));
        assert_eq!(chunk["object"], "chat.completion.chunk");
        let choices = chunk["choices"].as_array().expect("choices array");
        assert!(!choices.is_empty());
        if let Some(content) = choices[0]["delta"]["content"].as_str() {
            if !content.is_empty() {
                saw_content = true;
            }
        }
        if choices[0]["finish_reason"].is_string() {
            saw_finish = true;
        }
    }
    assert!(saw_content, "no content delta received");
    assert!(saw_finish, "no finish_reason chunk received");
    eprintln!("[✓] POST /v1/chat/completions (streaming SSE)");
}

// ---------------------------------------------------------------------------
// finish_reason = "length"
// ---------------------------------------------------------------------------

#[test]
fn chat_finish_reason_length() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [{"role":"user","content":"Tell me a very long story."}],
            "max_tokens": 3,
            "temperature": 0
        }),
        200,
    );
    let finish = body["choices"][0]["finish_reason"]
        .as_str()
        .expect("finish_reason");
    assert_eq!(finish, "length", "expected length, got {finish}");
    eprintln!("[✓] finish_reason=length");
}

// ---------------------------------------------------------------------------
// Stop sequences
// ---------------------------------------------------------------------------

#[test]
fn chat_stop_sequence() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [{"role":"user","content":"Count: one two three four five"}],
            "max_tokens": 64,
            "temperature": 0,
            "stop": ["three"]
        }),
        200,
    );
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.contains("three"),
        "output should not contain the stop sequence, got: {content:?}"
    );
    eprintln!("[✓] stop sequences");
}

// ---------------------------------------------------------------------------
// POST /v1/completions — non-streaming
// ---------------------------------------------------------------------------

#[test]
fn raw_completion_non_streaming() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/completions",
        json!({
            "prompt": "The colour of the sky is",
            "max_tokens": 16,
            "temperature": 0
        }),
        200,
    );
    assert_eq!(body["object"], "text_completion");
    let choices = body["choices"].as_array().expect("choices");
    assert!(!choices.is_empty());
    assert!(choices[0]["text"].is_string());
    let finish = choices[0]["finish_reason"].as_str().unwrap_or("");
    assert!(finish == "stop" || finish == "length");
    eprintln!("[✓] POST /v1/completions (non-streaming)");
}

// ---------------------------------------------------------------------------
// POST /v1/completions — streaming
// ---------------------------------------------------------------------------

#[test]
fn raw_completion_streaming() {
    let Some(base) = server_url() else { return };
    let url = format!("{base}/v1/completions");
    let resp = Client::new()
        .post(&url)
        .json(&json!({
            "prompt": "Hello",
            "max_tokens": 16,
            "temperature": 0,
            "stream": true
        }))
        .send()
        .expect("request failed");

    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().unwrap();
    assert!(text.contains("data: "));
    assert!(text.contains("data: [DONE]"));

    for line in text.lines() {
        let Some(raw) = line.strip_prefix("data: ") else {
            continue;
        };
        if raw == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(raw).expect("valid JSON chunk");
        assert_eq!(chunk["object"], "text_completion");
    }
    eprintln!("[✓] POST /v1/completions (streaming SSE)");
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings
// ---------------------------------------------------------------------------

#[test]
fn embeddings_single() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/embeddings",
        json!({"input": "Hello, world!"}),
        200,
    );
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let vec = data[0]["embedding"].as_array().expect("embedding array");
    assert!(!vec.is_empty(), "embedding must be non-empty");

    // L2 norm of a normalised vector must be ≈ 1.
    let norm: f64 = vec
        .iter()
        .map(|v| {
            let x = v.as_f64().expect("float");
            x * x
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "embedding must be L2-normalised (norm={norm})"
    );
    eprintln!(
        "[✓] POST /v1/embeddings (single, dim={}, norm≈{norm:.4})",
        vec.len()
    );
}

#[test]
fn embeddings_batch() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/embeddings",
        json!({"input": ["The cat sat on the mat.", "A dog ran in the park."]}),
        200,
    );
    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "batch of 2 must return 2 embeddings");
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);

    // Both vectors must have the same dimension.
    let dim0 = data[0]["embedding"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let dim1 = data[1]["embedding"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(dim0, dim1, "all embeddings must have the same dimension");
    assert!(dim0 > 0);
    eprintln!("[✓] POST /v1/embeddings (batch, dim={dim0})");
}

#[test]
fn embeddings_self_similarity() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/embeddings",
        json!({"input": ["Paris is the capital of France.", "Paris is the capital of France."]}),
        200,
    );
    let data = body["data"].as_array().expect("data");
    let v0: Vec<f64> = data[0]["embedding"]
        .as_array()
        .expect("vec")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let v1: Vec<f64> = data[1]["embedding"]
        .as_array()
        .expect("vec")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let dot: f64 = v0.iter().zip(&v1).map(|(a, b)| a * b).sum();
    assert!(dot > 0.99, "self-similarity must be ≈ 1.0, got {dot:.6}");
    eprintln!("[✓] embeddings self-similarity = {dot:.6}");
}

// ---------------------------------------------------------------------------
// Tool calling
// ---------------------------------------------------------------------------

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    })
}

/// `tool_choice: required` — model MUST call a tool (GBNF grammar enforced).
#[test]
fn tool_calling_required() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [{"role":"user","content":"What is the weather in Tokyo?"}],
            "tools": [weather_tool()],
            "tool_choice": "required",
            "max_tokens": 256,
            "temperature": 0
        }),
        200,
    );
    let choice = &body["choices"][0];
    assert_eq!(
        choice["finish_reason"], "tool_calls",
        "finish_reason must be tool_calls: {body}"
    );
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert!(!calls.is_empty(), "at least one tool call expected");
    let call = &calls[0];
    assert_eq!(call["type"], "function");
    assert!(call["id"].is_string());
    assert_eq!(call["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(
        call["function"]["arguments"]
            .as_str()
            .expect("arguments string"),
    )
    .expect("arguments must be valid JSON");
    assert!(
        args["city"].is_string(),
        "tool call must include 'city' argument: {args}"
    );
    eprintln!(
        "[✓] tool_choice=required → tool_calls[{}] city={:?}",
        calls.len(),
        args["city"]
    );
}

/// `tool_choice: none` — tools listed but model must NOT call any.
#[test]
fn tool_calling_none() {
    let Some(base) = server_url() else { return };
    let body = post_json(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [{"role":"user","content":"What is 2 + 2?"}],
            "tools": [weather_tool()],
            "tool_choice": "none",
            "max_tokens": 64,
            "temperature": 0
        }),
        200,
    );
    let choice = &body["choices"][0];
    // finish_reason must not be tool_calls
    let finish = choice["finish_reason"].as_str().unwrap_or("");
    assert_ne!(
        finish, "tool_calls",
        "tool_choice=none must not produce tool_calls"
    );
    // tool_calls field must be absent or null
    assert!(
        choice["message"]["tool_calls"].is_null()
            || choice["message"]["tool_calls"]
                .as_array()
                .map_or(false, |a| a.is_empty()),
        "tool_calls must be absent when tool_choice=none: {body}"
    );
    eprintln!("[✓] tool_choice=none → plain text response");
}

/// Multi-turn: send a tool result back and get a final answer.
#[test]
fn tool_calling_multi_turn() {
    let Some(base) = server_url() else { return };

    // Step 1 — get the tool call.
    let first = post_json(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [{"role":"user","content":"What is the weather in Paris?"}],
            "tools": [weather_tool()],
            "tool_choice": "required",
            "max_tokens": 256,
            "temperature": 0
        }),
        200,
    );
    let call = &first["choices"][0]["message"]["tool_calls"][0];
    let call_id = call["id"].as_str().expect("call id").to_owned();
    let call_name = call["function"]["name"]
        .as_str()
        .expect("call name")
        .to_owned();
    let call_args = call["function"]["arguments"]
        .as_str()
        .expect("args string")
        .to_owned();

    eprintln!("  step 1: {call_name}({call_args}) id={call_id}");

    // Step 2 — send the tool result back.
    let second = post_json(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [
                {"role":"user","content":"What is the weather in Paris?"},
                {
                    "role":"assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call_name, "arguments": call_args}
                    }]
                },
                {
                    "role":"tool",
                    "content": "{\"temp\":18,\"condition\":\"sunny\"}",
                    "tool_call_id": call_id
                }
            ],
            "tools": [weather_tool()],
            "tool_choice": "none",
            "max_tokens": 128,
            "temperature": 0.3
        }),
        200,
    );
    let finish = second["choices"][0]["finish_reason"].as_str().unwrap_or("");
    assert!(
        finish == "stop" || finish == "length",
        "step 2 must be a regular text response, got finish={finish}"
    );
    let content = second["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "step 2 must produce text content");
    eprintln!("[✓] multi-turn tool calling → final answer: {content:.60?}");
}

// ---------------------------------------------------------------------------
// API key authentication
// ---------------------------------------------------------------------------

#[test]
fn auth_health_is_unprotected() {
    let Some(base) = auth_server_url() else {
        return;
    };
    // /health must return 200 even without a key.
    let resp = Client::new()
        .get(format!("{base}/health"))
        .send()
        .expect("request failed");
    assert_eq!(resp.status().as_u16(), 200);
    eprintln!("[✓] /health is unprotected");
}

#[test]
fn auth_missing_key_returns_401() {
    let Some(base) = auth_server_url() else {
        return;
    };
    let resp = Client::new()
        .get(format!("{base}/v1/models"))
        .send()
        .expect("request failed");
    assert_eq!(resp.status().as_u16(), 401, "missing key must return 401");
    let body: Value = resp.json().unwrap_or_default();
    assert_eq!(body["error"]["type"], "authentication_error");
    eprintln!("[✓] missing key → 401");
}

#[test]
fn auth_wrong_key_returns_401() {
    let Some(base) = auth_server_url() else {
        return;
    };
    let resp = Client::new()
        .get(format!("{base}/v1/models"))
        .bearer_auth("totally_wrong_key")
        .send()
        .expect("request failed");
    assert_eq!(resp.status().as_u16(), 401, "wrong key must return 401");
    eprintln!("[✓] wrong key → 401");
}

#[test]
fn auth_correct_key_returns_200() {
    let Some(base) = auth_server_url() else {
        return;
    };
    let resp = Client::new()
        .get(format!("{base}/v1/models"))
        .bearer_auth(TEST_API_KEY)
        .send()
        .expect("request failed");
    assert_eq!(resp.status().as_u16(), 200, "correct key must return 200");
    eprintln!("[✓] correct key → 200");
}

#[test]
fn auth_chat_completion_with_key() {
    let Some(base) = auth_server_url() else {
        return;
    };
    let resp = post_json_auth(
        &base,
        "/v1/chat/completions",
        json!({
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 8,
            "temperature": 0
        }),
        TEST_API_KEY,
        200,
    );
    assert_eq!(resp["object"], "chat.completion");
    eprintln!("[✓] chat completion with correct key → 200");
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn invalid_json_returns_400() {
    let Some(base) = server_url() else { return };
    let url = format!("{base}/v1/chat/completions");
    let resp = Client::new()
        .post(&url)
        .header("Content-Type", "application/json")
        .body("{ this is not json }")
        .send()
        .expect("request failed");
    assert_eq!(resp.status().as_u16(), 400);
    eprintln!("[✓] invalid JSON → 400");
}

#[test]
fn missing_messages_returns_400() {
    let Some(base) = server_url() else { return };
    // Valid JSON but missing the `messages` field.
    let url = format!("{base}/v1/chat/completions");
    let resp = Client::new()
        .post(&url)
        .json(&json!({"max_tokens": 16}))
        .send()
        .expect("request failed");
    assert_eq!(resp.status().as_u16(), 400);
    let body: Value = resp.json().unwrap_or_default();
    assert!(body["error"]["message"].is_string());
    eprintln!("[✓] missing messages → 400");
}

#[test]
fn invalid_temperature_returns_400() {
    let Some(base) = server_url() else { return };
    let url = format!("{base}/v1/chat/completions");
    let resp = Client::new()
        .post(&url)
        .json(&json!({
            "messages": [{"role":"user","content":"hi"}],
            "temperature": -1.0
        }))
        .send()
        .expect("request failed");
    assert_eq!(resp.status().as_u16(), 400);
    eprintln!("[✓] negative temperature → 400");
}
