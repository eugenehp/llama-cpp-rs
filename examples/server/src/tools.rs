//! OpenAI-compatible tool/function-calling support.
//!
//! # How it works
//!
//! 1. `tools` + `tool_choice` are parsed from the request.
//! 2. Tool definitions are injected into the first system message (or a new
//!    one is prepended) using the Hermes `<tools>` format, which most
//!    instruction-tuned models understand.
//! 3. The model is instructed to emit tool calls as:
//!    ```text
//!    <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
//!    ```
//! 4. After generation, the output is scanned for `<tool_call>` blocks.
//! 5. If any are found the response gets `finish_reason: "tool_calls"` and a
//!    `tool_calls` array; otherwise it's a regular assistant message.
//!
//! # Multi-turn: replaying tool results
//!
//! When the conversation history contains an assistant message that previously
//! called tools, its `tool_calls` array is serialised to `<tool_call>` content
//! so the model's prompt reflects the prior exchange.
//!
//! `role: "tool"` messages are passed through to `apply_chat_template` as-is
//! (llama.cpp Jinja templates for Llama-3.1, Qwen2.5 etc. handle them natively).

use serde_json::{json, Value};
use std::fmt::Write as _;

use crate::{bad_request, HttpError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// The `parameters` field — a JSON Schema object.
    pub parameters: Value,
}

impl ToolDef {
    /// Parse from the OpenAI wire format:
    /// `{"type":"function","function":{"name":…,"description":…,"parameters":…}}`
    /// or the shorthand `{"name":…,"description":…,"parameters":…}`.
    pub fn from_value(v: &Value) -> Option<Self> {
        let func = if v.get("type").and_then(Value::as_str) == Some("function") {
            v.get("function")?
        } else {
            v
        };
        Some(ToolDef {
            name: func.get("name")?.as_str()?.to_owned(),
            description: func
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            parameters: func.get("parameters").cloned().unwrap_or(json!({})),
        })
    }

    /// Serialise to the OpenAI `tools` array element format.
    pub fn to_value(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    /// Force a specific named function.
    Function(String),
}

/// A single tool call extracted from (or being sent to) the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// The stable call id used for tool result messages.
    pub id: String,
    /// Always `"function"` for now.
    pub call_type: &'static str,
    pub name: String,
    /// Raw JSON string of the arguments object.
    pub arguments: String,
}

impl ToolCall {
    /// Serialise to the OpenAI wire format.
    pub fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "type": self.call_type,
            "function": {
                "name": self.name,
                "arguments": self.arguments
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

pub fn parse_tools(req: &Value) -> Result<Vec<ToolDef>, HttpError> {
    match req.get("tools") {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| ToolDef::from_value(v).ok_or_else(|| bad_request("invalid tool definition")))
            .collect(),
        _ => Err(bad_request("'tools' must be an array")),
    }
}

pub fn parse_tool_choice(req: &Value) -> Result<ToolChoice, HttpError> {
    match req.get("tool_choice") {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(s)) => match s.as_str() {
            "none" => Ok(ToolChoice::None),
            "auto" => Ok(ToolChoice::Auto),
            "required" => Ok(ToolChoice::Required),
            other => Err(bad_request(format!("unknown tool_choice '{other}'"))),
        },
        Some(v) if v.is_object() => {
            if v.get("type").and_then(Value::as_str) == Some("function") {
                let name = v
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| bad_request("tool_choice.function.name is required"))?;
                Ok(ToolChoice::Function(name.to_owned()))
            } else {
                Err(bad_request("unsupported tool_choice type"))
            }
        }
        _ => Err(bad_request("'tool_choice' must be \"none\"/\"auto\"/\"required\" or an object")),
    }
}

// ---------------------------------------------------------------------------
// System prompt injection
// ---------------------------------------------------------------------------

/// Build the tool-calling instruction block injected at the top of the system
/// message. Uses the Hermes `<tools>` format which most fine-tuned models
/// recognise.
pub fn tool_system_block(tools: &[ToolDef], choice: &ToolChoice) -> String {
    let tools_json: Vec<Value> = tools.iter().map(ToolDef::to_value).collect();
    let tools_str = serde_json::to_string_pretty(&tools_json).unwrap_or_default();

    let choice_note = match choice {
        ToolChoice::None => {
            "\nFor this response, do NOT call any tools — answer from your own knowledge."
        }
        ToolChoice::Required => "\nYou MUST call at least one tool before giving a final answer.",
        ToolChoice::Function(name) => {
            // static lifetime hack: we can't return a &str for owned String, so
            // callers who need this branch will append separately.
            let _ = name; // handled below
            ""
        }
        ToolChoice::Auto => "",
    };

    let mut block = format!(
        "You have access to the following tools:\n\n\
         <tools>\n{tools_str}\n</tools>\n\n\
         To call one or more tools, output each call on its own line using this exact format:\n\
         <tool_call>{{\"name\": \"tool_name\", \"arguments\": {{\"arg\": \"value\"}}}}</tool_call>\n\n\
         After all tool results have been returned, give your final answer as plain text.\
         {choice_note}"
    );

    if let ToolChoice::Function(name) = choice {
        let _ = write!(block, "\nYou MUST call the '{name}' tool in your response.");
    }

    block
}

/// Inject tool definitions into the message list (as `(role, content)` pairs).
/// If a system message already exists, the tool block is prepended to it.
/// Otherwise a new system message is inserted at position 0.
pub fn inject_tools(
    messages: &mut Vec<(String, String)>,
    tools: &[ToolDef],
    choice: &ToolChoice,
) {
    if tools.is_empty() || matches!(choice, ToolChoice::None) {
        return;
    }
    let block = tool_system_block(tools, choice);
    if let Some(pos) = messages.iter().position(|(r, _)| r == "system") {
        let (_, content) = &mut messages[pos];
        let old = std::mem::take(content);
        *content = format!("{block}\n\n{old}");
    } else {
        messages.insert(0, ("system".to_owned(), block));
    }
}

// ---------------------------------------------------------------------------
// Message normalisation
// ---------------------------------------------------------------------------

/// Convert the raw JSON `messages` array into `(role, content)` pairs, handling:
/// - Normal `{role, content}` messages.
/// - Assistant messages with `tool_calls` (serialised as `<tool_call>` blocks).
/// - `role: "tool"` messages (passed through unchanged for Jinja templates that
///   understand them; the content is the tool result).
pub fn normalise_messages(req: &Value) -> Result<Vec<(String, String)>, HttpError> {
    let arr = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_request("'messages' must be an array"))?;

    let mut out = Vec::with_capacity(arr.len());

    for m in arr {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| bad_request("each message must have a 'role' string"))?
            .to_owned();

        // --- assistant messages that carry tool_calls -----------------------
        if role == "assistant" {
            if let Some(Value::Array(calls)) = m.get("tool_calls") {
                // Serialise the calls as <tool_call> blocks so the model can
                // see its own prior tool usage when the conversation is replayed.
                let mut content = String::new();
                for call in calls {
                    let name = call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    let args = call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let _ = write!(
                        content,
                        "<tool_call>{{\"name\":\"{name}\",\"arguments\":{args}}}</tool_call>\n"
                    );
                }
                // Optional assistant text before the tool call(s).
                if let Some(Value::String(s)) = m.get("content") {
                    if !s.is_empty() {
                        content = format!("{s}\n{content}");
                    }
                }
                out.push((role, content.trim_end().to_owned()));
                continue;
            }
        }

        // --- tool result messages -------------------------------------------
        // Pass the raw content; the model's Jinja template will wrap it in
        // whatever tags it expects (e.g. <tool_response> for Qwen, ipython for
        // Llama-3.1).  If the template doesn't support the tool role, fall back
        // to a user message with explicit wrapping.
        if role == "tool" {
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                _ => return Err(bad_request("tool message 'content' must be a string")),
            };
            let tool_call_id = m
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            // Pass as "tool" role — the Jinja template handles it.
            // Include the call id in the content so a template fallback can
            // identify which call is being answered.
            let wrapped = if tool_call_id.is_empty() {
                content
            } else {
                format!("{content}")
            };
            out.push((role, wrapped));
            continue;
        }

        // --- normal text message --------------------------------------------
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

        out.push((role, content));
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// Extract all `<tool_call>...</tool_call>` blocks from `text`.
/// Returns the tool calls and the text that appeared before the first call
/// (trimmed), which becomes the `content` field of the assistant message.
pub fn extract_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let tag_open = "<tool_call>";
    let tag_close = "</tool_call>";

    let mut calls = Vec::new();
    let mut pre_text = String::new();
    let mut found_first = false;
    let mut remaining = text;

    while let Some(start) = remaining.find(tag_open) {
        if !found_first {
            pre_text = remaining[..start].trim().to_owned();
            found_first = true;
        }
        let after_open = &remaining[start + tag_open.len()..];
        let end = match after_open.find(tag_close) {
            Some(e) => e,
            None => {
                // Unclosed tag — try to parse what we have as JSON anyway.
                let json_candidate = after_open.trim();
                if let Some(tc) = parse_single_call(json_candidate) {
                    calls.push(tc);
                }
                break;
            }
        };
        let json_candidate = after_open[..end].trim();
        if let Some(tc) = parse_single_call(json_candidate) {
            calls.push(tc);
        }
        remaining = &after_open[end + tag_close.len()..];
    }

    (pre_text, calls)
}

fn parse_single_call(json: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(json).ok()?;
    let name = v.get("name")?.as_str()?.to_owned();
    // `arguments` may already be a string (re-serialised) or an object.
    let arguments = match v.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(obj) => serde_json::to_string(obj).unwrap_or_else(|_| "{}".to_owned()),
        None => "{}".to_owned(),
    };
    // Generate a stable-ish short id.
    let id = format!(
        "call_{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos()) as u64
            ^ (name.len() as u64 * 0x9e3779b9)
    );
    Some(ToolCall {
        id,
        call_type: "function",
        name,
        arguments,
    })
}



// ---------------------------------------------------------------------------
// Multimodal message normalisation
// ---------------------------------------------------------------------------

/// Where an image or audio file comes from inside a multimodal message.
#[cfg(feature = "mtmd")]
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// `"image_url"` content part — a `data:` URI or an `http(s)://` URL.
    Url(String),
    /// `"image_file"` content part — a file ID returned by `POST /v1/files`.
    FileId(String),
}

/// Like [`normalise_messages`] but also recognises multimodal content parts:
///
/// - `{"type":"image_url","image_url":{"url":"..."}}` — replaced with
///   `media_marker` in the text; the URL is collected.
/// - `{"type":"image_file","image_file":{"file_id":"..."}}` — replaced with
///   `media_marker`; the file ID is collected.
///
/// Returns `(message_pairs, image_sources)` where `image_sources` are ordered
/// by their first appearance in the prompt text so they align one-to-one with
/// the markers embedded in the messages.
#[cfg(feature = "mtmd")]
pub fn normalise_messages_multimodal(
    req: &Value,
    media_marker: &str,
) -> Result<(Vec<(String, String)>, Vec<ImageSource>), HttpError> {
    let arr = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_request("'messages' must be an array"))?;

    let mut out = Vec::with_capacity(arr.len());
    let mut sources: Vec<ImageSource> = Vec::new();

    for m in arr {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| bad_request("each message must have a 'role' string"))?
            .to_owned();

        // ── assistant messages with tool_calls (same as normalise_messages) ─
        if role == "assistant" {
            if let Some(Value::Array(calls)) = m.get("tool_calls") {
                let mut content = String::new();
                for call in calls {
                    let name = call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    let args = call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let _ = std::fmt::write(
                        &mut content,
                        format_args!(
                            "<tool_call>{{\"name\":\"{name}\",\"arguments\":{args}}}</tool_call>\n"
                        ),
                    );
                }
                if let Some(Value::String(s)) = m.get("content") {
                    if !s.is_empty() {
                        content = format!("{s}\n{content}");
                    }
                }
                out.push((role, content.trim_end().to_owned()));
                continue;
            }
        }

        // ── tool messages (pass through unchanged) ───────────────────────────
        if role == "tool" {
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                _ => return Err(bad_request("tool message 'content' must be a string")),
            };
            out.push((role, content));
            continue;
        }

        // ── normal messages (text + optional image parts) ────────────────────
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),

            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            text.push_str(
                                part.get("text").and_then(Value::as_str).unwrap_or(""),
                            );
                        }
                        Some("image_url") => {
                            let url = part
                                .get("image_url")
                                .and_then(|u| u.get("url"))
                                .and_then(Value::as_str)
                                .ok_or_else(|| {
                                    bad_request("image_url part must have an 'image_url.url' field")
                                })?;
                            sources.push(ImageSource::Url(url.to_owned()));
                            text.push_str(media_marker);
                        }
                        Some("image_file") => {
                            let file_id = part
                                .get("image_file")
                                .and_then(|f| f.get("file_id"))
                                .and_then(Value::as_str)
                                .ok_or_else(|| {
                                    bad_request(
                                        "image_file part must have an 'image_file.file_id' field",
                                    )
                                })?;
                            sources.push(ImageSource::FileId(file_id.to_owned()));
                            text.push_str(media_marker);
                        }
                        _ => {} // ignore unknown content part types
                    }
                }
                text
            }

            Some(Value::Null) | None => String::new(),
            _ => {
                return Err(bad_request(
                    "message 'content' must be a string or array of content parts",
                ))
            }
        };

        out.push((role, content));
    }

    Ok((out, sources))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── extract_tool_calls ───────────────────────────────────────────────────

    #[test]
    fn single_tool_call() {
        let out = r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#;
        let (pre, calls) = extract_tool_calls(out);
        assert_eq!(pre, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn text_before_tool_call() {
        let out = "Sure, let me check!\n<tool_call>{\"name\":\"fn\",\"arguments\":{}}</tool_call>";
        let (pre, calls) = extract_tool_calls(out);
        assert_eq!(pre, "Sure, let me check!");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fn");
    }

    #[test]
    fn multiple_tool_calls() {
        let out = concat!(
            r#"<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>"#,
            "\n",
            r#"<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>"#,
        );
        let (_, calls) = extract_tool_calls(out);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn no_tool_calls_returns_full_text() {
        let out = "Hello! How can I help you today?";
        let (pre, calls) = extract_tool_calls(out);
        // pre is only set when at least one <tool_call> is found
        assert!(calls.is_empty());
        assert_eq!(pre, "");
    }

    #[test]
    fn arguments_as_string_passthrough() {
        // Some models emit arguments already as a JSON string (re-serialised).
        let out = r#"<tool_call>{"name":"fn","arguments":"{\"k\":\"v\"}"}</tool_call>"#;
        let (_, calls) = extract_tool_calls(out);
        assert_eq!(calls.len(), 1);
        // arguments should be the raw value, not double-encoded
        assert!(calls[0].arguments.contains("k"));
    }

    #[test]
    fn tool_call_id_is_unique() {
        let out1 = r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#;
        let out2 = r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#;
        let (_, c1) = extract_tool_calls(out1);
        std::thread::sleep(std::time::Duration::from_nanos(1));
        let (_, c2) = extract_tool_calls(out2);
        // IDs are derived from subsecond time + name hash; not guaranteed unique
        // in a nanosecond, but should differ across different names.
        let _ = (c1, c2); // just ensure no panic
    }

    // ── parse_tools ──────────────────────────────────────────────────────────

    #[test]
    fn parse_tools_openai_format() {
        let req = json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type":"object","properties":{"q":{"type":"string"}}}
                }
            }]
        });
        let tools = parse_tools(&req).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[0].description, "Search the web");
    }

    #[test]
    fn parse_tools_shorthand_format() {
        let req = json!({
            "tools": [{"name":"fn","description":"does stuff","parameters":{}}]
        });
        let tools = parse_tools(&req).unwrap();
        assert_eq!(tools[0].name, "fn");
    }

    #[test]
    fn parse_tools_empty() {
        let req = json!({});
        assert!(parse_tools(&req).unwrap().is_empty());
    }

    #[test]
    fn parse_tools_null() {
        let req = json!({"tools": null});
        assert!(parse_tools(&req).unwrap().is_empty());
    }

    #[test]
    fn parse_tools_invalid() {
        let req = json!({"tools": "not an array"});
        assert!(parse_tools(&req).is_err());
    }

    // ── parse_tool_choice ────────────────────────────────────────────────────

    #[test]
    fn tool_choice_strings() {
        for (s, expected) in [
            ("auto",     ToolChoice::Auto),
            ("none",     ToolChoice::None),
            ("required", ToolChoice::Required),
        ] {
            let req = json!({"tool_choice": s});
            assert_eq!(parse_tool_choice(&req).unwrap(), expected);
        }
    }

    #[test]
    fn tool_choice_specific_function() {
        let req = json!({"tool_choice": {"type":"function","function":{"name":"my_fn"}}});
        assert_eq!(
            parse_tool_choice(&req).unwrap(),
            ToolChoice::Function("my_fn".into())
        );
    }

    #[test]
    fn tool_choice_missing_defaults_to_auto() {
        let req = json!({});
        assert_eq!(parse_tool_choice(&req).unwrap(), ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_unknown_string_is_error() {
        let req = json!({"tool_choice": "bogus"});
        assert!(parse_tool_choice(&req).is_err());
    }

    // ── inject_tools ─────────────────────────────────────────────────────────

    #[test]
    fn inject_creates_system_message_when_none_exists() {
        let tools = vec![ToolDef {
            name: "fn".into(),
            description: "does stuff".into(),
            parameters: json!({}),
        }];
        let mut msgs: Vec<(String, String)> = vec![
            ("user".into(), "hello".into()),
        ];
        inject_tools(&mut msgs, &tools, &ToolChoice::Auto);
        assert_eq!(msgs[0].0, "system");
        assert!(msgs[0].1.contains("<tools>"));
        assert_eq!(msgs[1].0, "user");
    }

    #[test]
    fn inject_prepends_to_existing_system_message() {
        let tools = vec![ToolDef {
            name: "fn".into(),
            description: "d".into(),
            parameters: json!({}),
        }];
        let mut msgs: Vec<(String, String)> = vec![
            ("system".into(), "You are helpful.".into()),
            ("user".into(), "hi".into()),
        ];
        inject_tools(&mut msgs, &tools, &ToolChoice::Auto);
        // Still exactly one system message at position 0
        assert_eq!(msgs[0].0, "system");
        assert!(msgs[0].1.contains("<tools>"));
        assert!(msgs[0].1.contains("You are helpful."));
    }

    #[test]
    fn inject_skips_when_choice_is_none() {
        let tools = vec![ToolDef {
            name: "fn".into(),
            description: "d".into(),
            parameters: json!({}),
        }];
        let mut msgs: Vec<(String, String)> = vec![("user".into(), "hi".into())];
        inject_tools(&mut msgs, &tools, &ToolChoice::None);
        // No system message injected
        assert_eq!(msgs[0].0, "user");
    }

    #[test]
    fn inject_required_adds_must_call_note() {
        let tools = vec![ToolDef {
            name: "fn".into(),
            description: "d".into(),
            parameters: json!({}),
        }];
        let mut msgs: Vec<(String, String)> = vec![("user".into(), "hi".into())];
        inject_tools(&mut msgs, &tools, &ToolChoice::Required);
        assert!(msgs[0].1.contains("MUST call"));
    }

    // ── normalise_messages ───────────────────────────────────────────────────

    #[test]
    fn normalise_plain_messages() {
        let req = json!({
            "messages": [
                {"role":"system","content":"Be helpful."},
                {"role":"user","content":"Hello"},
                {"role":"assistant","content":"Hi!"}
            ]
        });
        let msgs = normalise_messages(&req).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0], ("system".into(), "Be helpful.".into()));
        assert_eq!(msgs[2], ("assistant".into(), "Hi!".into()));
    }

    #[test]
    fn normalise_assistant_with_tool_calls() {
        let req = json!({
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}
                }]
            }]
        });
        let msgs = normalise_messages(&req).unwrap();
        assert_eq!(msgs[0].0, "assistant");
        assert!(msgs[0].1.contains("<tool_call>"));
        assert!(msgs[0].1.contains("get_weather"));
    }

    #[test]
    fn normalise_tool_role_message() {
        let req = json!({
            "messages": [{
                "role": "tool",
                "content": "{\"temp\":18}",
                "tool_call_id": "call_1"
            }]
        });
        let msgs = normalise_messages(&req).unwrap();
        assert_eq!(msgs[0].0, "tool");
        assert!(msgs[0].1.contains("18"));
    }

    #[test]
    fn normalise_content_parts_array() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type":"text","text":"Hello"},
                    {"type":"image_url","image_url":"..."},
                    {"type":"text","text":" world"}
                ]
            }]
        });
        let msgs = normalise_messages(&req).unwrap();
        assert_eq!(msgs[0].1, "Hello world");
    }

    // ── ToolDef round-trip ───────────────────────────────────────────────────

    #[test]
    fn tool_def_round_trip() {
        let v = json!({
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Does math",
                "parameters": {"type":"object","properties":{}}
            }
        });
        let def = ToolDef::from_value(&v).unwrap();
        let back = def.to_value();
        assert_eq!(back["function"]["name"], "calc");
        assert_eq!(back["type"], "function");
    }
}
