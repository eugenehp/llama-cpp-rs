//! `OpenAI`-compatible tool/function-calling support.
//!
//! # How it works
//!
//! Everything that used to be hand-rolled here — injecting a Hermes `<tools>`
//! block into the system prompt, then scanning the output for `<tool_call>`
//! markers — is now delegated to llama.cpp's own chat layer via
//! [`llama_cpp_4::chat`].
//!
//! That matters for three reasons the string-scraping approach could not solve:
//!
//! 1. **The tool-call syntax is per model family.** Hermes uses
//!    `<tool_call>`, Functionary uses `>>>name`, `DeepSeek` uses its own
//!    delimiters. Scraping for one marker silently fails on the others;
//!    llama.cpp picks the right parser from the template.
//! 2. **`tool_choice: "required"` is enforced by grammar, not by asking.** A
//!    prompt saying "you MUST call a tool" is advisory. A GBNF grammar makes
//!    any other output unrepresentable.
//! 3. **Lazy grammars keep reasoning intact.** Constraining from token zero
//!    stops a thinking model ever opening its `<think>` block — which is why
//!    grammar forcing was previously avoided altogether. A lazy grammar stays
//!    dormant until a trigger fires, so reasoning flows free and only the call
//!    itself is constrained.
//!
//! This module is now a thin adapter: it validates the `OpenAI` wire format,
//! hands JSON to the chat layer, and converts what comes back.

use llama_cpp_4::chat::{ChatApplyParams, ChatParams, ChatTemplates, ToolChoice as ChatToolChoice};
use serde_json::{json, Value};

use crate::{bad_request, HttpError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// What the caller asked the model to do with the tools.
///
/// This mirrors `OpenAI` rather than llama.cpp: `OpenAI` can name a specific
/// function, which `common_chat_tool_choice` cannot express. That case is
/// handled by narrowing the tool list to the named function and asking for
/// [`ChatToolChoice::Required`] — see [`ToolChoice::apply_to`].
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    /// Force a specific named function.
    Function(String),
}

impl ToolChoice {
    /// The llama.cpp choice this maps to.
    fn as_chat(&self) -> ChatToolChoice {
        match self {
            Self::None => ChatToolChoice::None,
            Self::Auto => ChatToolChoice::Auto,
            // A named function is "required", narrowed to one tool.
            Self::Required | Self::Function(_) => ChatToolChoice::Required,
        }
    }

    /// Narrow `tools` to the named function when this is
    /// [`ToolChoice::Function`], leaving it untouched otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the named function is not among the supplied tools —
    /// forcing a call to a tool the model was never told about would produce a
    /// grammar referencing nothing.
    fn narrow_tools(&self, tools: &[Value]) -> Result<Vec<Value>, HttpError> {
        let Self::Function(name) = self else {
            return Ok(tools.to_vec());
        };
        let matched: Vec<Value> = tools
            .iter()
            .filter(|t| tool_name(t) == Some(name.as_str()))
            .cloned()
            .collect();
        if matched.is_empty() {
            return Err(bad_request(format!(
                "tool_choice names '{name}' but it is not in 'tools'"
            )));
        }
        Ok(matched)
    }

    /// Fold this choice into a [`ChatApplyParams`] alongside `tools`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::narrow_tools`].
    pub fn apply_to(
        &self,
        params: ChatApplyParams,
        tools: &[Value],
    ) -> Result<ChatApplyParams, HttpError> {
        if tools.is_empty() || matches!(self, Self::None) {
            // Still tell the template about `none` so it can mention the tools
            // without offering them.
            let params = params.with_tool_choice(self.as_chat());
            return Ok(if tools.is_empty() {
                params
            } else {
                params.with_tools(Value::Array(tools.to_vec()).to_string())
            });
        }
        let narrowed = self.narrow_tools(tools)?;
        Ok(params
            .with_tools(Value::Array(narrowed).to_string())
            .with_tool_choice(self.as_chat()))
    }
}

/// A single tool call, in the `OpenAI` wire shape.
#[derive(Debug, Clone, PartialEq)]
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
    /// Serialise to the `OpenAI` wire format.
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

/// Name of a tool in either the `{"type":"function","function":{…}}` wrapper or
/// the bare shorthand.
fn tool_name(tool: &Value) -> Option<&str> {
    tool.pointer("/function/name")
        .or_else(|| tool.get("name"))
        .and_then(Value::as_str)
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

/// Extract and validate the `tools` array.
///
/// Validation is delegated to llama.cpp's own `common_chat_tools_parse_oaicompat`,
/// so a definition this server accepts is exactly one the template can render.
///
/// # Errors
///
/// Returns a 400 if `tools` is not an array, or if llama.cpp rejects a
/// definition.
pub fn parse_tools(req: &Value) -> Result<Vec<Value>, HttpError> {
    let tools = match req.get("tools") {
        None | Some(Value::Null) => return Ok(vec![]),
        Some(Value::Array(arr)) => arr.clone(),
        _ => return Err(bad_request("'tools' must be an array")),
    };
    if tools.is_empty() {
        return Ok(vec![]);
    }
    llama_cpp_4::chat::parse_tools_oaicompat(&Value::Array(tools.clone()).to_string())
        .map_err(|e| bad_request(format!("invalid tool definition: {e}")))?;
    Ok(tools)
}

/// Extract and validate `tool_choice`.
///
/// # Errors
///
/// Returns a 400 for an unknown string or a malformed object.
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
        _ => Err(bad_request(
            "'tool_choice' must be \"none\"/\"auto\"/\"required\" or an object",
        )),
    }
}

/// Extract the `messages` array and validate it through llama.cpp.
///
/// Returns the array as JSON text, ready for [`ChatTemplates::apply`]. The
/// shorthands `OpenAI` allows — typed content parts, legacy `function_call` —
/// are normalised by `common_chat_msgs_parse_oaicompat`, which is the same code
/// path upstream's own server uses.
///
/// # Errors
///
/// Returns a 400 if `messages` is missing, not an array, or rejected by
/// llama.cpp.
pub fn messages_json(req: &Value) -> Result<String, HttpError> {
    let arr = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_request("'messages' must be an array"))?;
    let json = Value::Array(arr.clone()).to_string();
    llama_cpp_4::chat::parse_messages_oaicompat(&json)
        .map_err(|e| bad_request(format!("invalid messages: {e}")))?;
    Ok(json)
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/// Render a request into a prompt plus its sampling constraints.
///
/// # Errors
///
/// Returns a 400 if the template cannot render the request.
pub fn build_chat_params(
    templates: &ChatTemplates,
    messages_json: &str,
    tools: &[Value],
    choice: &ToolChoice,
    json_schema: Option<&str>,
    grammar: Option<&str>,
) -> Result<ChatParams, HttpError> {
    let mut params = ChatApplyParams::new(messages_json);
    if let Some(schema) = json_schema {
        params = params.with_json_schema(schema);
    } else if let Some(gbnf) = grammar {
        // llama.cpp rejects a caller grammar combined with tools, so only pass
        // it through when there are none — the tool grammar wins otherwise.
        if tools.is_empty() || matches!(choice, ToolChoice::None) {
            params = params.with_grammar(gbnf);
        }
    }
    let params = choice.apply_to(params, tools)?;
    templates
        .apply(&params)
        .map_err(|e| bad_request(format!("chat template: {e}")))
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// The assistant message recovered from model output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedMessage {
    /// Plain assistant text, minus any tool calls or reasoning.
    pub content: String,
    /// Reasoning the model emitted before its answer, if the format splits it.
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Parse model output using the parser the template produced.
///
/// `chat` must be the [`ChatParams`] that produced the prompt: it carries the
/// format and the serialized PEG parser, which is what makes this work across
/// model families instead of only for Hermes.
///
/// Set `is_partial` while streaming so a truncated tail is tolerated.
///
/// # Errors
///
/// Returns a 500 if llama.cpp cannot parse the output.
pub fn parse_output(
    chat: &ChatParams,
    raw: &str,
    is_partial: bool,
) -> Result<ParsedMessage, HttpError> {
    let json = chat
        .parse(raw, is_partial)
        .map_err(|e| crate::internal_error(format!("chat parse: {e}")))?;
    let value: Value = serde_json::from_str(&json)
        .map_err(|e| crate::internal_error(format!("chat parse returned bad JSON: {e}")))?;
    Ok(parse_message_value(&value))
}

/// Convert llama.cpp's parsed-message JSON into [`ParsedMessage`].
///
/// Split out from [`parse_output`] so the shape can be tested without a model.
fn parse_message_value(value: &Value) -> ParsedMessage {
    let content = value
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let reasoning = value
        .get("reasoning_content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();

    let mut tool_calls = Vec::new();
    if let Some(Value::Array(calls)) = value.get("tool_calls") {
        for (i, call) in calls.iter().enumerate() {
            let Some(name) = call
                .pointer("/function/name")
                .or_else(|| call.get("name"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            let arguments = call
                .pointer("/function/arguments")
                .or_else(|| call.get("arguments"))
                .map_or_else(
                    || "{}".to_owned(),
                    |a| match a {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    },
                );
            // llama.cpp assigns ids when the model supplies them; otherwise
            // synthesize one that is stable for this response, so a client can
            // correlate its tool result without relying on wall-clock entropy.
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map_or_else(|| format!("call_{i}_{name}"), str::to_owned);
            tool_calls.push(ToolCall {
                id,
                call_type: "function",
                name: name.to_owned(),
                arguments,
            });
        }
    }

    ParsedMessage {
        content,
        reasoning,
        tool_calls,
    }
}

// ---------------------------------------------------------------------------
// Multimodal message rewriting
// ---------------------------------------------------------------------------

/// Where an image or audio file comes from inside a multimodal message.
#[cfg(feature = "mtmd")]
#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    /// `"image_url"` content part — a `data:` URI or an `http(s)://` URL.
    Url(String),
    /// `"image_file"` content part — a file ID returned by `POST /v1/files`.
    FileId(String),
}

/// Rewrite multimodal content parts into text carrying the mtmd media marker,
/// collecting the image sources in prompt order.
///
/// The chat layer takes `OpenAI` JSON directly, but it has no idea where this
/// server will splice image embeddings in — that is what the marker is for. So
/// the parts are collapsed to text *before* the JSON reaches the template, and
/// the returned sources line up one-to-one with the markers.
///
/// Returns the rewritten `messages` array as JSON text plus the sources.
///
/// # Errors
///
/// Returns a 400 if `messages` is malformed or a media part lacks its URL/id.
#[cfg(feature = "mtmd")]
pub fn rewrite_multimodal(
    req: &Value,
    media_marker: &str,
) -> Result<(String, Vec<ImageSource>), HttpError> {
    let arr = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_request("'messages' must be an array"))?;

    let mut out = Vec::with_capacity(arr.len());
    let mut sources: Vec<ImageSource> = Vec::new();

    for m in arr {
        let Some(Value::Array(parts)) = m.get("content") else {
            // Not a content-part message; leave it exactly as the client sent
            // it so llama.cpp's own normaliser sees the original shape.
            out.push(m.clone());
            continue;
        };

        let mut text = String::new();
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("text") => {
                    text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                }
                Some("image_url") => {
                    let url = part
                        .pointer("/image_url/url")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            bad_request("image_url part must have an 'image_url.url' field")
                        })?;
                    sources.push(ImageSource::Url(url.to_owned()));
                    text.push_str(media_marker);
                }
                Some("image_file") => {
                    let file_id = part
                        .pointer("/image_file/file_id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            bad_request("image_file part must have an 'image_file.file_id' field")
                        })?;
                    sources.push(ImageSource::FileId(file_id.to_owned()));
                    text.push_str(media_marker);
                }
                _ => {} // ignore unknown content part types
            }
        }

        let mut rewritten = m.clone();
        rewritten["content"] = Value::String(text);
        out.push(rewritten);
    }

    Ok((Value::Array(out).to_string(), sources))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        })
    }

    // ── parse_tools ─────────────────────────────────────────────────────────

    #[test]
    fn parse_tools_null() {
        assert!(parse_tools(&json!({})).unwrap().is_empty());
        assert!(parse_tools(&json!({"tools": null})).unwrap().is_empty());
        assert!(parse_tools(&json!({"tools": []})).unwrap().is_empty());
    }

    #[test]
    fn parse_tools_non_array_is_error() {
        assert!(parse_tools(&json!({"tools": "nope"})).is_err());
    }

    /// Validation is llama.cpp's, so a definition missing its name must be
    /// rejected here rather than blowing up inside the template.
    #[test]
    fn parse_tools_rejects_definition_without_a_name() {
        assert!(parse_tools(&json!({"tools": [{"type": "function", "function": {}}]})).is_err());
    }

    #[test]
    fn parse_tools_accepts_a_valid_definition() {
        let tools = parse_tools(&json!({"tools": [weather_tool()]})).unwrap();
        assert_eq!(tools.len(), 1);
    }

    // ── parse_tool_choice ───────────────────────────────────────────────────

    #[test]
    fn tool_choice_strings() {
        assert_eq!(
            parse_tool_choice(&json!({"tool_choice": "none"})).unwrap(),
            ToolChoice::None
        );
        assert_eq!(
            parse_tool_choice(&json!({"tool_choice": "auto"})).unwrap(),
            ToolChoice::Auto
        );
        assert_eq!(
            parse_tool_choice(&json!({"tool_choice": "required"})).unwrap(),
            ToolChoice::Required
        );
    }

    #[test]
    fn tool_choice_specific_function() {
        let c = parse_tool_choice(
            &json!({"tool_choice": {"type": "function", "function": {"name": "f"}}}),
        )
        .unwrap();
        assert_eq!(c, ToolChoice::Function("f".to_owned()));
    }

    #[test]
    fn tool_choice_missing_defaults_to_auto() {
        assert_eq!(parse_tool_choice(&json!({})).unwrap(), ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_unknown_string_is_error() {
        assert!(parse_tool_choice(&json!({"tool_choice": "sometimes"})).is_err());
    }

    #[test]
    fn tool_choice_object_without_function_type_is_error() {
        assert!(parse_tool_choice(&json!({"tool_choice": {"type": "magic"}})).is_err());
    }

    // ── narrowing for a named function ──────────────────────────────────────

    /// Naming a function must reduce the tool list to that one, so the grammar
    /// llama.cpp builds can only express a call to it.
    #[test]
    fn named_function_narrows_the_tool_list() {
        let tools = vec![
            weather_tool(),
            json!({"type":"function","function":{"name":"other","parameters":{}}}),
        ];
        let narrowed = ToolChoice::Function("get_weather".to_owned())
            .narrow_tools(&tools)
            .unwrap();
        assert_eq!(narrowed.len(), 1);
        assert_eq!(tool_name(&narrowed[0]), Some("get_weather"));
    }

    /// Forcing a tool the model was never given would build a grammar
    /// referencing nothing — reject it as a bad request instead.
    #[test]
    fn named_function_not_in_tools_is_error() {
        let tools = vec![weather_tool()];
        assert!(ToolChoice::Function("nope".to_owned())
            .narrow_tools(&tools)
            .is_err());
    }

    #[test]
    fn non_named_choices_leave_the_tool_list_alone() {
        let tools = vec![weather_tool(), json!({"name": "b", "parameters": {}})];
        for choice in [ToolChoice::Auto, ToolChoice::Required, ToolChoice::None] {
            assert_eq!(choice.narrow_tools(&tools).unwrap().len(), 2);
        }
    }

    #[test]
    fn named_function_maps_to_required() {
        assert_eq!(
            ToolChoice::Function("f".to_owned()).as_chat(),
            ChatToolChoice::Required
        );
    }

    #[test]
    fn tool_name_reads_both_wire_shapes() {
        assert_eq!(tool_name(&weather_tool()), Some("get_weather"));
        assert_eq!(tool_name(&json!({"name": "bare"})), Some("bare"));
        assert_eq!(tool_name(&json!({})), None);
    }

    // ── messages_json ───────────────────────────────────────────────────────

    #[test]
    fn messages_json_requires_an_array() {
        assert!(messages_json(&json!({})).is_err());
        assert!(messages_json(&json!({"messages": "hi"})).is_err());
    }

    #[test]
    fn messages_json_passes_a_simple_turn() {
        let json = messages_json(&json!({"messages": [{"role":"user","content":"hi"}]})).unwrap();
        assert!(json.contains("hi"));
    }

    // ── output parsing ──────────────────────────────────────────────────────

    #[test]
    fn parses_content_only_message() {
        let msg = parse_message_value(&json!({"role":"assistant","content":"Hello."}));
        assert_eq!(msg.content, "Hello.");
        assert!(msg.tool_calls.is_empty());
    }

    /// Arguments arrive as a JSON string from llama.cpp; a client expects that
    /// string verbatim, not a re-encoded object.
    #[test]
    fn parses_tool_call_with_string_arguments() {
        let msg = parse_message_value(&json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}
            }]
        }));
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].id, "call_abc");
        assert_eq!(msg.tool_calls[0].name, "get_weather");
        assert_eq!(msg.tool_calls[0].arguments, r#"{"city":"Tokyo"}"#);
    }

    /// Some formats hand back an arguments *object*; it must be re-serialised
    /// rather than dropped.
    #[test]
    fn parses_tool_call_with_object_arguments() {
        let msg = parse_message_value(&json!({
            "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]
        }));
        assert_eq!(msg.tool_calls[0].arguments, r#"{"x":1}"#);
    }

    /// Without an id from the model, a synthesized one must still be present
    /// and distinct per call — clients key their tool results on it.
    #[test]
    fn synthesizes_distinct_ids_when_the_model_gives_none() {
        let msg = parse_message_value(&json!({
            "tool_calls": [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": "{}"}}
            ]
        }));
        assert_eq!(msg.tool_calls.len(), 2);
        assert!(!msg.tool_calls[0].id.is_empty());
        assert_ne!(msg.tool_calls[0].id, msg.tool_calls[1].id);
    }

    #[test]
    fn parses_multiple_tool_calls() {
        let msg = parse_message_value(&json!({
            "tool_calls": [
                {"function": {"name": "a", "arguments": "{\"x\":1}"}},
                {"function": {"name": "b", "arguments": "{\"y\":2}"}}
            ]
        }));
        assert_eq!(msg.tool_calls.len(), 2);
        assert_eq!(msg.tool_calls[0].name, "a");
        assert_eq!(msg.tool_calls[1].name, "b");
    }

    /// Reasoning must come back separately so it can be surfaced (or dropped)
    /// without contaminating `content`.
    #[test]
    fn separates_reasoning_from_content() {
        let msg = parse_message_value(&json!({
            "content": "The answer.",
            "reasoning_content": "weighing it up"
        }));
        assert_eq!(msg.content, "The answer.");
        assert_eq!(msg.reasoning, "weighing it up");
    }

    /// A malformed entry must be skipped, not panic — this is model output.
    #[test]
    fn skips_tool_calls_without_a_name() {
        let msg = parse_message_value(&json!({
            "tool_calls": [{"function": {"arguments": "{}"}}, {"function": {"name": "ok"}}]
        }));
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "ok");
    }

    #[test]
    fn missing_arguments_default_to_empty_object() {
        let msg = parse_message_value(&json!({"tool_calls": [{"function": {"name": "f"}}]}));
        assert_eq!(msg.tool_calls[0].arguments, "{}");
    }

    #[test]
    fn tool_call_round_trips_to_the_wire_format() {
        let call = ToolCall {
            id: "call_1".to_owned(),
            call_type: "function",
            name: "f".to_owned(),
            arguments: r#"{"a":1}"#.to_owned(),
        };
        let v = call.to_value();
        assert_eq!(v["id"], "call_1");
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], "f");
        assert_eq!(v["function"]["arguments"], r#"{"a":1}"#);
    }

    // ── multimodal rewriting ────────────────────────────────────────────────

    #[cfg(feature = "mtmd")]
    #[test]
    fn rewrite_multimodal_replaces_image_parts_with_markers() {
        let req = json!({"messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is "},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "text", "text": "?"}
            ]
        }]});
        let (json_str, sources) = rewrite_multimodal(&req, "<__media__>").unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(
            sources[0],
            ImageSource::Url("data:image/png;base64,AAA".to_owned())
        );
        assert!(json_str.contains("What is <__media__>?"), "got {json_str}");
    }

    /// Sources must come back in prompt order, so they line up one-to-one with
    /// the markers spliced into the text.
    #[cfg(feature = "mtmd")]
    #[test]
    fn rewrite_multimodal_preserves_source_order() {
        let req = json!({"messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "first"}},
                {"type": "image_file", "image_file": {"file_id": "second"}}
            ]
        }]});
        let (_, sources) = rewrite_multimodal(&req, "M").unwrap();
        assert_eq!(
            sources,
            vec![
                ImageSource::Url("first".to_owned()),
                ImageSource::FileId("second".to_owned())
            ]
        );
    }

    /// Plain string messages must pass through untouched so llama.cpp's own
    /// normaliser sees the shape the client actually sent.
    #[cfg(feature = "mtmd")]
    #[test]
    fn rewrite_multimodal_leaves_plain_messages_alone() {
        let req = json!({"messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"}
        ]});
        let (json_str, sources) = rewrite_multimodal(&req, "M").unwrap();
        assert!(sources.is_empty());
        let parsed: Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed[0]["content"], "be nice");
        assert_eq!(parsed[1]["content"], "hi");
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn rewrite_multimodal_rejects_image_part_without_url() {
        let req = json!({"messages": [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {}}]
        }]});
        assert!(rewrite_multimodal(&req, "M").is_err());
    }

    /// An assistant turn carrying `tool_calls` must survive untouched — it is
    /// llama.cpp's job to render prior tool usage back into the prompt.
    #[cfg(feature = "mtmd")]
    #[test]
    fn rewrite_multimodal_preserves_assistant_tool_calls() {
        let req = json!({"messages": [{
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "f", "arguments": "{}"}}]
        }]});
        let (json_str, _) = rewrite_multimodal(&req, "M").unwrap();
        let parsed: Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed[0]["tool_calls"][0]["function"]["name"], "f");
    }
}
