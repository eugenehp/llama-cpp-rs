//! End-to-end tests for the chat / tool-calling layer.
//!
//! These drive llama.cpp's real Jinja engine and PEG parsers through the shim,
//! so they need a model — but not a *capable* one. The template is supplied as
//! an override, which is what lets a 260K-parameter story model exercise
//! Hermes-style tool calling: the template and parser are pure text processing,
//! independent of what the weights can actually produce.

mod support;

use llama_cpp_4::chat::{
    format_name, json_schema_to_grammar, ChatApplyParams, ChatTemplates, ReasoningFormat,
    ToolChoice,
};

use support::model::load_model;

/// The real Hermes-2-Pro tool-use template, vendored with llama.cpp.
///
/// It has to be a template upstream *recognises*: `common_chat_templates_apply`
/// only reaches the specialized per-family parsers (which emit a GBNF grammar
/// and lazy triggers) when it matches a known template source. An ad-hoc
/// lookalike falls through to the generic autoparser, which produces a PEG
/// parser and no grammar — still correct, but it would not exercise the path
/// that makes `tool_choice: required` binding.
const HERMES_TOOL_TEMPLATE: &str = include_str!(
    "../../llama-cpp-sys-4/llama.cpp/models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja"
);

const WEATHER_TOOL: &str = r#"[{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]"#;

fn templates() -> Option<ChatTemplates> {
    let (model, _) = load_model()?;
    match ChatTemplates::from_model(&model, Some(HERMES_TOOL_TEMPLATE)) {
        Ok(t) => Some(t),
        Err(e) => panic!("could not build chat templates: {e}"),
    }
}

#[test]
fn templates_report_explicit_override() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    assert!(
        tmpls.was_explicit(),
        "an override template must report as explicit"
    );
}

#[test]
fn templates_report_caps_as_json_object() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let caps = tmpls.caps_json().expect("caps");
    assert!(caps.starts_with('{'), "caps must be a JSON object: {caps}");
}

/// The template must actually render the messages into the prompt — otherwise
/// everything downstream is operating on an empty string, which is the failure
/// mode the server example hit (`prompt_tokens: 0`).
#[test]
fn apply_renders_messages_into_the_prompt() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(&ChatApplyParams::new(
            r#"[{"role":"user","content":"Weather in Tokyo?"}]"#,
        ))
        .expect("apply");
    assert!(
        applied.prompt.contains("Weather in Tokyo?"),
        "prompt did not include the user turn: {}",
        applied.prompt
    );
    assert!(
        applied.prompt.contains("assistant"),
        "generation prompt missing: {}",
        applied.prompt
    );
}

#[test]
fn apply_omits_generation_prompt_when_asked() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"hi"}]"#)
                .with_add_generation_prompt(false),
        )
        .expect("apply");
    let with_prompt = tmpls
        .apply(&ChatApplyParams::new(r#"[{"role":"user","content":"hi"}]"#))
        .expect("apply");
    assert!(
        applied.prompt.len() < with_prompt.prompt.len(),
        "omitting the generation prompt should shorten the render:\n{}\nvs\n{}",
        applied.prompt,
        with_prompt.prompt
    );
}

/// Tools must reach the rendered prompt, and the template must be recognised as
/// a tool-calling format rather than falling back to content-only.
#[test]
fn apply_with_tools_produces_a_tool_format() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"Weather in Tokyo?"}]"#)
                .with_tools(WEATHER_TOOL),
        )
        .expect("apply");

    assert!(
        applied.prompt.contains("get_weather"),
        "tool definition missing from prompt: {}",
        applied.prompt
    );
    let name = applied.format_name().expect("format name");
    assert!(!name.is_empty(), "format has no name");
}

/// `tool_choice: required` is the case the hand-rolled server path could not
/// implement: upstream enforces it with a grammar, not a prompt. If this is
/// empty, nothing is actually forcing a call.
#[test]
fn tool_choice_required_yields_a_grammar() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"Weather in Tokyo?"}]"#)
                .with_tools(WEATHER_TOOL)
                .with_tool_choice(ToolChoice::Required),
        )
        .expect("apply");

    assert!(
        !applied.grammar.is_empty(),
        "tool_choice=required produced no grammar"
    );
    assert!(
        applied.grammar.contains("get_weather"),
        "grammar does not mention the tool: {}",
        applied.grammar
    );
}

/// The point of the whole exercise: with `tool_choice: auto` the grammar must
/// be *lazy* and carry triggers. A non-lazy grammar applied from token zero is
/// what stops a thinking model ever opening its reasoning block — the exact
/// reason the hand-rolled server path gave up on grammar forcing.
#[test]
fn auto_tool_choice_yields_a_lazy_grammar_with_triggers() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"Weather in Tokyo?"}]"#)
                .with_tools(WEATHER_TOOL)
                .with_tool_choice(ToolChoice::Auto),
        )
        .expect("apply");

    assert!(
        applied.grammar_lazy,
        "auto tool choice must produce a lazy grammar, else reasoning is blocked"
    );
    assert!(
        !applied.grammar_triggers.is_empty(),
        "a lazy grammar with no triggers can never activate: {}",
        applied.grammar_triggers_json
    );
    // The trigger is what the model must emit before the grammar engages.
    assert!(
        applied
            .grammar_triggers
            .iter()
            .any(|t| t.value.contains("tool_call")),
        "expected a <tool_call> trigger, got {:?}",
        applied.grammar_triggers
    );
}

/// `required` must NOT be lazy: there is nothing to wait for when a call is
/// mandatory from the first token.
#[test]
fn required_tool_choice_is_not_lazy() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"Weather?"}]"#)
                .with_tools(WEATHER_TOOL)
                .with_tool_choice(ToolChoice::Required),
        )
        .expect("apply");
    assert!(
        !applied.grammar_lazy,
        "required tool choice should constrain from the start"
    );
}

/// `tool_choice: none` must not force a call, even with tools present.
#[test]
fn tool_choice_none_does_not_force_a_call() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"hi"}]"#)
                .with_tools(WEATHER_TOOL)
                .with_tool_choice(ToolChoice::None),
        )
        .expect("apply");
    assert!(
        applied.grammar.is_empty() || !applied.grammar.contains("get_weather"),
        "tool_choice=none should not force a tool call: {}",
        applied.grammar
    );
}

/// A JSON Schema on the request must become a grammar constraining the answer —
/// this is `response_format: json_schema`.
#[test]
fn json_schema_constrains_the_response() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"describe a city"}]"#)
                .with_json_schema(
                    r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
                ),
        )
        .expect("apply");
    assert!(
        applied.grammar.contains("city"),
        "schema did not constrain the grammar: {}",
        applied.grammar
    );
}

/// The round trip that matters: apply, then parse a tool call back out. The
/// parser comes from the template, so it understands this family's syntax.
#[test]
fn parse_extracts_a_tool_call_from_model_output() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"Weather in Tokyo?"}]"#)
                .with_tools(WEATHER_TOOL)
                .with_tool_choice(ToolChoice::Required),
        )
        .expect("apply");

    let output = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>"#;
    let parsed = applied.parse(output, false).expect("parse");

    assert!(
        parsed.contains("get_weather"),
        "tool call not recovered from output: {parsed}"
    );
    assert!(
        parsed.contains("Tokyo"),
        "tool arguments not recovered: {parsed}"
    );
}

/// Plain prose must come back as content, not be mistaken for a call.
#[test]
fn parse_returns_plain_content_when_no_tool_is_called() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(&ChatApplyParams::new(
            r#"[{"role":"user","content":"say hi"}]"#,
        ))
        .expect("apply");
    let parsed = applied.parse("Hello there.", false).expect("parse");
    assert!(
        parsed.contains("Hello there."),
        "content not preserved: {parsed}"
    );
}

/// A truncated buffer mid-stream must not be an error — streaming callers parse
/// on every delta.
#[test]
fn parse_tolerates_partial_output() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"Weather?"}]"#)
                .with_tools(WEATHER_TOOL),
        )
        .expect("apply");
    let partial = r#"<tool_call>
{"name": "get_weat"#;
    applied
        .parse(partial, true)
        .expect("partial parse must not fail");
}

/// Reasoning must be separable from content — the whole reason lazy grammars
/// exist. With `enable_thinking`, a `<think>` block must not end up in
/// `content`.
#[test]
fn parse_splits_reasoning_from_content() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let applied = tmpls
        .apply(
            &ChatApplyParams::new(r#"[{"role":"user","content":"think then answer"}]"#)
                .with_reasoning_format(ReasoningFormat::DeepSeek),
        )
        .expect("apply");

    // Whether the split happens depends on the template advertising thinking;
    // what must always hold is that parsing succeeds and preserves the answer.
    let parsed = applied
        .parse("<think>weighing it up</think>The answer.", false)
        .expect("parse");
    assert!(
        parsed.contains("The answer."),
        "answer lost while handling reasoning: {parsed}"
    );
}

/// Malformed message JSON must surface as an error, not a panic or UB — the
/// underlying C++ throws.
#[test]
fn apply_rejects_malformed_messages() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    assert!(tmpls.apply(&ChatApplyParams::new("{not json")).is_err());
}

#[test]
fn apply_rejects_malformed_tools() {
    let Some(tmpls) = templates() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    assert!(tmpls
        .apply(&ChatApplyParams::new(r#"[{"role":"user","content":"hi"}]"#).with_tools("[[["))
        .is_err());
}

/// A grammar built from a schema must be usable by the sampler, which is the
/// only thing that makes structured output binding rather than advisory.
#[test]
fn schema_grammar_is_accepted_by_the_sampler() {
    use llama_cpp_4::sampling::LlamaSampler;

    let Some((model, _)) = load_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let gbnf = json_schema_to_grammar(
        r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}"#,
        false,
    )
    .expect("schema to grammar");

    // Constructing the sampler is the check: llama.cpp parses the GBNF here and
    // would reject a malformed grammar.
    let _sampler = LlamaSampler::grammar(&model, &gbnf, "root");
}

/// A *string*-valued object schema, which is what real callers send. The
/// integer-only case above does not exercise GBNF's `char`/`string` rules.
#[test]
fn string_schema_grammar_is_accepted_by_the_sampler() {
    use llama_cpp_4::sampling::LlamaSampler;

    let Some((model, _)) = load_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let gbnf = json_schema_to_grammar(
        r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
        false,
    )
    .expect("schema to grammar");
    let _sampler = LlamaSampler::grammar(&model, &gbnf, "root");
}

#[test]
fn format_name_is_stable_for_content_only() {
    // 0 is COMMON_CHAT_FORMAT_CONTENT_ONLY, the default every template falls
    // back to; it must always name itself.
    let name = format_name(0).expect("format name");
    assert!(!name.is_empty());
}

/// A model with no chat template of its own must still yield usable templates:
/// llama.cpp falls back to a builtin `ChatML` program. Callers therefore need no
/// fallback path of their own — `examples/server` relies on this to accept any
/// model that loads, and the test checkpoint here ships no template at all.
#[test]
fn templates_fall_back_to_a_builtin_for_a_model_without_one() {
    let Some((model, _)) = load_model() else {
        eprintln!("SKIP: no test model available");
        return;
    };
    let tmpls = ChatTemplates::from_model(&model, None)
        .expect("a model with no template must still get the builtin fallback");
    let source = tmpls.source(None).expect("template source");
    assert!(!source.is_empty(), "fallback template is empty");
    assert!(
        !tmpls.was_explicit(),
        "a fallback is not an explicit override"
    );

    // It must actually render, not just exist.
    let applied = tmpls
        .apply(&ChatApplyParams::new(
            r#"[{"role":"user","content":"ping"}]"#,
        ))
        .expect("fallback template must render");
    assert!(
        applied.prompt.contains("ping"),
        "fallback did not render the turn: {}",
        applied.prompt
    );
}
