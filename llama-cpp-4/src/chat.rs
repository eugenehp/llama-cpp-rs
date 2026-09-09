//! Chat templates, tool calling, and JSON-Schema-driven grammars.
//!
//! This wraps llama.cpp's `common/chat.h` — the layer its own server uses to
//! turn an OpenAI-shaped request into a prompt, a GBNF grammar, and a parser
//! that can read tool calls back out of whatever the model emits.
//!
//! The value over hand-rolling it is [`ChatParams::grammar`] plus
//! [`ChatParams::grammar_triggers`]. Naive grammar forcing breaks models that
//! emit reasoning before a tool call: constrain from token zero and the model
//! can never open its `<think>` block. A *lazy* grammar stays dormant until a
//! trigger fires — usually the `<tool_call>` marker — so reasoning flows
//! unconstrained and only the call itself is forced to be well-formed.
//!
//! # Example
//!
//! ```no_run
//! # use llama_cpp_4::chat::{ChatTemplates, ChatApplyParams, ToolChoice};
//! # fn f(model: &llama_cpp_4::model::LlamaModel) -> Result<(), Box<dyn std::error::Error>> {
//! let templates = ChatTemplates::from_model(model, None)?;
//!
//! let applied = templates.apply(
//!     &ChatApplyParams::new(r#"[{"role":"user","content":"Weather in Tokyo?"}]"#)
//!         .with_tools(r#"[{"type":"function","function":{"name":"get_weather",
//!             "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#)
//!         .with_tool_choice(ToolChoice::Required),
//! )?;
//!
//! // Feed `applied.prompt` to the model, constrain with `applied.grammar`,
//! // then read the tool calls back out:
//! let msg = applied.parse("<tool_call>{\"name\":\"get_weather\"}</tool_call>", false)?;
//! # Ok(())
//! # }
//! ```

use std::ffi::{c_char, CString};
use std::ptr::NonNull;

use llama_cpp_sys_4 as sys;

use crate::model::LlamaModel;

/// Errors from the chat layer.
///
/// An alias for [`ShimError`](crate::shim::ShimError) — every shim-backed
/// module shares one error type, since they share one status enum and one error
/// buffer.
pub type ChatError = crate::shim::ShimError;

use crate::shim::{check_status, last_error, read_string};

/// What the model is allowed to do with the supplied tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    #[default]
    Auto,
    /// The model must call a tool. This is enforced by grammar, not by prompt.
    Required,
    /// Tools are visible to the template but must not be called.
    None,
}

impl ToolChoice {
    // The C enum is unsigned (no negative discriminants) but the struct field
    // it feeds is `int32_t`, so the two need bridging in both directions.
    #[allow(clippy::cast_possible_wrap)]
    fn as_raw(self) -> i32 {
        let raw = match self {
            Self::Auto => sys::CHAT_SHIM_TOOL_CHOICE_AUTO,
            Self::Required => sys::CHAT_SHIM_TOOL_CHOICE_REQUIRED,
            Self::None => sys::CHAT_SHIM_TOOL_CHOICE_NONE,
        };
        raw as i32
    }

    #[allow(clippy::cast_possible_wrap)]
    fn from_raw(raw: i32) -> Option<Self> {
        if raw == sys::CHAT_SHIM_TOOL_CHOICE_AUTO as i32 {
            Some(Self::Auto)
        } else if raw == sys::CHAT_SHIM_TOOL_CHOICE_REQUIRED as i32 {
            Some(Self::Required)
        } else if raw == sys::CHAT_SHIM_TOOL_CHOICE_NONE as i32 {
            Some(Self::None)
        } else {
            None
        }
    }

    /// Parse an `OpenAI` `tool_choice` value: `"auto"`, `"required"` or
    /// `"none"`.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::Failed`] if llama.cpp does not recognise `value`.
    pub fn parse_oaicompat(value: &str) -> Result<Self, ChatError> {
        let c_value = CString::new(value)?;
        let mut raw: i32 = 0;
        let status = unsafe {
            sys::chat_shim_tool_choice_parse_oaicompat(c_value.as_ptr(), &raw mut raw)
        };
        check_status(status)?;
        Self::from_raw(raw).ok_or_else(|| ChatError::Failed(format!("unknown tool_choice {raw}")))
    }
}

/// How reasoning (`<think>` blocks) should be handled when parsing output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// Do not treat reasoning specially.
    #[default]
    None,
    /// Split reasoning into `reasoning_content`. The usual choice.
    Auto,
    /// Like [`Self::Auto`], but leaves reasoning inline when streaming.
    DeepSeekLegacy,
    /// Split reasoning out, including in streaming deltas.
    DeepSeek,
}

impl ReasoningFormat {
    #[allow(clippy::cast_possible_wrap)]
    fn as_raw(self) -> i32 {
        let raw = match self {
            Self::None => sys::CHAT_SHIM_REASONING_NONE,
            Self::Auto => sys::CHAT_SHIM_REASONING_AUTO,
            Self::DeepSeekLegacy => sys::CHAT_SHIM_REASONING_DEEPSEEK_LEGACY,
            Self::DeepSeek => sys::CHAT_SHIM_REASONING_DEEPSEEK,
        };
        raw as i32
    }
}

/// Convert a JSON Schema into a GBNF grammar.
///
/// The output goes straight to
/// [`LlamaSampler::grammar`](crate::sampling::LlamaSampler::grammar), which is
/// how you implement `OpenAI`'s `response_format: json_schema` — the model then
/// *cannot* emit anything the schema rejects, rather than being asked nicely.
///
/// `force_gbnf` mirrors upstream's flag: leave it `false` and the converter may
/// pick a more compact representation for schemas it recognises.
///
/// ```
/// # use llama_cpp_4::chat::json_schema_to_grammar;
/// let gbnf = json_schema_to_grammar(r#"{"type":"integer"}"#, false).unwrap();
/// assert!(gbnf.contains("root"));
/// ```
///
/// # Errors
///
/// Returns [`ChatError::BadJson`] if `schema_json` is not valid JSON, or
/// [`ChatError::Failed`] if it is not a schema llama.cpp can convert.
pub fn json_schema_to_grammar(schema_json: &str, force_gbnf: bool) -> Result<String, ChatError> {
    let c_schema = CString::new(schema_json)?;
    read_string(|buf, len, expected| unsafe {
        sys::common_json_schema_to_grammar_c(c_schema.as_ptr(), force_gbnf, buf, len, expected)
    })
}

/// The chat templates a model ships, ready to apply.
///
/// Wraps `common_chat_templates`, which owns parsed Jinja programs — building
/// it is not free, so hold one per model rather than one per request.
#[derive(Debug)]
pub struct ChatTemplates {
    raw: NonNull<sys::chat_shim_templates>,
}

// SAFETY: the handle owns a `common_chat_templates_ptr` with no interior
// mutability reachable through `&self` — `apply` and the getters only read it.
// The shim's only mutable global is a `thread_local` error buffer.
unsafe impl Send for ChatTemplates {}
unsafe impl Sync for ChatTemplates {}

impl Drop for ChatTemplates {
    fn drop(&mut self) {
        unsafe { sys::chat_shim_templates_free(self.raw.as_ptr()) }
    }
}

impl ChatTemplates {
    /// Build the template set for `model`.
    ///
    /// Pass `None` for `template_override` to use whatever the model ships, or
    /// Jinja source to override it. Overriding is what lets you reach a
    /// model's `tool_use` variant — fetch it with
    /// [`LlamaModel::chat_template`] and hand it back here.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::Init`] if the model has no usable template
    /// or the override does not parse.
    pub fn from_model(
        model: &LlamaModel,
        template_override: Option<&str>,
    ) -> Result<Self, ChatError> {
        let c_override = template_override.map(CString::new).transpose()?;
        let override_ptr = c_override.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let raw = unsafe { sys::chat_shim_templates_init(model.model.as_ptr(), override_ptr) };
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| ChatError::Init(last_error()))
    }

    /// Where the active template came from, e.g. `"model"` or a builtin name.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::Failed`] if llama.cpp could not report a source.
    pub fn source(&self, variant: Option<&str>) -> Result<String, ChatError> {
        let c_variant = variant.map(CString::new).transpose()?;
        let variant_ptr = c_variant.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        read_string(|buf, len, expected| unsafe {
            sys::chat_shim_templates_source(self.raw.as_ptr(), variant_ptr, buf, len, expected)
        })
    }

    /// Whether the caller supplied the template rather than the model.
    #[must_use]
    pub fn was_explicit(&self) -> bool {
        unsafe { sys::chat_shim_templates_was_explicit(self.raw.as_ptr()) }
    }

    /// Whether this template understands `enable_thinking`.
    #[must_use]
    pub fn supports_enable_thinking(&self) -> bool {
        unsafe { sys::chat_shim_templates_support_enable_thinking(self.raw.as_ptr()) }
    }

    /// Template capabilities as a JSON object of `name -> bool`.
    ///
    /// This is what upstream's server reports on `/props`; it tells you whether
    /// the template can handle tools, parallel calls, a system role, and so on
    /// *before* you send a request it cannot render.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::Failed`] if llama.cpp could not report caps.
    pub fn caps_json(&self) -> Result<String, ChatError> {
        read_string(|buf, len, expected| unsafe {
            sys::chat_shim_templates_get_caps(self.raw.as_ptr(), buf, len, expected)
        })
    }

    /// Render messages and tools into a prompt plus its sampling constraints.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::BadJson`] for malformed `messages`/`tools` JSON, or
    /// [`ChatError::Failed`] if the template cannot render the request.
    pub fn apply(&self, params: &ChatApplyParams) -> Result<ChatParams, ChatError> {
        let messages = CString::new(params.messages_json.as_str())?;
        let tools = params.tools_json.as_deref().map(CString::new).transpose()?;
        let grammar = params.grammar.as_deref().map(CString::new).transpose()?;
        let schema = params.json_schema.as_deref().map(CString::new).transpose()?;
        let kwargs = params
            .template_kwargs_json
            .as_deref()
            .map(CString::new)
            .transpose()?;

        let raw_params = sys::chat_shim_apply_params {
            messages_json: messages.as_ptr(),
            tools_json: opt_ptr(tools.as_ref()),
            grammar: opt_ptr(grammar.as_ref()),
            json_schema: opt_ptr(schema.as_ref()),
            template_kwargs_json: opt_ptr(kwargs.as_ref()),
            tool_choice: params.tool_choice.as_raw(),
            reasoning_format: params.reasoning_format.as_raw(),
            add_generation_prompt: params.add_generation_prompt,
            enable_thinking: params.enable_thinking,
            parallel_tool_calls: params.parallel_tool_calls,
            use_jinja: params.use_jinja,
            add_bos: params.add_bos,
            add_eos: params.add_eos,
        };

        // Two-call protocol: size, then fill. The shim fills `result` on both
        // calls, so the template is only rendered once per call — but we take
        // the second call's offsets, which describe the buffer we actually got.
        // bindgen derives no `Default` for this struct; zeroing is correct
        // here because every field is a `usize` offset, an `i32`, or a `bool`,
        // and the shim overwrites all of them before reporting success.
        let mut result: sys::chat_shim_apply_result = unsafe { std::mem::zeroed() };
        let mut needed: usize = 0;
        let status = unsafe {
            sys::chat_shim_templates_apply(
                self.raw.as_ptr(),
                &raw const raw_params,
                &raw mut result,
                std::ptr::null_mut(),
                0,
                &raw mut needed,
            )
        };
        if status != sys::LLAMA_SHIM_BUFFER_TOO_SMALL {
            check_status(status)?;
        }

        let mut buf = vec![0u8; needed];
        let status = unsafe {
            sys::chat_shim_templates_apply(
                self.raw.as_ptr(),
                &raw const raw_params,
                &raw mut result,
                buf.as_mut_ptr().cast::<c_char>(),
                buf.len(),
                &raw mut needed,
            )
        };
        check_status(status)?;

        ChatParams::from_packed(&buf, &result)
    }
}

/// A request to render: messages, optional tools, and how to constrain output.
///
/// `messages_json` and `tools_json` are `OpenAI`-shaped JSON. They are passed as
/// text rather than typed structs so this crate need not take a JSON dependency
/// or track upstream's message schema, which grows most releases.
// Mirrors `common_chat_templates_inputs`, which is mostly independent flags;
// grouping them into sub-structs would diverge from the C layout for no gain.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct ChatApplyParams {
    messages_json: String,
    tools_json: Option<String>,
    grammar: Option<String>,
    json_schema: Option<String>,
    template_kwargs_json: Option<String>,
    tool_choice: ToolChoice,
    reasoning_format: ReasoningFormat,
    add_generation_prompt: bool,
    enable_thinking: bool,
    parallel_tool_calls: bool,
    use_jinja: bool,
    add_bos: bool,
    add_eos: bool,
}

impl ChatApplyParams {
    /// Start from an OpenAI-shaped `messages` JSON array.
    #[must_use]
    pub fn new(messages_json: impl Into<String>) -> Self {
        Self {
            messages_json: messages_json.into(),
            tools_json: None,
            grammar: None,
            json_schema: None,
            template_kwargs_json: None,
            tool_choice: ToolChoice::Auto,
            reasoning_format: ReasoningFormat::Auto,
            add_generation_prompt: true,
            enable_thinking: true,
            parallel_tool_calls: false,
            use_jinja: true,
            add_bos: false,
            add_eos: false,
        }
    }

    /// Supply an OpenAI-shaped `tools` JSON array.
    #[must_use]
    pub fn with_tools(mut self, tools_json: impl Into<String>) -> Self {
        self.tools_json = Some(tools_json.into());
        self
    }

    /// Constrain output with a GBNF grammar directly.
    #[must_use]
    pub fn with_grammar(mut self, grammar: impl Into<String>) -> Self {
        self.grammar = Some(grammar.into());
        self
    }

    /// Constrain output with a JSON Schema. llama.cpp converts it to GBNF.
    #[must_use]
    pub fn with_json_schema(mut self, schema_json: impl Into<String>) -> Self {
        self.json_schema = Some(schema_json.into());
        self
    }

    /// Extra Jinja variables, as a JSON object.
    #[must_use]
    pub fn with_template_kwargs(mut self, kwargs_json: impl Into<String>) -> Self {
        self.template_kwargs_json = Some(kwargs_json.into());
        self
    }

    /// Set what the model may do with the tools. Defaults to
    /// [`ToolChoice::Auto`].
    #[must_use]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Set how reasoning is handled. Defaults to [`ReasoningFormat::Auto`].
    #[must_use]
    pub fn with_reasoning_format(mut self, format: ReasoningFormat) -> Self {
        self.reasoning_format = format;
        self
    }

    /// Append the assistant generation prompt. Defaults to `true`; set `false`
    /// to render a transcript rather than a request.
    #[must_use]
    pub fn with_add_generation_prompt(mut self, add: bool) -> Self {
        self.add_generation_prompt = add;
        self
    }

    /// Let the model think before answering, on templates that support it.
    /// Defaults to `true`.
    #[must_use]
    pub fn with_enable_thinking(mut self, enable: bool) -> Self {
        self.enable_thinking = enable;
        self
    }

    /// Allow several tool calls in one turn. Defaults to `false`.
    #[must_use]
    pub fn with_parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = parallel;
        self
    }

    /// Render with the Jinja engine. Defaults to `true`; `false` selects the
    /// legacy `llama_chat_apply_template` path, which ignores tools.
    #[must_use]
    pub fn with_use_jinja(mut self, use_jinja: bool) -> Self {
        self.use_jinja = use_jinja;
        self
    }

    /// Prepend BOS / append EOS to the rendered prompt. Both default to
    /// `false`, since tokenization usually adds BOS itself.
    // The two parameter names mirror the upstream fields; renaming either to
    // satisfy `similar_names` would obscure which C field it sets.
    #[allow(clippy::similar_names)]
    #[must_use]
    pub fn with_add_bos_eos(mut self, add_bos: bool, add_eos: bool) -> Self {
        self.add_bos = add_bos;
        self.add_eos = add_eos;
        self
    }
}

/// A lazy-grammar trigger: what has to appear before the grammar engages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarTrigger {
    /// `"token"`, `"word"`, `"pattern"` or `"pattern_full"`.
    pub kind: String,
    /// The literal, word, or regex the trigger matches.
    pub value: String,
    /// Token id for `"token"` triggers; `-1` otherwise.
    pub token: i32,
}

/// The rendered prompt and everything needed to constrain and parse generation.
#[derive(Debug, Clone)]
pub struct ChatParams {
    /// The prompt to feed the model.
    pub prompt: String,
    /// GBNF grammar constraining output. Empty when unconstrained.
    pub grammar: String,
    /// When true, `grammar` must not be applied until a trigger in
    /// [`Self::grammar_triggers`] fires. Applying it from token zero is what
    /// stops thinking models emitting their reasoning prefix.
    pub grammar_lazy: bool,
    /// Triggers that activate a lazy grammar. Raw JSON, and parsed into
    /// [`Self::grammar_triggers`].
    pub grammar_triggers_json: String,
    /// Parsed form of [`Self::grammar_triggers_json`].
    pub grammar_triggers: Vec<GrammarTrigger>,
    /// JSON array of strings to keep verbatim while sampling.
    pub preserved_tokens_json: String,
    /// JSON array of extra stop strings this format needs.
    pub additional_stops_json: String,
    /// Whether this template supports reasoning.
    pub supports_thinking: bool,
    /// Opening reasoning tag, e.g. `"<think>"`. Empty when unsupported.
    pub thinking_start_tag: String,
    /// JSON array of closing reasoning tags.
    pub thinking_end_tags_json: String,
    /// `common_chat_format` discriminant, needed to parse output back.
    pub format: i32,
    /// Serialized PEG parser produced alongside the prompt. Opaque.
    parser: String,
    /// Prefix the parser expects to see before generated text.
    generation_prompt: String,
    reasoning_format: ReasoningFormat,
}

impl ChatParams {
    fn from_packed(
        buf: &[u8],
        result: &sys::chat_shim_apply_result,
    ) -> Result<Self, ChatError> {
        let at = |off: usize| -> Result<String, ChatError> {
            let rest = buf.get(off..).ok_or(ChatError::CorruptResult)?;
            let end = rest
                .iter()
                .position(|b| *b == 0)
                .ok_or(ChatError::CorruptResult)?;
            String::from_utf8(rest[..end].to_vec()).map_err(ChatError::from)
        };

        let grammar_triggers_json = at(result.grammar_triggers_off)?;
        let grammar_triggers = parse_triggers(&grammar_triggers_json);

        Ok(Self {
            prompt: at(result.prompt_off)?,
            grammar: at(result.grammar_off)?,
            grammar_lazy: result.grammar_lazy,
            grammar_triggers,
            grammar_triggers_json,
            preserved_tokens_json: at(result.preserved_tokens_off)?,
            additional_stops_json: at(result.additional_stops_off)?,
            supports_thinking: result.supports_thinking,
            thinking_start_tag: at(result.thinking_start_tag_off)?,
            thinking_end_tags_json: at(result.thinking_end_tags_off)?,
            format: result.format,
            parser: at(result.parser_off)?,
            generation_prompt: at(result.generation_prompt_off)?,
            reasoning_format: ReasoningFormat::Auto,
        })
    }

    /// Human-readable name of this chat format, e.g. `"Hermes 2 Pro"`.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::Failed`] if llama.cpp cannot name the format.
    pub fn format_name(&self) -> Result<String, ChatError> {
        format_name(self.format)
    }

    /// Convert [`Self::grammar_triggers`] into the `(patterns, tokens)` pair
    /// [`LlamaSampler::grammar_lazy_patterns`](crate::sampling::LlamaSampler::grammar_lazy_patterns)
    /// expects.
    ///
    /// The four trigger kinds do not map onto the sampler one-for-one, and
    /// getting the translation wrong silently produces a grammar that never
    /// activates:
    ///
    /// - `word` is a **literal**, so it is regex-escaped before becoming a
    ///   pattern. A raw `<tool_call>` would otherwise be a character class.
    /// - `pattern` passes through unchanged.
    /// - `pattern_full` is anchored with `^`/`$` unless it already is.
    /// - `token` becomes a trigger token rather than a pattern.
    ///
    /// This mirrors `common/sampling.cpp`, so callers do not have to.
    #[must_use]
    pub fn sampler_triggers(&self) -> (Vec<String>, Vec<crate::token::LlamaToken>) {
        let mut patterns = Vec::new();
        let mut tokens = Vec::new();
        for trigger in &self.grammar_triggers {
            match trigger.kind.as_str() {
                "word" => patterns.push(regex_escape(&trigger.value)),
                "pattern" => patterns.push(trigger.value.clone()),
                "pattern_full" => patterns.push(anchor_pattern(&trigger.value)),
                "token" => tokens.push(crate::token::LlamaToken(trigger.token)),
                // An unknown kind from a newer llama.cpp: dropping it is safer
                // than guessing, and the grammar simply stays dormant for it.
                _ => {}
            }
        }
        (patterns, tokens)
    }

    /// The prefix the template already placed at the end of the prompt, e.g.
    /// `"<|im_start|>assistant\n"`.
    ///
    /// This is not part of the model's output, but the grammar and the parser
    /// are both written as if it were — see [`Self::grammar_sampler`].
    #[must_use]
    pub fn generation_prompt(&self) -> &str {
        &self.generation_prompt
    }

    /// Build the grammar sampler this render needs, or `None` when
    /// unconstrained.
    ///
    /// Prefer this over constructing the sampler yourself: it handles three
    /// things that are each silently wrong if missed.
    ///
    /// - **Lazy vs eager.** A lazy grammar must be built with its triggers, or
    ///   it never activates. One built eagerly from a lazy grammar constrains
    ///   from token zero and blocks a thinking model's reasoning prefix.
    /// - **Trigger translation.** Literal, regex and token triggers map onto
    ///   the sampler differently — see [`Self::sampler_triggers`].
    /// - **Generation-prompt prefill.** llama.cpp writes tool-call grammars to
    ///   match `generation_prompt + output`, because that is what the parser
    ///   later sees. The sampler only sees `output`, so without advancing the
    ///   grammar past that prefix it forces the model to *re-emit*
    ///   `<|im_start|>assistant` as generated text. Prefill applies only to
    ///   non-lazy grammars — a lazy one has not started matching yet.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp cannot parse the grammar it just produced, or if
    /// that grammar contains an interior NUL.
    #[must_use]
    pub fn grammar_sampler(&self, model: &LlamaModel) -> Option<crate::sampling::LlamaSampler> {
        use crate::sampling::LlamaSampler;

        if self.grammar.is_empty() {
            return None;
        }

        let (patterns, tokens) = self.sampler_triggers();
        let lazy = self.grammar_lazy && !(patterns.is_empty() && tokens.is_empty());

        let mut sampler = if lazy {
            let refs: Vec<&str> = patterns.iter().map(String::as_str).collect();
            LlamaSampler::grammar_lazy_patterns(model, &self.grammar, "root", &refs, &tokens)
        } else {
            LlamaSampler::grammar(model, &self.grammar, "root")
        };

        if !lazy {
            for token in self.generation_prompt_tokens(model) {
                sampler.accept(token);
            }
        }
        Some(sampler)
    }

    /// Tokenize [`Self::generation_prompt`] the way llama.cpp does when
    /// prefilling a grammar.
    ///
    /// Some tokenizers prepend a space to the first token; upstream drops it
    /// when the prompt itself does not start with whitespace, since that space
    /// is an artefact rather than something the template emitted.
    fn generation_prompt_tokens(&self, model: &LlamaModel) -> Vec<crate::token::LlamaToken> {
        if self.generation_prompt.is_empty() {
            return Vec::new();
        }
        let Ok(tokens) = model.str_to_token(&self.generation_prompt, crate::model::AddBos::Never)
        else {
            return Vec::new();
        };
        let starts_with_space = self
            .generation_prompt
            .starts_with(char::is_whitespace);
        let mut out = Vec::with_capacity(tokens.len());
        for (i, token) in tokens.into_iter().enumerate() {
            if i == 0 && !starts_with_space {
                if let Ok(piece) = model.token_to_str(token, crate::model::Special::Tokenize) {
                    if piece.starts_with(char::is_whitespace) {
                        continue;
                    }
                }
            }
            out.push(token);
        }
        out
    }

    /// Parse model output back into an OpenAI-shaped message JSON object with
    /// `role`, `content`, `reasoning_content` and `tool_calls`.
    ///
    /// This uses the parser the template produced, so it understands that
    /// model family's tool-call syntax rather than scraping for a fixed marker.
    ///
    /// Set `is_partial` while streaming: the parser then tolerates a truncated
    /// tail instead of rejecting the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::Failed`] if the output cannot be parsed.
    pub fn parse(&self, text: &str, is_partial: bool) -> Result<String, ChatError> {
        self.parse_with(text, is_partial, true, false)
    }

    /// [`Self::parse`] with control over tool-call parsing and whether
    /// reasoning is left inline in `content`.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError::Failed`] if the output cannot be parsed.
    pub fn parse_with(
        &self,
        text: &str,
        is_partial: bool,
        parse_tool_calls: bool,
        reasoning_in_content: bool,
    ) -> Result<String, ChatError> {
        let c_text = CString::new(text)?;
        let c_parser = CString::new(self.parser.as_str())?;
        let c_gen_prompt = CString::new(self.generation_prompt.as_str())?;
        let params = sys::chat_shim_parse_params {
            text: c_text.as_ptr(),
            parser: c_parser.as_ptr(),
            generation_prompt: c_gen_prompt.as_ptr(),
            format: self.format,
            reasoning_format: self.reasoning_format.as_raw(),
            is_partial,
            parse_tool_calls,
            reasoning_in_content,
        };
        read_string(|buf, len, expected| unsafe {
            sys::chat_shim_parse(&raw const params, buf, len, expected)
        })
    }
}

/// Human-readable name of a `common_chat_format` discriminant.
///
/// # Errors
///
/// Returns [`ChatError::Failed`] if llama.cpp cannot name the format.
pub fn format_name(format: i32) -> Result<String, ChatError> {
    read_string(|buf, len, expected| unsafe {
        sys::chat_shim_format_name(format, buf, len, expected)
    })
}

/// Validate and normalise an OpenAI-shaped `messages` array.
///
/// Expands the shorthands llama.cpp accepts — typed content parts, legacy
/// `function_call` — into the canonical form, and rejects malformed input
/// before it reaches a template.
///
/// # Errors
///
/// Returns [`ChatError::BadJson`] if the input is not valid JSON, or
/// [`ChatError::Failed`] if it is not a valid message array.
pub fn parse_messages_oaicompat(messages_json: &str) -> Result<String, ChatError> {
    let c_messages = CString::new(messages_json)?;
    read_string(|buf, len, expected| unsafe {
        sys::chat_shim_msgs_parse_oaicompat(c_messages.as_ptr(), buf, len, expected)
    })
}

/// Validate and normalise an OpenAI-shaped `tools` array into a JSON array of
/// `{"name", "description", "parameters"}`.
///
/// # Errors
///
/// Returns [`ChatError::BadJson`] if the input is not valid JSON, or
/// [`ChatError::Failed`] if it is not a valid tool array.
pub fn parse_tools_oaicompat(tools_json: &str) -> Result<String, ChatError> {
    let c_tools = CString::new(tools_json)?;
    read_string(|buf, len, expected| unsafe {
        sys::chat_shim_tools_parse_oaicompat(c_tools.as_ptr(), buf, len, expected)
    })
}

// ── plumbing ────────────────────────────────────────────────────────────────

/// Escape the characters `std::regex` treats as special, matching
/// `regex_escape` in `common/common.cpp`.
fn regex_escape(s: &str) -> String {
    const SPECIAL: &[char] = &[
        '.', '^', '$', '|', '(', ')', '*', '+', '?', '[', ']', '{', '}', '\\',
    ];
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if SPECIAL.contains(&c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Anchor a `pattern_full` trigger, matching `common/sampling.cpp`. An empty
/// pattern becomes `^$` — matching only the empty string — rather than `^^$$`.
fn anchor_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_owned();
    }
    let mut out = String::with_capacity(pattern.len() + 2);
    if !pattern.starts_with('^') {
        out.push('^');
    }
    out.push_str(pattern);
    if !pattern.ends_with('$') {
        out.push('$');
    }
    out
}

fn opt_ptr(s: Option<&CString>) -> *const c_char {
    s.map_or(std::ptr::null(), |c| c.as_ptr())
}

/// Pull the fields out of the shim's trigger JSON without taking a JSON
/// dependency. The shape is fixed by `chat_shim.cpp`, so a hand parser is
/// enough — anything unexpected yields no triggers, and
/// [`ChatParams::grammar_triggers_json`] still carries the raw text.
fn parse_triggers(json: &str) -> Vec<GrammarTrigger> {
    let mut out = Vec::new();
    for chunk in json.split('{').skip(1) {
        let kind = json_str_field(chunk, "type");
        let value = json_str_field(chunk, "value");
        let token = json_int_field(chunk, "token");
        if let (Some(kind), Some(value)) = (kind, value) {
            out.push(GrammarTrigger {
                kind,
                value,
                token: token.unwrap_or(-1),
            });
        }
    }
    out
}

fn json_str_field(chunk: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let rest = &chunk[chunk.find(&needle)? + needle.len()..];
    let rest = rest.trim_start();
    let mut chars = rest.strip_prefix('"')?.chars();
    let mut value = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(value),
            '\\' => match chars.next()? {
                'n' => value.push('\n'),
                'r' => value.push('\r'),
                't' => value.push('\t'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    let code = u32::from_str_radix(&hex, 16).ok()?;
                    value.push(char::from_u32(code)?);
                }
                other => value.push(other),
            },
            other => value.push(other),
        }
    }
    None
}

fn json_int_field(chunk: &str, key: &str) -> Option<i32> {
    let needle = format!("\"{key}\":");
    let rest = &chunk[chunk.find(&needle)? + needle.len()..];
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schema the `structured` example ships in its docs must convert and
    /// be loadable by the sampler — an object schema with a required key is the
    /// shape every `response_format: json_schema` caller sends.
    #[test]
    fn realistic_object_schema_converts() {
        let gbnf = json_schema_to_grammar(
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
            false,
        )
        .unwrap();
        eprintln!("GBNF:\n{gbnf}");
        assert!(gbnf.contains("root"));
    }

    #[test]
    fn json_schema_to_grammar_produces_a_root_rule() {
        let gbnf = json_schema_to_grammar(r#"{"type":"integer"}"#, false).unwrap();
        assert!(gbnf.contains("root"), "no root rule in: {gbnf}");
    }

    /// An object schema must constrain its keys, otherwise the grammar is not
    /// actually enforcing the schema.
    #[test]
    fn json_schema_to_grammar_constrains_object_keys() {
        let gbnf = json_schema_to_grammar(
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
            false,
        )
        .unwrap();
        assert!(gbnf.contains("city"), "key not in grammar: {gbnf}");
    }

    /// Invalid JSON must be reported as such rather than crossing the FFI
    /// boundary as an exception.
    #[test]
    fn json_schema_to_grammar_rejects_invalid_json() {
        let err = json_schema_to_grammar("{not json", false).unwrap_err();
        assert!(
            matches!(err, ChatError::BadJson(_)),
            "expected BadJson, got {err:?}"
        );
    }

    #[test]
    fn json_schema_to_grammar_rejects_empty_input() {
        assert!(json_schema_to_grammar("", false).is_err());
    }

    /// Interior NULs must be caught in Rust — a `CString` cannot hold one, and
    /// truncating silently would change the schema.
    #[test]
    fn json_schema_to_grammar_rejects_interior_nul() {
        let err = json_schema_to_grammar("{\"type\":\"in\0teger\"}", false).unwrap_err();
        assert!(matches!(err, ChatError::Nul(_)), "got {err:?}");
    }

    #[test]
    fn tool_choice_parses_openai_values() {
        assert_eq!(ToolChoice::parse_oaicompat("auto").unwrap(), ToolChoice::Auto);
        assert_eq!(
            ToolChoice::parse_oaicompat("required").unwrap(),
            ToolChoice::Required
        );
        assert_eq!(ToolChoice::parse_oaicompat("none").unwrap(), ToolChoice::None);
    }

    #[test]
    fn tool_choice_rejects_unknown_value() {
        assert!(ToolChoice::parse_oaicompat("sometimes").is_err());
    }

    #[test]
    fn tools_parse_oaicompat_extracts_name_and_parameters() {
        let normalised = parse_tools_oaicompat(
            r#"[{"type":"function","function":{"name":"get_weather",
                "description":"Get weather","parameters":{"type":"object"}}}]"#,
        )
        .unwrap();
        assert!(normalised.contains("get_weather"), "got {normalised}");
    }

    #[test]
    fn tools_parse_oaicompat_rejects_garbage() {
        assert!(parse_tools_oaicompat("[[[").is_err());
    }

    #[test]
    fn messages_parse_oaicompat_roundtrips_a_simple_turn() {
        let normalised =
            parse_messages_oaicompat(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        assert!(normalised.contains("user"), "got {normalised}");
        assert!(normalised.contains("hi"), "got {normalised}");
    }

    #[test]
    fn messages_parse_oaicompat_rejects_non_array() {
        assert!(parse_messages_oaicompat(r#"{"role":"user"}"#).is_err());
    }

    /// The trigger parser is hand-rolled, so pin it against the exact shape
    /// `chat_shim.cpp` emits, including escapes.
    #[test]
    fn trigger_parser_reads_shim_output() {
        let json = r#"[{"type":"word","value":"<tool_call>","token":-1},
                       {"type":"token","value":"a\nb","token":42}]"#;
        let triggers = parse_triggers(json);
        assert_eq!(triggers.len(), 2);
        assert_eq!(triggers[0].kind, "word");
        assert_eq!(triggers[0].value, "<tool_call>");
        assert_eq!(triggers[0].token, -1);
        assert_eq!(triggers[1].kind, "token");
        assert_eq!(triggers[1].value, "a\nb");
        assert_eq!(triggers[1].token, 42);
    }

    /// A literal trigger like `<tool_call>` must be escaped before it becomes
    /// a regex — unescaped, `[` and `]` would make it a character class and the
    /// grammar would never fire.
    #[test]
    fn word_triggers_are_regex_escaped() {
        let params = ChatParams {
            grammar_triggers: vec![GrammarTrigger {
                kind: "word".to_owned(),
                value: "a.b[c]".to_owned(),
                token: -1,
            }],
            ..stub_params()
        };
        let (patterns, tokens) = params.sampler_triggers();
        assert_eq!(patterns, vec![r"a\.b\[c\]".to_owned()]);
        assert!(tokens.is_empty());
    }

    #[test]
    fn pattern_triggers_pass_through_unescaped() {
        let params = ChatParams {
            grammar_triggers: vec![GrammarTrigger {
                kind: "pattern".to_owned(),
                value: "a.b".to_owned(),
                token: -1,
            }],
            ..stub_params()
        };
        assert_eq!(params.sampler_triggers().0, vec!["a.b".to_owned()]);
    }

    #[test]
    fn pattern_full_triggers_are_anchored_once() {
        let cases = [
            ("abc", "^abc$"),
            ("^abc", "^abc$"),
            ("abc$", "^abc$"),
            ("^abc$", "^abc$"),
            ("", "^$"),
        ];
        for (input, want) in cases {
            let params = ChatParams {
                grammar_triggers: vec![GrammarTrigger {
                    kind: "pattern_full".to_owned(),
                    value: input.to_owned(),
                    token: -1,
                }],
                ..stub_params()
            };
            assert_eq!(
                params.sampler_triggers().0,
                vec![want.to_owned()],
                "anchoring {input:?}"
            );
        }
    }

    #[test]
    fn token_triggers_become_tokens_not_patterns() {
        let params = ChatParams {
            grammar_triggers: vec![GrammarTrigger {
                kind: "token".to_owned(),
                value: String::new(),
                token: 42,
            }],
            ..stub_params()
        };
        let (patterns, tokens) = params.sampler_triggers();
        assert!(patterns.is_empty());
        assert_eq!(tokens, vec![crate::token::LlamaToken(42)]);
    }

    /// A kind from a newer llama.cpp must be dropped rather than guessed at.
    #[test]
    fn unknown_trigger_kinds_are_dropped() {
        let params = ChatParams {
            grammar_triggers: vec![GrammarTrigger {
                kind: "something_new".to_owned(),
                value: "x".to_owned(),
                token: -1,
            }],
            ..stub_params()
        };
        let (patterns, tokens) = params.sampler_triggers();
        assert!(patterns.is_empty());
        assert!(tokens.is_empty());
    }

    fn stub_params() -> ChatParams {
        ChatParams {
            prompt: String::new(),
            grammar: String::new(),
            grammar_lazy: false,
            grammar_triggers_json: String::new(),
            grammar_triggers: Vec::new(),
            preserved_tokens_json: String::new(),
            additional_stops_json: String::new(),
            supports_thinking: false,
            thinking_start_tag: String::new(),
            thinking_end_tags_json: String::new(),
            format: 0,
            parser: String::new(),
            generation_prompt: String::new(),
            reasoning_format: ReasoningFormat::Auto,
        }
    }

    #[test]
    fn trigger_parser_handles_empty_array() {
        assert!(parse_triggers("[]").is_empty());
    }
}
