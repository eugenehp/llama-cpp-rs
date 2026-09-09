#pragma once

// Stable C linkage over llama.cpp's C++-only chat and grammar helpers.
//
// `common/chat.h` and `common/json-schema-to-grammar.h` traffic in
// `std::string`, `std::vector`, `std::map`, `common_json` and a PEG-parser
// arena, none of which bindgen can express — and several of their entry points
// throw. Every function here catches at the boundary and reports a negative
// status instead, because unwinding a C++ exception through `extern "C"` into
// Rust is undefined behaviour.
//
// Status codes, the error buffer and the size-then-fill string protocol are
// shared with every other shim; see `shim_support.h`.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "shim_support.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;

// ── JSON Schema → GBNF ──────────────────────────────────────────────────────

// Convert a JSON Schema (as JSON text) into a GBNF grammar.
//
// `force_gbnf` mirrors upstream's flag: left false, the converter may emit a
// more compact representation for schemas it recognises.
int32_t common_json_schema_to_grammar_c(
        const char * schema_json,
        bool         force_gbnf,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len);

// ── chat templates ──────────────────────────────────────────────────────────

// Opaque owner of a `common_chat_templates_ptr`.
struct chat_shim_templates;

// Build the template set for a model. `template_override` may be NULL to use
// whatever the model ships; pass Jinja source to override it.
// Returns NULL on failure — see `llama_shim_last_error`.
struct chat_shim_templates * chat_shim_templates_init(
        const struct llama_model * model,
        const char *               template_override);

void chat_shim_templates_free(struct chat_shim_templates * tmpls);

// Where the active template came from. `variant` may be NULL for the default.
int32_t chat_shim_templates_source(
        const struct chat_shim_templates * tmpls,
        const char *                       variant,
        char *                             out_buf,
        size_t                             out_len,
        size_t *                           expected_len);

// True when the caller supplied the template rather than the model.
bool chat_shim_templates_was_explicit(const struct chat_shim_templates * tmpls);

// True when this template understands `enable_thinking`.
bool chat_shim_templates_support_enable_thinking(const struct chat_shim_templates * tmpls);

// Template capabilities as a JSON object of `name -> bool`.
int32_t chat_shim_templates_get_caps(
        const struct chat_shim_templates * tmpls,
        char *                             out_buf,
        size_t                             out_len,
        size_t *                           expected_len);

// ── applying a template ─────────────────────────────────────────────────────

// Mirrors `common_chat_tool_choice`.
enum chat_shim_tool_choice {
    CHAT_SHIM_TOOL_CHOICE_AUTO     = 0,
    CHAT_SHIM_TOOL_CHOICE_REQUIRED = 1,
    CHAT_SHIM_TOOL_CHOICE_NONE     = 2,
};

// Mirrors `common_reasoning_format`.
enum chat_shim_reasoning_format {
    CHAT_SHIM_REASONING_NONE            = 0,
    CHAT_SHIM_REASONING_AUTO            = 1,
    CHAT_SHIM_REASONING_DEEPSEEK_LEGACY = 2,
    CHAT_SHIM_REASONING_DEEPSEEK        = 3,
};

// Inputs to `common_chat_templates_apply`. JSON fields may be NULL or "".
struct chat_shim_apply_params {
    // OpenAI-shaped JSON array of messages.
    const char * messages_json;
    // OpenAI-shaped JSON array of tool definitions.
    const char * tools_json;
    // A GBNF grammar to use directly, bypassing schema conversion.
    const char * grammar;
    // A JSON Schema to constrain output with.
    const char * json_schema;
    // Extra Jinja variables, as a JSON object of string -> string.
    const char * template_kwargs_json;
    int32_t      tool_choice;
    int32_t      reasoning_format;
    bool         add_generation_prompt;
    bool         enable_thinking;
    bool         parallel_tool_calls;
    bool         use_jinja;
    bool         add_bos;
    bool         add_eos;
};

// Everything `apply` produces, as offsets into one packed buffer. Each region
// is NUL-terminated, so a caller can read them without copying the whole thing.
struct chat_shim_apply_result {
    size_t prompt_off;
    size_t grammar_off;
    // JSON array of `{"type", "value", "token"}` lazy-grammar triggers.
    size_t grammar_triggers_off;
    // JSON array of strings to keep verbatim while sampling.
    size_t preserved_tokens_off;
    // JSON array of extra stop strings.
    size_t additional_stops_off;
    // Serialized PEG parser. Opaque — hand it straight back to
    // `chat_shim_parse`; without it the parser degrades to plain content.
    size_t parser_off;
    // Prefix the parser must see prepended to generated text.
    size_t generation_prompt_off;
    size_t thinking_start_tag_off;
    // JSON array of thinking end tags.
    size_t thinking_end_tags_off;
    // `common_chat_format`; hand back to `chat_shim_parse`.
    int32_t format;
    // Grammar should only engage once a trigger fires.
    bool    grammar_lazy;
    bool    supports_thinking;
};

int32_t chat_shim_templates_apply(
        const struct chat_shim_templates *    tmpls,
        const struct chat_shim_apply_params * params,
        struct chat_shim_apply_result *       out_result,
        char *                                out_buf,
        size_t                                out_len,
        size_t *                              expected_len);

// Human-readable name of a `common_chat_format` value, e.g. "Hermes 2 Pro".
int32_t chat_shim_format_name(
        int32_t  format,
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len);

// ── parsing generated text ──────────────────────────────────────────────────

struct chat_shim_parse_params {
    // Model output to parse.
    const char * text;
    // Serialized parser from `chat_shim_apply_result::parser_off`; NULL or ""
    // falls back to a pure-content parser.
    const char * parser;
    // `generation_prompt` from the same result; NULL or "" for none.
    const char * generation_prompt;
    int32_t      format;
    int32_t      reasoning_format;
    // Buffer is mid-stream: tolerate a truncated tail.
    bool         is_partial;
    bool         parse_tool_calls;
    // Keep reasoning inline in `content` rather than splitting it out.
    bool         reasoning_in_content;
};

// Parse model output into an OpenAI-shaped JSON object: `{"role", "content",
// "reasoning_content", "tool_calls"}`.
int32_t chat_shim_parse(
        const struct chat_shim_parse_params * params,
        char *                                out_buf,
        size_t                                out_len,
        size_t *                              expected_len);

// ── OpenAI-compatible input parsing ─────────────────────────────────────────

// Validate and normalise an OpenAI-shaped `messages` array, expanding the
// shorthands llama.cpp accepts (typed content parts, legacy function calls).
int32_t chat_shim_msgs_parse_oaicompat(
        const char * messages_json,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len);

// Same for an OpenAI-shaped `tools` array. Output is a JSON array of
// `{"name", "description", "parameters"}`.
int32_t chat_shim_tools_parse_oaicompat(
        const char * tools_json,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len);

// Parse an OpenAI `tool_choice` ("auto", "required", "none") into a
// `chat_shim_tool_choice`.
int32_t chat_shim_tool_choice_parse_oaicompat(const char * value, int32_t * out_choice);

#ifdef __cplusplus
}
#endif
