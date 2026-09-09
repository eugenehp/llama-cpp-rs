#include "chat_shim.h"

#include "chat.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "peg-parser.h"

#include "shim_support_impl.hpp"

#include <cstring>
#include <string>
#include <vector>

// ── error reporting ─────────────────────────────────────────────────────────
//
// Every entry point runs its body inside `guard`, which converts any escaping
// exception into a status code. Nothing throws across the C boundary.

namespace {

// Status codes, the error buffer and the size-then-fill helpers are shared by
// every shim; see shim_support_impl.hpp.
using llama_shim::blank;
using llama_shim::clear_error;
using llama_shim::emit;
using llama_shim::emit_tokens;
using llama_shim::guard;
using llama_shim::set_error;
using llama_shim::str_or_empty;

// Builds the packed multi-string buffer `apply` returns: append regions in
// order, each NUL-terminated, recording the offset of each.
struct packer {
    std::string buf;

    size_t add(const std::string & s) {
        const size_t off = buf.size();
        buf.append(s);
        buf.push_back('\0');
        return off;
    }
};

common_json parse_json_or_throw(const char * text, const char * what) {
    if (blank(text)) {
        throw std::runtime_error(std::string("empty ") + what);
    }
    return common_json::parse(text);
}

common_json strings_to_json(const std::vector<std::string> & v) {
    auto arr = common_json::array();
    for (const auto & s : v) {
        arr.push_back(s);
    }
    return arr;
}

const char * grammar_trigger_type_name(common_grammar_trigger_type type) {
    switch (type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:        return "token";
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:         return "word";
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:      return "pattern";
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: return "pattern_full";
    }
    return "unknown";
}

}  // namespace

// Owns the templates for the lifetime of the Rust handle.
struct chat_shim_templates {
    common_chat_templates_ptr ptr;
};

// ── JSON Schema → GBNF ──────────────────────────────────────────────────────

int32_t common_json_schema_to_grammar_c(
        const char * schema_json,
        bool         force_gbnf,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len) {
    if (blank(schema_json)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        common_json schema;
        try {
            schema = common_json::parse(schema_json);
        } catch (const std::exception & e) {
            set_error(e.what());
            return LLAMA_SHIM_BAD_JSON;
        }
        const std::string grammar = json_schema_to_grammar(schema, force_gbnf);
        return emit(grammar, out_buf, out_len, expected_len);
    });
}

// ── chat templates ──────────────────────────────────────────────────────────

struct chat_shim_templates * chat_shim_templates_init(
        const struct llama_model * model,
        const char *               template_override) {
    clear_error();
    try {
        auto tmpls = common_chat_templates_init(model, str_or_empty(template_override));
        if (!tmpls) {
            set_error("common_chat_templates_init returned null");
            return nullptr;
        }
        auto * handle = new chat_shim_templates{ std::move(tmpls) };
        return handle;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception");
        return nullptr;
    }
}

void chat_shim_templates_free(struct chat_shim_templates * tmpls) {
    delete tmpls;
}

int32_t chat_shim_templates_source(
        const struct chat_shim_templates * tmpls,
        const char *                       variant,
        char *                             out_buf,
        size_t                             out_len,
        size_t *                           expected_len) {
    if (!tmpls) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const std::string src =
            common_chat_templates_source(tmpls->ptr.get(), str_or_empty(variant));
        return emit(src, out_buf, out_len, expected_len);
    });
}

bool chat_shim_templates_was_explicit(const struct chat_shim_templates * tmpls) {
    if (!tmpls) {
        return false;
    }
    try {
        return common_chat_templates_was_explicit(tmpls->ptr.get());
    } catch (...) {
        return false;
    }
}

bool chat_shim_templates_support_enable_thinking(const struct chat_shim_templates * tmpls) {
    if (!tmpls) {
        return false;
    }
    try {
        return common_chat_templates_support_enable_thinking(tmpls->ptr.get());
    } catch (...) {
        return false;
    }
}

int32_t chat_shim_templates_get_caps(
        const struct chat_shim_templates * tmpls,
        char *                             out_buf,
        size_t                             out_len,
        size_t *                           expected_len) {
    if (!tmpls) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const auto caps = common_chat_templates_get_caps(tmpls->ptr.get());
        auto obj = common_json::object();
        for (const auto & [name, value] : caps) {
            obj[name] = value;
        }
        return emit(obj.dump(), out_buf, out_len, expected_len);
    });
}

// ── applying a template ─────────────────────────────────────────────────────

int32_t chat_shim_templates_apply(
        const struct chat_shim_templates *    tmpls,
        const struct chat_shim_apply_params * params,
        struct chat_shim_apply_result *       out_result,
        char *                                out_buf,
        size_t                                out_len,
        size_t *                              expected_len) {
    if (!tmpls || !params || !out_result) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        common_chat_templates_inputs inputs;

        try {
            if (!blank(params->messages_json)) {
                inputs.messages =
                    common_chat_msgs_parse_oaicompat(common_json::parse(params->messages_json));
            }
            if (!blank(params->tools_json)) {
                inputs.tools =
                    common_chat_tools_parse_oaicompat(common_json::parse(params->tools_json));
            }
            if (!blank(params->template_kwargs_json)) {
                const auto kwargs = common_json::parse(params->template_kwargs_json);
                for (const auto & [key, value] : kwargs.items()) {
                    inputs.chat_template_kwargs[key] =
                        value.is_string() ? value.get<std::string>() : value.dump();
                }
            }
        } catch (const std::exception & e) {
            set_error(e.what());
            return LLAMA_SHIM_BAD_JSON;
        }

        inputs.grammar               = str_or_empty(params->grammar);
        inputs.json_schema           = str_or_empty(params->json_schema);
        inputs.add_generation_prompt = params->add_generation_prompt;
        inputs.use_jinja             = params->use_jinja;
        inputs.parallel_tool_calls   = params->parallel_tool_calls;
        inputs.enable_thinking       = params->enable_thinking;
        inputs.add_bos               = params->add_bos;
        inputs.add_eos               = params->add_eos;
        inputs.tool_choice     = static_cast<common_chat_tool_choice>(params->tool_choice);
        inputs.reasoning_format = static_cast<common_reasoning_format>(params->reasoning_format);

        const common_chat_params applied = common_chat_templates_apply(tmpls->ptr.get(), inputs);

        auto triggers = common_json::array();
        for (const auto & trigger : applied.grammar_triggers) {
            auto entry = common_json::object();
            entry["type"]  = grammar_trigger_type_name(trigger.type);
            entry["value"] = trigger.value;
            entry["token"] = static_cast<int64_t>(trigger.token);
            triggers.push_back(entry);
        }

        packer p;
        struct chat_shim_apply_result r {};
        r.prompt_off             = p.add(applied.prompt);
        r.grammar_off            = p.add(applied.grammar);
        r.grammar_triggers_off   = p.add(triggers.dump());
        r.preserved_tokens_off   = p.add(strings_to_json(applied.preserved_tokens).dump());
        r.additional_stops_off   = p.add(strings_to_json(applied.additional_stops).dump());
        r.parser_off             = p.add(applied.parser);
        r.generation_prompt_off  = p.add(applied.generation_prompt);
        r.thinking_start_tag_off = p.add(applied.thinking_start_tag);
        r.thinking_end_tags_off  = p.add(strings_to_json(applied.thinking_end_tags).dump());
        r.format                 = static_cast<int32_t>(applied.format);
        r.grammar_lazy           = applied.grammar_lazy;
        r.supports_thinking      = applied.supports_thinking;

        // Report offsets even on a size query so the caller can size once and
        // fill on the second call without re-running the template.
        *out_result = r;

        const size_t needed = p.buf.size();
        if (expected_len) {
            *expected_len = needed;
        }
        if (!out_buf || out_len < needed) {
            return LLAMA_SHIM_BUFFER_TOO_SMALL;
        }
        std::memcpy(out_buf, p.buf.data(), needed);
        return LLAMA_SHIM_OK;
    });
}

int32_t chat_shim_format_name(
        int32_t  format,
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len) {
    return guard([&]() -> int32_t {
        const char * name = common_chat_format_name(static_cast<common_chat_format>(format));
        return emit(str_or_empty(name), out_buf, out_len, expected_len);
    });
}

// ── parsing generated text ──────────────────────────────────────────────────

int32_t chat_shim_parse(
        const struct chat_shim_parse_params * params,
        char *                                out_buf,
        size_t                                out_len,
        size_t *                              expected_len) {
    if (!params || !params->text) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        common_chat_parser_params pp;
        pp.format               = static_cast<common_chat_format>(params->format);
        pp.reasoning_format     = static_cast<common_reasoning_format>(params->reasoning_format);
        pp.reasoning_in_content = params->reasoning_in_content;
        pp.parse_tool_calls     = params->parse_tool_calls;
        pp.generation_prompt    = str_or_empty(params->generation_prompt);
        if (!blank(params->parser)) {
            pp.parser.load(params->parser);
        }

        const common_chat_msg msg = common_chat_parse(params->text, params->is_partial, pp);
        return emit(msg.to_json_oaicompat().dump(), out_buf, out_len, expected_len);
    });
}

// ── OpenAI-compatible input parsing ─────────────────────────────────────────

int32_t chat_shim_msgs_parse_oaicompat(
        const char * messages_json,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len) {
    return guard([&]() -> int32_t {
        common_json parsed;
        try {
            parsed = parse_json_or_throw(messages_json, "messages");
        } catch (const std::exception & e) {
            set_error(e.what());
            return LLAMA_SHIM_BAD_JSON;
        }
        const auto msgs = common_chat_msgs_parse_oaicompat(parsed);
        return emit(common_chat_msgs_to_json_oaicompat(msgs).dump(), out_buf, out_len, expected_len);
    });
}

int32_t chat_shim_tools_parse_oaicompat(
        const char * tools_json,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len) {
    return guard([&]() -> int32_t {
        common_json parsed;
        try {
            parsed = parse_json_or_throw(tools_json, "tools");
        } catch (const std::exception & e) {
            set_error(e.what());
            return LLAMA_SHIM_BAD_JSON;
        }
        const auto tools = common_chat_tools_parse_oaicompat(parsed);
        return emit(common_chat_tools_to_json_oaicompat(tools).dump(), out_buf, out_len,
                    expected_len);
    });
}

int32_t chat_shim_tool_choice_parse_oaicompat(const char * value, int32_t * out_choice) {
    if (blank(value) || !out_choice) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        *out_choice = static_cast<int32_t>(common_chat_tool_choice_parse_oaicompat(value));
        return LLAMA_SHIM_OK;
    });
}
