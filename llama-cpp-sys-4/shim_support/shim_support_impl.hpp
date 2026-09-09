#pragma once

// C++ helpers every shim shares. Not part of the bindgen surface — include it
// from a shim's `.cpp`, never from a header `wrapper.h` pulls in.

#include "shim_support.h"

#include <cstring>
#include <string>
#include <vector>

namespace llama_shim {

// Record / clear the thread-local detail behind `llama_shim_last_error`.
// Defined in shim_support.cpp so every shim shares one buffer.
void set_error(const std::string & what);
void clear_error();
const std::string & last_error();

// Run `fn`, converting any escaping C++ exception into a status code.
//
// Every fallible entry point must go through this: unwinding a C++ exception
// across `extern "C"` into Rust is undefined behaviour, and in practice aborts
// the process with "Rust cannot catch foreign exceptions".
//
// `fn` must return `int32_t` explicitly — a lambda that mixes an enum constant
// and an `int32_t` return will not compile otherwise.
template <typename Fn> int32_t guard(Fn && fn) {
    clear_error();
    try {
        return fn();
    } catch (const std::exception & e) {
        set_error(e.what());
        return LLAMA_SHIM_THROWN;
    } catch (...) {
        set_error("unknown C++ exception");
        return LLAMA_SHIM_THROWN;
    }
}

// Copy `src` into the caller's buffer under the size-then-fill protocol.
//
// Always reports the size needed, so a NULL or short buffer is a query rather
// than an error the caller has to tell apart from a real failure.
inline int32_t emit(const std::string & src, char * out_buf, size_t out_len, size_t * expected_len) {
    const size_t needed = src.size() + 1;
    if (expected_len) {
        *expected_len = needed;
    }
    if (!out_buf || out_len < needed) {
        return LLAMA_SHIM_BUFFER_TOO_SMALL;
    }
    std::memcpy(out_buf, src.data(), src.size());
    out_buf[src.size()] = '\0';
    return LLAMA_SHIM_OK;
}

// Same protocol for an array of tokens. Reports the true count so the caller
// can size up and retry rather than silently receiving a truncated draft.
template <typename T>
int32_t emit_tokens(const std::vector<T> & src, int32_t * out, size_t out_cap, size_t * out_len) {
    if (out_len) {
        *out_len = src.size();
    }
    if (!out || out_cap < src.size()) {
        return LLAMA_SHIM_BUFFER_TOO_SMALL;
    }
    for (size_t i = 0; i < src.size(); i++) {
        out[i] = static_cast<int32_t>(src[i]);
    }
    return LLAMA_SHIM_OK;
}

// A NULL `const char *` from Rust means "absent"; treat it as empty.
inline std::string str_or_empty(const char * s) {
    return s ? std::string(s) : std::string();
}

inline bool blank(const char * s) {
    return !s || *s == '\0';
}

}  // namespace llama_shim
