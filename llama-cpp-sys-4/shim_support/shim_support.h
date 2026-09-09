#pragma once

// Conventions shared by every shim in this crate.
//
// `ext_shim`, `mtp_shim`, `chat_shim` and `common_shim` all wrap C++-only
// llama.cpp APIs behind `extern "C"`. They agree on two things, and both live
// here so they cannot drift apart:
//
//   * one status enum, so a caller does not have to remember which shim's
//     `-2` means "bad JSON" and which means "it threw";
//   * one error buffer, so `llama_shim_last_error()` is always the detail for
//     whatever call just failed, whichever shim it was in.
//
// The C++ helpers that implement those conventions live in
// `shim_support_impl.hpp`, which is not part of the bindgen surface.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status returned by every fallible shim entry point.
//
// Negative values are failures. `BUFFER_TOO_SMALL` is positive because it is
// not one: it is the expected answer to a size query, and callers act on it by
// allocating and calling again.
enum llama_shim_status {
    LLAMA_SHIM_OK = 0,
    // Output buffer was absent or too small; `*expected_len` holds the size
    // needed, including any terminating NUL.
    LLAMA_SHIM_BUFFER_TOO_SMALL = 1,
    // A required pointer argument was NULL, or a value was out of range.
    LLAMA_SHIM_INVALID_ARG = -1,
    // Input was not valid JSON.
    LLAMA_SHIM_BAD_JSON = -2,
    // A C++ exception escaped the underlying call and was caught at the
    // boundary. Detail is in `llama_shim_last_error`.
    LLAMA_SHIM_THROWN = -3,
};

// Detail for the most recent failure on this thread, across all shims.
//
// Points at thread-local storage that stays valid until the next shim call on
// the same thread. Never NULL; empty when there has been no failure.
const char * llama_shim_last_error(void);

#ifdef __cplusplus
}
#endif
