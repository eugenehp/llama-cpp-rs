#include "shim_support_impl.hpp"

namespace llama_shim {

namespace {
// One buffer for every shim: whichever call last failed on this thread, its
// detail is here. Per-shim buffers would mean the caller had to know which
// shim to ask, and would silently return a stale message if it guessed wrong.
thread_local std::string g_last_error;
}  // namespace

void set_error(const std::string & what) {
    g_last_error = what;
}

void clear_error() {
    g_last_error.clear();
}

const std::string & last_error() {
    return g_last_error;
}

}  // namespace llama_shim

extern "C" const char * llama_shim_last_error(void) {
    return llama_shim::last_error().c_str();
}
