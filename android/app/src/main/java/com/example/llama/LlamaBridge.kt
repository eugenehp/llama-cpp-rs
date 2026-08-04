package com.example.llama

/**
 * JNI bridge to the Rust `llama-jni` crate (`libllama_jni.so`).
 *
 * The native symbol is `Java_com_example_llama_LlamaBridge_generate`, defined in
 * `android/llama-jni/src/lib.rs`.
 */
object LlamaBridge {
    init {
        System.loadLibrary("llama_jni")
    }

    /**
     * Load the GGUF at [modelPath], greedily generate up to [maxNewTokens] from
     * [prompt], and return the generated text. On failure the returned string is
     * prefixed with `error:` rather than throwing across the FFI boundary.
     */
    external fun generate(modelPath: String, prompt: String, maxNewTokens: Int): String
}
