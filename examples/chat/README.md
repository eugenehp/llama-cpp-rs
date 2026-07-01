# chat — interactive multi-turn REPL

Interactive chat that formats each turn with [`LlamaModel::apply_chat_template`],
so prompts match the GGUF's chat template (Llama, Qwen, Mistral, …).

This example uses [`llama_cpp_4::prelude`] for imports (`LlamaModel`,
`LlamaContext`, `LlamaChatMessage`, `apply_chat_template`, sampling, batching,
and related types) instead of listing individual crate paths.

## Run

```sh
cargo run -p chat -- local path/to/model.gguf
cargo run -p chat -- hf-model bartowski/Llama-3.2-1B-Instruct-GGUF Q4_K_M
```

With GPU backends:

```sh
cargo run -p chat --features metal -- local path/to/model.gguf
```

[`llama_cpp_4::prelude`]: https://docs.rs/llama-cpp-4/latest/llama_cpp_4/prelude/index.html
