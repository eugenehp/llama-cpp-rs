# openai-server

An OpenAI-compatible HTTP server backed by `llama-cpp-4`. For a quick overview
see the [root README server section](../../README.md#openai-compatible-server).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health`, `/v1/health` | Liveness check — `{"status":"ok"}` |
| `GET`  | `/v1/models` | List loaded model with context/embedding dimensions |
| `POST` | `/v1/chat/completions`, `/chat/completions` | Chat completion · streaming · tool calling |
| `POST` | `/v1/completions`, `/completions` | Raw text completion · streaming |
| `POST` | `/v1/embeddings`, `/embeddings` | Dense embedding vectors (L2-normalised) |
| `POST` | `/tokenize` | Tokenize text (llama.cpp-compatible) |
| `POST` | `/detokenize` | Detokenize token ids |
| `POST` | `/v1/files` | Upload image/audio for multimodal (`--mmproj`) |
| `GET`  | `/v1/files` | List uploaded files |
| `GET`  | `/v1/files/{id}` | File metadata |
| `GET`  | `/v1/files/{id}/content` | Download file bytes |
| `DELETE` | `/v1/files/{id}` | Delete uploaded file |

---

## Quick start

```console
# Local GGUF
cargo run -p openai-server -- local path/to/model.gguf

# Hugging Face — pick quant interactively
cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF

# Hugging Face — name the quant directly (downloads all shards)
cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF Q4_K_M

# Hugging Face — exact filename
cargo run -p openai-server -- \
    hf-model TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf

# GPU offload + auth key + custom port
cargo run -p openai-server --features metal -- \
    --n-gpu-layers 99 --api-key mysecret --port 11434 \
    hf-model bartowski/Llama-3.2-3B-Instruct-GGUF Q4_K_M
```

### Hugging Face interactive picker

When a repo has multiple quantizations (e.g. 194 files across BF16, Q3, Q4…)
and you omit the model name, the server lists all groups and prompts:

```
Available models in repo:
   1)  BF16  [17 shards]
   2)  Q3_K_M  [5 shards]
   3)  Q4_K_M  [6 shards]  ← auto-picked in non-TTY mode
   ...

Select a model [1–N]: 3

Downloading: Q4_K_M  [6 shards]
  shard 1/6: Q4_K_M/model-00001-of-00006.gguf
  ...
```

In non-interactive mode (piped / CI) the best quant is auto-selected:
`Q4_K_M` > `Q4_K_S` > `Q4_0` > `Q5_K_M` > … > `Q2_K`.

---

## CLI options

```
--host <HOST>            Bind address [default: 127.0.0.1]
--port <PORT>            Port [default: 8080]
--n-gpu-layers <N>       GPU layers to offload (0 = CPU only) [default: 0]
-c, --ctx-size <N>       Context length override
--api-key <KEY>          Require Authorization: Bearer <KEY> on protected routes
--parallel <N>           Max concurrent inferences [default: 1]
--print-path             Resolve model path and exit (for scripts/CI)
--mmproj <FILE>          Multimodal projector GGUF (requires `--features mtmd`)
--mmproj-n-threads <N>   Encoder thread count [default: 4]
--no-mmproj-gpu          Keep mmproj on CPU
```

---

## Authentication

When `--api-key` is set, send `Authorization: Bearer <key>` on all routes **except**
`GET /health` and `GET /v1/health` (always public).

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer mysecret" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

---

## Multimodal (`mtmd` feature)

Build and run with `--features mtmd`. If `--mmproj` is omitted, the server scans
the model directory for `mmproj-*.gguf` (same directory as the main GGUF).

1. Upload an image: `POST /v1/files` (`multipart/form-data`, field `file`).
2. Chat with an `image_url` or file reference in `messages` (see `tools.rs` normaliser).

Requires a vision-capable model + matching mmproj. Without `--mmproj`, image parts
in requests are ignored (warning logged).

---

## Upstream compatibility

Route names and `/tokenize` / `/detokenize` bodies follow
[llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
This crate does **not** implement upstream-only endpoints such as
`/v1/responses`, `/v1/messages`, `/rerank`, `/slots`, or `/props` — use
`llama-server` from the vendored submodule for those.

---

## Testing

```bash
# Unit tests (no model)
cargo test -p openai-server

# Integration tests (spawns server; needs a GGUF)
LLAMA_TEST_MODEL=/path/to/model.gguf \
  cargo test -p openai-server --test integration -- --nocapture

# Or download via the server's HF helper:
LLAMA_TEST_HF_REPO=bartowski/Llama-3.2-1B-Instruct-GGUF \
  LLAMA_TEST_HF_QUANT=Q4_K_M \
  cargo test -p openai-server --test integration -- --nocapture
```

---

## Chat completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is 2+2?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count to 5"}], "stream":true}'
```

Chunks follow the OpenAI SSE format (`data: {...}\n\ndata: [DONE]\n\n`).

---

## Tool calling

The server implements OpenAI-compatible function/tool calling.

### How it works

1. `tools` + `tool_choice` are extracted from the request.
2. Tool definitions are injected into the system message using the
   [Hermes `<tools>` format](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
   that most instruction-tuned models understand.
3. The model is instructed to emit tool calls as:
   ```
   <tool_call>{"name": "fn_name", "arguments": {"key": "value"}}</tool_call>
   ```
4. Output is scanned for `<tool_call>` blocks; if found, the response
   gets `finish_reason: "tool_calls"` and a `tool_calls` array.

### `tool_choice`

| Value | Behaviour |
|-------|-----------|
| `"auto"` (default) | Model decides whether to call a tool |
| `"none"` | Model must not call tools |
| `"required"` | Model **must** call at least one tool (enforced via GBNF grammar) |
| `{"type":"function","function":{"name":"fn"}}` | Must call that specific function |

When `"required"` or a specific function is forced, a GBNF grammar is
automatically generated to constrain the output to valid tool-call JSON —
the model physically cannot produce anything else.

### Single tool call

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

Response:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_a1b2c3",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Multi-turn (sending tool results back)

```json
{
  "messages": [
    {"role":"user",      "content":"What's the weather in Paris?"},
    {"role":"assistant", "content":null,
     "tool_calls":[{"id":"call_abc","type":"function",
                    "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
    {"role":"tool", "content":"{\"temp\":18,\"condition\":\"sunny\"}",
     "tool_call_id":"call_abc"}
  ],
  "tools": [...]
}
```

---

## Raw completions

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

---

## Embeddings

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "Bonjour le monde"]}'
```

Returns L2-normalised float32 vectors. Batch inputs are processed sequentially.

---

## Tokenize / detokenize

Compatible with [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) helpers:

```bash
curl http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello world", "add_special": false}'

curl http://127.0.0.1:8080/detokenize \
  -H "Content-Type: application/json" \
  -d '{"tokens": [1, 2, 3]}'
```

Optional: `"with_pieces": true` on `/tokenize` returns `[{"id": N, "piece": "..."}, ...]`.

---

## Supported request fields

### `/v1/chat/completions` and `/v1/completions`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `messages` / `prompt` | — | — | Required |
| `max_tokens` | integer | 1024 | Alias: `max_completion_tokens` |
| `temperature` | float | 1.0 | 0 = greedy |
| `top_p` | float | 1.0 | Nucleus sampling |
| `top_k` | integer | 0 | 0 = disabled |
| `seed` | integer | 0 | RNG seed |
| `stop` | string \| string[] | — | Stop sequences |
| `stream` | bool | false | SSE streaming |
| `grammar` | string | — | GBNF grammar |
| `chat_template` | string | — | Override model's Jinja template |
| `tools` | array | — | Tool definitions |
| `tool_choice` | string \| object | `"auto"` | Tool selection policy |

### `/v1/embeddings`

| Field | Type | Notes |
|-------|------|-------|
| `input` | string \| string[] | Required |
| `model` | string | Ignored (one model loaded) |

### `/tokenize`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `content` | string | — | Required (empty → `{"tokens":[]}`) |
| `add_special` | bool | `false` | Maps to BOS handling in `str_to_token` |
| `with_pieces` | bool | `false` | Return `[{"id", "piece"}, ...]` |

### `/detokenize`

| Field | Type | Notes |
|-------|------|-------|
| `tokens` | integer[] | Token ids to decode |
