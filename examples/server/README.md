# openai-server

A minimal OpenAI-compatible chat completion server backed by `llama-cpp-4`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status":"ok"}` |
| `GET` | `/v1/models` | Lists the loaded model |
| `POST` | `/v1/chat/completions` | Chat completion (non-streaming) |

## Usage

### Local model

```console
cargo run -p openai-server -- local path/to/model.gguf
```

### Hugging Face — explicit file

```console
cargo run -p openai-server -- hf-model TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf
```

### Hugging Face — interactive selection

Omit the filename and the server will list all available quantizations
(collapsing sharded files into single entries) and prompt you to pick one.
All shards for the chosen quantization are downloaded before the server starts.

```console
cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF
```

```
Available models in repo:
   1)  BF16  [17 shards]
   2)  MXFP4_MOE  [6 shards]
   3)  Q3_K_M  [5 shards]
   4)  Q3_K_S  [5 shards]
   5)  Q4_K_M  [6 shards]
   ...

Select a model [1–N]: 5

Downloading: Q4_K_M  [6 shards]
  shard 1/6: Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf
  ...
```

When stdin is not a terminal (e.g. piped or in CI) the best quantization is
auto-selected according to the preference order:
`Q4_K_M` > `Q4_K_S` > `Q4_0` > `Q5_K_M` > … > `Q2_K`.

### Options

```
Options:
      --host <HOST>              Host to listen on [default: 127.0.0.1]
      --port <PORT>              Port to listen on [default: 8080]
      --n-gpu-layers <N>         GPU layers to offload (0 = CPU only) [default: 0]
  -c, --ctx-size <CTX_SIZE>      Context size override
```

### GPU

```console
cargo run -p openai-server --features cuda -- --n-gpu-layers 35 local model.gguf
```

## Example request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is 2 + 2?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Supported request fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | array | — | Chat messages (required) |
| `max_tokens` | integer | 1024 | Max tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature (0 = greedy) |
| `top_p` | float | 1.0 | Nucleus sampling |
| `top_k` | integer | 0 | Top-k sampling (0 = disabled) |
| `seed` | integer | 0 | RNG seed for reproducibility |
| `stop` | string \| string[] | — | Stop sequences |
| `grammar` | string | — | GBNF grammar to constrain output |
| `chat_template` | string | — | Override the model's Jinja template |
| `stream` | bool | false | Not yet supported (returns error) |
