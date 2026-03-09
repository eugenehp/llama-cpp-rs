#!/usr/bin/env bash
# Test /v1/chat/completions — non-streaming and streaming.
set -euo pipefail

BASE="${BASE_URL:-http://127.0.0.1:8080}"
AUTH_HEADER=""
if [[ -n "${API_KEY:-}" ]]; then
  AUTH_HEADER="-H \"Authorization: Bearer $API_KEY\""
fi

echo "=== Non-streaming chat completion ==="
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system",    "content": "You are a concise assistant."},
      {"role": "user",      "content": "What is 2 + 2? Answer in one word."}
    ],
    "max_tokens": 16,
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo "=== Streaming chat completion (raw SSE) ==="
curl -sN "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Count from 1 to 5, comma-separated."}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
echo ""

echo ""
echo "=== Stop sequence ==="
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"List: A, B, C, D, E"}],
    "max_tokens": 64,
    "stop": ["C"],
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo "=== Grammar-constrained output (JSON only) ==="
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Give me a JSON object with name and age fields."}],
    "max_tokens": 128,
    "temperature": 0,
    "grammar": "root ::= \"{\" ws \"\\\"name\\\"\" ws \":\" ws string ws \",\" ws \"\\\"age\\\"\" ws \":\" ws number ws \"}\"\nstring ::= \"\\\"\" [a-zA-Z ]+ \"\\\"\"\nnumber ::= [0-9]+\nws ::= [ \\t\\n]*"
  }' | python3 -m json.tool
