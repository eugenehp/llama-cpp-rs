#!/usr/bin/env bash
# Test /v1/completions (raw text completion).
set -euo pipefail

BASE="${BASE_URL:-http://127.0.0.1:8080}"

echo "=== Non-streaming raw completion ==="
curl -sf "$BASE/v1/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of Japan is",
    "max_tokens": 16,
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo "=== Streaming raw completion ==="
curl -sN "$BASE/v1/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "One, two, three,",
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
echo ""

echo ""
echo "=== finish_reason=length ==="
RESP=$(curl -sf "$BASE/v1/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a very long story about a dragon",
    "max_tokens": 5,
    "temperature": 0
  }')
echo "$RESP" | python3 -m json.tool
REASON=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])")
echo "finish_reason = $REASON"
[[ "$REASON" == "length" ]] && echo "✓ Correctly reported 'length'" || echo "✗ Expected 'length', got '$REASON'"
