#!/usr/bin/env bash
# Test /v1/chat/completions tool calling.
set -euo pipefail

BASE="${BASE_URL:-http://127.0.0.1:8080}"

WEATHER_TOOL='{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
      "type": "object",
      "properties": {
        "city":  {"type": "string", "description": "City name"},
        "units": {"type": "string", "enum": ["celsius","fahrenheit"], "default": "celsius"}
      },
      "required": ["city"]
    }
  }
}'

CALC_TOOL='{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Evaluate a mathematical expression.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {"type": "string", "description": "e.g. 2 + 2 * 3"}
      },
      "required": ["expression"]
    }
  }
}'

# ── 1. tool_choice=auto ──────────────────────────────────────────────────────
echo "=== tool_choice=auto (model decides) ==="
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{\"role\":\"user\",\"content\":\"What is the weather in Tokyo?\"}],
    \"tools\": [$WEATHER_TOOL],
    \"tool_choice\": \"auto\",
    \"max_tokens\": 256,
    \"temperature\": 0
  }" | python3 -m json.tool

# ── 2. tool_choice=required ──────────────────────────────────────────────────
echo ""
echo "=== tool_choice=required (must call a tool, enforced by GBNF grammar) ==="
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{\"role\":\"user\",\"content\":\"Tell me today's weather in London.\"}],
    \"tools\": [$WEATHER_TOOL],
    \"tool_choice\": \"required\",
    \"max_tokens\": 256,
    \"temperature\": 0
  }" | python3 -m json.tool

# ── 3. specific function forced ──────────────────────────────────────────────
echo ""
echo "=== tool_choice=specific function ==="
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{\"role\":\"user\",\"content\":\"What is 17 * 23?\"}],
    \"tools\": [$WEATHER_TOOL, $CALC_TOOL],
    \"tool_choice\": {\"type\":\"function\",\"function\":{\"name\":\"calculator\"}},
    \"max_tokens\": 256,
    \"temperature\": 0
  }" | python3 -m json.tool

# ── 4. tool_choice=none ──────────────────────────────────────────────────────
echo ""
echo "=== tool_choice=none (tools listed but ignored) ==="
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{\"role\":\"user\",\"content\":\"What is 2 + 2?\"}],
    \"tools\": [$CALC_TOOL],
    \"tool_choice\": \"none\",
    \"max_tokens\": 64,
    \"temperature\": 0
  }" | python3 -m json.tool

# ── 5. Multi-turn: send the tool result back ─────────────────────────────────
echo ""
echo "=== Multi-turn: tool result → final answer ==="
# Step 1: get the tool call id from the first response
FIRST=$(curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],
    \"tools\": [$WEATHER_TOOL],
    \"tool_choice\": \"required\",
    \"max_tokens\": 256,
    \"temperature\": 0
  }")
echo "Step 1 (model calls tool):"
echo "$FIRST" | python3 -m json.tool

CALL_ID=$(echo "$FIRST" | python3 -c "
import sys, json
d = json.load(sys.stdin)
calls = d['choices'][0]['message'].get('tool_calls', [])
print(calls[0]['id'] if calls else 'no_id')
")
CALL_NAME=$(echo "$FIRST" | python3 -c "
import sys, json
d = json.load(sys.stdin)
calls = d['choices'][0]['message'].get('tool_calls', [])
print(calls[0]['function']['name'] if calls else 'unknown')
")
CALL_ARGS=$(echo "$FIRST" | python3 -c "
import sys, json
d = json.load(sys.stdin)
calls = d['choices'][0]['message'].get('tool_calls', [])
print(calls[0]['function']['arguments'] if calls else '{}')
")

echo ""
echo "Call id: $CALL_ID  name: $CALL_NAME  args: $CALL_ARGS"
echo ""

# Step 2: send back tool result and get the final answer
echo "Step 2 (send result back):"
curl -sf "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\":\"user\",\"content\":\"What is the weather in Paris?\"},
      {\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{
        \"id\":\"$CALL_ID\",\"type\":\"function\",
        \"function\":{\"name\":\"$CALL_NAME\",\"arguments\":\"$CALL_ARGS\"}
      }]},
      {\"role\":\"tool\",\"content\":\"{\\\"temp\\\":18,\\\"condition\\\":\\\"sunny\\\"}\",
       \"tool_call_id\":\"$CALL_ID\"}
    ],
    \"tools\": [$WEATHER_TOOL],
    \"tool_choice\": \"none\",
    \"max_tokens\": 128,
    \"temperature\": 0.3
  }" | python3 -m json.tool

# ── 6. Streaming with tools ──────────────────────────────────────────────────
echo ""
echo "=== Streaming + tools (raw SSE) ==="
curl -sN "$BASE/v1/chat/completions" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{\"role\":\"user\",\"content\":\"Weather in Berlin?\"}],
    \"tools\": [$WEATHER_TOOL],
    \"tool_choice\": \"required\",
    \"max_tokens\": 256,
    \"temperature\": 0,
    \"stream\": true
  }"
echo ""
