#!/usr/bin/env bash
# =============================================================================
# testbench.sh — Local test bench for llama-cpp-rs
#
# Runs unit tests (no model required) and, optionally, end-to-end integration
# tests against a live server.
#
# Usage:
#   ./scripts/testbench.sh                        # unit tests only
#   ./scripts/testbench.sh --model path/to/model.gguf
#   ./scripts/testbench.sh --hf bartowski/Llama-3.2-1B-Instruct-GGUF Q4_K_M
#   ./scripts/testbench.sh --auto                 # find any cached GGUF
#
# Environment variables:
#   LLAMA_TEST_MODEL      Path to a local GGUF file (overrides --model / --hf)
#   LLAMA_TEST_HF_REPO    HF repo id used by integration tests directly
#   LLAMA_TEST_HF_QUANT   Quant name or filename within the repo
#   CARGO_FLAGS           Extra flags forwarded to cargo (e.g. --features metal)
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

pass() { echo -e "${GREEN}[PASS]${RESET} $*"; }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; FAILURES=$((FAILURES+1)); }
info() { echo -e "${CYAN}[INFO]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
banner() { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

FAILURES=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Argument parsing ──────────────────────────────────────────────────────────
MODEL_PATH="${LLAMA_TEST_MODEL:-}"
HF_REPO=""
HF_QUANT=""
AUTO_FIND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL_PATH="$2"; shift 2 ;;
    --hf)      HF_REPO="$2"; HF_QUANT="${3:-}"; shift; shift; [[ -n "$HF_QUANT" ]] && shift ;;
    --auto)    AUTO_FIND=1; shift ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Print header ──────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════╗"
echo "║       llama-cpp-rs  test bench           ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${RESET}"
info "Workspace: $ROOT"
info "Date     : $(date)"
echo ""

# ── Step 1: Build ─────────────────────────────────────────────────────────────
banner "1 / 5  Build"
if cargo build -p openai-server ${CARGO_FLAGS:-} 2>&1; then
  pass "cargo build -p openai-server"
else
  fail "cargo build -p openai-server"
  exit 1
fi

# ── Step 2: Unit tests ────────────────────────────────────────────────────────
banner "2 / 5  Unit tests  (no model required)"
if cargo test -p openai-server ${CARGO_FLAGS:-} 2>&1; then
  pass "cargo test -p openai-server  ($(cargo test -p openai-server 2>&1 | grep -oE '[0-9]+ passed' | head -1 || echo "all tests") passed)"
else
  fail "cargo test -p openai-server"
fi

# ── Step 3: Locate / download a model ─────────────────────────────────────────
banner "3 / 5  Locate model"

if [[ -n "$HF_REPO" ]]; then
  info "Downloading from HF: $HF_REPO ${HF_QUANT:-}  (using --print-path)"
  BIN="$(find "$ROOT/target" -name "openai-server" -not -path "*/deps/*" 2>/dev/null \
        | head -1)"
  if [[ -z "$BIN" ]]; then
    info "Binary not found, building first…"
    cargo build -p openai-server ${CARGO_FLAGS:-} >/dev/null 2>&1
    BIN="$ROOT/target/debug/openai-server"
  fi
  MODEL_PATH=$(
    "$BIN" --print-path hf-model "$HF_REPO" ${HF_QUANT:-} 2>/dev/null \
      | grep -E '\.gguf$' | tail -1 || true
  )
  if [[ -z "$MODEL_PATH" ]]; then
    fail "HF download failed for $HF_REPO ${HF_QUANT:-}"
  else
    pass "Downloaded: $(basename "$MODEL_PATH")"
  fi
fi

if [[ -z "$MODEL_PATH" && $AUTO_FIND -eq 1 ]]; then
  info "Searching HF cache for any .gguf file…"
  HF_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"
  MODEL_PATH=$(find "$HF_CACHE" -name "*.gguf" -not -name "*.part" 2>/dev/null \
    | head -1 || true)
fi

if [[ -z "$MODEL_PATH" ]]; then
  warn "No model found. Integration tests will be skipped."
  warn "Provide a model with:"
  warn "  LLAMA_TEST_MODEL=/path/to/model.gguf $0"
  warn "  $0 --auto              (find any cached HF model)"
  warn "  $0 --hf bartowski/Llama-3.2-1B-Instruct-GGUF Q4_K_M"
else
  info "Model: $MODEL_PATH"
  if [[ ! -f "$MODEL_PATH" ]]; then
    fail "Model file not found: $MODEL_PATH"
    MODEL_PATH=""
  else
    # du -sh follows symlinks differently on macOS vs Linux; use ls as fallback.
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    if [[ -z "$MODEL_SIZE" || "$MODEL_SIZE" == "0B" ]]; then
      MODEL_SIZE=$(ls -lh "$MODEL_PATH" 2>/dev/null | awk '{print $5}')
    fi
    info "Size : $MODEL_SIZE"
    pass "Model file found"
  fi
fi
export LLAMA_TEST_MODEL="${MODEL_PATH:-}"

# ── Step 4: Rust integration tests ───────────────────────────────────────────
banner "4 / 5  Rust integration tests"
if [[ -z "$MODEL_PATH" ]]; then
  warn "Skipping (no model).  Set LLAMA_TEST_MODEL to enable."
else
  info "Running cargo test --test integration…"
  info "  (two server instances will start on ports 18080 and 18081)"
  if LLAMA_TEST_MODEL="$MODEL_PATH" \
     cargo test -p openai-server --test integration ${CARGO_FLAGS:-} \
      -- --test-threads=1 --nocapture 2>&1; then
    pass "cargo test --test integration"
  else
    fail "cargo test --test integration"
  fi
fi

# ── Step 5: Curl smoke tests ──────────────────────────────────────────────────
banner "5 / 5  curl smoke tests"
if [[ -z "$MODEL_PATH" ]]; then
  warn "Skipping (no model)."
else
  SMOKE_PORT=18082
  info "Starting server on port $SMOKE_PORT for smoke tests…"

  # Start server in background
  cargo run -p openai-server --quiet -- \
    --port $SMOKE_PORT local "$MODEL_PATH" \
    >/dev/null 2>&1 &
  SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null || true" EXIT

  BASE="http://127.0.0.1:$SMOKE_PORT"

  # Wait for ready
  for i in $(seq 1 120); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then break; fi
    sleep 1
  done

  run_curl_test() {
    local name="$1"; shift
    if curl -sf "$@" >/dev/null 2>&1; then
      pass "$name"
    else
      fail "$name"
    fi
  }

  # /health
  run_curl_test "GET /health" "$BASE/health"

  # /v1/models
  run_curl_test "GET /v1/models" "$BASE/v1/models"

  # /v1/chat/completions
  run_curl_test "POST /v1/chat/completions" \
    "$BASE/v1/chat/completions" \
    -X POST -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}'

  # /v1/chat/completions streaming
  SSE=$(curl -sN "$BASE/v1/chat/completions" \
    -X POST -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}' 2>&1)
  if echo "$SSE" | grep -q "data: \[DONE\]"; then
    pass "POST /v1/chat/completions (streaming)"
  else
    fail "POST /v1/chat/completions (streaming) — no [DONE] in response"
  fi

  # /v1/completions
  run_curl_test "POST /v1/completions" \
    "$BASE/v1/completions" \
    -X POST -H "Content-Type: application/json" \
    -d '{"prompt":"Hello","max_tokens":8,"temperature":0}'

  # /v1/embeddings
  EMBD=$(curl -sf "$BASE/v1/embeddings" \
    -X POST -H "Content-Type: application/json" \
    -d '{"input":"test"}' 2>&1)
  if echo "$EMBD" | python3 -c "
import sys, json, math
d = json.load(sys.stdin)
v = d['data'][0]['embedding']
n = math.sqrt(sum(x*x for x in v))
assert 0.99 < n < 1.01, f'norm={n}'
" 2>&1; then
    pass "POST /v1/embeddings (L2 norm ≈ 1.0)"
  else
    fail "POST /v1/embeddings"
  fi

  # Tool calling — required
  TOOL_RESP=$(curl -sf "$BASE/v1/chat/completions" \
    -X POST -H "Content-Type: application/json" \
    -d '{
      "messages":[{"role":"user","content":"What is the weather in Tokyo?"}],
      "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
      "tool_choice":"required",
      "max_tokens":256,"temperature":0
    }' 2>&1)
  if echo "$TOOL_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]
assert c['finish_reason'] == 'tool_calls', f\"finish={c['finish_reason']}\"
calls = c['message']['tool_calls']
assert calls, 'no tool_calls'
args = json.loads(calls[0]['function']['arguments'])
assert 'city' in args, f'no city in {args}'
print(f\"  city={args['city']!r}\")
" 2>&1; then
    pass "Tool calling (required) → tool_calls with city argument"
  else
    fail "Tool calling (required)"
  fi

  # Error handling — bad JSON
  BAD_STATUS=$(curl -so /dev/null -w "%{http_code}" \
    "$BASE/v1/chat/completions" \
    -X POST -H "Content-Type: application/json" \
    -d '{bad json}' 2>&1)
  if [[ "$BAD_STATUS" == "400" ]]; then
    pass "Invalid JSON → 400"
  else
    fail "Invalid JSON → expected 400, got $BAD_STATUS"
  fi

  kill "$SERVER_PID" 2>/dev/null || true
  trap - EXIT
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}══ Summary ══${RESET}"
if [[ $FAILURES -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}All checks passed ✓${RESET}"
  exit 0
else
  echo -e "${RED}${BOLD}$FAILURES check(s) failed ✗${RESET}"
  exit 1
fi
