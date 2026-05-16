#!/usr/bin/env bash
# MTP vs baseline benchmark script
# Usage: ./scripts/bench-mtp.sh
# Mirrors the test harness from https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090
set -euo pipefail

SERVER="$(dirname "$0")/../llama-cpp-sys-4/llama.cpp/build-mtp/bin/llama-server"
MTP_MODEL="${MTP_MODEL:-/Users/Shared/models/Qwen3.6-27B-Q4_K_M-mtp.gguf}"
BASE_MODEL="${BASE_MODEL:-/Users/Shared/models/Qwen3.6-27B-Q4_K_M.gguf}"
PORT=8099
CTX="${CTX:-16384}"
N_PREDICT="${N_PREDICT:-256}"
CURL_MAX_TIME="${CURL_MAX_TIME:-240}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"

PROMPTS=(
  "Write a Python function that implements binary search with full docstring and tests."
  "Explain the difference between transformer attention and linear attention in detail."
  "Summarize the key contributions of the paper 'Attention Is All You Need'."
  "Translate to French: The quick brown fox jumps over the lazy dog."
  "Write a short story about a robot learning to paint."
)

kill_server() {
  pkill -f "llama-server.*$PORT" 2>/dev/null || true
  lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  sleep 2
}

wait_server() {
  for i in $(seq 1 60); do
    if curl -sf --max-time 5 http://127.0.0.1:$PORT/health > /dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "Server did not start in time" >&2; exit 1
}

run_bench() {
  local label="$1"; shift
  local total_tps=0; local count=0
  echo ""
  echo "=== $label ==="
  for prompt in "${PROMPTS[@]}"; do
    result=$(curl -sf --max-time "$CURL_MAX_TIME" http://127.0.0.1:$PORT/completion \
      -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg p "$prompt" --argjson n $N_PREDICT \
        '{prompt:$p,n_predict:$n,temperature:0.7,top_k:20,top_p:0.95,cache_prompt:false,stream:false}')" )
    tps=$(echo "$result" | jq -r '.timings.predicted_per_second // 0')
    acc=$(echo "$result" | jq -r '.timings.draft_acceptance_rate // "n/a"')
    short_prompt="${prompt:0:60}..."
    printf "  prompt=%-63s  tps=%-8s  accept=%s\n" "$short_prompt" "$tps" "$acc"
    total_tps=$(echo "$total_tps + $tps" | bc)
    count=$((count+1))
  done
  avg=$(echo "scale=2; $total_tps / $count" | bc)
  echo "  --- avg tok/s: $avg ---"
}

echo "================================================"
echo " MTP vs Baseline benchmark  ($(date))"
echo "================================================"

# ensure port is free before starting
kill_server

# ---- BASELINE (no MTP) ----
echo ""
echo ">> Starting baseline server (no --spec-type draft-mtp) ..."
"$SERVER" -m "$BASE_MODEL" -ngl 99 -c $CTX -np 1 --no-webui --no-warmup \
  --port $PORT --host 127.0.0.1 2>/tmp/bench-base.log &
BASE_PID=$!
wait_server
# warmup
curl -sf --max-time 60 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"warmup","n_predict":32,"cache_prompt":false,"stream":false}' > /dev/null
run_bench "BASELINE (no MTP)  model=$BASE_MODEL"
kill_server

# ---- MTP ----
echo ""
echo ">> Starting MTP server (--spec-type draft-mtp --spec-draft-n-max $SPEC_DRAFT_N_MAX) ..."
"$SERVER" -m "$MTP_MODEL" -ngl 99 -c $CTX -np 1 --no-webui --no-warmup \
  --spec-type draft-mtp --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
  --port $PORT --host 127.0.0.1 2>/tmp/bench-mtp.log &
MTP_PID=$!
wait_server
# warmup
curl -sf --max-time 60 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"warmup","n_predict":32,"cache_prompt":false,"stream":false}' > /dev/null
run_bench "MTP (--spec-draft-n-max $SPEC_DRAFT_N_MAX)  model=$MTP_MODEL"
kill_server

echo ""
echo "Done. Server logs: /tmp/bench-base.log  /tmp/bench-mtp.log"
