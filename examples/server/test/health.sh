#!/usr/bin/env bash
# Smoke-test the /health endpoint.
set -euo pipefail

BASE="${BASE_URL:-http://127.0.0.1:8080}"

echo "=== GET /health ==="
curl -sf "$BASE/health" | python3 -m json.tool

echo "=== GET /v1/models ==="
curl -sf "$BASE/v1/models" | python3 -m json.tool
