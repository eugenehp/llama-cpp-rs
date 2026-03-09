#!/usr/bin/env bash
# Test API key authentication.  Start the server with --api-key testkey.
set -euo pipefail

BASE="${BASE_URL:-http://127.0.0.1:8080}"
KEY="${API_KEY:-testkey}"

echo "=== No auth header → 401 ==="
CODE=$(curl -so /dev/null -w "%{http_code}" "$BASE/v1/models")
echo "HTTP $CODE"
[[ "$CODE" == "401" ]] && echo "✓" || echo "✗ Expected 401, got $CODE"

echo ""
echo "=== Wrong key → 401 ==="
CODE=$(curl -so /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer wrong_key" "$BASE/v1/models")
echo "HTTP $CODE"
[[ "$CODE" == "401" ]] && echo "✓" || echo "✗ Expected 401, got $CODE"

echo ""
echo "=== Correct key → 200 ==="
CODE=$(curl -so /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $KEY" "$BASE/v1/models")
echo "HTTP $CODE"
[[ "$CODE" == "200" ]] && echo "✓" || echo "✗ Expected 200, got $CODE"

echo ""
echo "=== /health is unprotected → 200 ==="
CODE=$(curl -so /dev/null -w "%{http_code}" "$BASE/health")
echo "HTTP $CODE"
[[ "$CODE" == "200" ]] && echo "✓" || echo "✗ Expected 200, got $CODE"
