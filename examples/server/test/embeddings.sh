#!/usr/bin/env bash
# Test /v1/embeddings and verify basic properties of the returned vectors.
set -euo pipefail

BASE="${BASE_URL:-http://127.0.0.1:8080}"

echo "=== Single string embedding ==="
RESP=$(curl -sf "$BASE/v1/embeddings" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!"}')
echo "$RESP" | python3 -m json.tool

# Verify vector length and L2 norm ≈ 1
echo ""
echo "=== Vector sanity checks ==="
echo "$RESP" | python3 - << 'PY'
import sys, json, math

data = json.load(sys.stdin)
vec = data["data"][0]["embedding"]
print(f"  Embedding dimension : {len(vec)}")
norm = math.sqrt(sum(x*x for x in vec))
print(f"  L2 norm             : {norm:.6f}  (should be ≈ 1.0)")
assert 0.99 < norm < 1.01, f"Norm out of range: {norm}"
print("  ✓ L2-normalised")
PY

echo ""
echo "=== Batch embedding (2 inputs) ==="
BATCH=$(curl -sf "$BASE/v1/embeddings" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{"input": ["The cat sat on the mat.", "A dog ran in the park."]}')
echo "$BATCH" | python3 -m json.tool

# Cosine similarity between the two vectors
echo ""
echo "=== Cosine similarity between the two inputs ==="
echo "$BATCH" | python3 - << 'PY'
import sys, json, math

data = json.load(sys.stdin)
vecs = [d["embedding"] for d in data["data"]]
assert len(vecs) == 2

dot = sum(a*b for a, b in zip(vecs[0], vecs[1]))
# Already L2-normalised, so cosine = dot product
print(f"  Cosine similarity: {dot:.4f}")
print("  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
PY

echo ""
echo "=== Self-similarity (same string twice) ==="
SELF=$(curl -sf "$BASE/v1/embeddings" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -H "Content-Type: application/json" \
  -d '{"input": ["Paris is the capital of France.", "Paris is the capital of France."]}')
echo "$SELF" | python3 - << 'PY'
import sys, json
data = json.load(sys.stdin)
vecs = [d["embedding"] for d in data["data"]]
dot = sum(a*b for a, b in zip(vecs[0], vecs[1]))
print(f"  Self-cosine similarity: {dot:.6f}  (should be ≈ 1.0)")
assert dot > 0.99, f"Expected ≈1.0, got {dot}"
print("  ✓ Identical inputs produce identical embeddings")
PY
