#!/usr/bin/env bash
# Fail fast when Cargo.lock resolves bindgen and prettyplease onto different
# major versions of `syn`.
#
# bindgen 0.73 declares:
#
#     prettyplease = ">=0.2.7, <0.4"
#     syn          = ">=2, <4"
#
# and lets cargo resolve them independently. prettyplease 0.2 is built on syn 2
# and prettyplease 0.3 on syn 3, so cargo is free to pair bindgen's syn 2 with
# prettyplease 0.3 — a combination that cannot compile. When it happens the
# error surfaces inside bindgen's own source, not ours:
#
#     error[E0277]: the trait bound `syn::file::File: syn::parse_quote::ParseQuote`
#                   is not satisfied
#        --> .../bindgen-0.73.2/lib.rs:1017:50
#
# which gives no hint that the cause is a lockfile pairing. Nothing in this repo
# picks the combination, so ordinary lockfile churn flips it; it has done so
# more than once. This check turns twenty minutes of confusing CI into one line.
#
# Usage: ./scripts/check-bindgen-syn.sh [path/to/Cargo.lock]
set -euo pipefail

LOCK="${1:-$(cd "$(dirname "$0")/.." && pwd)/Cargo.lock}"
[[ -f "$LOCK" ]] || { echo "no such lockfile: $LOCK" >&2; exit 2; }

# The `syn "X.Y.Z"` entry inside a given package's dependency list. Lock entries
# name a version only when several of that crate are present, so a bare "syn"
# means there is exactly one and no mismatch is possible.
dep_of() {  # dep_of <package> <dependency>
  awk -v pkg="$1" -v dep="$2" '
    $0 == "name = \"" pkg "\"" { inpkg = 1; next }
    inpkg && /^\[\[package\]\]/ { exit }
    inpkg && $0 ~ "\"" dep " " { gsub(/[" ,]/, ""); sub("^" dep, ""); print; exit }
  ' "$LOCK"
}

version_of() {  # version_of <package> -> every version present
  awk -v pkg="$1" '
    $0 == "name = \"" pkg "\"" { want = 1; next }
    want && /^version = / { gsub(/[" ]/, ""); sub(/^version=/, ""); print; want = 0 }
  ' "$LOCK"
}

bindgen_syn=$(dep_of bindgen syn)
pp_versions=$(version_of prettyplease)
pp_syn=$(dep_of prettyplease syn)

# Only one syn in the graph: nothing can disagree.
if [[ -z "$bindgen_syn" && -z "$pp_syn" ]]; then
  echo "bindgen/syn check: one syn in the graph, nothing to reconcile"
  exit 0
fi

major() { cut -d. -f1 <<<"$1"; }

# prettyplease 0.2 -> syn 2, 0.3 -> syn 3. Derive the expectation from the
# version when the lock did not have to disambiguate syn for it.
if [[ -z "$pp_syn" ]]; then
  case "$pp_versions" in
    0.2.*) pp_syn="2" ;;
    0.3.*) pp_syn="3" ;;
    *)     echo "bindgen/syn check: unrecognised prettyplease $pp_versions, skipping"; exit 0 ;;
  esac
fi

if [[ "$(major "$bindgen_syn")" != "$(major "$pp_syn")" ]]; then
  cat >&2 <<EOF
error: Cargo.lock pairs bindgen and prettyplease with incompatible syn majors.

  bindgen       -> syn $bindgen_syn
  prettyplease $pp_versions -> syn $pp_syn

This does not compile. bindgen calls prettyplease::unparse() with its own
syn::File, so both must be built against the same syn major.

Fix by re-resolving the pair, e.g.:

  cargo update -p bindgen -p prettyplease -p syn

then re-run this check. See the comment at the top of this script.
EOF
  exit 1
fi

echo "bindgen/syn check: ok (bindgen syn $bindgen_syn, prettyplease $pp_versions syn $pp_syn)"
