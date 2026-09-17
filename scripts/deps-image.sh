#!/usr/bin/env bash
# Print a base's deps ref, <runtime repo>:deps-<hash of deps.Dockerfile>: a stale one cannot exist.
set -euo pipefail

BASE="${1:?usage: deps-image.sh <base> <runtime-repo>}"
REPO="${2:?usage: deps-image.sh <base> <runtime-repo>}"
REPO="${REPO%%:*}"

root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/bases/$BASE"
file="$dir/deps.Dockerfile"
[ -f "$file" ] || { echo "no $file" >&2; exit 1; }

# Hash deps.Dockerfile plus anything it COPY's so the tag moves when a copied file changes.
hash_inputs="$file"
for f in "$dir"/probe_*.py "$dir"/collect_*.py \
         "$root/bases/runtime/strip_unused_cuda.sh"; do
    [ -f "$f" ] || continue
    hash_inputs="$hash_inputs $f"
done

if command -v sha256sum >/dev/null; then
    h=$(cat $hash_inputs | sha256sum | cut -c1-12)
else
    h=$(cat $hash_inputs | shasum -a 256 | cut -c1-12)
fi
echo "${REPO}:deps-${h}"
