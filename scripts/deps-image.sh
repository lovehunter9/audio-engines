#!/usr/bin/env bash
# Print a base's deps ref, <runtime repo>:deps-<hash of deps.Dockerfile>: a stale one cannot exist.
set -euo pipefail

BASE="${1:?usage: deps-image.sh <base> <runtime-repo>}"
REPO="${2:?usage: deps-image.sh <base> <runtime-repo>}"
REPO="${REPO%%:*}"

root="$(cd "$(dirname "$0")/.." && pwd)"
file="$root/bases/$BASE/deps.Dockerfile"
[ -f "$file" ] || { echo "no $file" >&2; exit 1; }

if command -v sha256sum >/dev/null; then
    h=$(sha256sum "$file" | cut -c1-12)
else
    h=$(shasum -a 256 "$file" | cut -c1-12)
fi
echo "${REPO}:deps-${h}"
