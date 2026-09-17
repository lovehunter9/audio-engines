#!/usr/bin/env bash
# Print a base's vendor ref: <runtime repo>:vendor-<hash of vendor.Dockerfile>.
# The C++ patch is NOT in this hash — patch-only deps rebuilds pull this tag.
set -euo pipefail

BASE="${1:?usage: vendor-image.sh <base> <runtime-repo> [ovbase]}"
REPO="${2:?usage: vendor-image.sh <base> <runtime-repo> [ovbase]}"
REPO="${REPO%%:*}"
KIND="${3:-vendor}"

root="$(cd "$(dirname "$0")/.." && pwd)"
file="$root/bases/$BASE/vendor.Dockerfile"
[ -f "$file" ] || { echo "no $file" >&2; exit 1; }

if command -v sha256sum >/dev/null; then
    h=$(sha256sum "$file" | cut -c1-12)
else
    h=$(shasum -a 256 "$file" | cut -c1-12)
fi
case "$KIND" in
    vendor) echo "${REPO}:vendor-${h}" ;;
    ovbase) echo "${REPO}:ovbase-${h}" ;;
    *) echo "usage: vendor-image.sh <base> <runtime-repo> [vendor|ovbase]" >&2; exit 1 ;;
esac
