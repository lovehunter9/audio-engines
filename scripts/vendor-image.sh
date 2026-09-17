#!/usr/bin/env bash
# Print the pinned Hub vendor ref. One shot: bump VENDOR_VERSION only when
# the user nods a new vendor. The C++ patch is not here.
set -euo pipefail

BASE="${1:?usage: vendor-image.sh <base> <runtime-repo> [vendor|ovbase]}"
# runtime-repo is ignored: vendor lives on lovehunter9, not the qwen-ov tag.
: "${2:?usage: vendor-image.sh <base> <runtime-repo> [vendor|ovbase]}"
KIND="${3:-vendor}"

VENDOR_REPO=lovehunter9/ov-vendor
VENDOR_VERSION=v0.0.1

root="$(cd "$(dirname "$0")/.." && pwd)"
file="$root/bases/$BASE/vendor.Dockerfile"
[ -f "$file" ] || { echo "no $file" >&2; exit 1; }

case "$KIND" in
    vendor) echo "${VENDOR_REPO}:${VENDOR_VERSION}" ;;
    ovbase) echo "${VENDOR_REPO}:${VENDOR_VERSION}-base" ;;
    *) echo "usage: vendor-image.sh <base> <runtime-repo> [vendor|ovbase]" >&2; exit 1 ;;
esac
