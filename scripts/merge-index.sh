#!/usr/bin/env bash
# Merge per-arch tags IMAGE-amd64 / IMAGE-arm64 (etc.) into a multi-arch IMAGE index.
# Usage: merge-index.sh <image:tag> [arch ...]
# Default arches: amd64 arm64
set -euo pipefail

IMAGE="${1:?usage: merge-index.sh <image:tag> [arch ...]}"
shift || true
if [ "$#" -eq 0 ]; then
    set -- amd64 arm64
fi

command -v crane >/dev/null || { echo "crane not found" >&2; exit 1; }

args=()
for arch in "$@"; do
    ref="${IMAGE}-${arch}"
    if ! crane manifest "$ref" >/dev/null 2>&1; then
        echo "missing per-arch manifest: $ref" >&2
        exit 1
    fi
    args+=(-m "$ref")
done

echo "indexing ${args[*]} -> $IMAGE"
crane index append "${args[@]}" -t "$IMAGE"
echo "pushed multi-arch $IMAGE ($(crane digest "$IMAGE"))"
