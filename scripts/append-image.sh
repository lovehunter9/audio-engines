#!/usr/bin/env bash
# Wrapper into a base too big to unpack, via the registry: BASE=nemo IMAGE=<ref> $0 (README: why).
set -euo pipefail

BASE="${BASE:?BASE is required, e.g. BASE=nemo}"
IMAGE="${IMAGE:?IMAGE is required, e.g. IMAGE=docker.io/<ns>/audio-nemo:<tag>}"
VERSION="${VERSION:-dev}"
COMMIT="${COMMIT:-unknown}"
BUILD_DATE="${BUILD_DATE:-unknown}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
conf="$repo_root/bases/$BASE/append.env"
[ -f "$conf" ] || { echo "no $conf — this base is built from a Dockerfile" >&2; exit 1; }
# shellcheck disable=SC1090
. "$conf"
: "${BASE_IMAGE:?append.env must set BASE_IMAGE}"
: "${AUDIO_PYTHON:?append.env must set AUDIO_PYTHON}"
PLATFORM="${PLATFORM:-linux/amd64}"

command -v crane >/dev/null || { echo "crane not found (brew install crane)" >&2; exit 1; }

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/app" "$stage/usr/local/bin"
cp -R "$repo_root/wrapper" "$stage/app/wrapper"
find "$stage/app" -name '__pycache__' -type d -prune -exec rm -rf {} +

# audio-python is a RUN-made symlink in the other bases; with no RUN it must be a file in the layer.
cat > "$stage/usr/local/bin/audio-python" <<EOF
#!/bin/sh
exec $AUDIO_PYTHON "\$@"
EOF
chmod 755 "$stage/usr/local/bin/audio-python"

layer="$stage/layer.tar"
tar -C "$stage" -cf "$layer" app usr

echo "appending $(du -sh "$layer" | cut -f1) layer to $BASE_IMAGE -> $IMAGE"
crane append --platform "$PLATFORM" -b "$BASE_IMAGE" -f "$layer" -t "$IMAGE"

crane mutate "$IMAGE" -t "$IMAGE" \
  --workdir /app \
  --env "AUDIO_BASE=$BASE" \
  --env PYTHONPATH=/app \
  --env PYTHONUNBUFFERED=1 \
  --env WRAPPER_PORT=8000 \
  --cmd="audio-python,-m,wrapper.app" \
  --label "org.opencontainers.image.title=audio-$BASE" \
  --label "org.opencontainers.image.version=$VERSION" \
  --label "org.opencontainers.image.revision=$COMMIT" \
  --label "org.opencontainers.image.created=$BUILD_DATE"

echo "pushed $IMAGE ($(crane digest "$IMAGE"))"
