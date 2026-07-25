#!/usr/bin/env bash
# Publish a base image whose only delta from a HUGE upstream image is our wrapper
# code, without ever unpacking that upstream image.
#
# `docker build` has to unpack the base's whole root filesystem just to run one
# COPY. beclab/nvidia-nemo:26.02 is 25.7 GB compressed and ~55 GB unpacked, which
# does not fit on a GitHub-hosted runner (one ~72 GB filesystem; /mnt is a
# directory on it, not a second disk). crane works at the manifest level instead:
# it uploads our few-KB layer, cross-repo-mounts every upstream blob inside the
# registry, and rewrites the config. Disk cost is the tarball; wall clock is
# seconds.
#
# The trade-off: nothing RUNs at build time, so the upstream image must already
# carry every dependency and there is no build-time import check — wrapper/app.py
# fails loudly at startup instead. Only use this for a base that would otherwise
# be unbuildable; bases with a Dockerfile keep the normal path.
#
#   BASE=nemo IMAGE=docker.io/<ns>/audio-nemo:<tag> scripts/append-image.sh
#
# Requires crane (brew install crane) and a registry login (docker login).
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

# Every base exposes the interpreter that can import its engine as `audio-python`;
# elsewhere that is a symlink made by a RUN, here it has to be a file in the layer.
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
