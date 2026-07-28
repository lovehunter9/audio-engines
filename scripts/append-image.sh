#!/usr/bin/env bash
# Publish an engine image as its deps image plus ONE small wrapper layer, entirely registry-side:
# nothing is pulled, unpacked or run, so a wrapper change is seconds to build and a tiny delta for
# any mirror to copy (README: why every base is built this way).
#
#   BASE=pyannote IMAGE=docker.io/<ns>/audio-pyannote:<tag> scripts/append-image.sh
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
PLATFORM="${PLATFORM:-linux/amd64}"

command -v crane >/dev/null || { echo "crane not found (brew install crane)" >&2; exit 1; }

# A base whose deps are ours starts from the deps image; nemo's starts from an upstream image
# directly, since it needs nothing installed on top.
if [ -z "${BASE_IMAGE:-}" ]; then
    BASE_IMAGE="$("$repo_root/scripts/deps-image.sh" "$BASE" "$IMAGE")"
    if ! crane manifest "$BASE_IMAGE" >/dev/null 2>&1; then
        echo "deps image $BASE_IMAGE does not exist yet: bases/$BASE/deps.Dockerfile changed," >&2
        echo "so build and push it first (CI does this automatically)." >&2
        exit 1
    fi
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/app"
cp -R "$repo_root/wrapper" "$stage/app/wrapper"
find "$stage/app" -name '__pycache__' -type d -prune -exec rm -rf {} +
paths="app"

# Only a base with nothing installed on top needs this written here; a deps image already bakes it.
if [ -n "${AUDIO_PYTHON:-}" ]; then
    mkdir -p "$stage/usr/local/bin"
    cat > "$stage/usr/local/bin/audio-python" <<EOF
#!/bin/sh
exec $AUDIO_PYTHON "\$@"
EOF
    chmod 755 "$stage/usr/local/bin/audio-python"
    paths="app usr"
fi

layer="$stage/layer.tar"
# shellcheck disable=SC2086
tar -C "$stage" -cf "$layer" $paths

echo "appending $(du -sh "$layer" | cut -f1) layer to $BASE_IMAGE -> $IMAGE"
crane append --platform "$PLATFORM" -b "$BASE_IMAGE" -f "$layer" -t "$IMAGE"

env_args=(--env "AUDIO_BASE=$BASE" --env PYTHONPATH=/app --env PYTHONUNBUFFERED=1
          --env WRAPPER_PORT=8000)
for kv in ${EXTRA_ENV:-}; do env_args+=(--env "$kv"); done

crane mutate "$IMAGE" -t "$IMAGE" \
  --workdir /app \
  "${env_args[@]}" \
  --cmd="audio-python,-m,wrapper.app" \
  --label "org.opencontainers.image.title=audio-$BASE" \
  --label "org.opencontainers.image.version=$VERSION" \
  --label "org.opencontainers.image.revision=$COMMIT" \
  --label "org.opencontainers.image.created=$BUILD_DATE" \
  --label "audio.deps=$BASE_IMAGE@$(crane digest "$BASE_IMAGE")"

echo "pushed $IMAGE ($(crane digest "$IMAGE"))"
