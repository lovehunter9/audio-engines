#!/usr/bin/env bash
# Append one wrapper layer onto deps (PLATFORM default linux/amd64; PLATFORMS=amd64,arm64 builds per-arch then merge-index).
set -euo pipefail

BASE="${BASE:?BASE is required, e.g. BASE=nemo}"
IMAGE="${IMAGE:?IMAGE is required, e.g. IMAGE=docker.io/<ns>/audio-nemo:<tag>}"
VERSION="${VERSION:-dev}"
COMMIT="${COMMIT:-unknown}"
BUILD_DATE="${BUILD_DATE:-unknown}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
conf="$repo_root/bases/$BASE/append.env"
[ -f "$conf" ] || { echo "no $conf — this base is built from a Dockerfile" >&2; exit 1; }

# Caller (CI/Makefile) wins over append.env for platform / base image selection.
_REQUESTED_PLATFORMS="${PLATFORMS-}"
_REQUESTED_PLATFORM="${PLATFORM-}"
_REQUESTED_BASE_IMAGE="${BASE_IMAGE-}"

# shellcheck disable=SC1090
. "$conf"

if [ -n "${_REQUESTED_PLATFORMS}" ]; then
    PLATFORMS="${_REQUESTED_PLATFORMS}"
elif [ -n "${_REQUESTED_PLATFORM}" ]; then
    PLATFORMS="${_REQUESTED_PLATFORM}"
elif [ -n "${PLATFORM:-}" ]; then
    PLATFORMS="${PLATFORM}"
else
    PLATFORMS="linux/amd64"
fi
if [ -n "${_REQUESTED_BASE_IMAGE}" ]; then
    BASE_IMAGE="${_REQUESTED_BASE_IMAGE}"
fi

command -v crane >/dev/null || { echo "crane not found (brew install crane)" >&2; exit 1; }

# Ours to build means start from the deps image; nemo names its upstream image in append.env.
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

IFS=',' read -r -a platform_list <<< "$PLATFORMS"
manifest_args=()
single_only=false
if [ "${#platform_list[@]}" -eq 1 ]; then
    single_only=true
fi

append_one() {
    local platform="$1"
    local dest="$2"
    local deps_digest

    echo "appending $(du -sh "$layer" | cut -f1) layer to $BASE_IMAGE ($platform) -> $dest"
    crane append --platform "$platform" -b "$BASE_IMAGE" -f "$layer" -t "$dest"

    deps_digest="$(crane digest --platform "$platform" "$BASE_IMAGE")"

    local env_args=(--env "AUDIO_BASE=$BASE" --env PYTHONPATH=/app --env PYTHONUNBUFFERED=1
              --env WRAPPER_PORT=8000)
    local kv
    for kv in ${EXTRA_ENV:-}; do env_args+=(--env "$kv"); done

    # Do NOT --set-platform here: that would let an amd64 base be mislabeled as arm64. architecture must come from the selected BASE_IMAGE platform slice.
    crane mutate --platform "$platform" "$dest" -t "$dest" \
      --workdir /app \
      "${env_args[@]}" \
      --cmd="audio-python,-m,wrapper.app" \
      --label "org.opencontainers.image.title=audio-$BASE" \
      --label "org.opencontainers.image.version=$VERSION" \
      --label "org.opencontainers.image.revision=$COMMIT" \
      --label "org.opencontainers.image.created=$BUILD_DATE" \
      --label "audio.deps=$BASE_IMAGE@$deps_digest"

    # No Dockerfile declares this config, so read it back: a typo would only surface as a wrong pod.
    crane config --platform "$platform" "$dest" | python3 -c '
import json, sys
want_base, want_arch = sys.argv[1], sys.argv[2]
doc = json.load(sys.stdin)
cfg = doc["config"]
env = dict(e.split("=", 1) for e in cfg.get("Env", []))
assert env.get("AUDIO_BASE") == want_base, env.get("AUDIO_BASE")
assert env.get("PYTHONPATH") == "/app", env.get("PYTHONPATH")
assert env.get("WRAPPER_PORT") == "8000", env.get("WRAPPER_PORT")
assert cfg.get("WorkingDir") == "/app", cfg.get("WorkingDir")
assert cfg.get("Cmd") == ["audio-python", "-m", "wrapper.app"], cfg.get("Cmd")
assert doc.get("architecture") == want_arch, (doc.get("architecture"), want_arch)
' "$BASE" "$arch"

    echo "pushed $dest ($(crane digest --platform "$platform" "$dest"))"
}

for platform in "${platform_list[@]}"; do
    platform="$(echo "$platform" | tr -d '[:space:]')"
    [ -n "$platform" ] || continue
    arch="${platform#*/}"
    if [ "$single_only" = true ]; then
        dest="$IMAGE"
    else
        dest="${IMAGE}-${arch}"
    fi
    append_one "$platform" "$dest"
    manifest_args+=(-m "$dest")
done

if [ "$single_only" = false ]; then
    echo "indexing ${manifest_args[*]} -> $IMAGE"
    crane index append "${manifest_args[@]}" -t "$IMAGE"
    echo "pushed multi-arch $IMAGE ($(crane digest "$IMAGE"))"
fi
