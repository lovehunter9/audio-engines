#!/usr/bin/env bash
# Print the deps image ref for a base: <same repo as the runtime image>:deps-<hash of deps.Dockerfile>.
#
# The tag is the file's content hash, which is what makes a stale deps image impossible: change the
# recipe and the tag it resolves to simply does not exist yet, so the build makes it. Same repo as
# the runtime image on purpose, so pushing the runtime tag reuses blobs already there.
#
#   scripts/deps-image.sh <base> docker.io/<ns>/audio-<base>
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
