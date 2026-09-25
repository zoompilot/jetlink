#!/usr/bin/env bash
# Run the installer scenarios (scenarios.sh) in throwaway Ubuntu 24.04 and
# 22.04 containers. Needs Docker; changes nothing on this machine.
#
#   tests/installer/run.sh            both releases
#   tests/installer/run.sh 24.04      one
#
# The tree under test is what git tracks plus new files it does not ignore, so
# uncommitted work is tested and the multi-gigabyte model caches stay out.
set -euo pipefail
cd "$(dirname "$0")/../.."

releases=("${@:-24.04 22.04}")
# shellcheck disable=SC2206
[ $# -eq 0 ] && releases=(24.04 22.04)

tree="$(mktemp -d)"
trap 'rm -rf "$tree"' EXIT
git ls-files -z --cached --others --exclude-standard | xargs -0 tar -cf - | tar -xf - -C "$tree"

status=0
for release in "${releases[@]}"; do
  image="jetlink-installer-test:$release"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    printf 'FROM ubuntu:%s\nRUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*\n' "$release" \
      | docker build -q -t "$image" - >/dev/null
  fi
  docker run --rm -v "$tree:/src:ro" "$image" bash /src/tests/installer/scenarios.sh || status=1
done
exit "$status"
