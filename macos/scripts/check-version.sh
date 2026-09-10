#!/usr/bin/env bash
#
# Check that a release tag matches the package version.
#
#   scripts/check-version.sh v0.2.0
#
# The release workflow runs this before it builds anything, so a tag that does
# not match pyproject.toml fails in a second instead of after a notarization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

TAG="${1:-}"
[ -n "$TAG" ] || { echo "usage: check-version.sh vX.Y.Z" >&2; exit 1; }

TAG_VERSION="${TAG#v}"
PYPROJECT="$REPO_ROOT/pyproject.toml"
[ -f "$PYPROJECT" ] || { echo "error: no pyproject.toml at $PYPROJECT" >&2; exit 1; }

# The first `version = "..."` under [project]. Plain grep rather than a TOML
# parser so this runs with nothing installed.
PROJECT_VERSION="$(awk -F'"' '/^version[[:space:]]*=/ { print $2; exit }' "$PYPROJECT")"

if [ "$TAG_VERSION" != "$PROJECT_VERSION" ]; then
  echo "error: tag and package version disagree" >&2
  echo "  tag             $TAG (version $TAG_VERSION)" >&2
  echo "  pyproject.toml  $PROJECT_VERSION" >&2
  exit 1
fi

echo "$TAG matches pyproject.toml version $PROJECT_VERSION"
