#!/usr/bin/env bash
#
# Generate the Xcode project and build Release into macos/build/Jetlink.app.
#
# SIGN_IDENTITY defaults to "-" (ad hoc), which is what a machine with no
# Developer ID certificate can produce. Pass a real identity and
# DEVELOPMENT_TEAM for a release build; scripts/sign.sh then re-signs the
# nested Mach-O files with the right entitlements.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"

command -v xcodegen >/dev/null 2>&1 || { echo "error: xcodegen is missing; run: brew install xcodegen" >&2; exit 1; }

# The version the app shows. A tag like v0.2.0 becomes 0.2.0. An untagged tree
# makes `git describe --always` return a bare commit sha, which is not a legal
# CFBundleShortVersionString, so anything that does not look like a dotted
# version falls back to 0.0.0.
JETLINK_VERSION="${JETLINK_VERSION:-$(git -C "$MACOS_DIR" describe --tags --always --dirty 2>/dev/null || echo 0.0.0)}"
JETLINK_VERSION="${JETLINK_VERSION#v}"
case "$JETLINK_VERSION" in
  [0-9]*.[0-9]*) ;;
  *) JETLINK_VERSION=0.0.0 ;;
esac
JETLINK_BUILD="${JETLINK_BUILD:-$(git -C "$MACOS_DIR" rev-list --count HEAD 2>/dev/null || echo 1)}"
export JETLINK_VERSION JETLINK_BUILD

echo "==> generating the project (version $JETLINK_VERSION, build $JETLINK_BUILD)"
xcodegen generate --spec "$MACOS_DIR/project.yml" --project "$MACOS_DIR" --quiet

echo "==> building"
XCODEBUILD_ARGS=(
  -project "$MACOS_DIR/Jetlink.xcodeproj"
  -scheme Jetlink
  -configuration Release
  -derivedDataPath "$MACOS_DIR/build/DerivedData"
  build
  "CODE_SIGN_IDENTITY=${SIGN_IDENTITY:--}"
  CODE_SIGNING_ALLOWED=YES
)
if [ -n "${DEVELOPMENT_TEAM:-}" ]; then
  XCODEBUILD_ARGS+=("DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM")
fi
xcodebuild "${XCODEBUILD_ARGS[@]}"

PRODUCT="$MACOS_DIR/build/DerivedData/Build/Products/Release/Jetlink.app"
[ -d "$PRODUCT" ] || { echo "error: no product at $PRODUCT" >&2; exit 1; }

rm -rf "$MACOS_DIR/build/Jetlink.app"
ditto "$PRODUCT" "$MACOS_DIR/build/Jetlink.app"
echo "$MACOS_DIR/build/Jetlink.app"
