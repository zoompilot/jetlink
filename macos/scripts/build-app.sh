#!/usr/bin/env bash
#
# Generate the Xcode project and build Release into macos/build/Jetlink.app.
#
# The build is always ad hoc. scripts/sign.sh then signs everything inside out
# with SIGN_IDENTITY and the right entitlements.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"

command -v xcodegen >/dev/null 2>&1 || { echo "error: xcodegen is missing; run: brew install xcodegen" >&2; exit 1; }

# The version the app shows. A tag like v0.2.0 becomes 0.2.0. An untagged tree
# makes `git describe --always` return a bare commit sha, which is not a legal
# CFBundleShortVersionString, so anything that does not look like a dotted
# version falls back to 0.0.0. Only v* tags count, not the moving `edge` tag.
JETLINK_VERSION="${JETLINK_VERSION:-$(git -C "$MACOS_DIR" describe --tags --match 'v[0-9]*' --always 2>/dev/null || echo 0.0.0)}"
JETLINK_VERSION="${JETLINK_VERSION#v}"
case "$JETLINK_VERSION" in
  [0-9]*.[0-9]*) ;;
  *) JETLINK_VERSION=0.0.0 ;;
esac
JETLINK_BUILD="${JETLINK_BUILD:-$(git -C "$MACOS_DIR" rev-list --count HEAD 2>/dev/null || echo 1)}"

echo "==> generating the project (version $JETLINK_VERSION, build $JETLINK_BUILD)"
# The generated project is committed, so it is always generated with the
# placeholder version; writing this commit's version into it would make the
# checked in file stale after every commit. The real values go to xcodebuild
# below, where they override the project settings.
JETLINK_VERSION=0.0.0 JETLINK_BUILD=1 xcodegen generate --spec "$MACOS_DIR/project.yml" --project "$MACOS_DIR" --quiet

echo "==> building"
XCODEBUILD_ARGS=(
  -project "$MACOS_DIR/Jetlink.xcodeproj"
  -scheme Jetlink
  -configuration Release
  -derivedDataPath "$MACOS_DIR/build/DerivedData"
  build
  "MARKETING_VERSION=$JETLINK_VERSION"
  "CURRENT_PROJECT_VERSION=$JETLINK_BUILD"
  # Apple silicon only, for the package's targets too: with no destination a
  # Release build makes them universal. The app is for the Neural Engine, and
  # JetlinkONNX says so with an #error on Intel (its Float16 does not exist there).
  "ARCHS=arm64"
  "ONLY_ACTIVE_ARCH=NO"
  # A Developer ID (or a team) on this command line reaches the Swift package
  # targets too, which sign automatically, and xcodebuild refuses the pair.
  "CODE_SIGN_IDENTITY=-"
  CODE_SIGNING_ALLOWED=YES
)
# Sparkle's feed and public key default to zoompilot/jetlink's (project.yml).
# The release workflow points the feed at its own repository; a test build can
# point both at a feed and key of its own.
for setting in JETLINK_UPDATE_FEED_URL JETLINK_UPDATE_PUBLIC_KEY; do
  if [ -n "${!setting:-}" ]; then
    XCODEBUILD_ARGS+=("$setting=${!setting}")
  fi
done
xcodebuild "${XCODEBUILD_ARGS[@]}"

PRODUCT="$MACOS_DIR/build/DerivedData/Build/Products/Release/Jetlink.app"
[ -d "$PRODUCT" ] || { echo "error: no product at $PRODUCT" >&2; exit 1; }

rm -rf "$MACOS_DIR/build/Jetlink.app"
ditto "$PRODUCT" "$MACOS_DIR/build/Jetlink.app"
# Sparkle's XPC services are for sandboxed apps, and Jetlink is not one; its
# documentation allows removing them when the framework is copied in.
SPARKLE="$MACOS_DIR/build/Jetlink.app/Contents/Frameworks/Sparkle.framework"
rm -rf "$SPARKLE/XPCServices" "$SPARKLE/Versions/B/XPCServices"
# ditto keeps the product's dates, and Xcode never updates the bundle folder's
# own, so every build looked like the first one: the Dock went on showing the
# icon that bundle had then.
touch "$MACOS_DIR/build/Jetlink.app"
echo "$MACOS_DIR/build/Jetlink.app"
