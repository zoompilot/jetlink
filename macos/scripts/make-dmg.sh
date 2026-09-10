#!/usr/bin/env bash
#
# Build the release artifacts from a signed (and ideally stapled) app:
#
#   build/Jetlink-<version>.dmg   a compressed image with an /Applications link
#   build/Jetlink-<version>.zip   the same app, for an appcast or a direct download
#   build/SHA256SUMS              checksums of both
#
# No third party tooling: hdiutil and ditto only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"

APP="${1:-$MACOS_DIR/build/Jetlink.app}"
[ -d "$APP" ] || { echo "error: no app bundle at $APP" >&2; exit 1; }

VERSION="${JETLINK_VERSION:-$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist" 2>/dev/null || echo 0.0.0)}"
SIGN_IDENTITY="${SIGN_IDENTITY:--}"
OUT_DIR="$MACOS_DIR/build"
DMG="$OUT_DIR/Jetlink-$VERSION.dmg"
ZIP="$OUT_DIR/Jetlink-$VERSION.zip"
STAGE="$OUT_DIR/dmg-stage"

echo "==> staging"
rm -rf "$STAGE"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/Jetlink.app"
ln -s /Applications "$STAGE/Applications"

echo "==> building $DMG"
rm -f "$DMG"
# hdiutil sometimes fails with "resource busy" when a previous mount has not
# finished detaching, on CI in particular. One retry clears it.
if ! hdiutil create -volname Jetlink -srcfolder "$STAGE" -ov -format UDZO "$DMG"; then
  echo "warning: hdiutil failed, retrying once"
  sleep 5
  hdiutil create -volname Jetlink -srcfolder "$STAGE" -ov -format UDZO "$DMG"
fi

echo "==> signing the image"
if [ "$SIGN_IDENTITY" = "-" ]; then
  codesign --force --sign - "$DMG"
else
  codesign --force --timestamp --sign "$SIGN_IDENTITY" "$DMG"
fi

if [ "$SIGN_IDENTITY" != "-" ] && [ -n "${NOTARY_KEY_ID:-}" ] && [ -n "${NOTARY_ISSUER_ID:-}" ] && [ -n "${NOTARY_KEY_PATH:-}" ]; then
  echo "==> notarizing the image"
  xcrun notarytool submit "$DMG" \
    --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID" \
    --wait --timeout 30m
  xcrun stapler staple "$DMG"
else
  echo "==> skipping notarization of the image (no credentials, or an ad hoc signature)"
fi

echo "==> building $ZIP"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"

echo "==> checksums"
( cd "$OUT_DIR" && shasum -a 256 "$(basename "$DMG")" "$(basename "$ZIP")" > SHA256SUMS )
cat "$OUT_DIR/SHA256SUMS"

rm -rf "$STAGE"
echo "$DMG"
echo "$ZIP"
