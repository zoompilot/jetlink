#!/usr/bin/env bash
#
# Sign Jetlink.app, inside out.
#
#   scripts/sign.sh macos/build/Jetlink.app
#   SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" scripts/sign.sh ...
#
# Gotchas:
#   - codesign --deep is not used. It signs nested code with the outer
#     entitlements, which would give the app bundle's empty set to the
#     interpreter. The notary service checks each nested Mach-O anyway, so they
#     are signed explicitly here, deepest first.
#   - The Python binaries get python.entitlements: ctypes builds executable
#     trampolines and the interpreter loads extension modules out of its own
#     prefix.
#   - Ad hoc ("-") cannot carry a secure timestamp, so --timestamp is dropped
#     in that case. --options runtime stays on either way so a local build
#     behaves like a release one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"

APP="${1:-$MACOS_DIR/build/Jetlink.app}"
SIGN_IDENTITY="${SIGN_IDENTITY:--}"
APP_ENTITLEMENTS="$MACOS_DIR/Resources/Jetlink.entitlements"
PYTHON_ENTITLEMENTS="$MACOS_DIR/Resources/python.entitlements"

[ -d "$APP" ] || { echo "error: no app bundle at $APP" >&2; exit 1; }

# A plain string rather than an array: bash 3.2, which is what /bin/bash still
# is on macOS, treats an empty array under `set -u` as an unbound variable, so
# "${TIMESTAMP[@]}" would abort every ad hoc signing run.
TIMESTAMP=--timestamp
if [ "$SIGN_IDENTITY" = "-" ]; then
  TIMESTAMP=--timestamp=none
  echo "==> signing ad hoc; this build is for local use only"
fi

PYTHON_ROOT="$APP/Contents/Resources/python"
if [ -d "$PYTHON_ROOT" ]; then
  echo "==> signing the embedded runtime"
  # Everything that could be code: shared objects, dylibs, and anything
  # executable. `file -b` then decides what is really Mach-O, because the
  # prefix is full of executable shell and Python scripts.
  MACHO_LIST="$(mktemp)"
  BIN_LIST="$(mktemp)"
  trap 'rm -f "$MACHO_LIST" "$BIN_LIST"' EXIT
  while IFS= read -r f; do
    case "$(file -b "$f")" in
      *Mach-O*) printf '%s\n' "$f" ;;
    esac
  done < <(find "$PYTHON_ROOT" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \)) > "$MACHO_LIST"

  # bin/ last: the interpreter is what everything else is loaded into, and
  # signing it before its libraries would leave stale hashes behind. Matched
  # with a case pattern rather than a regex because the bundle path is not
  # under our control and may contain regex metacharacters.
  : > "$BIN_LIST"
  while IFS= read -r f; do
    case "$f" in
      "$PYTHON_ROOT"/bin/*) ;;
      *) printf '%s\n' "$f" >> "$BIN_LIST" ;;
    esac
  done < "$MACHO_LIST"
  while IFS= read -r f; do
    case "$f" in
      "$PYTHON_ROOT"/bin/*) printf '%s\n' "$f" >> "$BIN_LIST" ;;
    esac
  done < "$MACHO_LIST"

  COUNT=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    codesign --force --options runtime "$TIMESTAMP" \
      --entitlements "$PYTHON_ENTITLEMENTS" --sign "$SIGN_IDENTITY" "$f"
    COUNT=$((COUNT + 1))
  done < "$BIN_LIST"
  echo "    signed $COUNT Mach-O files"
else
  echo "warning: no embedded runtime at $PYTHON_ROOT; signing the app only"
fi

echo "==> signing the app"
codesign --force --options runtime "$TIMESTAMP" \
  --entitlements "$APP_ENTITLEMENTS" --sign "$SIGN_IDENTITY" "$APP"

echo "==> verifying"
codesign --verify --deep --strict --verbose=2 "$APP"
if [ -f "$PYTHON_ROOT/bin/python3.14" ]; then
  codesign -d --entitlements :- "$PYTHON_ROOT/bin/python3.14"
fi

if [ "$SIGN_IDENTITY" != "-" ]; then
  # Before notarization this reports "Unnotarized Developer ID". That is
  # expected; notarize.sh is the next step.
  spctl --assess --type execute --verbose "$APP" || true
fi
