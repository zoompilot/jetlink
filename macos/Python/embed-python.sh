#!/usr/bin/env bash
#
# Build the relocatable CPython prefix that ships inside Jetlink.app.
#
# The result is macos/build/python: a python-build-standalone install with the
# pinned wheels, tinygrad from git, the jetlink package, and Homebrew's
# libusb dropped into site-packages/usb1. `make app` rsyncs it into
# Jetlink.app/Contents/Resources/python.
#
# Gotchas this script exists to get right:
#   - The signed bundle must never be modified at runtime. Everything is
#     precompiled here with compileall and the app runs the interpreter with
#     PYTHONDONTWRITEBYTECODE=1, so no __pycache__ is written into the bundle.
#   - The onnxruntime backend runs in a multiprocessing spawn child, which
#     re-executes the interpreter. bin/python3 is a symlink to python3.14
#     inside the same prefix, which spawn handles; a symlink out to a Homebrew
#     cellar would not survive relocation.
#   - python-libusb1 looks for site-packages/usb1/libusb-1.0.dylib before it
#     falls back to the Homebrew path, so copying the dylib there is the whole
#     bundling story.
#   - Wheels are fetched with curl into $DOWNLOADS/wheels and installed with
#     --no-index --find-links. The build is then hermetic and the cache is
#     reused between runs; pip still verifies every hash from requirements.txt.
#     It also sidesteps filtered networks where a freshly extracted interpreter
#     cannot open a TLS connection of its own.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$MACOS_DIR")"

PBS_TAG="${PBS_TAG:-20260901}"
PBS_ASSET="${PBS_ASSET:-cpython-3.14.7+20260901-aarch64-apple-darwin-install_only_stripped.tar.gz}"
PBS_SHA256="${PBS_SHA256:-4632cb1a6edad9e73d3c81b6d2e69131637d995173e3e85005df14102b0592ba}"
PBS_URL="${PBS_URL:-https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_ASSET}}"
LIBUSB_DYLIB="${LIBUSB_DYLIB:-/opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib}"
OUT="${OUT:-$MACOS_DIR/build/python}"
DOWNLOADS="${DOWNLOADS:-$MACOS_DIR/build/downloads}"
WHEELHOUSE="$DOWNLOADS/wheels"
PYVER=3.14
TINYGRAD_COMMIT=e837e367aac9e1a66e689f4f32ce20ca9367df13

REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
REQUIREMENTS_GIT="$SCRIPT_DIR/requirements-git.txt"

command -v git >/dev/null 2>&1 || { echo "error: git is required to install tinygrad from source" >&2; exit 1; }
[ -f "$LIBUSB_DYLIB" ] || { echo "error: no libusb at $LIBUSB_DYLIB; run: brew install libusb" >&2; exit 1; }

mkdir -p "$DOWNLOADS" "$WHEELHOUSE"

# 1. The interpreter tarball, cached between runs and checksummed every time.
TARBALL="$DOWNLOADS/$PBS_ASSET"
if [ ! -f "$TARBALL" ]; then
  echo "==> downloading $PBS_ASSET"
  curl -fL --retry 3 --retry-delay 2 -o "$TARBALL.part" "$PBS_URL"
  mv "$TARBALL.part" "$TARBALL"
fi
ACTUAL_SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$PBS_SHA256" ]; then
  echo "error: $PBS_ASSET checksum mismatch" >&2
  echo "  expected $PBS_SHA256" >&2
  echo "  actual   $ACTUAL_SHA" >&2
  exit 1
fi

# 2. Extract. The archive's one top level directory is python/, so unpack into
#    a staging directory and move that directory into place.
echo "==> extracting into $OUT"
rm -rf "$OUT" "$OUT.staging"
mkdir -p "$OUT.staging"
tar -xzf "$TARBALL" -C "$OUT.staging"
mv "$OUT.staging/python" "$OUT"
rmdir "$OUT.staging"
PY="$OUT/bin/python3"
[ -x "$PY" ] || { echo "error: no interpreter at $PY after extracting" >&2; exit 1; }

# 3. Wheels. Prefetch every "wheel:" URL in requirements.txt, then install from
#    the local directory with the hashes still enforced.
echo "==> prefetching wheels into $WHEELHOUSE"
while read -r url; do
  [ -n "$url" ] || continue
  name="$(basename "$url")"
  if [ ! -f "$WHEELHOUSE/$name" ]; then
    echo "    $name"
    curl -fL --retry 3 --retry-delay 2 -o "$WHEELHOUSE/$name.part" "$url"
    mv "$WHEELHOUSE/$name.part" "$WHEELHOUSE/$name"
  fi
done < <(grep -oE 'https://files\.pythonhosted\.org/[^[:space:]]+' "$REQUIREMENTS")

echo "==> installing pinned wheels"
"$PY" -m pip install --no-deps --require-hashes --no-index --find-links "$WHEELHOUSE" -r "$REQUIREMENTS"

# tinygrad and the jetlink package both build with setuptools, which the line
# above just put in the prefix, so build isolation (which would need a second
# download) is off. Step 5 prunes setuptools again.
echo "==> installing tinygrad from git"
"$PY" -m pip install --no-deps --no-build-isolation -r "$REQUIREMENTS_GIT"

echo "==> installing the jetlink package"
"$PY" -m pip install --no-deps --no-build-isolation "$REPO_ROOT"

# Record the versions now: pip goes away in step 5.
PACKAGES_JSON="$("$PY" -m pip list --format json)"

# 4. libusb. ctypes loads the dylib by path, so LC_ID_DYLIB does not matter,
#    but setting it keeps otool -L honest about a relocatable bundle.
echo "==> copying libusb"
USB1_DIR="$OUT/lib/python$PYVER/site-packages/usb1"
[ -d "$USB1_DIR" ] || { echo "error: usb1 package missing at $USB1_DIR" >&2; exit 1; }
cp "$LIBUSB_DYLIB" "$USB1_DIR/libusb-1.0.dylib"
install_name_tool -id @loader_path/libusb-1.0.dylib "$USB1_DIR/libusb-1.0.dylib"
chmod 644 "$USB1_DIR/libusb-1.0.dylib"
# install_name_tool leaves Homebrew's signature invalid, and Apple silicon kills
# any process that maps a modified page: the server died with "SIGKILL (Code
# Signature Invalid)" the first time ctypes opened this file. scripts/sign.sh
# re-signs the release bundle, but Xcode's own Run copies this tree as it is.
codesign --force --sign - "$USB1_DIR/libusb-1.0.dylib"

# 5. Prune. Each of these has been checked to be unused at runtime. Every
#    *.dist-info stays: importlib.metadata version lookups read them.
echo "==> pruning"
LIB="$OUT/lib/python$PYVER"
SITE="$LIB/site-packages"
rm -rf "$LIB/test" "$LIB/idlelib" "$LIB/tkinter" "$LIB/turtledemo" "$LIB/ensurepip"
rm -rf "$SITE"/pip "$SITE"/pip-*.dist-info "$SITE"/pip-*.virtualenv
rm -rf "$SITE"/setuptools "$SITE"/setuptools-*.dist-info "$SITE"/_distutils_hack "$SITE"/distutils-precedence.pth
rm -rf "$SITE"/wheel "$SITE"/wheel-*.dist-info
find "$SITE/numpy" -type d -name tests -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$SITE/onnx/backend/test/data" "$SITE/onnx/test"
rm -rf "$OUT/share" "$OUT/include"
find "$OUT" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# 6. Precompile in tree. Read at runtime with PYTHONDONTWRITEBYTECODE=1, so the
#    signed bundle stays byte for byte what was notarized. A handful of files in
#    third party packages are samples that do not parse on this Python; they are
#    never imported, so a nonzero exit here is a warning, not a failure.
echo "==> compiling"
"$PY" -m compileall -q -j 0 "$LIB" || echo "warning: compileall reported files it could not compile (samples, not imports)"

# 7. Sanity, under the same environment the app uses (01-contracts.md section 9).
#    Every Mach-O must carry either no signature or a valid one: a signature
#    that no longer matches the file is a SIGKILL at load time, not an error.
echo "==> checking the runtime"
SIGNATURE_ERRORS=0
while IFS= read -r f; do
  if codesign -v "$f" 2>&1 | grep -q "invalid signature\|modified"; then
    echo "error: $f has a signature that no longer matches the file" >&2
    SIGNATURE_ERRORS=$((SIGNATURE_ERRORS + 1))
  fi
done < <(find "$OUT" -type f \( -name '*.so' -o -name '*.dylib' \))
[ "$SIGNATURE_ERRORS" -eq 0 ] || exit 1
CLEAN_ENV=(env -i "PATH=/usr/bin:/bin:/usr/sbin:/sbin" "HOME=$HOME" "TMPDIR=${TMPDIR:-/tmp}" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 LANG=en_US.UTF-8)
# USBContext() is what opens libusb; importing usb1 alone does not.
"${CLEAN_ENV[@]}" "$PY" -c \
  "import jetlink, numpy, onnx, usb1, tinygrad; usb1.USBContext().close(); import importlib.metadata as m; print('onnxruntime', m.version('onnxruntime'))"
BACKENDS="$("${CLEAN_ENV[@]}" "$PY" -m jetlink.server.main --list-backends)"
echo "$BACKENDS"
echo "$BACKENDS" | grep -q '\bort\b' || { echo "error: the ort backend did not come up" >&2; exit 1; }
echo "$BACKENDS" | grep -q '\btinygrad\b' || { echo "error: the tinygrad backend did not come up" >&2; exit 1; }

# 8. The manifest the app reads to show what it is running.
echo "==> writing MANIFEST.json"
# Homebrew keeps the real dylib under Cellar/libusb/<version>/lib, so the
# version is two directories above the resolved path.
LIBUSB_REAL="$(readlink -f "$LIBUSB_DYLIB" 2>/dev/null || echo "$LIBUSB_DYLIB")"
LIBUSB_VERSION="$(basename "$(dirname "$(dirname "$LIBUSB_REAL")")")"
case "$LIBUSB_VERSION" in
  [0-9]*) ;;
  *) LIBUSB_VERSION=unknown ;;
esac
JETLINK_GIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
MANIFEST_PY="$OUT/MANIFEST.json" \
PACKAGES_JSON="$PACKAGES_JSON" \
PBS_TAG="$PBS_TAG" PBS_SHA256="$PBS_SHA256" \
TINYGRAD_COMMIT="$TINYGRAD_COMMIT" LIBUSB_VERSION="$LIBUSB_VERSION" JETLINK_GIT="$JETLINK_GIT" \
"$PY" - <<'MANIFEST'
import datetime, json, os

packages = {p["name"]: p["version"] for p in json.loads(os.environ["PACKAGES_JSON"])}
for gone in ("pip", "setuptools", "wheel"):
  packages.pop(gone, None)
manifest = {
  "python": "3.14.7",
  "pbs_tag": os.environ["PBS_TAG"],
  "pbs_sha256": os.environ["PBS_SHA256"],
  "packages": packages,
  "tinygrad_commit": os.environ["TINYGRAD_COMMIT"],
  "libusb": os.environ["LIBUSB_VERSION"],
  "jetlink_git": os.environ["JETLINK_GIT"],
  "built_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(os.environ["MANIFEST_PY"], "w") as f:
  json.dump(manifest, f, indent=2, sort_keys=True)
  f.write("\n")
MANIFEST

du -sh "$OUT"
echo "==> embedded runtime ready at $OUT"
