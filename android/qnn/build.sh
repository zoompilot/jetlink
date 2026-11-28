#!/usr/bin/env bash
# Use one locally obtained SDK for headers AND every packaged QNN library.
# No vendor code or model is downloaded or stored in this repository.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the matching SDK}"
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to the Android NDK}"
OUT=${1:?usage: android/qnn/build.sh NEW_OUTPUT_JNILIBS_DIRECTORY}
[[ ! -e "$OUT" ]] || { echo "Output must not exist: $OUT" >&2; exit 1; }
bash "$HERE/test.sh"
INCLUDE="$QNN_SDK_ROOT/include/QNN"
[[ -d "$INCLUDE" ]] || INCLUDE="$QNN_SDK_ROOT/include"
TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
mkdir -p "$OUT/arm64-v8a"
"$TOOLCHAIN/aarch64-linux-android31-clang" -std=gnu11 -Wall -Wextra -Werror \
  -I "$INCLUDE" -fPIC -shared "$HERE/htp_no_activation_fusion.c" \
  -ldl -pthread -Wl,-z,relro,-z,now -o "$OUT/arm64-v8a/libJetlinkQnnHtpNoFusion.so"
python3 - "$QNN_SDK_ROOT" "$OUT/arm64-v8a" <<'PY'
import hashlib
from pathlib import Path
import re
import shutil
import sys
sdk, out = map(Path, sys.argv[1:])
core = {'libQnnHtp.so', 'libQnnGpu.so', 'libQnnSystem.so', 'libQnnHtpPrepare.so', 'libQnnIr.so', 'libQnnSaver.so'}
found = {}
for path in sdk.rglob('libQnn*.so'):
    name = path.name
    skel = re.fullmatch(r'libQnnHtpV\d+Skel\.so', name)
    stub = re.fullmatch(r'libQnnHtpV\d+Stub\.so', name)
    if name not in core and not skel and not stub:
        continue
    if skel:
        if 'unsigned' not in path.parts:
            continue
    elif 'aarch64-android' not in path.parts:
        continue
    with path.open('rb') as stream:
        header = stream.read(20)
    machine = 164 if skel else 183
    if header[:4] != b'\x7fELF' or header[5] != 1 or int.from_bytes(header[18:20], 'little') != machine:
        continue
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if name in found and found[name][0] != digest:
        raise SystemExit(f'Ambiguous SDK library: {name}; choose one SDK root')
    found[name] = (digest, path)
missing = core - found.keys()
stubs = {name.removesuffix('Stub.so') for name in found if name.endswith('Stub.so')}
skels = {name.removesuffix('Skel.so') for name in found if name.endswith('Skel.so')}
if missing or not stubs or stubs != skels:
    raise SystemExit(f'Incomplete matching SDK: missing={sorted(missing)}, stubs={sorted(stubs)}, skels={sorted(skels)}')
for _, path in found.values():
    shutil.copyfile(path, out / path.name)
print(f'Staged adapter and {len(found)} libraries from one SDK; Gradle fingerprints their actual bytes for cache identity.')
PY
