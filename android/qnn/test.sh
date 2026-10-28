#!/usr/bin/env bash
# Host-only public-API forwarding/lifetime test; no accelerator execution.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the matching SDK}"
INCLUDE="$QNN_SDK_ROOT/include/QNN"
[[ -d "$INCLUDE" ]] || INCLUDE="$QNN_SDK_ROOT/include"
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
FLAGS=(-std=gnu11 -Wall -Wextra -Werror -I "$INCLUDE")
cc "${FLAGS[@]}" -fPIC -shared "$HERE/test_htp_no_activation_fusion.c" -DFAKE_BACKEND -o "$BUILD/libQnnHtp.so"
cc "${FLAGS[@]}" -fPIC -shared "$HERE/htp_no_activation_fusion.c" -ldl -pthread -o "$BUILD/libJetlinkQnnHtpNoFusion.so"
cc "${FLAGS[@]}" "$HERE/test_htp_no_activation_fusion.c" -ldl -o "$BUILD/test-no-fusion"
"$BUILD/test-no-fusion" "$BUILD/libJetlinkQnnHtpNoFusion.so"
