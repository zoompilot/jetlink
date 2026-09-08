#!/usr/bin/env bash
# Serve from a Mac. Makes a venv on first run, then serves over TCP with
# CoreML through onnxruntime, the frame that makes the budget (39 ms on an
# M1 Pro, docs/platforms.md). JETLINK_BACKEND=tinygrad picks tinygrad on
# Metal instead: 66 ms a frame there, but a one-second start against CoreML's
# nine minutes. JETLINK_TRANSPORT=usb serves the comma on a USB-A port over
# an A-to-C cable.
#
# caffeinate -s: an idle Mac sleeps, and nothing wakes it on a USB edge the way
# the Jetson's hub does, so it is held awake for as long as the server runs.
#
# Build a model ahead of the first connect with:
#   scripts/run-mac.sh --build /path/to/big_driving_supercombo.onnx
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${JETLINK_VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -e ".[ort,tinygrad,usb]"
fi
exec caffeinate -s "$VENV/bin/python" -m jetlink.server.main --backend "${JETLINK_BACKEND:-auto}" \
  --transport "${JETLINK_TRANSPORT:-tcp}" "$@"
