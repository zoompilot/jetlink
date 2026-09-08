#!/usr/bin/env bash
# Serve from a Mac. Makes a venv on first run, then serves the comma over USB
# with CoreML through onnxruntime, the frame that makes the budget (39 ms on an
# M1 Pro, docs/platforms.md). JETLINK_BACKEND=tinygrad picks tinygrad on
# Metal instead: 66 ms a frame there, but a one-second start against CoreML's
# nine minutes. JETLINK_TRANSPORT=tcp serves a bench client on port 5599
# instead of the comma.
#
# USB is the default because the comma is the only client that matters and it
# only speaks USB in the car. Plug it into a USB-A port on a hub or dock with an
# A-to-C cable; the server waits for the gadget until then.
#
# The cache is models_cache/ beside this checkout unless JETLINK_CACHE says
# otherwise: a CoreML artifact is 5.5 GB and took nine minutes, so it should be
# where you can see it, back it up and move it with the checkout, not under
# ~/Library/Caches where a cleaner tool deletes it.
#
# caffeinate -s: an idle Mac sleeps, and nothing wakes it on a USB edge the way
# the Jetson's hub does, so it is held awake for as long as the server runs.
# -s only holds on AC power; on battery keep the lid open.
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
export JETLINK_CACHE="${JETLINK_CACHE:-$PWD/models_cache}"
mkdir -p "$JETLINK_CACHE"
exec caffeinate -s "$VENV/bin/python" -m jetlink.server.main --backend "${JETLINK_BACKEND:-auto}" \
  --transport "${JETLINK_TRANSPORT:-usb}" "$@"
