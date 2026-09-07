#!/usr/bin/env bash
#
# Copyright (c) 2026-, Zeph Leggett.
# This file is part of jetlink and is licensed under the MIT License.
#
# Push this package to a comma for testing, and set up the USB gadget.
#
# The repo lands at <openpilot>/jetlink_repo with a symlink <openpilot>/jetlink
# into its package dir, as launch_chffrplus.sh does for tinygrad and opendbc.
#
#   scripts/deploy_to_comma.sh comma@192.168.1.143
#
# The openpilot updater's reset --hard + clean deletes untracked files, so set
# DisableUpdates=1 on the device while testing.
set -euo pipefail

HOST="${1:?usage: deploy_to_comma.sh user@host [dest]}"
DEST="${2:-/data/openpilot/jetlink_repo}"

here="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> syncing $here -> $HOST:$DEST"
rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'tests' --exclude 'docker' \
  "$here/" "$HOST:$DEST/"

root="$(dirname "$DEST")"
echo "==> linking $root/jetlink -> $(basename "$DEST")/jetlink"
# ln -sfn refuses to replace a real directory, so clear one an older install left
ssh "$HOST" "[ -d '$root/jetlink' ] && [ ! -L '$root/jetlink' ] && rm -rf '$root/jetlink'; \
             ln -sfn '$(basename "$DEST")/jetlink' '$root/jetlink'"

echo "==> checking the package imports under the AGNOS venv"
ssh "$HOST" "cd '$root' && PYTHONPATH='$root' /usr/local/venv/bin/python3 -c '
import jetlink, jetlink.client, jetlink.transport.ffs, jetlink.queues
print(\"jetlink\", jetlink.__version__, \"ok\")'"

echo "==> configuring the USB gadget (idempotent)"
ssh "$HOST" "sudo bash $DEST/scripts/setup_gadget.sh"

cat <<'NEXT'

==> done. On the comma the gadget is configured but not bound; jetlinkd or
    modeld binds it when it opens ep0.

    A jetlinkd or modeld that was already running still has the OLD package
    imported, and manager never respawns a process that exited on its own.
    Restart the daemon yourself (get the pid first: pkill -f over ssh matches
    your own ssh command line and kills the session):

      pgrep -f "^/usr/local/venv/bin/python3 -m openpilot.sunnypilot.accelerators.jetlink.jetlinkd$"

    Sanity check with the Jetson cabled up and its server running
    (docker/run.sh --transport usb):

      ssh <comma> 'cat /sys/class/udc/*/state'      # want: configured
      ssh <jetson> 'lsusb -d 1209:0001'             # want: the gadget listed
NEXT
