#!/bin/sh
# Power the Jetson off because the comma asked (jetlink/server/power.py). Run by
# jetlink-poweroff.service when the flag appears; install to /usr/local/bin.
# Speaks through stdout because logger does not reach journald on this L4T.
FLAG="${1:-/mnt/data/jetlink/poweroff}"
DRY_RUN="$(dirname "$FLAG")/poweroff-dry-run"

[ -e "$FLAG" ] || exit 0
reason="$(cat "$FLAG" 2>/dev/null)"
flag_time="$(stat -c %Y "$FLAG" 2>/dev/null || echo 0)"
boot_time="$(( $(date +%s) - $(cut -d. -f1 /proc/uptime) ))"
rm -f "$FLAG"
sync

if [ "$flag_time" -lt "$boot_time" ]; then
  echo "ignoring a poweroff flag from before this boot: $reason"
  exit 0
fi
if [ -e "$DRY_RUN" ]; then
  echo "dry run, not powering off: $reason"
  exit 0
fi
echo "powering off: $reason"
exec systemctl poweroff
