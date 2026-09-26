#!/usr/bin/env bash
# Jetlink installer, for an NVIDIA Jetson or a Linux PC with an NVIDIA GPU.
#
#   curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
#
# It checks the computer, asks a few questions, installs Docker and NVIDIA's
# container toolkit if they are missing, gets the Jetlink server, and starts it
# at boot. Running it again is safe: it offers to keep your answers and brings
# everything up to date. Afterwards the `jetlink` command manages the install.
#
# Options, for support and scripts; the questions cover everything else:
#   --yes            take the recommended answer to every question
#   --update         keep the saved answers and update, asking nothing
#   --reconfigure    ask the questions again
#   --build          build the server image here instead of downloading one
#   --image IMAGE    run this image instead
#   --ref REF        install this branch or tag (default: main)
#   --dry-run        check and ask, then show the plan without changing anything
#   --uninstall      remove Jetlink
#
# Everything runs from main(), called on the last line, so a download cut off
# part way through runs nothing, and nothing reading stdin can eat the script.
set -Eeuo pipefail

REPO_URL="${JETLINK_REPO_URL:-https://github.com/zoompilot/jetlink.git}"
RAW_URL="${JETLINK_RAW_URL:-https://raw.githubusercontent.com/zoompilot/jetlink}"
REGISTRY="${JETLINK_REGISTRY:-ghcr.io/zoompilot/jetlink}"
ETC_DIR=/etc/jetlink
CONF="$ETC_DIR/install.conf"
ENV_FILE="$ETC_DIR/server.env"
LIB_DIR=/usr/local/lib/jetlink
BIN=/usr/local/bin/jetlink
SRC_ROOT=/opt/jetlink
UNIT_DIR=/etc/systemd/system
UNIT=jetlink-server
WAKE_RULE=/etc/udev/rules.d/99-jetlink-usb-wakeup.rules
JOURNALD_DROPIN=/etc/systemd/journald.conf.d/60-jetlink.conf
# CUDA 13 needs driver 580; TensorRT needs a Turing (7.5) or newer GPU
MIN_DRIVER=580
MIN_CC=75
MIN_DISK_GB=15
SWAP_GB=8
# units that hold up boot waiting for a network the car does not have
WAIT_ONLINE_UNITS="systemd-networkd-wait-online.service NetworkManager-wait-online.service"
# where detection looks; the installer's tests point these at fakes
DT_MODEL="${JETLINK_TEST_DT_MODEL:-/proc/device-tree/model}"
MEM_SLEEP="${JETLINK_TEST_MEM_SLEEP:-/sys/power/mem_sleep}"
SWAPS="${JETLINK_TEST_SWAPS:-/proc/swaps}"
# seconds between looks at something the installer waits on
POLL_S="${JETLINK_TEST_POLL_S:-5}"
# seconds a quick registry request may take before it is abandoned and tried again
NET_TIMEOUT_S="${JETLINK_TEST_NET_TIMEOUT_S:-60}"

OPT_YES=0 OPT_UPDATE=0 OPT_RECONFIGURE=0 OPT_BUILD=0 OPT_DRY_RUN=0 OPT_UNINSTALL=0
OPT_IMAGE="" OPT_REF="${JETLINK_REF:-}"

# ---------------------------------------------------------------------------
# Output

B='' D='' G='' Y='' R='' N=''
setup_colors() {
  if [ -t 1 ] && [ "${TERM:-dumb}" != dumb ]; then
    B=$'\e[1m' D=$'\e[2m' G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' N=$'\e[0m'
  fi
}
say() { printf '%s\n' "$*"; }
good() { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
note() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
bad() { printf '  %s✗%s %s\n' "$R" "$N" "$*"; }
heading() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
die() {
  local first="$1"
  shift
  printf '\n%s%s%s\n' "$R$B" "$first" "$N"
  local line
  for line in "$@"; do printf '  %s\n' "$line"; done
  printf '\n'
  save_log
  exit 1
}

LOG="${TMPDIR:-/tmp}/jetlink-install.$$.log"
save_log() {
  [ -s "$LOG" ] || return 0
  if [ "$OPT_DRY_RUN" != 1 ] && as_root cp "$LOG" /var/log/jetlink-install.log 2>/dev/null; then
    printf '  %sThe full log is in /var/log/jetlink-install.log%s\n\n' "$D" "$N"
  else
    printf '  %sThe full log is in %s%s\n\n' "$D" "$LOG" "$N"
  fi
}

on_error() {
  local rc=$? line=$1
  trap - ERR
  printf '\n%sSomething went wrong (line %s, exit %s).%s\n' "$R$B" "$line" "$rc" "$N"
  if [ -s "$LOG" ]; then
    printf '  Last lines of the log:\n'
    tail -n 15 "$LOG" | sed 's/^/    /'
  fi
  printf '\n  Running the installer again is safe. If it keeps failing, open an issue at\n'
  printf '  https://github.com/zoompilot/jetlink/issues with the log attached.\n\n'
  save_log
  exit "$rc"
}

# `step LABEL cmd...`: run a command with its output in the log, and a spinner
# with the elapsed time so a 20 minute download still looks alive.
step() {
  local label="$1"
  shift
  if [ "$OPT_DRY_RUN" = 1 ]; then
    printf '  %s·%s %s\n' "$D" "$N" "$label"
    return 0
  fi
  printf '\n==> %s\n' "$label" >>"$LOG"
  local rc=0 t0=$SECONDS
  if [ -t 1 ]; then
    "$@" </dev/null >>"$LOG" 2>&1 &
    local pid=$! i=0 frames=$'|/-\\'
    while kill -0 "$pid" 2>/dev/null; do
      printf '\r  %s%s%s %s %s%s%s ' "$D" "${frames:i++%4:1}" "$N" "$label" "$D" "$(elapsed $((SECONDS - t0)))" "$N"
      sleep 0.25
    done
    wait "$pid" || rc=$?
    printf '\r\e[K'
  else
    "$@" </dev/null >>"$LOG" 2>&1 || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    if [ $((SECONDS - t0)) -ge 2 ]; then
      good "$label ${D}$(elapsed $((SECONDS - t0)))${N}"
    else
      good "$label"
    fi
    return 0
  fi
  bad "$label"
  printf '\n  Last lines of the log:\n'
  tail -n 20 "$LOG" | sed 's/^/    /'
  die "That step failed." "Running the installer again is safe." \
    "If it keeps failing, open an issue at https://github.com/zoompilot/jetlink/issues"
}

elapsed() {
  local s=$1
  if [ "$s" -lt 60 ]; then printf '%ss' "$s"; else printf '%dm%02ds' $((s / 60)) $((s % 60)); fi
}

# ---------------------------------------------------------------------------
# Questions. Answers are read from the terminal, not stdin, which is the
# script itself under curl | bash.

INTERACTIVE=0
open_input() {
  if [ -n "${JETLINK_INPUT:-}" ]; then  # the installer's tests
    exec 3<"$JETLINK_INPUT"
    INTERACTIVE=1
  elif [ "$OPT_YES" = 1 ] || [ "$OPT_UPDATE" = 1 ]; then
    INTERACTIVE=0
  elif [ -r /dev/tty ] && (exec 3</dev/tty) 2>/dev/null; then
    exec 3</dev/tty
    INTERACTIVE=1
  else
    die "There is no terminal to ask questions in." \
      "Run this in a terminal, or add --yes to take the recommended answers:" \
      "  curl -fsSL $RAW_URL/main/install.sh | bash -s -- --yes"
  fi
}

# The helpers below set the caller's variable by name, so their own locals are
# all __-prefixed: a local with the caller's name would take the answer.
read_answer() {
  local __ra_reply=''
  IFS= read -r __ra_reply <&3 || true
  printf -v "$1" '%s' "$__ra_reply"
}

# ask_yn VAR default(y|n) "question" ["explanation"...]
ask_yn() {
  local __yn_var=$1 __yn_def=$2 __yn_q=$3
  shift 3
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__yn_var" '%s' "$__yn_def"
    return
  fi
  printf '\n  %s%s%s\n' "$B" "$__yn_q" "$N"
  local __yn_line
  for __yn_line in "$@"; do printf '  %s%s%s\n' "$D" "$__yn_line" "$N"; done
  local __yn_hint="Y/n"
  [ "$__yn_def" = n ] && __yn_hint="y/N"
  local __yn_a
  while true; do
    printf '  [%s] ' "$__yn_hint"
    read_answer __yn_a
    __yn_a="$(printf '%s' "$__yn_a" | tr '[:upper:]' '[:lower:]')"
    case "$__yn_a" in
      '') __yn_a=$__yn_def; break ;;
      y|yes) __yn_a=y; break ;;
      n|no) __yn_a=n; break ;;
      *) printf '  Please type y or n.\n' ;;
    esac
  done
  printf -v "$__yn_var" '%s' "$__yn_a"
}

# ask_choice VAR default_number "question" "option 1" "option 2" ...
ask_choice() {
  local __ch_var=$1 __ch_def=$2 __ch_q=$3
  shift 3
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__ch_var" '%s' "$__ch_def"
    return
  fi
  printf '\n  %s%s%s\n' "$B" "$__ch_q" "$N"
  local __ch_i=1 __ch_opt
  for __ch_opt in "$@"; do
    printf '    %s%d)%s %s\n' "$B" "$__ch_i" "$N" "$__ch_opt"
    __ch_i=$((__ch_i + 1))
  done
  local __ch_a
  while true; do
    printf '  Type a number and press Enter [%s] ' "$__ch_def"
    read_answer __ch_a
    [ -z "$__ch_a" ] && __ch_a=$__ch_def
    if [[ "$__ch_a" =~ ^[0-9]+$ ]] && [ "$__ch_a" -ge 1 ] && [ "$__ch_a" -le $# ]; then break; fi
    printf '  Please type a number from 1 to %d.\n' "$#"
  done
  printf -v "$__ch_var" '%s' "$__ch_a"
}

# ---------------------------------------------------------------------------
# Root. The script runs as the user so curl | bash works without sudo, and
# asks sudo for each change.

SUDO=''
as_root() {
  if [ -z "$SUDO" ]; then "$@"; else sudo -n -- "$@"; fi
}

get_root() {
  if [ "$(id -u)" -eq 0 ]; then
    SUDO=''
    return
  fi
  command -v sudo >/dev/null 2>&1 || die "Jetlink needs administrator rights to install, and sudo is missing." \
    "Run the installer as root instead."
  SUDO=sudo
  if ! sudo -n true 2>/dev/null; then
    say ""
    say "  Jetlink needs administrator rights to install. Enter your password if asked."
    # the password prompt comes from the terminal, not the piped script
    # shellcheck disable=SC2024
    sudo -v </dev/tty || die "Could not get administrator rights."
  fi
  # a build or a download can outlast sudo's 15 minutes
  ( while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 50; done ) &
}

# root_write PATH MODE: stdin into a root-owned file
root_write() {
  local path=$1 mode=${2:-644} tmp
  tmp="$(mktemp)"
  cat >"$tmp"
  as_root install -D -m "$mode" "$tmp" "$path"
  rm -f "$tmp"
}

apt_get() {
  # a fresh Jetson runs unattended-upgrades for a while after its first boot
  as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 -y "$@"
}

# ---------------------------------------------------------------------------
# What this computer is

ARCH='' OS_ID='' OS_LIKE='' OS_CODENAME='' OS_NAME=''
JETSON=0 L4T='' L4T_MAJOR=0 L4T_MINOR=0 JETPACK='' MODEL=''
GPU_NAME='' DRIVER='' DRIVER_MAJOR=0 GPU_CC=0 GPU_PRESENT=0
FLAVOR='' PLATFORM_NAME=''
HAVE_DOCKER=0 DOCKER_VERSION='' HAVE_TOOLKIT=0 HAVE_NVIDIA_RUNTIME=0 SNAP_DOCKER=0
DISK_GB=0
DEEP_SLEEP=0
PM_BEST_ID='' PM_BEST_NAME='' PM_CURRENT=''

detect() {
  [ "$(uname -s)" = Linux ] || die "This installer is for Linux: a Jetson, or a PC running Linux." \
    "On a Mac, use the Jetlink app from https://github.com/zoompilot/jetlink/releases"
  if grep -qi microsoft /proc/version 2>/dev/null; then
    die "Windows (WSL) is not supported by the installer yet." \
      "See https://github.com/zoompilot/jetlink/blob/main/docs/platforms.md"
  fi
  ARCH="$(uname -m)"
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS_ID="${ID:-}" OS_LIKE="${ID_LIKE:-}" OS_NAME="${PRETTY_NAME:-Linux}"
    OS_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
  fi
  command -v apt-get >/dev/null 2>&1 || die "This installer needs Ubuntu or Debian (apt)." \
    "See https://github.com/zoompilot/jetlink/blob/main/docs/platforms.md for other systems."

  if [ -f /etc/nv_tegra_release ] || grep -qa tegra /proc/device-tree/compatible 2>/dev/null; then
    JETSON=1
    detect_jetson
  else
    detect_pc
  fi

  local where=/var/lib
  [ -d /var/lib/docker ] && where=/var/lib/docker
  DISK_GB=$(df -Pk "$where" | awk 'NR == 2 {printf "%d", $4 / 1048576}')
  DISK_GB="${JETLINK_TEST_FREE_GB:-$DISK_GB}"

  detect_docker
}

detect_docker() {
  HAVE_DOCKER=0 DOCKER_VERSION='' HAVE_TOOLKIT=0 HAVE_NVIDIA_RUNTIME=0 SNAP_DOCKER=0
  if [ -x /snap/bin/docker ]; then SNAP_DOCKER=1; fi
  if command -v docker >/dev/null 2>&1; then
    HAVE_DOCKER=1
    DOCKER_VERSION="$(docker --version 2>/dev/null | sed -n 's/^Docker version \([^,]*\).*/\1/p')"
    if [ "$OPT_DRY_RUN" != 1 ] && as_root docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
      HAVE_NVIDIA_RUNTIME=1
    fi
  fi
  command -v nvidia-ctk >/dev/null 2>&1 && HAVE_TOOLKIT=1
  return 0
}

detect_jetson() {
  local line rev
  line="$(head -n 1 /etc/nv_tegra_release 2>/dev/null || true)"
  if [[ "$line" =~ R([0-9]+)\ \(release\),\ REVISION:\ ([0-9.]+) ]]; then
    L4T_MAJOR="${BASH_REMATCH[1]}"
    rev="${BASH_REMATCH[2]}"
  else
    # r39 and later may drop the file; the core package has the version
    rev="$(dpkg-query -W -f '${Version}' nvidia-l4t-core 2>/dev/null || true)"
    L4T_MAJOR="${rev%%.*}"
    rev="${rev#*.}"
    rev="${rev%%-*}"
  fi
  L4T_MINOR="${rev%%.*}"
  [[ "$L4T_MAJOR" =~ ^[0-9]+$ ]] || L4T_MAJOR=0
  [[ "$L4T_MINOR" =~ ^[0-9]+$ ]] || L4T_MINOR=0
  L4T="$L4T_MAJOR.$rev"
  MODEL="$(tr -d '\0' <"$DT_MODEL" 2>/dev/null || echo "NVIDIA Jetson")"
  MODEL="${MODEL/ Engineering Reference Developer Kit/}"

  case "$L4T_MAJOR" in
    39)
      [ "$L4T_MINOR" -ge 2 ] || die "This Jetson runs Jetson Linux $L4T; Jetlink needs JetPack 7.2 (Jetson Linux 39.2) or newer." \
        "Flash JetPack 7.2.1: https://developer.nvidia.com/embedded/jetpack"
      FLAVOR=cuda JETPACK="JetPack 7" ;;
    36)
      [ "$L4T_MINOR" -ge 4 ] || die "This Jetson runs JetPack 6 with Jetson Linux $L4T, which is too old." \
        "Update to JetPack 7.2.1 (recommended) or 6.2: https://developer.nvidia.com/embedded/jetpack"
      FLAVOR=jetpack6 JETPACK="JetPack 6" ;;
    38)
      die "JetPack 7.0 and 7.1 are not supported. Update to JetPack 7.2.1:" \
        "https://developer.nvidia.com/embedded/jetpack" ;;
    *)
      die "This Jetson's software (Jetson Linux $L4T) is not supported." \
        "Flash JetPack 7.2.1 (recommended) or 6.2: https://developer.nvidia.com/embedded/jetpack" ;;
  esac
  [ "$ARCH" = aarch64 ] || die "Unexpected: a Jetson that is not aarch64 ($ARCH)."
  PLATFORM_NAME="$MODEL, $JETPACK (Jetson Linux $L4T)"

  if grep -qw deep "$MEM_SLEEP" 2>/dev/null; then DEEP_SLEEP=1; fi
  detect_power_modes
}

detect_power_modes() {
  command -v nvpmodel >/dev/null 2>&1 || return 0
  local conf=/etc/nvpmodel.conf best_rank=-1 id name rank watts
  [ -r "$conf" ] || return 0
  while read -r id name; do
    case "$name" in
      MAXN_SUPER) rank=1000 ;;
      MAXN) rank=900 ;;
      *W*)
        watts="${name%%W*}"
        watts="${watts//[!0-9]/}"
        rank="${watts:-0}" ;;
      *) rank=0 ;;
    esac
    if [ "$rank" -gt "$best_rank" ]; then
      best_rank=$rank PM_BEST_ID=$id PM_BEST_NAME=$name
    fi
  done < <(sed -n 's/.*POWER_MODEL ID=\([0-9]*\) NAME=\([^ >]*\).*/\1 \2/p' "$conf")
  PM_CURRENT="$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: *//p' | head -n 1)"
  return 0
}

detect_pc() {
  [ "$ARCH" = x86_64 ] || die "On an Arm computer, Jetlink supports NVIDIA Jetson only." \
    "This one is $ARCH and does not look like a Jetson."
  FLAVOR=cuda
  local q
  if command -v nvidia-smi >/dev/null 2>&1 \
      && q="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -n 1)" \
      && [ -n "$q" ]; then
    GPU_PRESENT=1
    GPU_NAME="$(printf '%s' "$q" | cut -d, -f1 | sed 's/^ *//; s/ *$//')"
    DRIVER="$(printf '%s' "$q" | cut -d, -f2 | tr -d ' ')"
    DRIVER_MAJOR="${DRIVER%%.*}"
    GPU_CC="$(printf '%s' "$q" | cut -d, -f3 | tr -d ' .')"
  else
    local dev
    for dev in /sys/bus/pci/devices/*; do
      if [ "$(cat "$dev/vendor" 2>/dev/null)" = 0x10de ] && grep -q '^0x03' "$dev/class" 2>/dev/null; then
        GPU_PRESENT=1
      fi
    done
    GPU_NAME="NVIDIA GPU"
  fi
  [ "$GPU_PRESENT" = 1 ] || die "No NVIDIA GPU found." \
    "Jetlink needs an NVIDIA GPU (GeForce RTX 20 series or newer) or a Jetson."
  [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] || DRIVER_MAJOR=0
  [[ "$GPU_CC" =~ ^[0-9]+$ ]] || GPU_CC=0
  if [ "$GPU_CC" -gt 0 ] && [ "$GPU_CC" -lt "$MIN_CC" ]; then
    die "The $GPU_NAME is too old for TensorRT." "Jetlink needs a GeForce RTX 20 series (Turing) or newer GPU."
  fi
  PLATFORM_NAME="$GPU_NAME, $OS_NAME"
}

# ---------------------------------------------------------------------------
# Answers, saved in install.conf so an update asks nothing

POWER='' SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0 FAST_MODE=0 ADD_SWAP=0 AUTOSTART=1
CACHE_DIR='' REF='' SOURCE='' SOURCE_DIR='' COMMIT=''
SWAP_FILE='' MASKED_UNITS='' JOURNALD_CAPPED=0 NEED_REBOOT=0
IMAGE_REF='' IMAGE_ID='' IMAGE_SOURCE='' GPU_ARGS='' GPU_REPORT=''
HAD_INSTALL=0

load_previous() {
  [ -r "$CONF" ] || return 0
  HAD_INSTALL=1
  # shellcheck disable=SC1090
  . "$CONF"
  POWER="${JETLINK_POWER:-}"
  SLEEP_AFTER="${JETLINK_SLEEP_AFTER:-0}"
  POWEROFF_WITH_COMMA="${JETLINK_POWEROFF_WITH_COMMA:-0}"
  FAST_MODE="${JETLINK_FAST_MODE:-0}"
  AUTOSTART="${JETLINK_AUTOSTART:-1}"
  CACHE_DIR="${JETLINK_CACHE_DIR:-}"
  SWAP_FILE="${JETLINK_SWAP_FILE:-}"
  MASKED_UNITS="${JETLINK_MASKED_UNITS:-}"
  JOURNALD_CAPPED="${JETLINK_JOURNALD_CAPPED:-0}"
  REF="${JETLINK_REF:-}"
  [ -n "$SWAP_FILE" ] && ADD_SWAP=1
  return 0
}

ask_questions() {
  heading "A few questions"
  say "  Press Enter to take the recommended answer."

  if [ "$JETSON" = 1 ]; then
    # always on is the recommended wiring, for a Jetson that can deep-sleep
    local prev="$POWER" def=1 choice always
    if [ "$prev" = switched ] || { [ -z "$prev" ] && [ "$DEEP_SLEEP" = 0 ]; }; then def=2; fi
    always="Always on ${D}(recommended)${N}: sleeps when the car is off to save battery, wakes when you start the car"
    [ "$DEEP_SLEEP" = 1 ] || always="Always on: stays awake when the car is off ${D}(this Jetson cannot sleep)${N}"
    ask_choice choice "$def" "How is the Jetson powered in the car?" \
      "$always" \
      "Switched: turns on and off with the car"
    if [ "$choice" = 1 ]; then
      set_always_on
      local off offdef=y
      [ "$prev" = always ] && [ "$POWEROFF_WITH_COMMA" = 0 ] && offdef=n
      ask_yn off "$offdef" "Allow the comma to shut down the Jetson to protect the car battery?" \
        "The comma does this when it shuts itself down for low battery. The Jetson then" \
        "stays off until its power is reconnected."
      if [ "$off" = y ]; then POWEROFF_WITH_COMMA=1; else POWEROFF_WITH_COMMA=0; fi
    else
      POWER=switched SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0
    fi

    AUTOSTART=1
  else
    local auto
    ask_yn auto y "Start Jetlink automatically when this computer starts?" \
      "If you say no, start it yourself with: jetlink start"
    if [ "$auto" = y ]; then AUTOSTART=1; else AUTOSTART=0; fi
  fi
}

set_always_on() {
  POWER=always SLEEP_AFTER=0
  if [ "$DEEP_SLEEP" = 1 ]; then
    SLEEP_AFTER=120
  else
    note "This Jetson's software cannot deep-sleep, so it stays awake when the car is off (about 7 W)."
  fi
}

take_defaults() {
  # --yes on a fresh install: the recommended answers
  if [ "$JETSON" = 1 ] && [ -z "$POWER" ]; then
    if [ "$DEEP_SLEEP" = 1 ]; then
      set_always_on
      POWEROFF_WITH_COMMA=1
    else
      POWER=switched SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0
    fi
  fi
  return 0
}

# Not questions: the large models need both, so every Jetson install gets
# them, updates included.
jetson_musts() {
  [ "$JETSON" = 1 ] || return 0
  FAST_MODE=0
  if [ -n "$PM_BEST_NAME" ]; then
    FAST_MODE=1
    if [ "$PM_BEST_NAME" != MAXN_SUPER ] && [[ "$MODEL" == *"Orin Nano"* ]]; then
      note "This JetPack install does not offer the Orin Nano's Super modes; JetPack 7.2.1's"
      note "installer sets them up. Jetlink still works, a little slower, in $PM_BEST_NAME."
    fi
  fi
  ADD_SWAP=0
  if [ -n "$SWAP_FILE" ]; then
    ADD_SWAP=1
  elif swap_short; then
    if [ "$DISK_GB" -ge $((MIN_DISK_GB + SWAP_GB + 5)) ]; then
      ADD_SWAP=1
    else
      note "Not enough disk space for ${SWAP_GB} GB of swap, so the 1.7 GB models may fail to prepare."
    fi
  fi
  return 0
}

# less than the swap the 1.7 GB models need; JetPack's zram is compressed
# memory, not swap, and does not count
swap_short() {
  local kb
  kb=$(awk 'NR > 1 && $1 !~ /zram/ {s += $3} END {printf "%d", s}' "$SWAPS" 2>/dev/null || echo 0)
  [ "${kb:-0}" -lt $(((SWAP_GB - 1) * 1048576)) ]
}

# ---------------------------------------------------------------------------
# The plan

show_found() {
  heading "This computer"
  if [ "$JETSON" = 1 ]; then
    good "$MODEL"
    good "$JETPACK (Jetson Linux $L4T)"
  else
    good "$GPU_NAME"
    if [ "$DRIVER_MAJOR" -ge "$MIN_DRIVER" ]; then
      good "NVIDIA driver $DRIVER"
    elif [ -n "$DRIVER" ]; then
      bad "NVIDIA driver $DRIVER ${D}(needs $MIN_DRIVER or newer)${N}"
    else
      bad "No NVIDIA driver installed"
    fi
    good "$OS_NAME"
  fi
  if [ "$DISK_GB" -ge "$MIN_DISK_GB" ]; then
    good "$DISK_GB GB of free disk space"
  else
    bad "$DISK_GB GB of free disk space ${D}(needs $MIN_DISK_GB GB)${N}"
  fi
  if [ "$HAVE_DOCKER" = 1 ]; then good "Docker $DOCKER_VERSION"; fi
  return 0
}

preflight() {
  [ "$DISK_GB" -ge "$MIN_DISK_GB" ] || die "Not enough free disk space: $DISK_GB GB, and Jetlink needs $MIN_DISK_GB GB." \
    "Free some space (or use a bigger drive) and run the installer again."
  [ "$SNAP_DOCKER" = 0 ] || die "Docker is installed from the Snap Store, which cannot use the NVIDIA GPU reliably." \
    "Remove it with: sudo snap remove docker" "then run the installer again; it installs Docker the supported way."
  if [ "$OPT_DRY_RUN" != 1 ] && ! curl -fsS --max-time 15 -o /dev/null https://github.com 2>/dev/null; then
    die "No internet connection." "The installer downloads Docker and the Jetlink server; connect and try again."
  fi
  if [ "$JETSON" = 0 ] && [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER" ]; then
    offer_driver
  fi
  return 0
}

offer_driver() {
  local what="needs an NVIDIA driver"
  [ -n "$DRIVER" ] && what="has NVIDIA driver $DRIVER and needs"
  if [ "$OS_ID" != ubuntu ]; then
    die "This computer $what $MIN_DRIVER or newer." \
      "Install it from your distribution or https://www.nvidia.com/drivers, restart," \
      "and run the installer again."
  fi
  local go
  ask_yn go y "This computer $what $MIN_DRIVER or newer. Install it now?" \
    "A restart is needed afterwards. With Secure Boot on, you will be asked to" \
    "choose a password now and confirm it on a blue screen when the computer restarts."
  [ "$go" = y ] || die "Jetlink cannot run without NVIDIA driver $MIN_DRIVER or newer." \
    "Install it, restart, and run the installer again."
  [ "$OPT_DRY_RUN" = 1 ] && { step "Install NVIDIA driver $MIN_DRIVER" true; return 0; }
  get_root
  step "Getting the driver list" apt_get update
  step "Installing Ubuntu's driver tool" apt_get install ubuntu-drivers-common
  # open kernel modules: what NVIDIA recommends for Turing and newer, and all
  # CUDA 13 supports
  # from the terminal when there is one: with Secure Boot on, the driver asks
  # for the password it will want confirmed at the next boot
  local input=/dev/null
  if [ -r /dev/tty ] && (exec 3</dev/tty) 2>/dev/null; then input=/dev/tty; fi
  if ! as_root ubuntu-drivers install "nvidia:${MIN_DRIVER}-open" <"$input" >>"$LOG" 2>&1; then
    die "Installing the NVIDIA driver failed." \
      "Install driver $MIN_DRIVER or newer yourself (Software & Updates > Additional Drivers)," \
      "restart, and run the installer again."
  fi
  heading "The NVIDIA driver is installed."
  say "  Restart the computer, then run the installer again:"
  say "    curl -fsSL $RAW_URL/${REF:-main}/install.sh | bash"
  say ""
  save_log
  exit 0
}

show_plan() {
  heading "Here is the plan"
  if [ "$HAVE_DOCKER" = 0 ]; then
    if [ "$FLAVOR" = jetpack6 ]; then
      say "  • Install Docker, the container system Jetlink runs in ${D}(Ubuntu's docker.io)${N}"
    else
      say "  • Install Docker, the container system Jetlink runs in"
    fi
  fi
  if [ "$HAVE_TOOLKIT" = 0 ] || [ "$HAVE_NVIDIA_RUNTIME" = 0 ]; then
    say "  • Let Docker use the NVIDIA GPU ${D}(NVIDIA Container Toolkit)${N}"
  fi
  if [ -n "$OPT_IMAGE" ]; then
    say "  • Use the server image $OPT_IMAGE"
  elif [ "$OPT_BUILD" = 1 ] || [ "$SOURCE" = local ]; then
    say "  • Build the Jetlink server here ${D}(about 4 GB of downloads, 5 to 30 minutes)${N}"
  else
    say "  • Download the Jetlink server ${D}(about 4 GB)${N}"
  fi
  if [ "$AUTOSTART" = 1 ]; then
    say "  • Start Jetlink every time this computer starts"
  else
    say "  • Start Jetlink now ${D}(not at every boot)${N}"
  fi
  if [ "$JETSON" = 1 ]; then
    if [ "$SLEEP_AFTER" != 0 ]; then
      say "  • Sleep when the car is off to save battery, and wake when you start the car"
    fi
    if [ "$POWEROFF_WITH_COMMA" = 1 ]; then
      say "  • Let the comma shut down the Jetson to protect the car battery"
    fi
    if [ "$FAST_MODE" = 1 ] && [ "$PM_CURRENT" != "$PM_BEST_NAME" ]; then
      say "  • Switch to the fastest power mode, $PM_BEST_NAME, which the large models need ${D}(may need a restart)${N}"
    fi
    if [ "$ADD_SWAP" = 1 ] && [ -z "$SWAP_FILE" ]; then
      say "  • Add ${SWAP_GB} GB of swap, which the largest models need while they are prepared"
    fi
    say "  • Start up without waiting for a network, and keep the system log small"
  fi
  say "  ${D}Models and prepared engines go in $CACHE_DIR${N}"
}

# ---------------------------------------------------------------------------
# Doing it

prepare_source() {
  local here=''
  if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  if [ -n "$here" ] && [ "$here" != "$SRC_ROOT/src" ] \
      && [ -f "$here/docker/Dockerfile" ] && [ -f "$here/scripts/jetlink-run-server" ]; then
    # run from a checkout: install exactly what is in it
    SOURCE=local SOURCE_DIR="$here"
    COMMIT="$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo local)"
    return 0
  fi
  SOURCE=git SOURCE_DIR="$SRC_ROOT/src"
  [ "$OPT_DRY_RUN" = 1 ] && { COMMIT=''; return 0; }
  if [ -d "$SOURCE_DIR/.git" ]; then
    step "Getting Jetlink ($REF)" as_root sh -c "git -C '$SOURCE_DIR' fetch --depth 1 origin '$REF' && git -C '$SOURCE_DIR' reset --hard FETCH_HEAD"
  else
    step "Getting Jetlink ($REF)" as_root sh -c "rm -rf '$SOURCE_DIR' && mkdir -p '$SRC_ROOT' && git clone --depth 1 --branch '$REF' '$REPO_URL' '$SOURCE_DIR'"
  fi
  COMMIT="$(as_root git -C "$SOURCE_DIR" rev-parse --short HEAD)"
}

install_base_packages() {
  local missing=() p
  for p in curl git ca-certificates gnupg; do
    case "$p" in
      ca-certificates) [ -d /etc/ssl/certs ] || missing+=("$p") ;;
      gnupg) command -v gpg >/dev/null 2>&1 || missing+=("$p") ;;
      *) command -v "$p" >/dev/null 2>&1 || missing+=("$p") ;;
    esac
  done
  [ ${#missing[@]} -eq 0 ] && return 0
  step "Getting the package list" apt_get update
  step "Installing ${missing[*]}" apt_get install --no-install-recommends "${missing[@]}"
}

# JetPack's `nvidia-container` package, which nvidia-jetpack pulls in, carries
# its own Docker installer: installing it starts nv-install-docker.service,
# which stops Docker, removes every Docker package, installs the newest Docker
# CE and deletes itself, failing and retrying every 30 s while apt is busy.
# Whatever the installer does to Docker meanwhile is undone under it (a pull
# dies with "failed to send write: EOF"), so a run in progress finishes first.
# The installer never starts one itself: see install_toolkit.
NV_DOCKER_UNIT=nv-install-docker.service

nvidia_docker_setup_running() {
  case "$(as_root systemctl show -p ActiveState --value "$NV_DOCKER_UNIT" 2>/dev/null)" in
    activating|active|reloading|deactivating) return 0 ;;
    *) return 1 ;;
  esac
}

wait_nvidia_docker_setup() {
  local deadline=$((SECONDS + 900))
  while nvidia_docker_setup_running; do
    if [ $SECONDS -ge $deadline ]; then
      echo "JetPack's Docker setup ($NV_DOCKER_UNIT) is still running after 15 minutes;"
      echo "see: systemctl status $NV_DOCKER_UNIT"
      return 1
    fi
    sleep "$POLL_S"
  done
}

settle_docker() {
  [ "$JETSON" = 1 ] && nvidia_docker_setup_running || return 0
  step "Waiting for JetPack to finish installing Docker" wait_nvidia_docker_setup
  detect_docker
}

install_docker() {
  if [ "$HAVE_DOCKER" = 1 ]; then
    as_root systemctl is-active --quiet docker || step "Starting Docker" as_root systemctl enable --now docker
    return 0
  fi
  if [ "$FLAVOR" = jetpack6 ]; then
    # Docker 28 and later cannot run containers on a JetPack 6 kernel (no
    # iptables raw table); Ubuntu 22.04's docker.io is older and works.
    step "Installing Docker" apt_get install docker.io
  else
    local dist=ubuntu
    case "$OS_ID $OS_LIKE" in
      *ubuntu*) dist=ubuntu ;;
      *debian*) dist=debian ;;
    esac
    [ "$OS_ID" = debian ] && dist=debian
    step "Adding Docker's package source" add_docker_repo "$dist"
    step "Installing Docker" apt_get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  fi
  step "Starting Docker" as_root systemctl enable --now docker
  HAVE_DOCKER=1
}

add_docker_repo() {
  local dist=$1 arch
  arch="$(dpkg --print-architecture)"
  as_root install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/$dist/gpg" | as_root tee /etc/apt/keyrings/docker.asc >/dev/null
  as_root chmod a+r /etc/apt/keyrings/docker.asc
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/%s %s stable\n' \
    "$arch" "$dist" "$OS_CODENAME" | root_write /etc/apt/sources.list.d/docker.list
  apt_get update
}

install_toolkit() {
  if [ "$HAVE_TOOLKIT" = 0 ]; then
    if [ "$JETSON" = 1 ]; then
      # From the JetPack package source every Jetson already has. The toolkit
      # itself, not JetPack's `nvidia-container`: that one would replace the
      # Docker just installed, in the background (see settle_docker), and on
      # JetPack 6 with a Docker too new for its kernel.
      step "Getting the package list" apt_get update
      step "Installing the NVIDIA Container Toolkit" apt_get install nvidia-container-toolkit
    else
      step "Adding NVIDIA's package source" add_toolkit_repo
      step "Installing the NVIDIA Container Toolkit" apt_get install nvidia-container-toolkit
    fi
  fi
  if [ "$HAVE_NVIDIA_RUNTIME" = 0 ]; then
    step "Letting Docker use the GPU" as_root nvidia-ctk runtime configure --runtime=docker
    step "Restarting Docker" as_root systemctl restart docker
  fi
}

add_toolkit_repo() {
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | as_root gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | root_write /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt_get update
}

# The published image for a ref: main -> edge-<flavor>, vX.Y.Z -> X.Y.Z-<flavor>
published_tag() {
  case "$REF" in
    main) printf 'edge-%s' "$FLAVOR" ;;
    v[0-9]*) printf '%s-%s' "${REF#v}" "$FLAVOR" ;;
    *) return 1 ;;
  esac
}

get_image() {
  if [ -n "$OPT_IMAGE" ]; then
    IMAGE_REF="$OPT_IMAGE" IMAGE_SOURCE=given
    if ! as_root docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
      step "Downloading $IMAGE_REF" pull_image "$IMAGE_REF"
    fi
  elif [ "$OPT_BUILD" = 1 ] || [ "$SOURCE" = local ]; then
    build_image
  else
    local tag
    if tag="$(published_tag)" && try_pull "$REGISTRY:$tag"; then
      IMAGE_REF="$REGISTRY:$tag" IMAGE_SOURCE=pull
    else
      note "There is no ready-made Jetlink server for this computer yet, so it will be built here."
      build_image
    fi
  fi
  IMAGE_ID="$(as_root docker image inspect --format '{{.Id}}' "$IMAGE_REF")"
}

try_pull() {
  local ref=$1 attempt rc=0
  # A missing tag fails in seconds; only then is it worth the spinner. The time
  # limit is for a connection that died under the request, as when Wi-Fi hands
  # the computer a new address: docker waits on it for a quarter of an hour.
  # Only a timeout (124) is tried again; any other failure means no image.
  for attempt in 1 2 3; do
    rc=0
    as_root timeout "$NET_TIMEOUT_S" docker manifest inspect "$ref" >/dev/null 2>&1 || rc=$?
    [ "$rc" = 124 ] || break
    printf '\n==> registry check %s timed out after %ss\n' "$attempt" "$NET_TIMEOUT_S" >>"$LOG"
  done
  [ "$rc" = 0 ] || return 1
  step "Downloading the Jetlink server (about 4 GB)" pull_image "$ref"
}

# Docker keeps the layers an interrupted pull finished, so another try costs
# only the rest: a Wi-Fi drop three gigabytes in should not end the install.
pull_image() {
  local attempt
  for attempt in 1 2 3; do
    as_root docker pull "$1" && return 0
    if [ "$attempt" != 3 ]; then
      echo "the download was interrupted; trying again"
      sleep "$POLL_S"
    fi
  done
  return 1
}

build_image() {
  local file=docker/Dockerfile
  [ "$FLAVOR" = jetpack6 ] && file=docker/Dockerfile.jetpack6
  IMAGE_REF="jetlink:local-$FLAVOR" IMAGE_SOURCE=build
  # host networking: Docker 28 on a JetPack 6 kernel cannot give a build step
  # a bridge network
  step "Building the Jetlink server (5 to 30 minutes)" \
    as_root docker build --network host -f "$SOURCE_DIR/$file" -t "$IMAGE_REF" "$SOURCE_DIR"
}

# Which way of handing the GPU to a container works here, proven by running
# TensorRT in the image. JetPack 6, JetPack 7 and PCs each prefer a different
# one, and the container toolkit decides some of it at run time.
check_gpu() {
  local candidates=() c out
  if [ "$FLAVOR" = jetpack6 ]; then
    candidates=("--runtime nvidia" "--runtime nvidia --gpus all")
  else
    candidates=("--runtime nvidia --gpus all" "--gpus all" "--runtime nvidia" "--device nvidia.com/gpu=all")
  fi
  local probe='import tensorrt; from jetlink.server.backends.trt import cudart; name, major, minor = cudart.device_name(0); print(f"{name} (compute {major}.{minor}), TensorRT {tensorrt.__version__}")'
  local attempt
  for attempt in 1 2; do
    for c in "${candidates[@]}"; do
      printf '\n==> GPU check with: %s\n' "$c" >>"$LOG"
      # shellcheck disable=SC2086
      if out="$(as_root docker run --rm --network host $c --entrypoint python3 "$IMAGE_ID" -c "$probe" 2>>"$LOG")"; then
        GPU_ARGS="$c" GPU_REPORT="$(printf '%s' "$out" | tail -n 1)"
        good "The server can use the GPU: $GPU_REPORT"
        return 0
      fi
    done
    # a toolkit that only speaks CDI needs its device list written once
    if [ "$attempt" != 1 ] || ! command -v nvidia-ctk >/dev/null 2>&1; then break; fi
    as_root nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >>"$LOG" 2>&1 || break
  done
  bad "The Jetlink server cannot reach the GPU."
  local hint="Restart the computer and run the installer again: a new driver needs a restart."
  [ "$JETSON" = 1 ] && hint="Check that JetPack installed completely (sudo apt install nvidia-jetpack), then run the installer again."
  die "Docker could not give the server access to the GPU." "$hint"
}

configure_jetson() {
  [ "$JETSON" = 1 ] || return 0
  if [ "$FAST_MODE" = 1 ] && [ -n "$PM_BEST_ID" ] && [ "$PM_CURRENT" != "$PM_BEST_NAME" ]; then
    set_power_mode
  fi
  if [ "$ADD_SWAP" = 1 ] && [ -z "$SWAP_FILE" ]; then
    SWAP_FILE="$(dirname "$CACHE_DIR")/jetlink-swapfile"
    step "Adding ${SWAP_GB} GB of swap" add_swap "$SWAP_FILE"
  fi
  local u masked=''
  for u in $WAIT_ONLINE_UNITS; do
    if as_root systemctl list-unit-files "$u" 2>/dev/null | grep -q "^$u"; then
      if [ "$(as_root systemctl is-enabled "$u" 2>/dev/null || true)" != masked ]; then
        as_root systemctl mask "$u" >>"$LOG" 2>&1 || true
      fi
      masked="$masked $u"
    fi
  done
  MASKED_UNITS="${masked# }"
  [ -n "$MASKED_UNITS" ] && good "Starts without waiting for a network"
  if [ "$JOURNALD_CAPPED" != 1 ]; then
    printf '# Jetlink: keep the system log from filling a small root partition\n[Journal]\nSystemMaxUse=200M\n' \
      | root_write "$JOURNALD_DROPIN"
    as_root systemctl restart systemd-journald >>"$LOG" 2>&1 || true
    JOURNALD_CAPPED=1
    good "System log limited to 200 MB"
  fi
}

set_power_mode() {
  printf '\n==> nvpmodel -m %s\n' "$PM_BEST_ID" >>"$LOG"
  # nvpmodel asks whether to reboot when the new mode needs one; say no here
  # and tell the user at the end
  printf 'no\n' | as_root nvpmodel -m "$PM_BEST_ID" >>"$LOG" 2>&1 || true
  local now
  now="$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: *//p' | head -n 1)"
  if [ "$now" = "$PM_BEST_NAME" ]; then
    good "Power mode set to $PM_BEST_NAME"
    PM_CURRENT="$now"
  else
    NEED_REBOOT=1
    note "Power mode $PM_BEST_NAME takes effect after a restart."
  fi
}

add_swap() {
  local file=$1
  as_root mkdir -p "$(dirname "$file")"
  as_root fallocate -l "${SWAP_GB}G" "$file"
  as_root chmod 600 "$file"
  as_root mkswap "$file"
  as_root swapon "$file"
  grep -q "^$file " /etc/fstab || printf '%s none swap sw 0 0\n' "$file" | as_root tee -a /etc/fstab >/dev/null
}

install_files() {
  local src="$SOURCE_DIR/scripts"
  as_root install -d -m 755 "$ETC_DIR" "$LIB_DIR"
  as_root mkdir -p "$CACHE_DIR"
  as_root install -D -m 755 "$src/jetlink-run-server" "$LIB_DIR/run-server"
  as_root install -D -m 755 "$src/jetlink" "$BIN"
  as_root install -D -m 644 "$src/jetlink-server.service" "$UNIT_DIR/$UNIT.service"
  printf '# Jetlink: the cache has to be mounted before the server starts\n[Unit]\nRequiresMountsFor=%s\n' "$CACHE_DIR" \
    | root_write "$UNIT_DIR/$UNIT.service.d/10-cache.conf"

  if [ "$SLEEP_AFTER" != 0 ]; then
    as_root install -D -m 755 "$src/jetlink-wake-setup.sh" "$LIB_DIR/wake-setup"
    as_root install -D -m 644 "$src/99-jetlink-usb-wakeup.rules" "$WAKE_RULE"
    as_root udevadm control --reload-rules >>"$LOG" 2>&1 || true
    as_root udevadm trigger --subsystem-match=usb --action=add >>"$LOG" 2>&1 || true
  else
    as_root rm -f "$LIB_DIR/wake-setup" "$WAKE_RULE"
  fi

  if [ "$POWEROFF_WITH_COMMA" = 1 ]; then
    sed "s#/mnt/data/jetlink#$CACHE_DIR#g" "$src/jetlink-poweroff.sh" | root_write "$LIB_DIR/poweroff" 755
    sed "s#/mnt/data/jetlink#$CACHE_DIR#g" "$src/jetlink-poweroff.path" | root_write "$UNIT_DIR/jetlink-poweroff.path"
    sed -e "s#/usr/local/bin/jetlink-poweroff.sh#$LIB_DIR/poweroff#" -e "s#/mnt/data/jetlink#$CACHE_DIR#g" \
      "$src/jetlink-poweroff.service" | root_write "$UNIT_DIR/jetlink-poweroff.service"
  else
    as_root systemctl disable --now jetlink-poweroff.path >>"$LOG" 2>&1 || true
    as_root rm -f "$UNIT_DIR/jetlink-poweroff.path" "$UNIT_DIR/jetlink-poweroff.service" "$LIB_DIR/poweroff"
  fi

  write_env
  write_conf
}

write_env() {
  {
    echo "# Written by the Jetlink installer; run it again (jetlink setup) to change these."
    printf 'JETLINK_IMAGE=%q\n' "$IMAGE_ID"
    printf 'JETLINK_IMAGE_REF=%q\n' "$IMAGE_REF"
    printf 'JETLINK_FLAVOR=%q\n' "$FLAVOR"
    printf 'JETLINK_CACHE_DIR=%q\n' "$CACHE_DIR"
    printf 'JETLINK_JETSON=%q\n' "$JETSON"
    printf 'JETLINK_SLEEP_AFTER=%q\n' "$SLEEP_AFTER"
    printf 'JETLINK_TRANSPORT=usb\n'
    printf 'JETLINK_GPU_ARGS=%q\n' "$GPU_ARGS"
  } | root_write "$ENV_FILE"
}

write_conf() {
  {
    echo "# The answers the Jetlink installer was given; it reads them back on an update."
    printf 'JETLINK_REF=%q\n' "$REF"
    printf 'JETLINK_SOURCE=%q\n' "$SOURCE"
    printf 'JETLINK_SOURCE_DIR=%q\n' "$SOURCE_DIR"
    printf 'JETLINK_COMMIT=%q\n' "$COMMIT"
    printf 'JETLINK_IMAGE_SOURCE=%q\n' "$IMAGE_SOURCE"
    printf 'JETLINK_PLATFORM_NAME=%q\n' "$PLATFORM_NAME"
    printf 'JETLINK_POWER=%q\n' "$POWER"
    printf 'JETLINK_SLEEP_AFTER=%q\n' "$SLEEP_AFTER"
    printf 'JETLINK_POWEROFF_WITH_COMMA=%q\n' "$POWEROFF_WITH_COMMA"
    printf 'JETLINK_FAST_MODE=%q\n' "$FAST_MODE"
    printf 'JETLINK_AUTOSTART=%q\n' "$AUTOSTART"
    printf 'JETLINK_CACHE_DIR=%q\n' "$CACHE_DIR"
    printf 'JETLINK_SWAP_FILE=%q\n' "$SWAP_FILE"
    printf 'JETLINK_MASKED_UNITS=%q\n' "$MASKED_UNITS"
    printf 'JETLINK_JOURNALD_CAPPED=%q\n' "$JOURNALD_CAPPED"
    printf 'JETLINK_INSTALLED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } | root_write "$CONF"
}

start_server() {
  as_root systemctl daemon-reload
  if [ "$POWEROFF_WITH_COMMA" = 1 ]; then
    as_root systemctl enable --now jetlink-poweroff.path >>"$LOG" 2>&1
  fi
  if [ "$AUTOSTART" = 1 ]; then
    as_root systemctl enable "$UNIT" >>"$LOG" 2>&1
  else
    as_root systemctl disable "$UNIT" >>"$LOG" 2>&1 || true
  fi
  local since
  since="$(date '+%Y-%m-%d %H:%M:%S')"
  as_root systemctl restart "$UNIT"
  step "Starting the Jetlink server" wait_ready "$since"
}

# Up means the server chose its backend and is waiting for the comma (or
# already has it). A crash loop shows as a restart count.
wait_ready() {
  local since=$1 deadline=$((SECONDS + 180)) out restarts
  while [ $SECONDS -lt $deadline ]; do
    out="$(as_root journalctl -u "$UNIT" --since "$since" --no-pager -o cat 2>/dev/null || true)"
    if printf '%s' "$out" | grep -qE 'waiting for a jetlink gadget|client connected'; then
      printf '%s\n' "$out" | tail -n 5
      return 0
    fi
    restarts="$(as_root systemctl show -p NRestarts --value "$UNIT" 2>/dev/null || echo 0)"
    if [ "${restarts:-0}" -ge 3 ]; then
      printf '%s\n' "$out" | tail -n 30
      echo "the server keeps restarting"
      return 1
    fi
    sleep 2
  done
  printf '%s\n' "$out" | tail -n 30
  echo "the server did not report ready within 3 minutes"
  return 1
}

finish() {
  save_log >/dev/null
  heading "${G}Jetlink is installed and running.${N}"
  say ""
  say "  ${B}Next, on your comma:${N}"
  say "    1. Settings > Software > Target Branch: choose ${B}jetson-trt${N} (zoompilot),"
  say "       and let it update and restart."
  say "    2. Settings > Models: turn on ${B}Accelerator Link${N}."
  if [ "$JETSON" = 1 ]; then
    say "    3. Connect the comma's USB-C port to one of this Jetson's ${B}USB-A${N} ports"
  else
    say "    3. Connect the comma's USB-C port to one of this computer's ${B}USB-A${N} ports"
  fi
  say "       with a USB 3 data cable (charge-only cables do not work)."
  say "    4. Stay parked and wait for the comma's icon to turn ${G}green${N}. The first model"
  say "       takes a few minutes to prepare."
  if [ "$JETSON" = 0 ]; then
    say ""
    note "Keep this computer plugged in and awake while driving: sleep drops the link."
  fi
  say ""
  say "  ${B}Handy commands:${N}"
  say "    jetlink status    is it running, and is the comma connected"
  say "    jetlink logs      watch what it is doing"
  say "    jetlink update    get the newest version"
  say "    jetlink setup     change your answers"
  if [ "$NEED_REBOOT" = 1 ]; then
    say ""
    note "${B}Restart this computer once${N} to finish switching the power mode: sudo reboot"
  fi
  say ""
}

# ---------------------------------------------------------------------------
# Removing it

uninstall() {
  load_previous
  heading "Remove Jetlink"
  if [ "$HAD_INSTALL" = 0 ] && [ ! -f "$UNIT_DIR/$UNIT.service" ]; then
    say "  Jetlink is not installed here."
    exit 0
  fi
  local go
  ask_yn go n "Remove Jetlink from this computer?" \
    "Docker and the NVIDIA Container Toolkit stay installed."
  [ "$go" = y ] || { say "  Nothing changed."; exit 0; }
  [ "$OPT_DRY_RUN" = 1 ] && { say "  (dry run: nothing changed)"; exit 0; }
  get_root
  as_root systemctl disable --now "$UNIT" jetlink-poweroff.path >>"$LOG" 2>&1 || true
  as_root docker rm -f jetlink >>"$LOG" 2>&1 || true
  as_root rm -rf "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$UNIT.service.d" \
    "$UNIT_DIR/jetlink-poweroff.path" "$UNIT_DIR/jetlink-poweroff.service"
  as_root systemctl daemon-reload
  as_root rm -f "$BIN" "$WAKE_RULE"
  as_root rm -rf "$LIB_DIR"
  as_root udevadm control --reload-rules >>"$LOG" 2>&1 || true
  good "Server and its settings removed"
  if [ -n "$MASKED_UNITS" ]; then
    # shellcheck disable=SC2086
    as_root systemctl unmask $MASKED_UNITS >>"$LOG" 2>&1 || true
  fi
  if [ -f "$JOURNALD_DROPIN" ]; then
    as_root rm -f "$JOURNALD_DROPIN"
    as_root systemctl restart systemd-journald >>"$LOG" 2>&1 || true
  fi
  if [ -n "$SWAP_FILE" ] && [ -f "$SWAP_FILE" ]; then
    as_root swapoff "$SWAP_FILE" >>"$LOG" 2>&1 || true
    as_root sed -i "\#^$SWAP_FILE #d" /etc/fstab
    as_root rm -f "$SWAP_FILE"
    good "Swap file removed"
  fi
  local images
  images="$(as_root docker image ls --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep -E "^(jetlink|${REGISTRY//./\\.}):" || true)"
  if [ -n "$images" ]; then
    local rm_images
    ask_yn rm_images y "Delete the Jetlink server image to free its disk space (about 4 GB)?"
    if [ "$rm_images" = y ]; then
      # shellcheck disable=SC2086
      as_root docker rmi $images >>"$LOG" 2>&1 || true
      good "Server image deleted"
    fi
  fi
  if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
    local size rm_cache
    size="$(as_root du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
    ask_yn rm_cache n "Also delete the downloaded models in $CACHE_DIR ($size)?" \
      "Keep them if you might install Jetlink again: they take a while to download."
    if [ "$rm_cache" = y ]; then
      as_root rm -rf "$CACHE_DIR"
      good "Models deleted"
    fi
  fi
  as_root rm -rf "$ETC_DIR" "$SRC_ROOT"
  heading "Jetlink is removed."
  say ""
  exit 0
}

# ---------------------------------------------------------------------------

usage() {
  sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" 2>/dev/null | sed '$d; s/^# \{0,1\}//'
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --yes|-y) OPT_YES=1 ;;
      --update) OPT_UPDATE=1 ;;
      --reconfigure) OPT_RECONFIGURE=1 ;;
      --build) OPT_BUILD=1 ;;
      --image) OPT_IMAGE="${2:?--image needs an image}"; shift ;;
      --image=*) OPT_IMAGE="${1#*=}" ;;
      --ref) OPT_REF="${2:?--ref needs a branch or tag}"; shift ;;
      --ref=*) OPT_REF="${1#*=}" ;;
      --dry-run) OPT_DRY_RUN=1 ;;
      --uninstall) OPT_UNINSTALL=1 ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown option: $1" "Run with --help to see the options." ;;
    esac
    shift
  done
}

main() {
  # the script has been read in full by now; nothing below may read stdin
  exec </dev/null
  parse_args "$@"
  setup_colors
  trap 'on_error $LINENO' ERR
  : >"$LOG"

  printf '\n%sJetlink installer%s\n' "$B" "$N"
  say "  Run openpilot's large driving models on this computer, for your comma."
  [ "$OPT_DRY_RUN" = 1 ] && note "Dry run: nothing will be changed."

  open_input
  if [ "$OPT_UNINSTALL" = 1 ]; then
    uninstall
  fi

  detect
  load_previous
  REF="${OPT_REF:-${REF:-main}}"
  if [ -z "$CACHE_DIR" ]; then
    CACHE_DIR="${JETLINK_CACHE_DIR:-/var/lib/jetlink}"
    [ "$JETSON" = 1 ] && CACHE_DIR="${JETLINK_CACHE_DIR:-/mnt/data/jetlink}"
  fi
  show_found
  preflight

  if [ "$HAD_INSTALL" = 1 ] && [ "$OPT_UPDATE" = 0 ] && [ "$OPT_RECONFIGURE" = 0 ] && [ "$INTERACTIVE" = 1 ]; then
    local keep
    ask_yn keep y "Jetlink is already installed. Keep your current settings and update it?"
    [ "$keep" = y ] && OPT_UPDATE=1
  fi
  if [ "$OPT_UPDATE" = 1 ] && [ "$HAD_INSTALL" = 1 ]; then
    : # the saved answers stand
  elif [ "$INTERACTIVE" = 1 ]; then
    ask_questions
  else
    take_defaults
  fi
  jetson_musts

  local here=''
  [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ] && here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -n "$here" ] && [ "$here" != "$SRC_ROOT/src" ] && [ -f "$here/scripts/jetlink-run-server" ]; then
    SOURCE=local
  fi
  show_plan

  if [ "$OPT_UPDATE" = 0 ] || [ "$HAD_INSTALL" = 0 ]; then
    local go
    ask_yn go y "Go ahead?"
    [ "$go" = y ] || { say ""; say "  Nothing changed."; say ""; exit 0; }
  fi
  if [ "$OPT_DRY_RUN" = 1 ]; then
    heading "Dry run: stopping here. Nothing was changed."
    say ""
    exit 0
  fi

  get_root
  heading "Installing"
  settle_docker
  install_base_packages
  prepare_source
  install_docker
  install_toolkit
  get_image
  check_gpu
  configure_jetson
  install_files
  start_server
  finish
}

main "$@"
