#!/usr/bin/env bash
# Jetlink installer, for an NVIDIA Jetson or a Linux PC with an NVIDIA GPU.
#
#   curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
#
# It checks the computer, asks a few questions, installs NVIDIA's TensorRT if
# it is missing, gets the Jetlink server, and starts it at boot. Running it
# again is safe: it offers to keep your answers and brings everything up to
# date. An install that ran the server in Docker moves to the native server
# and keeps its answers, models and engines. Afterwards the `jetlink` command
# manages the install.
#
# Options, for support and scripts; the questions cover everything else:
#   --yes            take the recommended answer to every question
#   --update         keep the saved answers and update, asking nothing
#   --reconfigure    ask the questions again
#   --ref REF        a tag or branch (default: latest, the newest release)
#   --binary FILE    install this server tarball instead of downloading one
#   --dry-run        check and ask, then show the plan without changing anything
#   --uninstall      remove Jetlink
#   --set KEY=VALUE  change one answer and apply it, asking nothing and keeping
#                    the installed server; repeatable. On a Jetson:
#                    power=always|switched, comma_poweroff=yes|no,
#                    desktop=on|off, swap=on|off, power_mode=fastest|NAME,
#                    clocks=on|off. On a PC: autostart=yes|no
#
# Everything runs from main(), called on the last line, so a download cut off
# part way through runs nothing, and nothing reading stdin can eat the script.
set -Eeuo pipefail

REPO_URL="${JETLINK_REPO_URL:-https://github.com/zoompilot/jetlink.git}"
RAW_URL=https://raw.githubusercontent.com/zoompilot/jetlink
API_URL=https://api.github.com/repos/zoompilot/jetlink
RELEASES_URL=https://github.com/zoompilot/jetlink/releases/download
ETC_DIR=/etc/jetlink
CONF="$ETC_DIR/install.conf"
ENV_FILE="$ETC_DIR/server.env"
# the web page's password: a salted hash and the key its sign-ins are signed
# with, root's alone, written by the server's web-password
AUTH_FILE="$ETC_DIR/web-auth.json"
# what the Docker era installed, kept for a failed move and for going back by hand
DOCKER_ERA_DIR="$ETC_DIR/docker-era"
BIN=/usr/local/bin/jetlink
# the Docker era's launcher and helpers; nothing is installed here any more
LIB_DIR=/usr/local/lib/jetlink
# a directory per server version, `current` and `previous` links to two of
# them, and `src`, the source the jetlink command updates from
SRC_ROOT=/opt/jetlink
UNIT_DIR=/etc/systemd/system
UNIT=jetlink-server
# in a server's directory: the unit and udev rules built with it
SHARE=share/jetlink
# what the Docker-era installers made (the status page's own unit was never
# released, but a bench has it): a move from Docker saves these, and a failed
# one puts them back
DOCKER_ERA_UNITS="$UNIT.service $UNIT.service.d jetlink-poweroff.path jetlink-poweroff.service jetlink-web.service jetlink-web.service.d"
# jetson_clocks at boot, as v0.9.0 and older ran it (a unit) and 0.7.4 and
# older (a drop-in): the server sets the clocks itself now, so both go
CLOCKS_UNIT=jetlink-clocks.service
CLOCKS_DROPIN="$UNIT_DIR/$UNIT.service.d/20-jetson-clocks.conf"
# on a Jetson the server waits for the GPU's driver instead
GPU_DROPIN="$UNIT_DIR/$UNIT.service.d/20-jetson-gpu.conf"
WAKE_RULE=/etc/udev/rules.d/99-jetlink-usb-wakeup.rules
JOURNALD_DROPIN=/etc/systemd/journald.conf.d/60-jetlink.conf
# the last release that ran in Docker: `jetlink update --ref` it to go back
DOCKER_LAST=v0.6.0
# CUDA 13 needs driver 580; TensorRT needs a Turing (7.5) or newer GPU
MIN_DRIVER=580
MIN_CC=75
# the server is built on Ubuntu 22.04 (JetPack 6's), and runs on no older glibc
GLIBC_MIN=2.35
MIN_DISK_GB=15
# Building the largest model's engine: what it holds besides TensorRT's
# workspace, what the system and a desktop keep, and the margin on top
# (installation-reference.md, Swap, has the measurements)
BUILD_MB=5800 SYSTEM_MB=2048 SWAP_MARGIN_MB=512
# free space an installed Jetlink keeps beside its swap: one more 1.7 GB model,
# its engine and the patched copy a build makes
KEEP_FREE_GB=6
# JetPack 7.2 runs the newest TensorRT the Jetson repository has, and nothing
# older than this.
JP7_MIN_TRT=10.16.2.10
# PCs get no TensorRT package: NVIDIA's x86_64 runtime wheel of the build the
# server is compiled against (what pip's tensorrt-cu13==11.3.0.99 fetches from
# NVIDIA's index) is unpacked into a directory of its own, which every server
# version shares and the unit names with --tensorrt-libs
PC_TRT=11.3.0.99
PC_TRT_WHEEL=https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-11.3.0.99-py3-none-manylinux_2_28_x86_64.whl
PC_TRT_SHA256="${JETLINK_TEST_TRT_SHA256:-cef957e0b48a525f2bcc8849742de06883cedcc6b96185b9cf0b17c9c75a858a}"
TRT_ROOT="$SRC_ROOT/tensorrt"
PC_TRT_DIR="$TRT_ROOT/$PC_TRT"
# free space that installing TensorRT takes, the download and the files
# together: NVIDIA's package indexes give libnvinfer, its ONNX parser and its
# plugins as 2.33 GB + 3.05 GB on JetPack 7.2 and 0.11 + 0.40 on 6.2; a PC's
# wheel is 3.81 GB, and the libraries unpacked from it 2.69 GB
TRT_GB_JP7=6
TRT_GB_JP6=1
TRT_GB_PC=7
# units that hold up boot waiting for a network the car does not have
WAIT_ONLINE_UNITS="systemd-networkd-wait-online.service NetworkManager-wait-online.service"
# seconds the firmware's boot menu waits at power-on; JetPack's is 5, and 1
# still leaves a moment for Esc
UEFI_TIMEOUT=1
# where detection looks; the installer's tests point these at fakes
OS_RELEASE="${JETLINK_TEST_OS_RELEASE:-/etc/os-release}"
PKG_PATH="${JETLINK_TEST_PKG_PATH:-$PATH}"
DT_MODEL="${JETLINK_TEST_DT_MODEL:-/proc/device-tree/model}"
MEM_SLEEP="${JETLINK_TEST_MEM_SLEEP:-/sys/power/mem_sleep}"
SECURE_BOOT="${JETLINK_TEST_SECURE_BOOT:-/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c}"
SWAPS="${JETLINK_TEST_SWAPS:-/proc/swaps}"
MEMINFO="${JETLINK_TEST_MEMINFO:-/proc/meminfo}"
PROC_VERSION="${JETLINK_TEST_PROC_VERSION:-/proc/version}"
SYSTEMD_RUN="${JETLINK_TEST_SYSTEMD_RUN:-/run/systemd/system}"
AWAKE_LOCK="${JETLINK_TEST_AWAKE_LOCK:-/run/jetlink-awake.lock}"
EXTLINUX="${JETLINK_TEST_EXTLINUX:-/boot/extlinux/extlinux.conf}"
EFIVARS="${JETLINK_TEST_EFIVARS:-/sys/firmware/efi/efivars}"
# seconds between looks at something the installer waits on
POLL_S="${JETLINK_TEST_POLL_S:-5}"
# A new server has 3 minutes to say it serves, and then has to stay up. One
# that loads the model it ran last is watched until that is done: a TensorRT
# that changed makes it rebuild the plan, which takes minutes, so the 20
# minute wait is only how long the installer watches, and the server serves
# all the while.
READY_S=180
SETTLE_S="${JETLINK_TEST_SETTLE_S:-10}"
PRELOAD_S="${JETLINK_TEST_PRELOAD_S:-1200}"

OPT_YES=0 OPT_UPDATE=0 OPT_RECONFIGURE=0 OPT_DRY_RUN=0 OPT_UNINSTALL=0
OPT_REF="" OPT_BINARY=""
# --set's KEY=VALUE pairs
OPT_SET=()
# 1 for a run that changes answers only: it keeps the server, its source and
# the TensorRT there are, and needs no network (--set)
KEEP_INSTALLED=0
# as given, for the installer of an older release to take over with
ARGS=()

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
  # a second Ctrl-C must not cut the way back short
  trap '' INT TERM HUP
  restore_previous_server
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
  trap '' INT TERM HUP
  restore_previous_server
  save_log
  exit "$rc"
}

# Ctrl-C, a closed terminal or ssh session, or a kill: the way back a failure
# takes, then out with the signal's status. A closed terminal takes the output
# with it, so from then on everything goes to the log.
on_signal() {
  local sig=$1 rc=$2
  trap '' INT TERM HUP
  trap - ERR
  set +e
  if [ "$sig" = HUP ]; then exec >>"$LOG" 2>&1; else exec 1>&4 2>&5; fi
  printf '\n%sStopped by %s.%s\n' "$R$B" "SIG$sig" "$N"
  restore_previous_server
  save_log
  exit "$rc"
}

# `run_step LABEL cmd...`: run a command with its output in the log, and a
# spinner with the elapsed time so a 20 minute download still looks alive.
# Returns the command's status; `step` stops the installer on a failure.
run_step() {
  local label="$1"
  shift
  if [ "$OPT_DRY_RUN" = 1 ]; then
    printf '  %s·%s %s\n' "$D" "$N" "$label"
    return 0
  fi
  printf '\n==> %s\n' "$label" >>"$LOG"
  local rc=0 t0=$SECONDS
  if [ -t 1 ]; then
    # a failure is the step's to report: the ERR trap, inherited, would put
    # the old server back from in here, and then the step's die again
    ( trap - ERR; "$@" ) </dev/null >>"$LOG" 2>&1 &
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
  return "$rc"
}

step() {
  run_step "$@" && return 0
  printf '\n  Last lines of the log:\n'
  tail -n 20 "$LOG" | sed 's/^/    /'
  die "That step failed." "Running the installer again is safe." \
    "If it keeps failing, open an issue at https://github.com/zoompilot/jetlink/issues"
}

elapsed() {
  local s=$1
  if [ "$s" -lt 60 ]; then printf '%ss' "$s"; else printf '%dm%02ds' $((s / 60)) $((s % 60)); fi
}

# a >= b, for version numbers
version_ge() {
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1)" = "$1" ]
}

# ---------------------------------------------------------------------------
# Questions. Answers are read from the terminal, not stdin, which is the
# script itself under curl | bash.

INTERACTIVE=0
open_input() {
  if [ ${#OPT_SET[@]} -gt 0 ]; then
    # --set asks nothing, so it runs where nobody can answer, like the web page
    INTERACTIVE=0
  elif [ -n "${JETLINK_INPUT:-}" ]; then  # the installer's tests
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

# read_answer without echo, for a password; the Enter typed does not show either
read_secret() {
  local __rs_reply=''
  IFS= read -rs __rs_reply <&3 || true
  printf '\n'
  printf -v "$1" '%s' "$__rs_reply"
}

# ask_intro "question" ["explanation"...]
ask_intro() {
  printf '\n  %s%s%s\n' "$B" "$1" "$N"
  shift
  local __in_line
  for __in_line in "$@"; do printf '  %s%s%s\n' "$D" "$__in_line" "$N"; done
}

# ask_number VAR default MIN MAX: until a number from MIN to MAX, or Enter
ask_number() {
  local __nb_a
  while true; do
    printf '  Type a number and press Enter [%s] ' "$2"
    read_answer __nb_a
    [ -z "$__nb_a" ] && __nb_a=$2
    if [[ "$__nb_a" =~ ^[0-9]{1,5}$ ]] && [ "$((10#$__nb_a))" -ge "$3" ] && [ "$((10#$__nb_a))" -le "$4" ]; then break; fi
    printf '  Please type a number from %d to %d.\n' "$3" "$4"
  done
  printf -v "$1" '%s' "$((10#$__nb_a))"
}

# ask_yn VAR default(y|n) "question" ["explanation"...]
ask_yn() {
  local __yn_var=$1 __yn_def=$2
  shift 2
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__yn_var" '%s' "$__yn_def"
    return
  fi
  ask_intro "$@"
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
  local __ch_var=$1 __ch_def=$2
  shift 2
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__ch_var" '%s' "$__ch_def"
    return
  fi
  ask_intro "$1"
  shift
  local __ch_i=1 __ch_opt
  for __ch_opt in "$@"; do
    printf '    %s%d)%s %s\n' "$B" "$__ch_i" "$N" "$__ch_opt"
    __ch_i=$((__ch_i + 1))
  done
  ask_number "$__ch_var" "$__ch_def" 1 $#
}

# ask_port VAR default "question" ["explanation"...]: 0 to 65535
ask_port() {
  local __pt_var=$1 __pt_def=$2
  shift 2
  if [ "$INTERACTIVE" != 1 ]; then
    printf -v "$__pt_var" '%s' "$__pt_def"
    return
  fi
  ask_intro "$@"
  ask_number "$__pt_var" "$__pt_def" 0 65535
}

# The web page's password, typed twice; Enter alone has the server make one.
# Sets WEB_AUTH to typed (the password in WEB_PASSWORD) or generate.
ask_web_password() {
  ask_intro "Which password should the web page ask for?" \
    "8 to 128 characters. Press Enter to have one made; it is shown at the end."
  local __wp_a __wp_b
  while true; do
    printf '  Password: '
    read_secret __wp_a
    if [ -z "$__wp_a" ]; then
      WEB_AUTH=generate WEB_PASSWORD=''
      return
    fi
    if [ ${#__wp_a} -lt 8 ] || [ ${#__wp_a} -gt 128 ]; then
      printf '  Please use 8 to 128 characters.\n'
      continue
    fi
    printf '  Once more: '
    read_secret __wp_b
    [ "$__wp_a" = "$__wp_b" ] && break
    printf '  The two did not match; please type it again.\n'
  done
  WEB_AUTH=typed WEB_PASSWORD=$__wp_a
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
  # a download can outlast sudo's 15 minutes
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

# ---------------------------------------------------------------------------
# Packages. A Jetson's come from apt: JetPack's, TensorRT among them. A PC
# takes only curl, git, unzip and libcurl from its own package manager (apt,
# dnf, pacman or zypper), and TensorRT from NVIDIA's wheel, so a PC with none
# of those four still installs when the packages are already there.

PKG=''
detect_pkg() {
  local p
  for p in apt-get dnf pacman zypper; do
    if PATH="$PKG_PATH" command -v "$p" >/dev/null 2>&1; then
      PKG="${p%-get}"
      return 0
    fi
  done
  return 0
}

# the package list, where the package manager keeps one apart from upgrades;
# before a pkg_install, outside its step
pkg_refresh() {
  case "$PKG" in
    apt) apt_update ;;
  esac
  return 0
}

apt_get() {
  # a fresh Jetson runs unattended-upgrades for a while after its first boot
  as_root env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 -y "$@"
}

# the package list, once a run; a new package source clears APT_UPDATED
APT_UPDATED=0
apt_update() {
  [ "$APT_UPDATED" = 1 ] && return 0
  step "Getting the package list" apt_get update
  APT_UPDATED=1
}

# packages, named as this package manager names them
pkg_install() {
  case "$PKG" in
    apt) apt_get install --no-install-recommends "$@" ;;
    dnf) as_root dnf -y install "$@" ;;
    # the package list is refreshed only when a stale one fails, since a
    # refresh without a full upgrade is what Arch warns against
    pacman) as_root pacman -S --needed --noconfirm "$@" || as_root pacman -Sy --needed --noconfirm "$@" ;;
    zypper) as_root zypper --non-interactive install "$@" ;;
    *) return 1 ;;
  esac
}

# the loader's cache holds this library
has_lib() {
  local libs
  libs="$({ ldconfig -p || /sbin/ldconfig -p; } 2>/dev/null || true)"
  [[ $libs == *"$1 "* ]]
}

# an installed apt package's version; empty elsewhere
pkg_version() {
  dpkg-query -W -f '${Version}' "$1" 2>/dev/null || true
}

# the firmware's SecureBoot variable: its fifth byte is the state
secure_boot_on() {
  [ "$(od -An -tu1 -j4 -N1 "$SECURE_BOOT" 2>/dev/null | tr -d ' ')" = 1 ]
}

# on the filesystem that holds $1, or will once it is made
free_gb() {
  local where=$1
  while [ ! -d "$where" ]; do where="$(dirname "$where")"; done
  df -Pk "$where" 2>/dev/null | awk 'NR == 2 {printf "%d", $4 / 1048576}'
}

# ---------------------------------------------------------------------------
# What this computer is

ARCH='' OS_ID='' OS_LIKE='' OS_VERSION_ID='' OS_NAME='' OS_FAMILY=''
JETSON=0 L4T='' L4T_MAJOR=0 L4T_MINOR=0 JETPACK='' JP_MAJOR=0 MODEL='' WSL=0
GPU_NAME='' DRIVER='' DRIVER_MAJOR=0 GPU_CC=0 GPU_PRESENT=0
# FLAVOR names the server build: linux-aarch64 (Jetson) or linux-x86_64 (PC)
FLAVOR='' PLATFORM_NAME=''
TRT_GB=0 TRT_PRESENT=0 TRT_VERSION=''
# the flags that point the server at a PC's TensorRT; none on a Jetson
TRT_ARGS=()
DISK_GB=0
DEEP_SLEEP=0
# 1 when the Jetson starts its desktop (stock JetPack's graphical.target)
HAS_DESKTOP=0
PM_BEST_ID='' PM_BEST_NAME='' PM_CURRENT=''
# every power mode as ID:NAME and their names alone, slowest first;
# PM_WANT_* the one to run in
PM_MODES='' PM_NAMES='' PM_WANT_ID='' PM_WANT_NAME=''

detect() {
  [ "$(uname -s)" = Linux ] || die "This installer is for Linux: a Jetson, or a PC running Linux." \
    "On a Mac, use the Jetlink app from https://github.com/zoompilot/jetlink/releases"
  if grep -qi microsoft "$PROC_VERSION" 2>/dev/null; then WSL=1; fi
  ARCH="$(uname -m)"
  if [ -r "$OS_RELEASE" ]; then
    # shellcheck disable=SC1090
    . "$OS_RELEASE"
    OS_ID="${ID:-}" OS_LIKE="${ID_LIKE:-}" OS_VERSION_ID="${VERSION_ID:-}" OS_NAME="${PRETTY_NAME:-Linux}"
  fi
  OS_FAMILY="$(os_family)"
  detect_pkg

  if [ -f /etc/nv_tegra_release ] || grep -qa tegra /proc/device-tree/compatible 2>/dev/null; then
    JETSON=1
    detect_jetson
    if has_lib libnvinfer.so.10 && has_lib libnvonnxparser.so.10; then
      TRT_PRESENT=1 TRT_VERSION="$(pkg_version libnvinfer10)"
    fi
  else
    detect_pc
    if [ -f "$PC_TRT_DIR/libnvinfer.so.11" ] && [ -f "$PC_TRT_DIR/libnvonnxparser.so.11" ]; then
      TRT_PRESENT=1 TRT_VERSION=$PC_TRT
    fi
  fi
}

# the distribution's family, from os-release's ID and ID_LIKE: ubuntu (Mint
# and Pop!_OS too), debian, fedora, rhel, arch or suse; empty for another
os_family() {
  local id
  for id in $OS_ID $OS_LIKE; do
    case "$id" in
      ubuntu|debian|fedora|arch) echo "$id"; return ;;
      rhel|centos) echo rhel; return ;;
      suse|opensuse*|sles) echo suse; return ;;
    esac
  done
}

check_glibc() {
  local v
  v="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)"
  [ -n "$v" ] || return 0
  version_ge "$v" "$GLIBC_MIN" || die "This system's glibc is $v, and the Jetlink server needs $GLIBC_MIN or newer." \
    "Ubuntu 22.04, Debian 12, Fedora, Arch, openSUSE Tumbleweed and newer have it; Debian 11 and RHEL 9 do not."
}

detect_jetson() {
  local line rev
  line="$(head -n 1 /etc/nv_tegra_release 2>/dev/null || true)"
  if [[ "$line" =~ R([0-9]+)\ \(release\),\ REVISION:\ ([0-9.]+) ]]; then
    L4T_MAJOR="${BASH_REMATCH[1]}"
    rev="${BASH_REMATCH[2]}"
  else
    # r39 and later may drop the file; the core package has the version
    rev="$(pkg_version nvidia-l4t-core)"
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
      JP_MAJOR=7 JETPACK="JetPack 7" TRT_GB=$TRT_GB_JP7 ;;
    36)
      [ "$L4T_MINOR" -ge 4 ] || die "This Jetson runs JetPack 6 with Jetson Linux $L4T, which is too old." \
        "Update to JetPack 7.2.1 (recommended) or 6.2: https://developer.nvidia.com/embedded/jetpack"
      JP_MAJOR=6 JETPACK="JetPack 6" TRT_GB=$TRT_GB_JP6 ;;
    38)
      die "JetPack 7.0 and 7.1 are not supported. Update to JetPack 7.2.1:" \
        "https://developer.nvidia.com/embedded/jetpack" ;;
    *)
      die "This Jetson's software (Jetson Linux $L4T) is not supported." \
        "Flash JetPack 7.2.1 (recommended) or 6.2: https://developer.nvidia.com/embedded/jetpack" ;;
  esac
  [ "$ARCH" = aarch64 ] || die "Unexpected: a Jetson that is not aarch64 ($ARCH)."
  # TensorRT 10 from JetPack's own package source, on both JetPacks
  FLAVOR=linux-aarch64
  PLATFORM_NAME="$MODEL, $JETPACK (Jetson Linux $L4T)"

  if grep -qw deep "$MEM_SLEEP" 2>/dev/null; then DEEP_SLEEP=1; fi
  if [ "$(systemctl get-default 2>/dev/null || true)" = graphical.target ]; then HAS_DESKTOP=1; fi
  detect_power_modes
}

detect_power_modes() {
  command -v nvpmodel >/dev/null 2>&1 || return 0
  local conf=/etc/nvpmodel.conf id name rank watts ranked=''
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
    ranked+="$rank $id:$name"$'\n'
  done < <(sed -n 's/.*POWER_MODEL ID=\([0-9]*\) NAME=\([^ >]*\).*/\1 \2/p' "$conf")
  PM_MODES="$(printf '%s' "$ranked" | sort -s -n -k 1,1 | cut -d ' ' -f 2 | paste -sd ' ' -)"
  [ -n "$PM_MODES" ] || return 0
  # the last is the fastest
  local best="${PM_MODES##* }"
  PM_BEST_ID="${best%%:*}" PM_BEST_NAME="${best#*:}"
  PM_NAMES="$(echo "$PM_MODES" | sed -E 's/(^| )[0-9]+:/\1/g')"
  PM_CURRENT="$(power_mode_now)"
  return 0
}

power_mode_now() {
  nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: *//p' | head -n 1
}

detect_pc() {
  [ "$ARCH" = x86_64 ] || die "On an Arm computer, Jetlink supports NVIDIA Jetson only." \
    "This one is $ARCH and does not look like a Jetson."
  FLAVOR=linux-x86_64 TRT_GB=$TRT_GB_PC TRT_ARGS=(--tensorrt-libs "$PC_TRT_DIR")
  local q smi=nvidia-smi
  # WSL keeps the Windows driver's tools here, not always on the PATH
  if ! command -v nvidia-smi >/dev/null 2>&1 && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then smi=/usr/lib/wsl/lib/nvidia-smi; fi
  if command -v "$smi" >/dev/null 2>&1 \
      && q="$("$smi" --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -n 1)" \
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
  [ "$WSL" = 1 ] && PLATFORM_NAME="$PLATFORM_NAME (WSL)"
  return 0
}

# ---------------------------------------------------------------------------
# Answers, saved in install.conf (the questions) and server.env (what the
# server runs with), so an update asks nothing

POWER='' SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0 AUTOSTART=1 STATUS_PORT=5600
# swap for building the largest models: SWAP is the answer; ADD_SWAP_GB the
# file this run adds and REMOVE_SWAP whether it removes Jetlink's; SWAP_FILE_GB
# the file there is, SWAP_SHORT_GB what did not fit on the disk, SWAP_NEED_GB
# all the swap the build needs here
SWAP=on ADD_SWAP_GB=0 REMOVE_SWAP=0 SWAP_FILE_GB=0 SWAP_SHORT_GB=0 SWAP_NEED_GB=0
# a Jetson's power mode, `fastest` or one of its mode names, and whether the
# server holds the clocks at that mode's top while it serves (on), or leaves
# them to NVIDIA's scaling (off)
POWER_MODE=fastest CLOCKS=on CLOCKS_SAVED=on
# 1 when the Jetson starts without its desktop; the default target changes
# only when the answer does, so a desktop turned back on by hand stays on
DESKTOP_OFF=0 DESKTOP_OFF_SAVED=0
CACHE_DIR='' REF='' SOURCE='' SOURCE_DIR='' COMMIT=''
# REF is what the install follows: latest (the newest release), a tag or a
# branch. RESOLVED is the tag or branch that gave, or `local` for a checkout,
# saved as JETLINK_VERSION (not VERSION, which /etc/os-release sets).
RESOLVED=''
# what a restart at the end would finish, said in the closing note
SWAP_FILE='' MASKED_UNITS='' JOURNALD_CAPPED=0 REBOOT_FOR=''
# the firmware's boot menu wait before Jetlink changed it, for uninstall
# (`none` when it had none set)
UEFI_TIMEOUT_PREV=''
HAD_INSTALL=0
# 1 when the install to update runs the server in Docker (0.6.0 and older)
DOCKER_ERA=0

load_previous() {
  # a Docker-era unit, even from before the installer wrote install.conf
  if grep -qs docker "$UNIT_DIR/$UNIT.service" || grep -qs '^JETLINK_IMAGE=' "$ENV_FILE"; then
    DOCKER_ERA=1
  fi
  [ -r "$CONF" ] || return 0
  HAD_INSTALL=1
  # only from the files: the jetlink command runs this with them exported
  local JETLINK_REF='' JETLINK_VERSION='' JETLINK_STATUS_PORT='' JETLINK_SLEEP_AFTER_HELD=''
  # shellcheck disable=SC1090
  . "$CONF"
  # what the server runs with, the sleep delay and the cache among it
  # shellcheck disable=SC1090
  [ -r "$ENV_FILE" ] && . "$ENV_FILE"
  POWER="${JETLINK_POWER:-}"
  # a run stopped while hold_sleep held the server awake left the answer here
  SLEEP_AFTER="${JETLINK_SLEEP_AFTER_HELD:-${JETLINK_SLEEP_AFTER:-0}}"
  POWEROFF_WITH_COMMA="${JETLINK_POWEROFF_WITH_COMMA:-0}"
  DESKTOP_OFF="${JETLINK_DESKTOP_OFF:-0}" DESKTOP_OFF_SAVED="${JETLINK_DESKTOP_OFF:-0}"
  AUTOSTART="${JETLINK_AUTOSTART:-1}"
  CACHE_DIR="${JETLINK_CACHE_DIR:-}"
  STATUS_PORT="${JETLINK_STATUS_PORT:-$STATUS_PORT}"
  SWAP_FILE="${JETLINK_SWAP_FILE:-}"
  SWAP="${JETLINK_SWAP:-on}"
  # installers before the size was recorded made 8 GB
  SWAP_FILE_GB="${JETLINK_SWAP_GB:-8}"
  [ -n "$SWAP_FILE" ] || SWAP_FILE_GB=0
  POWER_MODE="${JETLINK_POWER_MODE:-fastest}"
  CLOCKS=on
  [ "${JETLINK_CLOCKS:-on}" != off ] || CLOCKS=off
  CLOCKS_SAVED=$CLOCKS
  MASKED_UNITS="${JETLINK_MASKED_UNITS:-}"
  JOURNALD_CAPPED="${JETLINK_JOURNALD_CAPPED:-0}"
  UEFI_TIMEOUT_PREV="${JETLINK_UEFI_TIMEOUT_PREV:-}"
  REF="${JETLINK_REF:-}"
  RESOLVED="${JETLINK_VERSION:-}"
  return 0
}

# --ref, else the saved choice, else latest. Installers before 0.5.0 saved
# main, their default, without asking, and no JETLINK_VERSION: such an
# install follows releases now. So does one pinned to a release that ran in
# Docker (a rollback, `--ref v0.6.0`) that --binary moves to the native
# server: kept, the pin would hold every later update to that release.
choose_ref() {
  # a checkout installs what is in it, so a ref there would do nothing
  if [ "$SOURCE" = local ] && [ -n "$OPT_REF" ]; then
    die "--ref does nothing from a checkout, which installs what is in it." \
      "For $OPT_REF, run the installer of that ref instead:" \
      "  curl -fsSL $RAW_URL/$OPT_REF/install.sh | bash -s -- --update --ref $OPT_REF"
  fi
  if [ -n "$OPT_REF" ]; then
    REF="$OPT_REF"
  elif [ "$REF" = main ] && [ -z "$RESOLVED" ]; then
    REF=latest RESOLVED=main
    if [ "$SOURCE" != local ]; then
      note "Jetlink now follows releases; for development builds, use --ref main."
    fi
  elif [ -n "$OPT_BINARY" ] && docker_release "$REF"; then
    note "Jetlink follows releases again: it was pinned to $REF, which runs the server in Docker."
    REF=latest
  fi
  REF="${REF:-latest}"
}

# Without a terminal every question takes its default, which is the
# recommended answer, or the saved one on a reinstall.
ask_questions() {
  if [ "$INTERACTIVE" = 1 ]; then
    heading "A few questions"
    say "  Press Enter to take the recommended answer."
  fi

  if [ "$JETSON" = 1 ]; then
    # always on is the recommended wiring, for a Jetson that can deep-sleep
    local prev="$POWER" def=1 choice always
    if [ "$prev" = switched ] || { [ -z "$prev" ] && [ "$(default_answer power)" = switched ]; }; then def=2; fi
    always="Always on ${D}(recommended)${N}: sleeps while parked, wakes when the car starts"
    [ "$DEEP_SLEEP" = 1 ] || always="Always on: stays awake while parked ${D}(this Jetson cannot sleep)${N}"
    ask_choice choice "$def" "Does the Jetson's power stay on when the car is off?" \
      "$always" \
      "Switched: loses power with the car, boots at every start"
    if [ "$choice" = 1 ]; then
      set_always_on
      local off offdef=n
      [ "$(default_answer comma_poweroff)" = no ] || offdef=y
      [ "$prev" = always ] && [ "$POWEROFF_WITH_COMMA" = 0 ] && offdef=n
      ask_yn off "$offdef" "Let the comma turn the Jetson off to protect the car battery?" \
        "When the comma shuts down (low battery, or parked 30 hours), the Jetson turns" \
        "off too and stays off until its power is reconnected."
      if [ "$off" = y ]; then POWEROFF_WITH_COMMA=1; else POWEROFF_WITH_COMMA=0; fi
    else
      POWER=switched SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0
    fi

    # only for a Jetson that starts its desktop, or one Jetlink turned it off on
    if [ "$HAS_DESKTOP" = 1 ] || [ "$DESKTOP_OFF" = 1 ]; then
      local desk deskdef=n
      [ "$(default_answer desktop)" = on ] || deskdef=y
      [ "$HAD_INSTALL" = 1 ] && [ "$DESKTOP_OFF" = 0 ] && deskdef=n
      ask_yn desk "$deskdef" "Turn off the desktop? ${D}(recommended unless you use it)${N}" \
        "Jetlink doesn't need it: more memory for the models, and a faster start." \
        "jetlink setup turns it back on."
      if [ "$desk" = y ]; then DESKTOP_OFF=1; else DESKTOP_OFF=0; fi
    fi

    AUTOSTART=1
  else
    local auto autodef=n
    [ "$(default_answer autostart)" = no ] || autodef=y
    [ "$AUTOSTART" = 0 ] && autodef=n
    ask_yn auto "$autodef" "Start Jetlink when this computer starts?" \
      "Otherwise, start it with: jetlink start"
    if [ "$auto" = y ]; then AUTOSTART=1; else AUTOSTART=0; fi
  fi

  ask_port STATUS_PORT "$STATUS_PORT" "Which port for the web page?" \
    "A page for your phone or computer on the same network, like the comma's" \
    "hotspot: the status, the setup and the models, behind a password." \
    "0 turns it off."
}

# The web page's password, when the page is on. A run that asks nothing (an
# update, --set, --yes, no terminal) keeps the one there is, or has the server
# make one, shown at the end; one that asks offers to keep it. WEB_AUTH: keep,
# typed (WEB_PASSWORD holds it until the server writes the file), generate,
# or empty with the page off.
WEB_AUTH='' WEB_PASSWORD=''
choose_web_password() {
  WEB_AUTH='' WEB_PASSWORD=''
  [ "$STATUS_PORT" != 0 ] || return 0
  local asks=1 keep
  if [ "$INTERACTIVE" != 1 ] || { [ "$OPT_UPDATE" = 1 ] && [ "$HAD_INSTALL" = 1 ]; }; then asks=0; fi
  if [ -f "$AUTH_FILE" ]; then
    WEB_AUTH=keep
    [ "$asks" = 1 ] || return 0
    ask_yn keep y "Keep the web page's password?" "No sets a new one."
    [ "$keep" = n ] || return 0
  fi
  if [ "$asks" = 1 ]; then ask_web_password; else WEB_AUTH=generate; fi
}

# --set: the install it changes, and the answers it gives, checked before
# anything is shown. It keeps the server and the source that are here, so it
# needs no network, and the web page can run it in the car.
SET_KEYS="On a Jetson: power=always|switched, comma_poweroff=yes|no, desktop=on|off, swap=on|off, power_mode=fastest|<a mode it has>, clocks=on|off. On a PC: autostart=yes|no."
SET_POWER='' SET_POWEROFF='' SET_DESKTOP='' SET_SWAP='' SET_POWER_MODE='' SET_CLOCKS='' SET_AUTOSTART=''
check_set() {
  [ ${#OPT_SET[@]} -gt 0 ] || return 0
  [ "$DOCKER_ERA" = 0 ] || die "This install runs the server in Docker, and --set changes only a native one." \
    "Move it out of Docker first: jetlink update"
  { [ "$HAD_INSTALL" = 1 ] && [ -L "$SRC_ROOT/current" ]; } || die "Jetlink is not installed here, so --set has nothing to change." \
    "Install it first: curl -fsSL $RAW_URL/main/install.sh | bash"
  [ -f "$SOURCE_DIR/scripts/jetlink" ] || die "--set keeps the Jetlink files in $SOURCE_DIR, and they are missing." \
    "Get them back with: jetlink update"
  local kv key value
  for kv in "${OPT_SET[@]}"; do
    [[ $kv == ?*=* ]] || die "--set takes KEY=VALUE, not '$kv'." "$SET_KEYS"
    key="${kv%%=*}" value="${kv#*=}"
    case "$key" in
      power|comma_poweroff|desktop|swap|power_mode|clocks)
        [ "$JETSON" = 1 ] || die "--set $key is for a Jetson, and this computer is a PC." "$SET_KEYS" ;;
      autostart)
        [ "$JETSON" = 0 ] || die "--set autostart is for a PC: a Jetson always starts Jetlink." "$SET_KEYS" ;;
      *) die "--set does not know '$key'." "$SET_KEYS" ;;
    esac
    case "$key=$value" in
      power=always|power=switched) SET_POWER=$value ;;
      comma_poweroff=yes|comma_poweroff=no) SET_POWEROFF=$value ;;
      desktop=on|desktop=off) SET_DESKTOP=$value ;;
      swap=on|swap=off) SET_SWAP=$value ;;
      clocks=on|clocks=off) SET_CLOCKS=$value ;;
      power_mode=*)
        [ "$value" = fastest ] || [[ " $PM_NAMES " == *" $value "* ]] \
          || die "--set power_mode takes fastest or one of this Jetson's modes (${PM_NAMES// /, }), not '$value'."
        # the fastest by name is the fastest
        [ "$value" != "$PM_BEST_NAME" ] || value=fastest
        SET_POWER_MODE=$value ;;
      autostart=yes|autostart=no) SET_AUTOSTART=$value ;;
      power=*) die "--set power takes always or switched, not '$value'." ;;
      desktop=*|swap=*|clocks=*) die "--set $key takes on or off, not '$value'." ;;
      *) die "--set $key takes yes or no, not '$value'." ;;
    esac
  done
  KEEP_INSTALLED=1 KEEP_SOURCE=1 REUSE_SERVER=1
}

# --set's answers over the saved ones, in the questions' order, so the power
# comes first whatever order they were given in. An answer the install already
# has changes nothing: the web page sends every one that applies.
apply_set() {
  if [ "$SET_POWER" = always ] && [ "$POWER" != always ]; then set_always_on; fi
  if [ "$SET_POWER" = switched ]; then POWER=switched SLEEP_AFTER=0 POWEROFF_WITH_COMMA=0; fi
  case "$SET_POWEROFF" in
    yes)
      if [ "$POWER" = always ]; then
        POWEROFF_WITH_COMMA=1
      else
        note "comma_poweroff=yes does nothing on switched power: the Jetson loses power with the car."
      fi ;;
    no) POWEROFF_WITH_COMMA=0 ;;
  esac
  case "$SET_DESKTOP" in
    on) DESKTOP_OFF=0 ;;
    off)
      # as the question: one that starts no desktop has none to turn off
      if [ "$HAS_DESKTOP" = 1 ] || [ "$DESKTOP_OFF" = 1 ]; then
        DESKTOP_OFF=1
      else
        note "This Jetson starts no desktop, so there is none to turn off."
      fi ;;
  esac
  [ -z "$SET_SWAP" ] || SWAP=$SET_SWAP
  [ -z "$SET_POWER_MODE" ] || POWER_MODE=$SET_POWER_MODE
  [ -z "$SET_CLOCKS" ] || CLOCKS=$SET_CLOCKS
  case "$SET_AUTOSTART" in
    yes) AUTOSTART=1 ;;
    no) AUTOSTART=0 ;;
  esac
  return 0
}

# The default answers here, as `--set` words: the questions offer them, and
# the web page marks them and resets to them.
default_answers() {
  if [ "$JETSON" = 0 ]; then
    echo "autostart=yes"
  elif [ "$DEEP_SLEEP" = 1 ]; then
    echo "power=always comma_poweroff=yes desktop=off swap=on power_mode=fastest clocks=on"
  else
    echo "power=switched comma_poweroff=yes desktop=off swap=on power_mode=fastest clocks=on"
  fi
}

# One of them: default_answer power
default_answer() {
  local word
  for word in $(default_answers); do
    [ "${word%%=*}" != "$1" ] || echo "${word#*=}"
  done
}

set_always_on() {
  POWER=always SLEEP_AFTER=0
  if [ "$DEEP_SLEEP" = 1 ]; then
    SLEEP_AFTER=120
  else
    note "This Jetson's software cannot deep-sleep, so it stays awake when the car is off (about 7 W)."
  fi
}

# The power mode and the swap the answers name, on every Jetson install and
# update. A move from Docker keeps the Jetson as it was set up.
jetson_musts() {
  [ "$JETSON" = 1 ] || return 0
  want_power_mode
  if [ -n "$PM_BEST_NAME" ]; then
    if [ "$PM_BEST_NAME" != MAXN_SUPER ] && [[ "$MODEL" == *"Orin Nano"* ]]; then
      note "This JetPack install does not offer the Orin Nano's Super modes; JetPack 7.2.1's"
      note "installer sets them up. Jetlink still works, a little slower, in $PM_BEST_NAME."
    fi
  fi
  plan_swap
  return 0
}

# The mode the answer names, or the fastest; one this Jetson no longer has
# (a JetPack update renamed it) goes back to the fastest.
want_power_mode() {
  PM_WANT_ID=$PM_BEST_ID PM_WANT_NAME=$PM_BEST_NAME
  [ "$POWER_MODE" != fastest ] || return 0
  local m
  for m in $PM_MODES; do
    if [ "${m#*:}" = "$POWER_MODE" ]; then
      PM_WANT_ID=${m%%:*} PM_WANT_NAME=$POWER_MODE
      return 0
    fi
  done
  [ -z "$PM_MODES" ] || note "This Jetson has no power mode $POWER_MODE now, so Jetlink uses the fastest, $PM_BEST_NAME."
  POWER_MODE=fastest
}

# Jetlink's swap file tops up the swap this Jetson has to what building the
# largest model needs here, and no more; turned off, the file goes. A file an
# older installer made (8 GB) stays as it is.
plan_swap() {
  local need gb keep
  ADD_SWAP_GB=0 REMOVE_SWAP=0 SWAP_SHORT_GB=0
  need=$(swap_need_mb)
  SWAP_NEED_GB=$(((need + 1023) / 1024))
  if [ "$SWAP" = off ]; then
    [ -z "$SWAP_FILE" ] || REMOVE_SWAP=1
    return 0
  fi
  { [ -z "$SWAP_FILE" ] && [ "$DOCKER_ERA" = 0 ]; } || return 0
  gb=$(((need - $(other_swap_mb) + 1023) / 1024))
  [ "$gb" -gt 0 ] || return 0
  # a new install needs its own room too; an installed one, room for a model
  keep=$MIN_DISK_GB
  [ "$HAD_INSTALL" = 0 ] || keep=$KEEP_FREE_GB
  if [ $((DISK_GB - gb)) -ge "$keep" ]; then
    ADD_SWAP_GB=$gb
    return 0
  fi
  SWAP_SHORT_GB=$gb
  local free_more=$((keep + gb - DISK_GB))
  # asked for from the web page: say why nothing changed
  [ "$SET_SWAP" != on ] || die "Not enough free disk space for $gb GB of swap: $DISK_GB GB is free, and Jetlink keeps $keep GB free besides." \
    "Free $free_more GB and try again."
  note "Not enough disk space for $gb GB of swap, so the 1.7 GB models may fail to build."
  note "Free $free_more GB, then add it with: jetlink setup --set swap=on"
}

# The swap building the largest model needs here, in MB: what the build holds
# beyond the memory the system and a desktop leave, with the margin.
swap_need_mb() {
  local mem free ws need
  mem=$(awk '$1 == "MemTotal:" {printf "%d", $2 / 1024}' "$MEMINFO")
  free=$((mem - SYSTEM_MB))
  ws=$((free * 2 / 5))
  [ "$ws" -ge 256 ] || ws=256
  [ "$ws" -le 4096 ] || ws=4096
  need=$((BUILD_MB + ws - free + SWAP_MARGIN_MB))
  [ "$need" -gt 0 ] || need=0
  echo "$need"
}

# The swap this computer has besides Jetlink's file, in MB. JetPack's zram is
# compressed memory, not swap, and does not count.
other_swap_mb() {
  awk -v own="$SWAP_FILE" 'NR > 1 && $1 !~ /zram/ && $1 != own {s += $3} END {printf "%d", s / 1024}' "$SWAPS" 2>/dev/null || echo 0
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
    if [ "$WSL" = 1 ]; then
      note "Windows (WSL) support is untested."
    elif [ "$OS_ID" != ubuntu ]; then
      note "Jetlink is untested on $OS_NAME (Ubuntu is tested); please report how it goes."
    fi
  fi
  if [ "$TRT_PRESENT" = 1 ]; then good "TensorRT ${TRT_VERSION%%-*}"; fi
  if [ "$DISK_GB" -ge "$MIN_DISK_GB" ]; then
    good "$DISK_GB GB of free disk space"
  else
    bad "$DISK_GB GB of free disk space ${D}(needs $MIN_DISK_GB GB)${N}"
  fi
  return 0
}

preflight() {
  if [ "$HAD_INSTALL" = 0 ] && [ "$DOCKER_ERA" = 0 ] && [ "$DISK_GB" -lt "$MIN_DISK_GB" ]; then
    die "Not enough free disk space: $DISK_GB GB, and Jetlink needs $MIN_DISK_GB GB." \
      "Free some space (or use a bigger drive) and run the installer again."
  fi
  if [ ! -d "$SYSTEMD_RUN" ]; then
    if [ "$WSL" = 1 ]; then
      die "Jetlink runs as a systemd service, and this WSL runs without systemd." \
        "Add these two lines to /etc/wsl.conf, run 'wsl --shutdown' in Windows, and try again:" \
        "  [boot]" "  systemd=true"
    fi
    die "Jetlink runs as a systemd service, and this system does not run systemd." \
      "Installing by hand: https://github.com/zoompilot/jetlink/blob/main/docs/installation-reference.md"
  fi
  check_glibc
  if [ -z "$PKG" ]; then
    local missing
    mapfile -t missing < <(base_packages_missing)
    [ ${#missing[@]} -eq 0 ] || die "Jetlink needs ${missing[*]}, and this system's package manager is not one the installer knows (apt, dnf, pacman, zypper)." \
      "Install them and run the installer again."
  fi
  if [ "$OPT_DRY_RUN" != 1 ] && [ -z "$OPT_BINARY" ] && [ "$KEEP_INSTALLED" = 0 ] \
      && ! curl -fsS --max-time 15 -o /dev/null https://github.com 2>/dev/null; then
    die "No internet connection." "The installer downloads TensorRT and the Jetlink server; connect and try again."
  fi
  if [ "$JETSON" = 0 ] && [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER" ]; then
    offer_driver
  fi
  return 0
}

offer_driver() {
  local what="needs an NVIDIA driver"
  [ -n "$DRIVER" ] && what="has NVIDIA driver $DRIVER and needs"
  if [ "$WSL" = 1 ]; then
    # WSL uses the Windows driver; one installed inside would break it
    die "This computer $what $MIN_DRIVER or newer." \
      "Update the NVIDIA driver in Windows, then run the installer again."
  fi
  # the installer puts it in where that is one package from a source the
  # computer already has: Ubuntu's own tool on its family, Arch's nvidia-open
  # on its (Manjaro's comes from its own tool). Anywhere else it is a
  # repository, a key and a kernel module build of its own, so the
  # distribution's documented steps are printed instead.
  local how=''
  case "$OS_FAMILY" in
    ubuntu) how=install_driver_ubuntu ;;
    arch) [ "$OS_ID" = manjaro ] || how=install_driver_arch ;;
  esac
  [ -n "$how" ] || driver_by_hand "$what"
  local go lines=("A restart is needed afterwards.")
  if [ "$how" = install_driver_ubuntu ]; then
    lines+=("With Secure Boot on, you will be asked to choose a password now and confirm it" \
      "on a blue screen when the computer restarts.")
  fi
  ask_yn go y "This computer $what $MIN_DRIVER or newer. Install it now?" "${lines[@]}"
  [ "$go" = y ] || die "Jetlink cannot run without NVIDIA driver $MIN_DRIVER or newer." \
    "Install it, restart, and run the installer again."
  [ "$OPT_DRY_RUN" = 1 ] && { step "Install NVIDIA driver $MIN_DRIVER" true; return 0; }
  get_root
  "$how"
  heading "The NVIDIA driver is installed."
  say "  Restart the computer, then run the installer again:"
  if [ "$REF" = latest ]; then
    say "    curl -fsSL $RAW_URL/main/install.sh | bash"
  else
    say "    curl -fsSL $RAW_URL/$REF/install.sh | bash -s -- --ref $REF"
  fi
  say ""
  save_log
  exit 0
}

install_driver_ubuntu() {
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
}

# Arch's open kernel modules, built for its own kernels (6.12.4-arch1-1,
# 6.12.4-1-lts); another kernel (zen, cachyos) gets the DKMS build and its
# headers. None of them is signed.
install_driver_arch() {
  local kernel pkgs
  kernel="$(uname -r)"
  case "$kernel" in
    *-lts) pkgs=(nvidia-open-lts) ;;
    *-arch*) pkgs=(nvidia-open) ;;
    *) pkgs=(nvidia-open-dkms "linux-${kernel##*-}-headers") ;;
  esac
  pkg_refresh
  if ! run_step "Installing ${pkgs[*]}" pkg_install "${pkgs[@]}"; then
    die "Installing the NVIDIA driver failed." \
      "Install driver $MIN_DRIVER or newer yourself (https://wiki.archlinux.org/title/NVIDIA)," \
      "restart, and run the installer again."
  fi
  if secure_boot_on; then
    note "Secure Boot is on, and these kernel modules are not signed: sign them, or turn Secure"
    note "Boot off in the firmware, or the driver will not load after the restart."
  fi
}

driver_by_hand() {
  local lines
  mapfile -t lines < <(driver_steps)
  if secure_boot_on; then
    lines+=("Secure Boot is on: the driver's kernel modules have to be signed, or Secure Boot turned off.")
  fi
  die "This computer $1 $MIN_DRIVER or newer." \
    "Install it, restart, and run the installer again. On $OS_NAME:" "${lines[@]}"
}

# the distribution's own way to the driver, as its documentation gives it
driver_steps() {
  local rel="${OS_VERSION_ID%%.*}" repo
  case "$OS_FAMILY" in
    debian)
      [[ $rel =~ ^[0-9]+$ ]] || rel=12
      echo "  curl -fLO https://developer.download.nvidia.com/compute/cuda/repos/debian$rel/x86_64/cuda-keyring_1.1-1_all.deb"
      echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
      echo "  sudo apt update && sudo apt install linux-headers-amd64 nvidia-open" ;;
    fedora)
      echo "  sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$rel.noarch.rpm \\"
      echo "    https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$rel.noarch.rpm"
      echo "  sudo dnf install akmod-nvidia"
      echo "  and wait for the module to build (modinfo -F version nvidia) before restarting: https://rpmfusion.org/Howto/NVIDIA" ;;
    rhel)
      [[ $rel =~ ^[0-9]+$ ]] || rel=10
      echo "  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel$rel/x86_64/cuda-rhel$rel.repo | sudo tee /etc/yum.repos.d/cuda-rhel$rel.repo"
      echo "  sudo dnf install nvidia-open" ;;
    arch)
      # Manjaro: the others install it
      echo "  sudo mhwd -a pci nonfree 0300" ;;
    suse)
      repo=https://download.nvidia.com/opensuse/tumbleweed
      case "$OS_ID" in *leap*|sles) repo="https://download.nvidia.com/opensuse/leap/$OS_VERSION_ID" ;; esac
      echo "  sudo zypper addrepo $repo NVIDIA"
      echo "  sudo zypper install-new-recommends --repo NVIDIA" ;;
    *)
      echo "  your distribution's NVIDIA driver package, or https://www.nvidia.com/drivers" ;;
  esac
}

show_plan() {
  heading "Here is the plan"
  if [ "$GOING_BACK" = 1 ]; then
    say "  • Go back to $RESOLVED, which runs Jetlink in Docker: its own installer takes over"
    return 0
  fi
  if [ "$DOCKER_ERA" = 1 ]; then
    say "  • Move Jetlink out of Docker ${D}(your settings, models and engines stay)${N}"
  fi
  if [ "$TRT_PRESENT" = 0 ]; then
    if [ "$JP_MAJOR" = 7 ]; then
      say "  • Install NVIDIA TensorRT from JetPack's package source ${D}(about 2.3 GB)${N}"
    elif [ "$JETSON" = 1 ]; then
      say "  • Install NVIDIA TensorRT from JetPack's package source"
    else
      say "  • Download NVIDIA TensorRT $PC_TRT into $PC_TRT_DIR ${D}(3.8 GB, 2.7 GB once unpacked)${N}"
    fi
  fi
  if [ -n "$OPT_BINARY" ]; then
    say "  • Install the Jetlink server from $OPT_BINARY"
  elif [ "$REUSE_SERVER" = 1 ]; then
    say "  • Keep the Jetlink server that is installed"
  else
    say "  • Download the Jetlink server $RESOLVED"
  fi
  if [ "$AUTOSTART" = 1 ]; then
    say "  • Start Jetlink every time this computer starts"
  else
    say "  • Start Jetlink now ${D}(not at every boot)${N}"
  fi
  if [ "$JETSON" = 1 ]; then
    if [ "$SLEEP_AFTER" != 0 ]; then
      say "  • Sleep while parked, and wake when the car starts"
    fi
    if [ "$POWEROFF_WITH_COMMA" = 1 ]; then
      say "  • Let the comma turn the Jetson off to protect the car battery"
    fi
    if [ "$DOCKER_ERA" = 0 ] && [ -n "$PM_WANT_ID" ] && [ "$PM_CURRENT" != "$PM_WANT_NAME" ]; then
      if [ "$POWER_MODE" = fastest ]; then
        say "  • Switch to the fastest power mode, $PM_WANT_NAME"
      else
        say "  • Switch to power mode $PM_WANT_NAME"
      fi
    fi
    if [ "$CLOCKS" != "$CLOCKS_SAVED" ]; then
      if [ "$CLOCKS" = on ]; then
        say "  • Hold the clocks at the power mode's top speed while serving"
      else
        say "  • Leave the clocks to NVIDIA's scaling"
      fi
    fi
    if [ "$ADD_SWAP_GB" -gt 0 ]; then
      say "  • Add ${ADD_SWAP_GB} GB of swap, so the 1.7 GB models can be built"
    fi
    if [ "$REMOVE_SWAP" = 1 ]; then
      say "  • Remove Jetlink's swap file ${D}(the 1.7 GB models may then fail to build)${N}"
    fi
    if [ "$DESKTOP_OFF" != "$DESKTOP_OFF_SAVED" ]; then
      if [ "$DESKTOP_OFF" = 1 ]; then
        say "  • Turn off the desktop ${D}(from the next restart)${N}"
      else
        say "  • Turn the desktop back on ${D}(from the next restart)${N}"
      fi
    fi
    say "  • Start up faster, and keep the system log small"
  fi
  case "$WEB_AUTH" in
    generate) say "  • Serve the web page on port $STATUS_PORT ${D}(with a new password, shown at the end)${N}" ;;
    typed) say "  • Serve the web page on port $STATUS_PORT ${D}(with the password you chose)${N}" ;;
    keep) say "  • Serve the web page on port $STATUS_PORT ${D}(with the password it has)${N}" ;;
  esac
  if [ "$DOCKER_ERA" = 1 ]; then
    say "  • Delete Jetlink's Docker images once the new server runs ${D}(Docker itself stays)${N}"
  fi
  say "  ${D}Models and prepared engines go in $CACHE_DIR${N}"
}

# ---------------------------------------------------------------------------
# Doing it

detect_source() {
  local here=''
  if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  if [ -n "$here" ] && [ "$here" != "$SRC_ROOT/src" ] \
      && [ -f "$here/scripts/jetlink" ] && [ -f "$here/scripts/jetlink-server.service" ]; then
    # run from a checkout: install exactly what is in it
    SOURCE=local SOURCE_DIR="$here"
  else
    SOURCE=git SOURCE_DIR="$SRC_ROOT/src"
  fi
}

# --binary: install a build made elsewhere, with the scripts from the source
# that is already here (a checkout, or the clone an earlier install made)
# rather than moving that source to a release.
KEEP_SOURCE=0
check_binary() {
  [ -n "$OPT_BINARY" ] || return 0
  [ -f "$OPT_BINARY" ] || die "No such file: $OPT_BINARY"
  OPT_BINARY="$(cd "$(dirname "$OPT_BINARY")" && pwd)/$(basename "$OPT_BINARY")"
  case "$(basename "$OPT_BINARY")" in
    *-linux-aarch64.tar.gz|*-linux-x86_64.tar.gz)
      [[ "$(basename "$OPT_BINARY")" == *"-$FLAVOR.tar.gz" ]] \
        || die "$(basename "$OPT_BINARY") is for another kind of computer; this one needs a $FLAVOR build." ;;
  esac
  if [ "$SOURCE" = local ] || [ -f "$SOURCE_DIR/install.sh" ]; then KEEP_SOURCE=1; fi
  # the jetlink command comes from that source too, and a release's from
  # before the native server drives the Docker one
  if [ "$KEEP_SOURCE" = 1 ] && [ -f "$SOURCE_DIR/scripts/jetlink-run-server" ]; then
    die "$SOURCE_DIR holds Jetlink from before the native server, and --binary installs its scripts." \
      "Move it to the tree the server was built from first, for example:" \
      "  sudo git -C $SOURCE_DIR fetch --depth 1 origin <branch> && sudo git -C $SOURCE_DIR reset --hard FETCH_HEAD" \
      "  sudo $SOURCE_DIR/install.sh --update --binary $OPT_BINARY"
  fi
  return 0
}

# The tag or branch to check out. latest is looked up on every run, so an
# update moves to the newest release; with no answer an update stays on the
# one it has, and a first install stops rather than guess.
resolve_ref() {
  local had="$RESOLVED"
  if [ "$SOURCE" = local ]; then
    RESOLVED=local
  elif [ "$KEEP_SOURCE" = 1 ]; then
    RESOLVED="${had:-local}"
  elif [ "$REF" != latest ]; then
    RESOLVED="$REF"
  else
    RESOLVED="$(latest_release)"
    if [ -n "$RESOLVED" ]; then
      printf '\n==> the newest release is %s\n' "$RESOLVED" >>"$LOG"
    elif [ -n "$had" ] && [ "$had" != local ]; then
      RESOLVED="$had"
      note "Could not look up the newest release; staying on $had."
    else
      die "Could not find the newest Jetlink release." \
        "GitHub did not answer. Check the connection and run the installer again," \
        "or name a release: --ref v0.5.0"
    fi
  fi
  stay_native "$had"
}

# An update or a setup does not take a native install back to Docker by
# itself, while the release it follows (the newest, until one runs natively)
# is one that ran in Docker: only --ref goes there. The server here stays, and
# so does its source.
stay_native() {
  local had=$1
  if [ -n "$OPT_REF" ] || [ "$SOURCE" = local ] || [ "$KEEP_SOURCE" = 1 ] || [ "$DOCKER_ERA" = 1 ] \
      || [ ! -L "$SRC_ROOT/current" ] || ! docker_release "$RESOLVED"; then
    return 0
  fi
  note "$RESOLVED runs Jetlink in Docker, so the server installed here stays. To go back to it:"
  note "  curl -fsSL $RAW_URL/$RESOLVED/install.sh | bash -s -- --update --ref $RESOLVED"
  RESOLVED="$had" KEEP_SOURCE=1 REUSE_SERVER=1
}

# a release from before the native server
docker_release() {
  [[ $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] && version_ge "${DOCKER_LAST#v}" "${1#v}"
}

# The newest release's tag, or nothing: GitHub's latest release, never a draft
# or a prerelease, else the highest vX.Y.Z tag (the API allows 60 requests an
# hour from one address). Nothing in here fails, since the ERR trap would fire
# inside the command substitution that calls it.
latest_release() {
  local json refs re='"tag_name"[[:space:]]*:[[:space:]]*"(v[0-9]+\.[0-9]+\.[0-9]+)"'
  json="$(curl -fsSL --max-time 20 "$API_URL/releases/latest" 2>>"$LOG" || true)"
  if [[ $json =~ $re ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  refs="$(git ls-remote --tags --refs "$REPO_URL" 'v*' 2>>"$LOG" || true)"
  printf '%s\n' "$refs" \
    | sed -n 's#.*refs/tags/\(v[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)$#\1#p' \
    | sort -t. -k1.2,1n -k2,2n -k3,3n | tail -n 1
}

# Where the server for RESOLVED is: a release's own assets, or for main the
# edge prerelease, which CI refreshes on every push under fixed names.
# GOING_BACK: a release from before the native server, which only its own
# installer can put back (the switch itself is decided on its source).
ASSET_URL='' GOING_BACK=0 REUSE_SERVER=0
choose_server() {
  if [ -n "$OPT_BINARY" ] || [ "$REUSE_SERVER" = 1 ]; then return 0; fi
  if [ "$SOURCE" = local ]; then
    if [ "$DOCKER_ERA" = 0 ] && [ -L "$SRC_ROOT/current" ]; then
      REUSE_SERVER=1
      return 0
    fi
    die "From a checkout, the installer installs a server you built:" \
      "  scripts/build-linux.sh $FLAVOR" \
      "  ./install.sh --binary dist/jetlink-server-<version>-$FLAVOR.tar.gz"
  fi
  case "$RESOLVED" in
    main) ASSET_URL="$RELEASES_URL/edge/jetlink-server-edge-$FLAVOR.tar.gz" ;;
    v[0-9]*)
      ASSET_URL="$RELEASES_URL/$RESOLVED/jetlink-server-${RESOLVED#v}-$FLAVOR.tar.gz"
      if docker_release "$RESOLVED"; then GOING_BACK=1; fi ;;
    *) no_server ;;
  esac
}

no_server() {
  die "There is no ready-made Jetlink server for $RESOLVED." \
    "Build one with scripts/build-linux.sh $FLAVOR and install it with --binary."
}

prepare_source() {
  if [ "$SOURCE" = local ] || [ "$KEEP_SOURCE" = 1 ]; then
    COMMIT="$(git -C "$SOURCE_DIR" rev-parse --short HEAD 2>/dev/null || echo local)"
    return 0
  fi
  if [ -d "$SOURCE_DIR/.git" ]; then
    step "Getting Jetlink ($RESOLVED)" as_root sh -c "git -C '$SOURCE_DIR' fetch --depth 1 origin '$RESOLVED' && git -C '$SOURCE_DIR' reset --hard FETCH_HEAD"
  else
    step "Getting Jetlink ($RESOLVED)" as_root sh -c "rm -rf '$SOURCE_DIR' && mkdir -p '$SRC_ROOT' && git clone --depth 1 --branch '$RESOLVED' '$REPO_URL' '$SOURCE_DIR'"
  fi
  COMMIT="$(as_root git -C "$SOURCE_DIR" rev-parse --short HEAD)"
  hand_over
}

# A release that ran the server in Docker (the launcher is the sign) is
# installed by its own installer: `jetlink update --ref v0.6.0` goes back to
# it. That one reads the same answers and puts its Docker server over this
# one; the native files stay, unused, in /opt/jetlink.
hand_over() {
  [ -f "$SOURCE_DIR/scripts/jetlink-run-server" ] || return 0
  [ -z "$OPT_BINARY" ] || die "$RESOLVED runs Jetlink in Docker, so --binary does not apply to it."
  note "$RESOLVED runs Jetlink in Docker; its own installer takes over from here."
  note "To come back later: curl -fsSL $RAW_URL/main/install.sh | bash -s -- --update --ref latest"
  save_log >/dev/null
  exec bash "$SOURCE_DIR/install.sh" "${ARGS[@]}"
}

# the base packages this computer lacks, one a line, as its package manager
# names them (pacman's curl holds libcurl too, hence the sort -u)
base_packages_missing() {
  {
    command -v curl >/dev/null 2>&1 || echo curl
    command -v git >/dev/null 2>&1 || echo git
    [ -d /etc/ssl/certs ] || echo ca-certificates
    # the server's one library beyond the C and C++ runtimes
    if ! has_lib libcurl.so.4; then
      case "$PKG" in dnf) echo libcurl ;; pacman) echo curl ;; *) echo libcurl4 ;; esac
    fi
    # a PC unpacks TensorRT's wheel
    if [ "$JETSON" = 0 ] && ! command -v unzip >/dev/null 2>&1; then echo unzip; fi
    # a Jetson shortens its firmware's boot menu wait
    if [ "$JETSON" = 1 ] && [ -d "$EFIVARS" ] && ! command -v efibootmgr >/dev/null 2>&1; then echo efibootmgr; fi
  } | sort -u
}

install_base_packages() {
  # without a package manager, preflight has seen that nothing is missing
  [ -n "$PKG" ] || return 0
  local missing
  mapfile -t missing < <(base_packages_missing)
  [ ${#missing[@]} -eq 0 ] && return 0
  pkg_refresh
  step "Installing ${missing[*]}" pkg_install "${missing[@]}"
}

# The running server keeps serving while the slow parts download and install,
# and stops only for the switch, so a failure before then leaves it as it was
# (hold_sleep has the one exception). Its settings are kept as server.env.prev
# and install.conf.prev: a failed update puts them back and starts it again
# (restore_previous_server), and after a good one they are a way back by hand.
SERVER_STOPPED=0 CHANGED=0 IMAGES_REMOVED=0 SLEEP_HELD=0
ENV_PREV="$ETC_DIR/server.env.prev"
CONF_PREV="$ETC_DIR/install.conf.prev"

backup_install() {
  local held
  held="$(sed -n 's/^JETLINK_SLEEP_AFTER_HELD=//p' "$ENV_FILE" 2>/dev/null | tail -n 1 || true)"
  if [ -n "$held" ]; then
    # a run stopped while it held the server awake: its sleep is the setting
    { grep -Ev '^JETLINK_SLEEP_AFTER(_HELD)?=' "$ENV_FILE"; echo "JETLINK_SLEEP_AFTER=$held"; } | root_write "$ENV_PREV"
  elif [ -f "$ENV_FILE" ]; then
    as_root cp -p "$ENV_FILE" "$ENV_PREV"
  fi
  if [ -f "$CONF" ]; then as_root cp -p "$CONF" "$CONF_PREV"; fi
  [ "$DOCKER_ERA" = 1 ] || return 0
  # everything the move replaces, and which of the units were enabled
  as_root rm -rf "$DOCKER_ERA_DIR"
  as_root install -d -m 755 "$DOCKER_ERA_DIR/systemd"
  local f u enabled=''
  for u in $DOCKER_ERA_UNITS; do
    f="$UNIT_DIR/$u"
    [ -e "$f" ] || continue
    as_root cp -a "$f" "$DOCKER_ERA_DIR/systemd/"
    case "$u" in
      *.service|*.path)
        if [ "$(as_root systemctl is-enabled "$u" 2>/dev/null || true)" = enabled ]; then
          enabled="$enabled$u"$'\n'
        fi ;;
    esac
  done
  printf '%s' "$enabled" | root_write "$DOCKER_ERA_DIR/enabled"
  if [ -d "$LIB_DIR" ]; then as_root cp -a "$LIB_DIR" "$DOCKER_ERA_DIR/lib"; fi
  for f in "$BIN" "$CONF"; do
    if [ -f "$f" ]; then as_root cp -p "$f" "$DOCKER_ERA_DIR/"; fi
  done
  # the settings as they were, not as a held sleep left them
  if [ -f "$ENV_FILE" ]; then as_root cp -p "$ENV_PREV" "$DOCKER_ERA_DIR/$(basename "$ENV_FILE")"; fi
  good "The Docker setup is saved in $DOCKER_ERA_DIR"
}

# A server that sleeps would suspend the computer under a long download the
# moment the comma lets go. A native one does not while its awake lock is held
# (jetlink caffeinate), so this run holds it until it exits; one from before
# the lock stops now instead of at the switch. The Docker era's does not know
# the lock: it restarts without sleeping, and a failure or an interruption puts
# its server.env back. The sleep it had stays in server.env beside the 0, for
# the next run, should this one be killed before it can put it back.
hold_sleep() {
  local after
  [ -f "$UNIT_DIR/$UNIT.service" ] && [ -f "$ENV_FILE" ] || return 0
  # from the copy backup_install made, which a held sleep does not change
  after="$(sed -n 's/^JETLINK_SLEEP_AFTER=//p' "$ENV_PREV" 2>/dev/null | tail -n 1 || true)"
  [ "${after%.*}" -gt 0 ] 2>/dev/null || return 0
  as_root systemctl is-active --quiet "$UNIT" || return 0
  if [ "$DOCKER_ERA" = 0 ]; then
    if [ -r "$AWAKE_LOCK" ] && exec 8<"$AWAKE_LOCK" && flock --shared --wait 10 8; then
      good "Holding this computer awake for the update"
    else
      stop_running_server
    fi
    return 0
  fi
  SLEEP_HELD=1
  { grep -v '^JETLINK_SLEEP_AFTER=' "$ENV_PREV"; echo 'JETLINK_SLEEP_AFTER=0'; echo "JETLINK_SLEEP_AFTER_HELD=$after"; } \
    | root_write "$ENV_FILE"
  step "Keeping this computer awake for the update" as_root systemctl restart "$UNIT"
}

stop_running_server() {
  [ "$SERVER_STOPPED" = 0 ] || return 0
  [ -f "$UNIT_DIR/$UNIT.service" ] || return 0
  if as_root systemctl is-active --quiet "$UNIT"; then
    step "Stopping the running Jetlink server for the update" as_root systemctl stop "$UNIT"
    SERVER_STOPPED=1
  fi
  # the unit removes its container on the way down; one a crash left holds the
  # name, and the comma's USB interface, which the native server cannot share
  if [ "$DOCKER_ERA" = 1 ] && command -v docker >/dev/null 2>&1; then
    as_root docker rm -f jetlink >>"$LOG" 2>&1 || true
    if as_root docker ps -q --filter 'name=^/?jetlink$' 2>>"$LOG" | grep -q .; then
      die "The Docker server did not stop." "Stop it (sudo docker rm -f jetlink) and run the installer again."
    fi
  fi
  return 0
}

# After a failed update: the previous files and settings, and the previous
# server running if it was.
restore_previous_server() {
  if [ -n "$TRT_STAGE" ]; then as_root rm -rf "$TRT_STAGE" >>"$LOG" 2>&1 || true; fi
  [ "$CHANGED" = 1 ] || [ "$SERVER_STOPPED" = 1 ] || [ "$SLEEP_HELD" = 1 ] || return 0
  local changed=$CHANGED was_running=$SERVER_STOPPED held=$SLEEP_HELD
  CHANGED=0 SERVER_STOPPED=0 SLEEP_HELD=0
  if [ "$changed" = 1 ] || [ "$held" = 1 ]; then
    if [ -f "$ENV_PREV" ]; then as_root cp -p "$ENV_PREV" "$ENV_FILE" >>"$LOG" 2>&1 || true; fi
  fi
  if [ "$changed" = 1 ]; then
    if [ -f "$CONF_PREV" ]; then as_root cp -p "$CONF_PREV" "$CONF" >>"$LOG" 2>&1 || true; fi
    if [ "$DOCKER_ERA" = 1 ]; then
      restore_docker_era
    elif [ -n "$OLD_CURRENT" ]; then
      { point_current "$OLD_CURRENT" && install_unit "$OLD_CURRENT"; } >>"$LOG" 2>&1 || true
    fi
  fi
  as_root systemctl daemon-reload >>"$LOG" 2>&1 || true
  if [ "$IMAGES_REMOVED" = 1 ]; then
    as_root systemctl stop "$UNIT" >>"$LOG" 2>&1 || true
    note "The Docker server's images were deleted to make room, so it cannot start again."
    note "Run the installer again, or go back to Docker with: jetlink update --ref $DOCKER_LAST"
  elif [ "$was_running" = 1 ] || [ "$held" = 1 ]; then
    if as_root systemctl restart "$UNIT" >>"$LOG" 2>&1; then
      note "The previous Jetlink server is running again."
    else
      note "The previous Jetlink server did not start again; see: journalctl -u $UNIT"
    fi
  elif [ "$changed" = 1 ]; then
    as_root systemctl stop "$UNIT" >>"$LOG" 2>&1 || true
  fi
}

restore_docker_era() {
  [ -d "$DOCKER_ERA_DIR/systemd" ] || return 0
  {
    as_root rm -rf "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$UNIT.service.d"
    as_root cp -a "$DOCKER_ERA_DIR/systemd/." "$UNIT_DIR/"
    if [ -d "$DOCKER_ERA_DIR/lib" ]; then
      as_root rm -rf "$LIB_DIR"
      as_root cp -a "$DOCKER_ERA_DIR/lib" "$LIB_DIR"
    fi
    if [ -f "$DOCKER_ERA_DIR/jetlink" ]; then as_root cp -p "$DOCKER_ERA_DIR/jetlink" "$BIN"; fi
    as_root systemctl daemon-reload
    local u
    while read -r u; do
      case "$u" in
        '') ;;
        "$UNIT.service") as_root systemctl enable "$u" ;;
        *) as_root systemctl enable --now "$u" ;;
      esac
    done <"$DOCKER_ERA_DIR/enabled"
  } >>"$LOG" 2>&1 || true
}

# A jetlink-* unit the installer did not make, set up by hand on a bench,
# stays as it is, and is not saved or started again with the Docker era's.
# One that runs may hold the comma's USB interface the server needs.
others_units() {
  local f u
  for f in "$UNIT_DIR"/jetlink-*.service; do
    [ -e "$f" ] || continue
    u="$(basename "$f")"
    case " $DOCKER_ERA_UNITS $CLOCKS_UNIT " in *" $u "*) continue ;; esac
    if as_root systemctl is-active --quiet "$u" || [ "$(as_root systemctl is-enabled "$u" 2>/dev/null || true)" = enabled ]; then
      note "$u is not the installer's, and stays as it is. If it serves the comma too, stop it:"
      note "  sudo systemctl disable --now $u"
    fi
  done
  return 0
}

# ---------------------------------------------------------------------------
# The runtime: TensorRT on the host, where the Docker era had it in the image

# TensorRT's library, ONNX parser and plugins for major $1, as apt names them
trt_packages() {
  printf '%s ' "libnvinfer$1" "libnvonnxparsers$1" "libnvinfer-plugin$1"
}

# which of those, for each major given, apt has installed
apt_trt_packages() {
  local m p
  for m in "$@"; do
    for p in $(trt_packages "$m"); do
      if [ -n "$(pkg_version "$p")" ]; then printf ' %s' "$p"; fi
    done
  done
}

ensure_runtime() {
  if [ "$JETSON" = 0 ]; then
    pc_trt
    good "TensorRT $PC_TRT"
    return 0
  fi
  # --set keeps the TensorRT there is: no network, and no newer one that
  # would prepare every model again
  if [ "$KEEP_INSTALLED" = 0 ] || [ "$TRT_PRESENT" = 0 ]; then
    jetson_trt
    trt_plugins
  fi
  # what apt downloaded is as big again as what it installed
  if [ "$APT_UPDATED" = 1 ]; then apt_get clean >>"$LOG" 2>&1 || true; fi
  if ! { has_lib libnvinfer.so.10 && has_lib libnvonnxparser.so.10; }; then
    die "TensorRT 10 is not where the server can load it." \
      "Check that libnvinfer.so.10 appears in: ldconfig -p"
  fi
  TRT_VERSION="$(pkg_version libnvinfer10)"
  printf '\n==> TensorRT %s\n' "${TRT_VERSION:-from outside the package manager}" >>"$LOG"
  if [ "$JP_MAJOR" = 7 ] && [ -n "$TRT_VERSION" ] && ! version_ge "${TRT_VERSION%%-*}" "$JP7_MIN_TRT"; then
    die "This Jetson has TensorRT ${TRT_VERSION%%-*}, and Jetlink needs $JP7_MIN_TRT or newer." \
      "Update JetPack (sudo apt update && sudo apt upgrade) and run the installer again."
  fi
  if [ -n "$TRT_VERSION" ]; then good "TensorRT ${TRT_VERSION%%-*}"; else good "TensorRT 10"; fi
}

jetson_trt() {
  if [ "$TRT_PRESENT" = 1 ]; then
    if [ "$JP_MAJOR" = 7 ]; then newest_jetson_trt; fi
    return 0
  fi
  make_room_for_trt
  apt_update
  # shellcheck disable=SC2046
  step "Installing TensorRT" apt_get install --no-install-recommends $(trt_packages 10)
}

# JetPack 7.2 follows the newest TensorRT 10 in NVIDIA's Jetson repository; a
# plan built by the one before fails to load and is built again, once.
newest_jetson_trt() {
  local have want
  apt_update
  have="$(pkg_version libnvinfer10)"
  want="$(apt-cache policy libnvinfer10 2>/dev/null | sed -n 's/^ *Candidate: *//p' | head -n 1 || true)"
  if [ -z "$have" ] || [ -z "$want" ] || [ "$want" = '(none)' ] || version_ge "$have" "$want"; then
    return 0
  fi
  if [ "$(free_gb /)" -lt "$TRT_GB" ]; then
    note "Not enough room on / to update TensorRT; staying on ${have%%-*}."
    return 0
  fi
  # shellcheck disable=SC2046
  if ! run_step "Updating TensorRT to ${want%%-*}" apt_get install --only-upgrade --no-install-recommends $(trt_packages 10); then
    note "Could not update TensorRT; staying on ${have%%-*}."
  fi
}

# A PC's TensorRT: the wheel's libraries (libnvinfer, its ONNX parser and
# plugins, and the Linux builder resources, not the ones for building Windows
# plans) into $PC_TRT_DIR, which later versions reuse. It downloads beside
# that directory, whose filesystem the space check was for, and appears whole
# or not at all. A TensorRT an earlier installer put in with apt stays, unused.
TRT_STAGE=''
pc_trt() {
  local old wheel
  old="$(apt_trt_packages 11)"
  [ -z "$old" ] || note "TensorRT from apt is no longer used here; to remove it: sudo apt remove$old"
  [ "$TRT_PRESENT" = 1 ] && return 0
  make_room_for_trt
  # owned by the user, who downloads into it; a failure removes it
  TRT_STAGE="$TRT_ROOT/.new" wheel="$TRT_ROOT/.new/${PC_TRT_WHEEL##*/}"
  as_root rm -rf "$TRT_STAGE"
  as_root install -d -m 755 -o "$(id -u)" -g "$(id -g)" "$TRT_STAGE"
  step "Downloading TensorRT $PC_TRT (3.8 GB)" download "$PC_TRT_WHEEL" "$wheel"
  printf '%s  %s\n' "$PC_TRT_SHA256" "$wheel" | sha256sum -c --status \
    || die "The TensorRT download is damaged: its checksum does not match." "Run the installer again."
  step "Unpacking TensorRT" as_root unzip -q -j -o "$wheel" 'tensorrt_libs/lib*.so*' '*/LICENSE.txt' \
    -x '*_win_*' -d "$TRT_STAGE/lib"
  as_root rm -rf "$PC_TRT_DIR"
  as_root mv -T "$TRT_STAGE/lib" "$PC_TRT_DIR"
  as_root rm -rf "$TRT_STAGE"
  TRT_STAGE=''
}

# TensorRT's plugin library, where TensorRT came without it: the servers in
# Docker had it, so a plan with a plugin layer loads here too. The same build
# as libnvinfer, which it depends on exactly. Jetlink's own models use no
# plugin, so doing without is not an error.
trt_plugins() {
  local have
  has_lib libnvinfer_plugin.so.10 && return 0
  have="$(pkg_version libnvinfer10)"
  # a TensorRT from outside the package manager brings its own, or none
  [ -n "$have" ] || return 0
  apt_update
  if ! run_step "Installing TensorRT's plugins" \
      apt_get install --no-install-recommends "libnvinfer-plugin10=$have"; then
    note "Could not install TensorRT's plugins; Jetlink's models do not need them."
  fi
}

# TensorRT goes on / (a PC's under /opt/jetlink). When the Docker era's images
# are what fills it they go first, which stops the old server now rather than
# at the switch; going back to it downloads them again. Images Docker keeps on
# another disk would free nothing there, so they stay.
make_room_for_trt() {
  local where=/ free docker_root
  [ "$JETSON" = 1 ] || where=$TRT_ROOT
  free="$(free_gb "$where")"
  [ "$free" -ge "$TRT_GB" ] && return 0
  if [ "$DOCKER_ERA" = 1 ] && [ -n "$(docker_images)" ]; then
    docker_root="$(as_root docker info -f '{{.DockerRootDir}}' 2>>"$LOG" || true)"
    if [ -n "$docker_root" ] && [ "$(mount_of "$docker_root")" = "$(mount_of "$where")" ]; then
      note "$free GB free for TensorRT, which needs $TRT_GB GB: deleting Jetlink's Docker images first."
      stop_running_server
      remove_docker_images
      free="$(free_gb "$where")"
      [ "$free" -ge "$TRT_GB" ] && return 0
    else
      note "Jetlink's Docker images are not on the disk TensorRT goes on (${docker_root:-Docker did not say where}), so they stay."
    fi
  fi
  die "Not enough free space for TensorRT: $free GB on $where, and it needs $TRT_GB GB." \
    "Free some space and run the installer again."
}

# the mount point of the filesystem that holds $1, or will once it is made
mount_of() {
  local where=$1
  while [ ! -d "$where" ]; do where="$(dirname "$where")"; done
  { df -P "$where" 2>/dev/null || true; } | awk 'NR == 2 {print $NF}'
}

# Every image the Docker-era installers pulled or built, as docker rmi takes
# it: by name, or by ID once a newer pull took its tag ("<none>"), since
# docker rmi refuses repo:<none>. Only Jetlink's repositories: an untagged
# image of another repository, or one with none at all, is not ours.
docker_images() {
  command -v docker >/dev/null 2>&1 || return 0
  as_root docker image ls --format '{{.Repository}} {{.Tag}} {{.ID}}' 2>/dev/null \
    | awk '$1 == "jetlink" || $1 == "ghcr.io/zoompilot/jetlink" { print ($2 == "<none>" ? $3 : $1 ":" $2) }' \
    | sort -u || true
}

remove_docker_images() {
  local images
  images="$(docker_images)"
  [ -n "$images" ] || return 0
  # shellcheck disable=SC2086
  as_root docker rmi $images >>"$LOG" 2>&1 || true
  IMAGES_REMOVED=1
  good "Jetlink's Docker images deleted"
}

# ---------------------------------------------------------------------------
# The server: /opt/jetlink/<version>/, and `current` pointing at it

NEW_DIR='' SERVER_VERSION='' OLD_CURRENT=''

get_server() {
  if [ "$REUSE_SERVER" = 1 ]; then
    NEW_DIR="$(readlink -f "$SRC_ROOT/current")"
    SERVER_VERSION="$(tr -d '[:space:]' <"$NEW_DIR/VERSION" 2>/dev/null || basename "$NEW_DIR")"
    return 0
  fi
  local tmp name rc=0
  tmp="$(mktemp -d)"
  if [ -n "$OPT_BINARY" ]; then
    name="$(basename "$OPT_BINARY")"
    cp "$OPT_BINARY" "$tmp/$name"
    if [ -f "$OPT_BINARY.sha256" ]; then cp "$OPT_BINARY.sha256" "$tmp/$name.sha256"; fi
  else
    name="${ASSET_URL##*/}"
    printf '\n==> %s\n' "$ASSET_URL.sha256" >>"$LOG"
    download "$ASSET_URL.sha256" "$tmp/$name.sha256" >>"$LOG" 2>&1 || rc=$?
    if [ "$rc" = 22 ]; then
      no_server
    elif [ "$rc" != 0 ]; then
      die "Could not download the Jetlink server." "Check the connection and run the installer again."
    fi
    step "Downloading the Jetlink server ($RESOLVED)" download "$ASSET_URL" "$tmp/$name"
  fi
  # the .sha256 names the tarball beside it, as build-linux.sh and CI write it
  if [ -f "$tmp/$name.sha256" ] && ! (cd "$tmp" && sha256sum -c --status "$name.sha256"); then
    die "The Jetlink server download is damaged: its checksum does not match." "Run the installer again."
  fi
  unpack_server "$tmp/$name"
  rm -rf "$tmp"
  # a build of a release names it, so an update later knows where it stands
  if [ -n "$OPT_BINARY" ] && [ "$SOURCE" != local ] && [[ $SERVER_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    RESOLVED="v$SERVER_VERSION"
  fi
  good "Jetlink server $SERVER_VERSION"
}

# A dropped connection is tried again; a file that is not there (curl's 22,
# an HTTP error) is not.
download() {
  local url=$1 out=$2 attempt rc=0
  for attempt in 1 2 3; do
    rc=0
    # a minute without progress abandons the attempt
    curl -fL --connect-timeout 20 --speed-limit 1024 --speed-time 60 -o "$out" "$url" || rc=$?
    case "$rc" in
      0) return 0 ;;
      22) return 22 ;;
    esac
    if [ "$attempt" != 3 ]; then
      echo "the download was interrupted; trying again"
      sleep "$POLL_S"
    fi
  done
  return "$rc"
}

# Into a directory named for the version inside. The files sit at the top of
# the tarball or in its one folder. A version already here is replaced whole;
# a server running from it keeps the files it has open.
unpack_server() {
  local tarball=$1 stage="$SRC_ROOT/.new-$$" top ver
  as_root rm -rf "$stage"
  as_root mkdir -p "$stage"
  # owned by root, not by whoever built it: root runs it
  as_root tar -xzf "$tarball" --no-same-owner -C "$stage" >>"$LOG" 2>&1 \
    || die "$(basename "$tarball") could not be unpacked."
  top="$stage"
  if [ ! -e "$stage/bin/jetlink-server" ]; then
    top="$(find "$stage" -mindepth 3 -maxdepth 3 -path '*/bin/jetlink-server' | head -n 1 || true)"
    top="${top%/bin/jetlink-server}"
  fi
  if [ -z "$top" ] || [ ! -x "$top/bin/jetlink-server" ] || [ ! -f "$top/$SHARE/systemd/$UNIT.service" ]; then
    as_root rm -rf "$stage"
    die "$(basename "$tarball") has no bin/jetlink-server and $SHARE/systemd/$UNIT.service in it."
  fi
  # a tree from before the native server ships a unit that runs Docker, which
  # systemd would start and fail with every 2 seconds
  if ! grep -q "^ExecStart=$SRC_ROOT/current/bin/jetlink-server" "$top/$SHARE/systemd/$UNIT.service"; then
    as_root rm -rf "$stage"
    die "The unit in $(basename "$tarball") does not start $SRC_ROOT/current/bin/jetlink-server." \
      "It was built from an older tree. Build the tarball from the current one:" \
      "  scripts/build-linux.sh $FLAVOR"
  fi
  ver="$(tr -d '[:space:]' <"$top/VERSION" 2>/dev/null || true)"
  if [ -z "$ver" ]; then
    ver="$(basename "$tarball" | sed -n 's/^jetlink-server-\(.*\)-linux-[a-z0-9_]*\.tar\.gz$/\1/p')"
  fi
  if ! [[ $ver =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]] || [ "$ver" = current ] || [ "$ver" = previous ] || [ "$ver" = src ] || [ "$ver" = tensorrt ]; then
    as_root rm -rf "$stage"
    die "Cannot tell which version $(basename "$tarball") is."
  fi
  SERVER_VERSION="$ver" NEW_DIR="$SRC_ROOT/$ver"
  as_root rm -rf "$NEW_DIR"
  as_root mv "$top" "$NEW_DIR"
  as_root rm -rf "$stage"
}

# The installer's GPU check is the server's own: TensorRT has to load and see
# a GPU. It runs before the old server stops, so a server that cannot work
# never replaces one that does.
check_gpu() {
  local out rc=0 trt
  printf '\n==> %s/bin/jetlink-server backends --backend trt %s\n' "$NEW_DIR" "${TRT_ARGS[*]}" >>"$LOG"
  out="$(as_root "$NEW_DIR/bin/jetlink-server" backends --backend trt "${TRT_ARGS[@]}" 2>&1)" || rc=$?
  printf '%s\n' "$out" >>"$LOG"
  # "trt: usable: TensorRT 10.16.2.10 on Orin-sm87", or "trt: not usable: why"
  trt="$(printf '%s\n' "$out" | sed -n 's/^trt: //p' | head -n 1)"
  if [ "$rc" = 0 ]; then
    good "The server can use the GPU: ${trt#usable: }"
    return 0
  fi
  if [ -n "$trt" ]; then
    bad "The Jetlink server cannot use the GPU: ${trt#not usable: }"
  else
    bad "The Jetlink server did not start."
    printf '%s\n' "$out" | tail -n 5 | sed 's/^/    /'
  fi
  local hint="Restart the computer and run the installer again: a new driver needs a restart."
  [ "$JETSON" = 1 ] && hint="Check that JetPack installed completely (sudo apt install nvidia-jetpack), then run the installer again."
  die "TensorRT cannot run on this computer." "$hint"
}

# An atomic switch: a new link renamed over the old one.
point_current() {
  as_root ln -sfn "$1" "$SRC_ROOT/current.new"
  as_root mv -Tf "$SRC_ROOT/current.new" "$SRC_ROOT/current"
}

switch_server() {
  OLD_CURRENT=''
  if [ -L "$SRC_ROOT/current" ]; then OLD_CURRENT="$(readlink -f "$SRC_ROOT/current")"; fi
  CHANGED=1
  point_current "$NEW_DIR"
}

# After a good start: the one before becomes `previous`, the way back by hand,
# and any older one goes.
keep_previous() {
  if [ -n "$OLD_CURRENT" ] && [ "$OLD_CURRENT" != "$NEW_DIR" ] && [ -d "$OLD_CURRENT" ]; then
    as_root ln -sfn "$OLD_CURRENT" "$SRC_ROOT/previous"
  fi
  local prev='' d
  if [ -L "$SRC_ROOT/previous" ]; then prev="$(readlink -f "$SRC_ROOT/previous")"; fi
  for d in "$SRC_ROOT"/*; do
    if [ -L "$d" ] || [ ! -x "$d/bin/jetlink-server" ] || [ "$d" = "$NEW_DIR" ] || [ "$d" = "$prev" ]; then
      continue
    fi
    as_root rm -rf "$d"
    printf '\n==> removed the old server %s\n' "$d" >>"$LOG"
  done
  # a PC's TensorRT from before the one this server uses
  for d in "$TRT_ROOT"/*; do
    if [ -d "$d" ] && [ "$d" != "$PC_TRT_DIR" ]; then as_root rm -rf "$d"; fi
  done
}

# ---------------------------------------------------------------------------
# The rest of the computer

configure_jetson() {
  [ "$JETSON" = 1 ] || return 0
  if [ "$DOCKER_ERA" = 0 ] && [ -n "$PM_WANT_ID" ] && [ "$PM_CURRENT" != "$PM_WANT_NAME" ]; then
    set_power_mode
  fi
  if [ "$ADD_SWAP_GB" -gt 0 ]; then
    SWAP_FILE="$(dirname "$CACHE_DIR")/jetlink-swapfile"
    step "Adding ${ADD_SWAP_GB} GB of swap" add_swap "$SWAP_FILE"
    SWAP_FILE_GB=$ADD_SWAP_GB
  fi
  if [ "$REMOVE_SWAP" = 1 ]; then
    step "Removing Jetlink's swap file" remove_swap
    SWAP_FILE='' SWAP_FILE_GB=0
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
  shorten_uefi_wait
  quiet_kernel
  set_desktop
  if [ "$JOURNALD_CAPPED" != 1 ]; then
    printf '# Jetlink: keep the system log from filling a small root partition\n[Journal]\nSystemMaxUse=200M\n' \
      | root_write "$JOURNALD_DROPIN"
    as_root systemctl restart systemd-journald >>"$LOG" 2>&1 || true
    JOURNALD_CAPPED=1
    good "System log limited to 200 MB"
  fi
}

# Boot. On switched power the Jetson boots at every start of the car, and the
# big model drives only once the server is up. On an Orin Nano with JetPack
# 7.2.1, a reboot to the model loaded took 4 s less with a 1 s firmware menu,
# 4 s less with quiet (the serial console runs at 115200 baud), and 4 s less
# with the server waiting for the GPU's driver (GPU_DROPIN) instead of for
# nvpmodel.

# the firmware's boot menu wait in seconds after `efibootmgr ARGS`, which
# lists the variables once it has changed any; `none` when none is set, or
# nothing when efibootmgr cannot say
uefi_timeout() {
  local out
  out="$(as_root efibootmgr "$@" 2>>"$LOG" || true)"
  [ -n "$out" ] || return 0
  if [[ $out =~ Timeout:\ ([0-9]+) ]]; then echo "${BASH_REMATCH[1]}"; else echo none; fi
}

# down to UEFI_TIMEOUT, never up: a shorter wait is the user's own
shorten_uefi_wait() {
  [ -d "$EFIVARS" ] && command -v efibootmgr >/dev/null 2>&1 || return 0
  local now
  now="$(uefi_timeout)"
  case "$now" in
    '') note "Could not read the firmware's boot menu wait, so it stays as it is."; return 0 ;;
    none) ;;
    *) [ "$now" -gt "$UEFI_TIMEOUT" ] || return 0 ;;
  esac
  if [ "$(uefi_timeout -t "$UEFI_TIMEOUT")" = "$UEFI_TIMEOUT" ]; then
    # what it was before Jetlink first changed it, which a later run keeps
    [ -n "$UEFI_TIMEOUT_PREV" ] || UEFI_TIMEOUT_PREV="$now"
    good "Firmware boot menu waits $UEFI_TIMEOUT s"
  else
    note "The firmware kept its boot menu wait; efibootmgr could not change it."
  fi
}

restore_uefi_wait() {
  [ -n "$UEFI_TIMEOUT_PREV" ] && command -v efibootmgr >/dev/null 2>&1 || return 0
  # a wait changed since is the user's own
  [ "$(uefi_timeout)" = "$UEFI_TIMEOUT" ] || return 0
  local back
  if [ "$UEFI_TIMEOUT_PREV" = none ]; then back="$(uefi_timeout -T)"; else back="$(uefi_timeout -t "$UEFI_TIMEOUT_PREV")"; fi
  if [ "$back" = "$UEFI_TIMEOUT_PREV" ]; then good "The firmware's boot menu wait is as it was"; fi
}

# extlinux.conf on stdout with quiet added to (add) or taken off the end of
# (remove) the APPEND line of the entry that boots: DEFAULT's, else the first
# LABEL's. Exits 0 with that line changed, 1 with nothing to change, and 2
# when that entry has no APPEND line or more than one, which is left alone.
extlinux_quiet() {
  awk -v mode="$1" '
    NR == FNR {
      key = toupper($1)
      if (key == "DEFAULT" && def == "") def = $2
      if (key == "LABEL" && first == "") first = $2
      next
    }
    FNR == 1 && def == "" { def = first }
    toupper($1) == "LABEL" { inside = ($2 == def) }
    inside && toupper($1) == "APPEND" {
      lines++
      if (mode == "add") {
        has = 0
        for (i = 2; i <= NF; i++) if ($i == "quiet") has = 1
        if (!has) { $0 = $0 " quiet"; changed++ }
      } else if (sub(/ quiet$/, "")) {
        changed++
      }
    }
    { print }
    END { exit (lines != 1 ? 2 : (changed ? 0 : 1)) }
  ' "$EXTLINUX" "$EXTLINUX"
}

# quiet added (add) or taken off (remove) as extlinux_quiet says, with its
# status. The new file is written beside the old one and renamed over it, so a
# power cut leaves one whole file or the other. The first add keeps the file as
# it was in extlinux.conf.jetlink-bak, which is also how uninstall knows the
# quiet is Jetlink's.
edit_extlinux() {
  local new rc=0
  new="$(mktemp)"
  extlinux_quiet "$1" >"$new" || rc=$?
  if [ "$rc" = 0 ]; then
    [ -e "$EXTLINUX.jetlink-bak" ] || as_root cp -p "$EXTLINUX" "$EXTLINUX.jetlink-bak"
    root_write "$EXTLINUX.jetlink-new" <"$new"
    as_root mv -f "$EXTLINUX.jetlink-new" "$EXTLINUX"
    as_root sync "$EXTLINUX" "$(dirname "$EXTLINUX")"
  fi
  rm -f "$new"
  return "$rc"
}

# Kernel messages stay in the journal and dmesg. A quiet the user added stays.
quiet_kernel() {
  [ -f "$EXTLINUX" ] || return 0
  local rc=0
  edit_extlinux add || rc=$?
  case "$rc" in
    0) good "Kernel messages kept off the console" ;;
    1) ;;
    *) note "$EXTLINUX has no boot entry Jetlink can add quiet to, so it stays as it is." ;;
  esac
}

restore_kernel_messages() {
  [ -e "$EXTLINUX.jetlink-bak" ] || return 0
  if edit_extlinux remove; then good "Kernel messages are on the console at boot again"; fi
  as_root rm -f "$EXTLINUX.jetlink-bak"
}

# The desktop goes or comes back at the next start: stopping it now could end
# the session the installer runs in.
set_desktop() {
  [ "$DESKTOP_OFF" != "$DESKTOP_OFF_SAVED" ] || return 0
  local target=graphical.target why="bring the desktop back"
  [ "$DESKTOP_OFF" = 0 ] || target=multi-user.target why="turn the desktop off"
  if as_root systemctl set-default "$target" >>"$LOG" 2>&1; then
    REBOOT_FOR="${REBOOT_FOR:+$REBOOT_FOR and }$why"
  else
    DESKTOP_OFF=$DESKTOP_OFF_SAVED
    note "Could not change whether the desktop starts; see $LOG"
  fi
}

restore_desktop() {
  [ "$DESKTOP_OFF" = 1 ] && [ "$(systemctl get-default 2>/dev/null || true)" = multi-user.target ] || return 0
  as_root systemctl set-default graphical.target >>"$LOG" 2>&1 || true
  good "The desktop comes back at the next start"
}

set_power_mode() {
  printf '\n==> nvpmodel -m %s\n' "$PM_WANT_ID" >>"$LOG"
  # nvpmodel asks whether to reboot when the new mode needs one; say no here
  # and tell the user at the end
  printf 'no\n' | as_root nvpmodel -m "$PM_WANT_ID" >>"$LOG" 2>&1 || true
  local now
  now="$(power_mode_now)"
  # nvpmodel switches live only when nothing used the GPU before it ran at
  # boot, and otherwise wants a restart; the server holds the clocks to the
  # mode's limits either way (--power-mode)
  local held=''
  [ "$now" = "$PM_WANT_NAME" ] || held=" ${D}(Jetlink holds its clock limits)${N}"
  good "Power mode set to $PM_WANT_NAME$held"
}

add_swap() {
  local file=$1
  as_root mkdir -p "$(dirname "$file")"
  as_root fallocate -l "${ADD_SWAP_GB}G" "$file"
  as_root chmod 600 "$file"
  as_root mkswap "$file"
  as_root swapon "$file"
  grep -q "^$file " /etc/fstab || printf '%s none swap sw 0 0\n' "$file" | as_root tee -a /etc/fstab >/dev/null
}

# A swap file the memory cannot take back yet is still deleted: it is gone
# from fstab, and its space comes back at the next restart.
remove_swap() {
  if [ -f "$SWAP_FILE" ]; then
    as_root swapoff "$SWAP_FILE" >>"$LOG" 2>&1 || true
    as_root rm -f "$SWAP_FILE"
  fi
  as_root sed -i "\#^$SWAP_FILE #d" /etc/fstab
}

# The jetson_clocks unit and drop-in older installs have (CLOCKS_UNIT).
remove_clocks_units() {
  as_root rm -f "$CLOCKS_DROPIN"
  [ -e "$UNIT_DIR/$CLOCKS_UNIT" ] || return 0
  as_root systemctl disable "$CLOCKS_UNIT" >>"$LOG" 2>&1 || true
  as_root rm -f "$UNIT_DIR/$CLOCKS_UNIT"
}

install_files() {
  as_root install -d -m 755 "$ETC_DIR"
  as_root mkdir -p "$CACHE_DIR"
  as_root install -D -m 755 "$SOURCE_DIR/scripts/jetlink" "$BIN"
  install_unit "$NEW_DIR"
  printf '# Jetlink: the cache has to be mounted before the server starts\n[Unit]\nRequiresMountsFor=%s\n' "$CACHE_DIR" \
    | root_write "$UNIT_DIR/$UNIT.service.d/10-cache.conf"
  remove_clocks_units
  if [ "$JETSON" = 1 ]; then
    gpu_dropin | root_write "$GPU_DROPIN"
  fi
  if [ "$DOCKER_ERA" = 1 ]; then set_aside_docker_dropins; fi

  # the hubs are armed for remote wakeup at boot by the rule, and again by
  # the server before every suspend
  if [ "$SLEEP_AFTER" != 0 ]; then
    as_root install -D -m 644 "$NEW_DIR/$SHARE/udev/99-jetlink-usb-wakeup.rules" "$WAKE_RULE"
    as_root udevadm control --reload-rules >>"$LOG" 2>&1 || true
    as_root udevadm trigger --subsystem-match=usb --action=add >>"$LOG" 2>&1 || true
  else
    as_root rm -f "$WAKE_RULE"
  fi

  remove_docker_era_files
  # the "no" to the poweroff question before the server took a flag for it
  if grep -qs "Written by the Jetlink installer" "$CACHE_DIR/poweroff-dry-run"; then
    as_root rm -f "$CACHE_DIR/poweroff-dry-run"
  fi
  write_env
  write_conf
}

# udev loads the GPU's driver during boot, after the server could start, and
# CUDA fails for good in a process that asked before then: systemd's restart
# cost 3.8 s on an Orin Nano. So the server waits for the control node the
# driver makes last, which CUDA opens: nvgpu's on an Orin, by either of its
# names, or the nvidia driver's; at most 30 s, then it starts anyway.
gpu_dropin() {
  cat <<'EOF'
# Written by the Jetlink installer: the server starts once the GPU's driver is up
[Service]
ExecStartPre=/bin/sh -c 'n=0; until [ -e /dev/nvhost-ctrl-gpu ] || [ -e /dev/nvgpu/igpu0/ctrl ] || [ -e /dev/nvidia0 ] || [ $$n -ge 300 ]; do sleep 0.1; n=$$((n + 1)); done'
EOF
}

# the unit of the server in directory $1, so the two always match
install_unit() {
  as_root install -D -m 644 "$1/$SHARE/systemd/$UNIT.service" "$UNIT_DIR/$UNIT.service"
}

# A drop-in of the user's for the Docker-era unit that runs docker would stop
# the native server starting. The backup has it.
set_aside_docker_dropins() {
  local f
  for f in "$UNIT_DIR/$UNIT.service.d"/*.conf; do
    [ -f "$f" ] || continue
    case "$f" in */10-cache.conf|"$GPU_DROPIN") continue ;; esac
    grep -qi docker "$f" || continue
    as_root rm -f "$f"
    note "Your drop-in $(basename "$f") runs Docker, so it is set aside in $DOCKER_ERA_DIR/systemd/$UNIT.service.d"
  done
  return 0
}

# What the server does itself now: the launcher, the hub wakeup script, the
# poweroff flag's units, and the status page's own process (never released,
# now in the server).
remove_docker_era_files() {
  local u
  for u in jetlink-poweroff.path jetlink-web.service; do
    if [ -e "$UNIT_DIR/$u" ]; then as_root systemctl disable --now "$u" >>"$LOG" 2>&1 || true; fi
  done
  as_root rm -rf "$UNIT_DIR/jetlink-poweroff.path" "$UNIT_DIR/jetlink-poweroff.service" \
    "$UNIT_DIR/jetlink-web.service" "$UNIT_DIR/jetlink-web.service.d" "$LIB_DIR"
}

write_env() {
  # the unit passes the server --poweroff only on a Jetson whose user said yes
  local poweroff=''
  if [ "$JETSON" = 1 ] && [ "$POWEROFF_WITH_COMMA" = 1 ]; then poweroff=--poweroff; fi
  # and how to set a Jetson's clocks, inside the limits of the mode it runs in
  local clocks=''
  if [ "$JETSON" = 1 ] && [ -n "$PM_WANT_NAME" ]; then clocks="--clocks $CLOCKS --power-mode $PM_WANT_NAME"; fi
  {
    echo "# Written by the Jetlink installer; run it again (jetlink setup) to change these."
    printf 'JETLINK_CACHE_DIR=%q\n' "$CACHE_DIR"
    printf 'JETLINK_SLEEP_AFTER=%q\n' "$SLEEP_AFTER"
    printf 'JETLINK_STATUS_PORT=%q\n' "$STATUS_PORT"
    # flags the unit passes as they are, or nothing when empty
    printf 'JETLINK_POWEROFF="%s"\n' "$poweroff"
    printf 'JETLINK_CLOCK_FLAGS="%s"\n' "$clocks"
    printf 'JETLINK_TENSORRT="%s"\n' "${TRT_ARGS[*]}"
    printf 'JETLINK_JETSON=%q\n' "$JETSON"
    printf 'JETLINK_FLAVOR=%q\n' "$FLAVOR"
    printf 'JETLINK_SERVER_VERSION=%q\n' "$SERVER_VERSION"
  } | root_write "$ENV_FILE"
}

write_conf() {
  {
    echo "# The answers the Jetlink installer was given; it reads them back on an update."
    printf 'JETLINK_REF=%q\n' "$REF"
    printf 'JETLINK_VERSION=%q\n' "$RESOLVED"
    printf 'JETLINK_SOURCE=%q\n' "$SOURCE"
    printf 'JETLINK_SOURCE_DIR=%q\n' "$SOURCE_DIR"
    printf 'JETLINK_COMMIT=%q\n' "$COMMIT"
    printf 'JETLINK_PLATFORM_NAME=%q\n' "$PLATFORM_NAME"
    printf 'JETLINK_POWER=%q\n' "$POWER"
    printf 'JETLINK_POWEROFF_WITH_COMMA=%q\n' "$POWEROFF_WITH_COMMA"
    printf 'JETLINK_DESKTOP_OFF=%q\n' "$DESKTOP_OFF"
    printf 'JETLINK_AUTOSTART=%q\n' "$AUTOSTART"
    printf 'JETLINK_POWER_MODE=%q\n' "$POWER_MODE"
    printf 'JETLINK_POWER_MODES=%q\n' "$PM_NAMES"
    printf 'JETLINK_CLOCKS=%q\n' "$CLOCKS"
    printf 'JETLINK_SWAP=%q\n' "$SWAP"
    printf 'JETLINK_SWAP_FILE=%q\n' "$SWAP_FILE"
    printf 'JETLINK_SWAP_GB=%q\n' "$SWAP_FILE_GB"
    printf 'JETLINK_SWAP_NEED_GB=%q\n' "$SWAP_NEED_GB"
    printf 'JETLINK_SWAP_SHORT_GB=%q\n' "$SWAP_SHORT_GB"
    printf 'JETLINK_DEFAULTS=%q\n' "$(default_answers)"
    printf 'JETLINK_MASKED_UNITS=%q\n' "$MASKED_UNITS"
    printf 'JETLINK_JOURNALD_CAPPED=%q\n' "$JOURNALD_CAPPED"
    printf 'JETLINK_UEFI_TIMEOUT_PREV=%q\n' "$UEFI_TIMEOUT_PREV"
    printf 'JETLINK_INSTALLED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } | root_write "$CONF"
}

start_server() {
  as_root systemctl daemon-reload
  if [ "$AUTOSTART" = 1 ]; then
    as_root systemctl enable "$UNIT" >>"$LOG" 2>&1
  else
    as_root systemctl disable "$UNIT" >>"$LOG" 2>&1 || true
  fi
  local since
  since="$(date '+%Y-%m-%d %H:%M:%S')"
  as_root systemctl restart "$UNIT"
  step "Starting the Jetlink server" wait_ready "$since"
  if [[ $(server_journal "$since") == *"$PRELOAD_LINE"* ]]; then
    step "Loading the model it ran last" wait_engine "$since"
    if [ -z "$(engine_line "$since")" ]; then
      note "It is still preparing that model, and goes on in the background: jetlink logs"
    fi
  fi
  # only now, with the new server up and staying up: until then the old one,
  # its settings and its images are the way back
  SERVER_STOPPED=0 CHANGED=0 SLEEP_HELD=0
  keep_previous
  if [ "$DOCKER_ERA" = 1 ]; then remove_docker_images; fi
}

# The server says the first once it serves, whatever the comma is doing; the
# others are what builds before that line said, for a --binary of one.
READY_RE='jetlink-server is serving|waiting for a jetlink gadget|client connected|nothing on the comma is serving it yet'
# systemd's line when it starts a server that exited again
RESTART_RE='restart counter is at [0-9]'
# said before the server serves when it loads the model it ran last
PRELOAD_LINE='preloading the engine loaded last'

server_journal() {
  as_root journalctl -u "$UNIT" --since "$1" --no-pager -o cat 2>/dev/null || true
}

# Up means the server said it serves and then stayed up; a restart means it
# crashed. The journal is followed, so either shows the moment it is written.
# The follower ends at its next line or at the timeout.
wait_ready() {
  local since=$1 line
  line="$(grep -m1 -E "$READY_RE|$RESTART_RE" \
    < <(as_root timeout "$READY_S" journalctl -f -u "$UNIT" --since "$since" -o cat 2>/dev/null) || true)"
  case "$line" in
    *"restart counter"*|'')
      server_journal "$since" | tail -n 30
      if [ -n "$line" ]; then echo "the server crashed, and systemd started it again"; else echo "the server did not report ready within 3 minutes"; fi
      return 1 ;;
  esac
  echo "$line"
  # one loading a model is watched until that is done, by wait_engine
  [[ $(server_journal "$since") == *"$PRELOAD_LINE"* ]] && return 0
  stays_up "$since"
}

# The model the server ran last, loaded, or its preparation failed (the
# server serves without it then, and the comma sends it again); a restart
# means loading it crashed the server. After PRELOAD_S the server is left to it.
wait_engine() {
  local since=$1 line deadline=$((SECONDS + PRELOAD_S))
  while :; do
    line="$(engine_line "$since")"
    case "$line" in
      *"restart counter"*)
        server_journal "$since" | tail -n 30
        echo "loading the model crashed the server, and systemd started it again"
        return 1 ;;
      ?*) echo "$line"; break ;;
    esac
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "still preparing the model after $(elapsed "$PRELOAD_S"); the server serves meanwhile"
      break
    fi
    sleep "$POLL_S"
  done
  stays_up "$since"
}

engine_line() {
  server_journal "$1" | grep -m1 -E "engine ready|engine preparation failed|$RESTART_RE" || true
}

# Still running a moment later and never restarted: a server can crash just
# after it said it serves, when the comma is first seen or the model loads.
stays_up() {
  local since=$1 restarts
  sleep "$SETTLE_S"
  restarts="$(as_root systemctl show -p NRestarts --value "$UNIT" 2>/dev/null || true)"
  if as_root systemctl is-active --quiet "$UNIT" && [ "${restarts:-0}" = 0 ]; then
    return 0
  fi
  server_journal "$since" | tail -n 30
  echo "the server did not stay up: systemd started it again ${restarts:-?} times"
  return 1
}

# The web page's password goes in once the new server is up and staying up,
# so a run that fails leaves no password nobody was shown; the server reads
# the file again whenever it changes. The server writes it, since it knows
# the format, and a typed one goes in on stdin, where ps cannot see it.
write_web_auth() {
  case "$WEB_AUTH" in generate|typed) ;; *) return 0 ;; esac
  # a server from before the sign-in has no web-password, and its unit no --web-auth
  if ! grep -qs -- '--web-auth' "$NEW_DIR/$SHARE/systemd/$UNIT.service"; then
    note "This server's web page has no sign-in; it shows the status only."
    WEB_AUTH='' WEB_PASSWORD=''
    return 0
  fi
  local server="$NEW_DIR/bin/jetlink-server" err rc=0
  err="$(mktemp)"
  if [ "$WEB_AUTH" = generate ]; then
    WEB_PASSWORD="$(as_root "$server" web-password --file "$AUTH_FILE" --generate 2>"$err")" || rc=$?
  else
    printf '%s\n' "$WEB_PASSWORD" | as_root "$server" web-password --file "$AUTH_FILE" >/dev/null 2>"$err" || rc=$?
  fi
  if [ "$rc" = 0 ] && [ -n "$WEB_PASSWORD" ]; then
    good "Web page password set"
  else
    note "Could not set the web page's password: $(tail -n 1 "$err")"
    note "Set one with: sudo jetlink password"
    WEB_AUTH='' WEB_PASSWORD=''
  fi
  rm -f "$err"
}

finish() {
  save_log >/dev/null
  heading "${G}Jetlink is installed and running.${N}"
  if [ "$DOCKER_ERA" = 1 ]; then
    say "  It no longer runs in Docker. Docker stays installed for anything else that uses it,"
    say "  and the old setup is kept in $DOCKER_ERA_DIR."
  fi
  say ""
  say "  ${B}Next, on your comma:${N}"
  say "    1. Settings > Software > Target Branch: choose ${B}main${N} (zoompilot),"
  say "       and let it update and restart."
  say "    2. Settings > Models: set ${B}Jetlink${N} to ${B}USB${N}."
  if [ "$JETSON" = 1 ]; then
    say "    3. Connect the comma's USB-C port to one of this Jetson's ${B}USB-A${N} ports"
  else
    say "    3. Connect the comma's USB-C port to one of this computer's ${B}USB-A${N} ports"
  fi
  say "       with a USB 3 data cable (charge-only cables do not work)."
  say "    4. Stay parked and wait for the comma's icon to turn ${G}green${N}. The first model"
  say "       takes a few minutes to prepare."
  if [ "$WSL" = 1 ]; then
    say ""
    note "In Windows, attach the comma to WSL with usbipd: https://learn.microsoft.com/windows/wsl/connect-usb"
  fi
  if [ "$JETSON" = 0 ]; then
    say ""
    note "Keep this computer plugged in and awake while driving: sleep drops the link."
  fi
  if [ "$STATUS_PORT" != 0 ]; then
    say ""
    say "  ${B}Web page:${N} http://$(hostname 2>/dev/null || uname -n).local:$STATUS_PORT"
    say "    from a phone or computer on the same network, like the comma's hotspot."
    case "$WEB_AUTH" in
      generate) say "    ${B}Password:${N} $WEB_PASSWORD ${D}(change it with: sudo jetlink password)${N}" ;;
      typed) say "    Sign in with the password you chose; sudo jetlink password sets a new one." ;;
      keep) say "    Sign in with its password, as before; sudo jetlink password sets a new one." ;;
    esac
  fi
  say ""
  say "  ${B}Handy commands:${N}"
  say "    jetlink status    is it running, and is the comma connected"
  say "    jetlink logs      watch what it is doing"
  say "    jetlink update    get the newest version"
  say "    jetlink setup     change your answers"
  if [ -n "$REBOOT_FOR" ]; then
    say ""
    note "${B}Restart this computer once${N} to $REBOOT_FOR: sudo reboot"
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
    "TensorRT from apt and Docker stay installed."
  [ "$go" = y ] || { say "  Nothing changed."; exit 0; }
  [ "$OPT_DRY_RUN" = 1 ] && { say "  (dry run: nothing changed)"; exit 0; }
  get_root
  as_root systemctl disable --now "$UNIT" >>"$LOG" 2>&1 || true
  remove_docker_era_files
  if command -v docker >/dev/null 2>&1; then as_root docker rm -f jetlink >>"$LOG" 2>&1 || true; fi
  remove_clocks_units
  as_root rm -rf "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$UNIT.service.d"
  as_root systemctl daemon-reload
  as_root rm -f "$BIN" "$WAKE_RULE"
  as_root udevadm control --reload-rules >>"$LOG" 2>&1 || true
  good "Server and its settings removed"
  restore_uefi_wait
  restore_kernel_messages
  restore_desktop
  if [ -n "$MASKED_UNITS" ]; then
    # shellcheck disable=SC2086
    as_root systemctl unmask $MASKED_UNITS >>"$LOG" 2>&1 || true
  fi
  if [ -f "$JOURNALD_DROPIN" ]; then
    as_root rm -f "$JOURNALD_DROPIN"
    as_root systemctl restart systemd-journald >>"$LOG" 2>&1 || true
  fi
  if [ -n "$SWAP_FILE" ] && [ -f "$SWAP_FILE" ]; then
    remove_swap
    good "Swap file removed"
  fi
  if [ -n "$(docker_images)" ]; then
    local rm_images
    ask_yn rm_images y "Delete Jetlink's old Docker images? ${D}(about 4 GB each)${N}"
    if [ "$rm_images" = y ]; then
      remove_docker_images
    fi
  fi
  if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
    local size rm_cache
    size="$(as_root du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
    ask_yn rm_cache n "Also delete the downloaded models in $CACHE_DIR? ${D}($size)${N}" \
      "Keep them if you might reinstall: they take a while to download."
    if [ "$rm_cache" = y ]; then
      as_root rm -rf "$CACHE_DIR"
      good "Models deleted"
    fi
  fi
  # the answers, the settings and the web page's password with them
  as_root rm -rf "$ETC_DIR" "$SRC_ROOT"
  heading "Jetlink is removed."
  local trt
  trt="$(apt_trt_packages 10 11)"
  [ -z "$trt" ] || say "  TensorRT from apt stays installed; to remove it: sudo apt remove$trt"
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
      --ref) OPT_REF="${2:?--ref needs a branch or tag}"; shift ;;
      --ref=*) OPT_REF="${1#*=}" ;;
      --binary) OPT_BINARY="${2:?--binary needs a server tarball}"; shift ;;
      --binary=*) OPT_BINARY="${1#*=}" ;;
      --dry-run) OPT_DRY_RUN=1 ;;
      --uninstall) OPT_UNINSTALL=1 ;;
      --set) OPT_SET+=("${2:?--set needs KEY=VALUE}"); shift ;;
      --set=*) OPT_SET+=("${1#*=}") ;;
      --build|--image|--image=*)
        die "Jetlink no longer runs in Docker, so $1 is gone." \
          "To install a server you built: --binary jetlink-server-<version>-<flavor>.tar.gz" ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown option: $1" "Run with --help to see the options." ;;
    esac
    shift
  done
  if [ ${#OPT_SET[@]} -gt 0 ]; then
    if [ "$OPT_UPDATE" = 1 ] || [ "$OPT_UNINSTALL" = 1 ] || [ -n "$OPT_REF" ] || [ -n "$OPT_BINARY" ]; then
      die "--set does not go with --update, --uninstall, --ref or --binary." \
        "It changes answers, and keeps the server that is installed."
    fi
    OPT_RECONFIGURE=1
  fi
}

main() {
  # the script has been read in full by now; nothing below may read stdin
  exec </dev/null
  ARGS=("$@")
  parse_args "$@"
  setup_colors
  # the output as it is here, for on_signal: a step sends its own to the log
  exec 4>&1 5>&2
  trap 'on_error $LINENO' ERR
  trap 'on_signal INT 130' INT
  trap 'on_signal TERM 143' TERM
  trap 'on_signal HUP 129' HUP
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
  detect_source
  check_set
  choose_ref
  if [ -z "$CACHE_DIR" ]; then
    CACHE_DIR=/var/lib/jetlink
    [ "$JETSON" = 1 ] && CACHE_DIR=/mnt/data/jetlink
  fi
  DISK_GB="${JETLINK_TEST_FREE_GB:-$(free_gb "$CACHE_DIR")}"
  check_binary
  show_found
  preflight

  if [ "$HAD_INSTALL" = 1 ] && [ "$OPT_UPDATE" = 0 ] && [ "$OPT_RECONFIGURE" = 0 ] && [ "$INTERACTIVE" = 1 ]; then
    local keep
    ask_yn keep y "Jetlink is already installed. Update it and keep your answers?" \
      "No asks the questions again."
    [ "$keep" = y ] && OPT_UPDATE=1
  fi
  if [ ${#OPT_SET[@]} -gt 0 ]; then
    apply_set
  elif [ "$OPT_UPDATE" = 0 ] || [ "$HAD_INSTALL" = 0 ]; then
    ask_questions   # on an update the saved answers stand
  fi
  choose_web_password
  jetson_musts

  resolve_ref
  choose_server
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
  install_base_packages
  prepare_source
  backup_install
  others_units
  hold_sleep
  get_server
  ensure_runtime
  check_gpu
  stop_running_server
  switch_server
  configure_jetson
  install_files
  start_server
  write_web_auth
  finish
}

main "$@"
