#!/usr/bin/env bash
# The installer, end to end, inside a throwaway Ubuntu container: real files
# written, real units and scripts installed, and every system command the
# installer calls (apt, systemd, the server, nvpmodel, ...) replaced by
# fake.sh. run.sh starts the container; this runs in it, as root, with the
# source tree at /src and the Docker releases' installers under /releases.
#
# Each scenario is a computer (a JetPack 7.2 Jetson, a JetPack 6 one, a PC,
# an install one of the Docker releases left) plus the answers typed at the
# questions, and checks what the installer left behind, what it ran, and what
# it told the user.
set -uo pipefail

SRC=/src
FAKE_BIN=/tmp/fakebin
export FAKE_BIN FAKE_LOG=/tmp/fake.log FAKE_STATE=/tmp/fake-state
export JETLINK_TEST_DT_MODEL=/tmp/dt-model JETLINK_TEST_MEM_SLEEP=/tmp/mem-sleep
export JETLINK_TEST_PROC_VERSION=/tmp/proc-version JETLINK_TEST_SYSTEMD_RUN=/tmp
export JETLINK_TEST_OS_RELEASE=/tmp/os-release JETLINK_TEST_SECURE_BOOT=/tmp/secure-boot
export JETLINK_TEST_PKG_PATH=$FAKE_BIN
# a Jetson's boot loader configuration, and its firmware's variables
export JETLINK_TEST_EXTLINUX=/tmp/boot/extlinux/extlinux.conf JETLINK_TEST_EFIVARS=/tmp/efivars
EXTLINUX=$JETLINK_TEST_EXTLINUX
# the lock a native server makes to be held awake; none until a scenario says
export JETLINK_TEST_AWAKE_LOCK=/tmp/jetlink-awake.lock
LOCK=$JETLINK_TEST_AWAKE_LOCK
# the same questions on every machine: plenty of disk, an 8 GB Orin Nano's
# memory, and no swap yet
export JETLINK_TEST_FREE_GB=100 JETLINK_TEST_SWAPS=/tmp/swaps JETLINK_TEST_MEMINFO=/tmp/meminfo
# nothing waited on here is real, so there is nothing to wait for
export JETLINK_TEST_POLL_S=0 JETLINK_TEST_SETTLE_S=0
printf 'Filename\tType\tSize\tUsed\tPriority\n/dev/zram0 partition 1000000 0 5\n' >/tmp/swaps
printf 'MemTotal:        7727244 kB\n' >/tmp/meminfo
PATH="$FAKE_BIN:$PATH"
OUT=/tmp/out.txt
UNITS=/etc/systemd/system
FAILED=0 PASSED=0 SCENARIO=''

ok() { PASSED=$((PASSED + 1)); }
fail() { FAILED=$((FAILED + 1)); printf '    FAIL [%s] %s\n' "$SCENARIO" "$*"; }
check() {  # check "failure message" command...: pass when the command succeeds
  local msg=$1
  shift
  if "$@"; then ok; else fail "$msg"; fi
}
refute() {  # refute "failure message" command...: pass when the command fails
  local msg=$1
  shift
  if "$@"; then fail "$msg"; else ok; fi
}
expect_out() { check "output lacks: $1" grep -qF -- "$1" "$OUT"; }
expect_no_out() { refute "output has: $1" grep -qF -- "$1" "$OUT"; }
expect_ran() { check "never ran: $1" grep -qF -- "$1" "$FAKE_LOG"; }
expect_not_ran() { refute "ran: $1" grep -qF -- "$1" "$FAKE_LOG"; }
expect_file() { check "missing file: $1" test -e "$1"; }
expect_no_file() { refute "file should be gone: $1" test -e "$1"; }
expect_in() { check "$1 lacks: $2" grep -qF -- "$2" "$1"; }
expect_not_in() { refute "$1 has: $2" grep -qF -- "$2" "$1"; }
expect_rc() { check "exit $RC, wanted $1" test "$RC" = "$1"; }
expect_link() { check "$1 points at $(readlink -f "$1" 2>/dev/null), wanted $2" test "$(readlink -f "$1" 2>/dev/null)" = "$2"; }
first_line() { grep -nF -- "$1" "$FAKE_LOG" | head -n 1 | cut -d: -f1; }
expect_before() {  # expect_before A B: the first run of A came before the first run of B
  local a b
  a="$(first_line "$1")" b="$(first_line "$2")"
  check "never ran: $1" test -n "$a"
  check "never ran: $2" test -n "$b"
  if [ -n "$a" ] && [ -n "$b" ]; then check "$1 ran after $2" test "$a" -lt "$b"; fi
}
apt_install() { printf 'apt-get -o DPkg::Lock::Timeout=900 -y install --no-install-recommends %s' "$1"; }

reset_box() {
  rm -rf /etc/jetlink /usr/local/lib/jetlink /usr/local/bin/jetlink /opt/jetlink /var/lib/jetlink /mnt/data \
    "$UNITS"/jetlink-* /etc/udev/rules.d/99-jetlink-usb-wakeup.rules \
    /etc/systemd/journald.conf.d/60-jetlink.conf "$FAKE_STATE" "$FAKE_LOG" "$FAKE_BIN" \
    /etc/nv_tegra_release /etc/nvpmodel.conf /tmp/dt-model /tmp/mem-sleep /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    "$LOCK" /tmp/secure-boot /tmp/boot /tmp/efivars
  cp /tmp/fstab.orig /etc/fstab
  mkdir -p "$FAKE_STATE"
  echo "Linux version 6.8.0-fake (gcc) #1 SMP" >/tmp/proc-version
  os_release ubuntu "Ubuntu 24.04.2 LTS" 24.04 debian
  mkdir -p "$FAKE_BIN" "$UNITS"
  local c
  for c in uname getconf apt-get apt-cache dpkg dpkg-query ldconfig df systemctl journalctl nvpmodel ubuntu-drivers \
      udevadm fallocate mkswap swapon swapoff jetson_clocks curl gpg; do
    ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/$c"
  done
  unset FAKE_ARCH FAKE_SMI FAKE_PUBLISHED FAKE_PM_REBOOT FAKE_SERVER_BROKEN FAKE_GPU_BROKEN \
    FAKE_TRT10 FAKE_NO_CURL FAKE_ROOT_FREE_GB FAKE_IMAGE_GB FAKE_DOWNLOAD_FAILS FAKE_BAD_SUM FAKE_NO_PLUGIN \
    FAKE_DOCKER_STUCK FAKE_BAD_WHEEL FAKE_SERVER_CRASHLOOP FAKE_PRELOAD FAKE_SERVER_OLD FAKE_DOWNLOAD_HANG \
    FAKE_DOCKER_ROOT FAKE_OTHER_FS FAKE_KERNEL FAKE_GLIBC FAKE_PACMAN_STALE FAKE_UEFI_LOCKED JETLINK_TEST_PRELOAD_S
  export JETLINK_REPO_URL=file:///tmp/repo FAKE_LATEST=v0.10.0 JETLINK_TEST_SYSTEMD_RUN=/tmp
}

os_release() {  # os_release ID "PRETTY NAME" VERSION_ID [ID_LIKE]: what the installer reads
  printf 'ID=%s\nPRETTY_NAME="%s"\nVERSION_ID="%s"\nID_LIKE="%s"\n' "$1" "$2" "$3" "${4:-}" >/tmp/os-release
}

# a distribution without apt: the installer looks for package managers in
# $FAKE_BIN alone (JETLINK_TEST_PKG_PATH), so apt is there exactly when its
# fake is, and with_pkg puts the distribution's own in its place
without_apt() { rm -f "$FAKE_BIN/apt-get"; }
with_pkg() { without_apt; ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/$1"; }

distro() {  # distro NAME: the computer's os-release, and its package manager
  case "$1" in
    fedora) with_pkg dnf; os_release fedora "Fedora Linux 42 (Workstation Edition)" 42 ;;
    rocky) with_pkg dnf; os_release rocky "Rocky Linux 10.0 (Red Quartz)" 10.0 "rhel centos fedora" ;;
    arch) with_pkg pacman; os_release arch "Arch Linux" "" ;;
    cachyos) with_pkg pacman; os_release cachyos "CachyOS Linux" "" arch ;;
    manjaro) with_pkg pacman; os_release manjaro "Manjaro Linux" "" arch ;;
    tumbleweed) with_pkg zypper; os_release opensuse-tumbleweed "openSUSE Tumbleweed" 20260928 "opensuse suse" ;;
    leap) with_pkg zypper; os_release opensuse-leap "openSUSE Leap 15.6" 15.6 "suse opensuse" ;;
    debian) os_release debian "Debian GNU/Linux 12 (bookworm)" 12 ;;
    mint) os_release linuxmint "Linux Mint 22.1" 22.1 ubuntu ;;
    gentoo) without_apt; os_release gentoo "Gentoo Linux" "" ;;
  esac
}

# the firmware's SecureBoot variable, on: 4 bytes of attributes, then 1
secure_boot() { printf '\x06\x00\x00\x00\x01' >/tmp/secure-boot; }

jetson() {  # jetson L4T_RELEASE REVISION
  printf '# R%s (release), REVISION: %s, GCID: 1, BOARD: generic, EABI: aarch64, DATE: now\n' "$1" "$2" \
    >/etc/nv_tegra_release
  printf 'NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super\0' >/tmp/dt-model
  echo 's2idle [deep]' >/tmp/mem-sleep
  cat >/etc/nvpmodel.conf <<'EOF'
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=25W >
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
< POWER_MODEL ID=3 NAME=7W >
EOF
  # no TensorRT on the host: the Docker era had it only in the image
  export FAKE_ARCH=aarch64
  [ "$1" = 36 ] && export FAKE_TRT10=10.3.0.30-1+cuda12.5
  # UEFI, efibootmgr, and JetPack's extlinux.conf as a flash leaves it
  mkdir -p /tmp/efivars
  ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/efibootmgr"
  local append='root=PARTUUID=c6679549-196f-44be-9e3f-e4b5751013cb rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 firmware_class.path=/etc/firmware fbcon=map:0'
  if [ "$1" = 36 ]; then
    append="$append nospectre_bhb video=efifb:off console=tty0"
  else
    append="$append video=efifb:off console=tty0 efi_pstore.pstore_disable=1 pstore.backend=ramoops efi=runtime pci=pcie_bus_perf nvme.use_threaded_interrupts=1 swiotlb=2048 "
  fi
  extlinux "TIMEOUT 30
DEFAULT primary

MENU TITLE L4T boot options

LABEL primary
      MENU LABEL primary kernel
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND \${cbootargs} $append

# When testing a custom kernel, it is recommended that you create a backup of
# the original kernel and add a new entry to this file so that the device can
# fallback to the original kernel. To do this:
#
# 1, Make a backup of the original kernel
#      sudo cp /boot/Image /boot/Image.backup
#
# 2, Copy your custom kernel into /boot/Image
#
# 3, Uncomment below menu setting lines for the original kernel
#
# 4, Reboot

# LABEL backup
#    MENU LABEL backup kernel
#    LINUX /boot/Image.backup
#    INITRD /boot/initrd
#    APPEND \${cbootargs}"
  return 0
}

EXTLINUX_ORIG=/tmp/boot/extlinux.orig
extlinux() {  # extlinux TEXT: the boot loader's configuration, and a copy as it was
  mkdir -p "$(dirname "$EXTLINUX")"
  printf '%s\n' "$1" >"$EXTLINUX"
  cp "$EXTLINUX" "$EXTLINUX_ORIG"
}
expect_one_quiet() { check "quiet is on more than the one line" test "$(grep -c quiet "$EXTLINUX")" = 1; }
# the file as it was with quiet on the end of its one APPEND line, and nothing else changed
expect_quiet_added() {
  check "extlinux.conf is not the one it was with quiet added" \
    cmp -s <(sed '/^ *APPEND/s/$/ quiet/' "$EXTLINUX_ORIG") "$EXTLINUX"
}
expect_extlinux_as_was() { check "extlinux.conf is not as it was" cmp -s "$EXTLINUX_ORIG" "$EXTLINUX"; }

pc() {  # pc DRIVER
  export FAKE_ARCH=x86_64 FAKE_SMI="NVIDIA GeForce RTX 4070 Laptop GPU, $1, 8.9"
  ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/nvidia-smi"
}

wsl() { echo "Linux version 6.6.87.2-microsoft-standard-WSL2 (root@fake) #1 SMP" >/tmp/proc-version; }

with_docker() { ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/docker"; }

with_trt() {  # with_trt VERSION MAJOR: TensorRT already on the host
  echo "$1" >"$FAKE_STATE/pkg-libnvinfer$2"
  echo "$1" >"$FAKE_STATE/pkg-libnvonnxparsers$2"
}

answers() { printf '%b' "$1" >/tmp/answers; }

# run_installer curl|checkout "answers" [installer args...]: as curl | bash
# runs it, the script on stdin, or from the checkout; "" means no terminal
run_installer() {
  local how=$1 input=''
  if [ -n "$2" ]; then answers "$2"; input=/tmp/answers; fi
  shift 2
  if [ "$how" = curl ]; then
    JETLINK_INPUT=$input bash -s -- "$@" <"$SRC/install.sh" >"$OUT" 2>&1
  else
    JETLINK_INPUT=$input bash "$SRC/install.sh" "$@" </dev/null >"$OUT" 2>&1
  fi
  RC=$?
}

old_install() {  # old_install TAG [args...]: that Docker release's curl | bash, with --yes
  local tag=$1 rc
  shift
  FAKE_LATEST=$tag FAKE_PUBLISHED=1 bash <"/releases/$tag/install.sh" -s -- --yes "$@" >/tmp/old.txt 2>&1
  rc=$?
  check "the $tag install failed" test "$rc" = 0
  check "$tag left no Docker server" grep -q '^JETLINK_IMAGE=' /etc/jetlink/server.env
  [ "$rc" = 0 ] || sed 's/^/    | /' /tmp/old.txt
  : >"$FAKE_LOG"
}

cli() {  # cli ARGS...: the jetlink command, as installed
  jetlink "$@" >"$OUT" 2>&1
  RC=$?
}

# interrupted SIGNAL COMMAND...: COMMAND in a session of its own, as a terminal
# runs it, and SIGNAL to all of it once the server download has started, as
# Ctrl-C (INT) or a dropped ssh session (HUP) sends it, or KILL for a power cut.
# A command started with & ignores SIGINT, as a shell without job control
# starts it, and a terminal's would not: perl puts it back.
interrupted() {
  local sig=$1 pid tries=300
  shift
  rm -f "$FAKE_STATE/download-hanging"
  # shellcheck disable=SC2016  # perl's variables, not the shell's
  FAKE_DOWNLOAD_HANG=1 setsid perl -e '$SIG{INT} = "DEFAULT"; exec @ARGV or die "exec: $!"' -- "$@" >"$OUT" 2>&1 &
  pid=$!
  while [ ! -e "$FAKE_STATE/download-hanging" ] && [ "$tries" -gt 0 ]; do
    tries=$((tries - 1))
    sleep 0.1
  done
  check "the download never started" test -e "$FAKE_STATE/download-hanging"
  kill -s "$sig" -- "-$pid"
  wait "$pid"
  RC=$?
}

# scenario "name": the one before shows its output if it failed, and this one
# starts with an empty command log
FAILED_BEFORE=0
scenario() {
  show_on_failure
  : >"$FAKE_LOG"
  FAILED_BEFORE=$FAILED SCENARIO="$1"
  printf '  %s\n' "$1"
}

show_on_failure() {
  [ -n "$SCENARIO" ] || return 0
  if [ "$FAILED" -gt "$FAILED_BEFORE" ] || [ -n "${SHOW_OUTPUT:-}" ]; then
    echo "    --- installer output ---"
    sed 's/^/    | /' "$OUT"
    echo "    --- commands run ---"
    sed 's/^/    | /' "$FAKE_LOG" 2>/dev/null | head -80
  fi
}

make_release() {  # make_release TAG VERSION [ASSET]: a tarball per arch with its .sha256, as CI publishes
  local tag=$1 ver=$2 asset=${3:-$2} arch d name
  mkdir -p "/tmp/releases/$tag"
  for arch in aarch64 x86_64; do
    d="$(mktemp -d)"
    mkdir -p "$d/bin" "$d/share/jetlink/systemd" "$d/share/jetlink/udev"
    ln -s "$SRC/tests/installer/fake.sh" "$d/bin/jetlink-server"
    echo "$ver" >"$d/VERSION"
    cp "$SRC/LICENSE" "$d/"
    cp "$SRC/scripts/jetlink-server.service" "$d/share/jetlink/systemd/"
    cp "$SRC"/scripts/*.rules "$d/share/jetlink/udev/"
    name="jetlink-server-$asset-linux-$arch.tar.gz"
    tar -czf "/tmp/releases/$tag/$name" -C "$d" .
    (cd "/tmp/releases/$tag" && sha256sum "$name" >"$name.sha256")
    rm -rf "$d"
  done
}

cp /etc/fstab /tmp/fstab.orig 2>/dev/null || : >/tmp/fstab.orig
# the certificates every computer with curl has; the container has none
mkdir -p /etc/ssl/certs
# what `curl | bash` clones: the tree under test, committed
rm -rf /tmp/repo /tmp/releases /tmp/dev
git init -q -b main /tmp/repo
# -R, not -a: the bind-mounted tree belongs to the CI runner's user, and a repo
# owned by someone else is "dubious ownership" to the root git below
cp -R "$SRC/." /tmp/repo/
git -C /tmp/repo add -A
git -C /tmp/repo -c user.name=test -c user.email=test@example.invalid commit -qm "tree under test"
# releases, as git ls-remote sees them: v0.10.0 is the highest by number, not
# v0.9.0, and v0.11.0rc1 is a prerelease
for t in v0.9.0 v0.10.0 v0.11.0rc1; do git -C /tmp/repo tag "$t"; done
# the Docker releases, each its own commit, tagged as on GitHub
for dir in /releases/v*; do
  t="$(basename "$dir")"
  rm -rf "/tmp/old-$t" /tmp/old-index
  cp -R "$dir" "/tmp/old-$t"
  (cd "/tmp/old-$t" && GIT_DIR=/tmp/repo/.git GIT_WORK_TREE="/tmp/old-$t" GIT_INDEX_FILE=/tmp/old-index git add -A)
  tree="$(GIT_DIR=/tmp/repo/.git GIT_INDEX_FILE=/tmp/old-index git write-tree)"
  commit="$(git -C /tmp/repo -c user.name=test -c user.email=test@example.invalid commit-tree "$tree" -m "$t")"
  git -C /tmp/repo tag "$t" "$commit"
done
rm -f /tmp/old-index
# the server tarballs the fake GitHub hands out: two releases, and main's
# build on the edge prerelease
make_release v0.9.0 0.9.0
make_release v0.10.0 0.10.0
make_release edge 0.11.0-dev.1 edge
# a build of the kind the hardware bench installs by hand: a dev version, its
# files inside one folder, and a unit of its own
dev=/tmp/dev/jetlink-server-0.12.0-dev
mkdir -p "$dev/bin" "$dev/share/jetlink/systemd" "$dev/share/jetlink/udev"
ln -s "$SRC/tests/installer/fake.sh" "$dev/bin/jetlink-server"
echo 0.12.0-dev >"$dev/VERSION"
{ cat "$SRC/scripts/jetlink-server.service"; echo "# the 0.12.0-dev build's unit"; } >"$dev/share/jetlink/systemd/jetlink-server.service"
cp "$SRC"/scripts/*.rules "$dev/share/jetlink/udev/"
tar -czf /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz -C /tmp/dev jetlink-server-0.12.0-dev
# shellcheck disable=SC1091
. /etc/os-release
echo "installer scenarios on $PRETTY_NAME"
# NVIDIA's TensorRT wheel for PCs, as the fake index hands it out: stand-ins
# where the real libraries are, and its sha256 in place of the pinned one
PC_TRT="$(sed -n 's/^PC_TRT=//p' "$SRC/install.sh")"
PC_TRT_WHEEL="$(sed -n 's/^PC_TRT_WHEEL=//p' "$SRC/install.sh")"
PC_TRT_DIR=/opt/jetlink/tensorrt/$PC_TRT
rm -rf /tmp/pypi /tmp/wheel
mkdir -p /tmp/pypi /tmp/wheel/tensorrt_libs "/tmp/wheel/tensorrt_cu13_libs-$PC_TRT.dist-info/licenses"
for f in __init__.py libnvinfer.so.11 libnvonnxparser.so.11 libnvinfer_plugin.so.11 libnvinfer_builder_resource_sm89.so.11.3.0 \
    libnvinfer_builder_resource_ptx.so.11.3.0 libnvinfer_builder_resource_win_sm89.so.11.3.0; do
  echo "stand-in $f" >"/tmp/wheel/tensorrt_libs/$f"
done
echo "NVIDIA's license" >"/tmp/wheel/tensorrt_cu13_libs-$PC_TRT.dist-info/licenses/LICENSE.txt"
(cd /tmp/wheel && zip -q -r "/tmp/pypi/${PC_TRT_WHEEL##*/}" .)
JETLINK_TEST_TRT_SHA256="$(sha256sum "/tmp/pypi/${PC_TRT_WHEEL##*/}" | cut -d' ' -f1)"
export JETLINK_TEST_TRT_SHA256

scenario "JetPack 7.2 Jetson, always-on power, fresh install"
reset_box; jetson 39 2.1
# questions: power (1 = always on), let the comma shut it down, turn off the
# desktop, the web page's port (Enter), its password (too short, then typed
# differently the second time, then right), go ahead
run_installer curl '1\ny\ny\n\nshort\nhunter2-long\nhunter2-other\nhunter2-long\nhunter2-long\ny\n'
expect_rc 0
expect_out "Orin Nano"
expect_out "JetPack 7 (Jetson Linux 39.2.1)"
expect_out "Does the Jetson's power stay on when the car is off?"
expect_out "Turn off the desktop?"
expect_out "Which port for the web page?"
expect_out "Which password should the web page ask for?"
expect_out "Please use 8 to 128 characters."
expect_out "The two did not match; please type it again."
expect_out "Serve the web page on port 5600 (with the password you chose)"
expect_out "Sign in with the password you chose; sudo jetlink password sets a new one."
# the server wrote it, from stdin: on no command line, in no log, never shown
expect_in /etc/jetlink/web-auth.json '"fake_password": "hunter2-long"'
check "the password file is not root's alone" test "$(stat -c '%a %U' /etc/jetlink/web-auth.json)" = "600 root"
expect_ran "jetlink-server web-password --file /etc/jetlink/web-auth.json"
expect_not_ran "hunter2"
expect_not_in /var/log/jetlink-install.log "hunter2"
expect_no_out "hunter2"
expect_in "$UNITS/jetlink-server.service" "--web-auth /etc/jetlink/web-auth.json"
# the desktop goes at the next start, not under the installer
expect_out "Turn off the desktop (from the next restart)"
expect_ran "systemctl set-default multi-user.target"
expect_not_ran "systemctl stop gdm"
expect_out "Restart this computer once to turn the desktop off: sudo reboot"
expect_in /etc/jetlink/install.conf "JETLINK_DESKTOP_OFF=1"
expect_out "Install NVIDIA TensorRT from JetPack's package source"
expect_no_out "fastest power mode ("
expect_out "Jetlink is installed and running"
expect_out "Web page: http://"
expect_ran "$(apt_install "libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10")"
# the plugins came with it
expect_not_ran "libnvinfer-plugin10="
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y clean"
expect_out "TensorRT 10.16.2.10"
refute "Docker or its toolkit was touched" grep -qE '^(docker|nvidia-ctk) |install .*(docker|nvidia-container)' "$FAKE_LOG"
expect_ran "https://github.com/zoompilot/jetlink/releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz.sha256"
expect_ran "https://github.com/zoompilot/jetlink/releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_before "jetlink-server backends --backend trt" "systemctl restart jetlink-server"
expect_out "The server can use the GPU: TensorRT 10.16.2.10 on Orin-sm87"
expect_ran "nvpmodel -m 2"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/mnt/data/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=5600"
expect_in /etc/jetlink/server.env "JETLINK_JETSON=1"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-aarch64"
expect_in /etc/jetlink/server.env "JETLINK_SERVER_VERSION=0.10.0"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF="--poweroff"'
# JetPack's TensorRT, on the loader path
expect_in /etc/jetlink/server.env 'JETLINK_TENSORRT=""'
expect_not_ran "pypi.nvidia.com"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=git"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
check "the unit is not the server's own" cmp -s "$UNITS/jetlink-server.service" /opt/jetlink/0.10.0/share/jetlink/systemd/jetlink-server.service
expect_in "$UNITS/jetlink-server.service.d/10-cache.conf" "RequiresMountsFor=/mnt/data/jetlink"
# the server sets the clocks itself, told how in server.env, inside the
# limits of the mode it names; no jetson_clocks, and no wait for nvpmodel
expect_no_file "$UNITS/jetlink-clocks.service"
expect_in /etc/jetlink/server.env 'JETLINK_CLOCK_FLAGS="--clocks on --power-mode MAXN_SUPER"'
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
refute "the server waits for nvpmodel" grep -rqs nvpmodel "$UNITS/jetlink-server.service" "$UNITS/jetlink-server.service.d"
expect_not_ran "jetlink-clocks.service"
# it waits for the GPU's driver instead: as systemd runs the wait, it ends at
# once with the driver's control node there
GPU_DROPIN="$UNITS/jetlink-server.service.d/20-jetson-gpu.conf"
expect_in "$GPU_DROPIN" "[ -e /dev/nvhost-ctrl-gpu ]"
gpu_wait="$(sed -n "s/^ExecStartPre=\/bin\/sh -c '\(.*\)'$/\1/p" "$GPU_DROPIN" | sed 's/\$\$/$/g')"
check "no wait for the GPU in $GPU_DROPIN" test -n "$gpu_wait"
touch /dev/nvhost-ctrl-gpu
check "the wait for the GPU did not end with its node there" timeout 5 sh -c "$gpu_wait"
rm -f /dev/nvhost-ctrl-gpu
# boot: the firmware's menu waits 1 s (JetPack's 5 s kept for uninstall), and
# quiet on the kernel's command line, its one change to extlinux.conf
expect_ran "efibootmgr -t 1"
expect_in "$FAKE_STATE/uefi-timeout" 1
expect_in /etc/jetlink/install.conf "JETLINK_UEFI_TIMEOUT_PREV=5"
expect_quiet_added
check "the backup is not the file as it was" cmp -s "$EXTLINUX_ORIG" "$EXTLINUX.jetlink-bak"
expect_no_file "$EXTLINUX.jetlink-new"
expect_out "Start up faster, and keep the system log small"
expect_out "Firmware boot menu waits 1 s"
expect_out "Kernel messages kept off the console"
# up once it says it serves: the line the Swift server says, whatever the comma does
expect_in /var/log/jetlink-install.log "jetlink-server is serving"
check "Serve.swift no longer says the line install.sh waits for" \
  grep -qF 'servingLine = "jetlink-server is serving"' "$SRC/JetlinkKit/Sources/jetlink-server/Serve.swift"
expect_ran "systemctl show -p NRestarts --value jetlink-server"
expect_file /usr/local/bin/jetlink
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
expect_no_file /usr/local/lib/jetlink
expect_no_file "$UNITS/jetlink-poweroff.path"
expect_file /etc/systemd/journald.conf.d/60-jetlink.conf
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_file "$FAKE_STATE/masked-systemd-networkd-wait-online.service"
expect_ran "systemctl enable jetlink-server"
# the helper
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "Jetlink is running"
expect_in /tmp/status.txt "server         0.10.0 (TensorRT 10.16.2.10)"
expect_in /tmp/status.txt "comma          not connected"
expect_in /tmp/status.txt "web page       http://"
expect_in /tmp/status.txt ".local:5600 (sign in with the web page's password)"
expect_in /tmp/status.txt "always on: sleeps while parked; the comma can turn it off"
jetlink models list >/dev/null 2>&1
expect_ran "jetlink-server models list --cache /mnt/data/jetlink"
jetlink models --help >/dev/null 2>&1
expect_ran "jetlink-server models --help"
# the installed unit's command line, with server.env's settings; the server
# sets the clocks itself
: >"$FAKE_LOG"
jetlink run --log-level debug >/dev/null 2>&1
expect_ran "jetlink-server --usb --backend trt --cache /mnt/data/jetlink --sleep-after 120 --status-port 5600 --web-auth /etc/jetlink/web-auth.json --poweroff --clocks on --power-mode MAXN_SUPER --log-level debug"
expect_not_ran "jetson_clocks"
systemctl start jetlink-server

scenario "update keeps the answers and asks nothing"
# a native server that sleeps, with the awake lock it makes at start
: >"$LOCK" && chmod 644 "$LOCK"
# and the clocks drop-in 0.7.4 and older wrote, which held the server back,
# and the jetson_clocks unit 0.9.0 and older installed
printf '[Unit]\nAfter=nvpmodel.service\n[Service]\nExecStartPre=-/usr/bin/jetson_clocks\n' \
  >"$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
printf '[Service]\nExecStart=/usr/bin/jetson_clocks\n' >"$UNITS/jetlink-clocks.service"
# and a desktop turned back on by hand, which an update leaves on
echo graphical.target >"$FAKE_STATE/default-target"
cli update
expect_rc 0
expect_not_ran "systemctl set-default"
expect_in "$FAKE_STATE/default-target" graphical.target
expect_no_out "Restart this computer once"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_no_file "$UNITS/jetlink-clocks.service"
expect_ran "systemctl disable jetlink-clocks.service"
expect_no_out "jetlink-clocks.service is not the installer's"
expect_in /etc/jetlink/server.env 'JETLINK_CLOCK_FLAGS="--clocks on --power-mode MAXN_SUPER"'
# boot is as the install left it, and what it was before stays known
expect_not_ran "efibootmgr -t"
expect_quiet_added
expect_in /etc/jetlink/install.conf "JETLINK_UEFI_TIMEOUT_PREV=5"
check "the backup is not the file as it was" cmp -s "$EXTLINUX_ORIG" "$EXTLINUX.jetlink-bak"
expect_out "Getting the newest Jetlink (v0.10.0)"
expect_no_out "A few questions"
expect_no_out "Go ahead?"
expect_out "Jetlink is installed and running"
# it serves through the downloads, held awake so it cannot suspend the Jetson
# under them, and stops only for the switch
expect_out "Holding this computer awake for the update"
expect_ran "jetlink-server-0.10.0-linux-aarch64.tar.gz [held awake]"
expect_ran "apt-cache policy libnvinfer10 [held awake]"
expect_before "releases/download/v0.10.0" "systemctl stop jetlink-server"
expect_before "jetlink-server backends" "systemctl stop jetlink-server"
expect_out "Stopping the running Jetlink server for the update"
expect_ran "jetlink-server started: native, sleep 120"
expect_file /etc/jetlink/server.env.prev
expect_no_out "The previous Jetlink server is running again."
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_not_ran "nvpmodel -m"
# TensorRT is there; JetPack 7.2 only looks for a newer one
expect_not_ran "$(apt_install "libnvinfer10")"
expect_ran "apt-cache policy libnvinfer10"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_no_file /opt/jetlink/previous

scenario "jetlink caffeinate holds the server awake: a command, -t, until stopped"
lock_free() { flock --exclusive --nonblock "$LOCK" true; }
# a command: held while it runs, and its exit status comes back
jetlink caffeinate sh -c "flock -xn $LOCK true && echo free || echo held; exit 7" >"$OUT" 2>&1; RC=$?
expect_rc 7
expect_out "held"
check "still held after the command" lock_free
jetlink caffeinate jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "held awake by jetlink caffeinate"
jetlink status >/tmp/status.txt 2>&1
expect_not_in /tmp/status.txt "held awake"
# -t SECONDS
jetlink caffeinate -t 2 >"$OUT" 2>&1 &
pid=$!
sleep 1
refute "not held during -t" lock_free
wait "$pid"; RC=$?
expect_rc 0
expect_out "Holding the Jetson awake for 2 s"
check "still held after -t" lock_free
# no arguments: until it is stopped
jetlink caffeinate >"$OUT" 2>&1 &
pid=$!
sleep 1
refute "not held" lock_free
kill -TERM "$pid"; wait "$pid"; RC=$?
expect_rc 0
expect_out "Holding the Jetson awake; Ctrl-C to let it sleep."
check "still held after it stopped" lock_free
# a server without the lock (from before it, or in Docker): nothing to hold
rm -f "$LOCK"
cli caffeinate
expect_rc 1
expect_out "The server is not running natively; nothing to hold."

scenario "a second run offers to keep the settings"
run_installer curl 'y\n'
expect_rc 0
expect_out "Jetlink is already installed. Update it and keep your answers?"
expect_no_out "Does the Jetson's power stay on when the car is off?"

scenario "a failed update puts the previous server back"
echo '# the previous settings' >>/etc/jetlink/server.env
echo "# 0.10.0's own unit" >>/opt/jetlink/0.10.0/share/jetlink/systemd/jetlink-server.service
# the new server never gets as far as waiting for the comma
export FAKE_SERVER_BROKEN=1
run_installer curl '' --update --ref v0.9.0
expect_rc 1
# systemd's restart of it, seen as it is logged
expect_in /var/log/jetlink-install.log "the server crashed, and systemd started it again"
expect_out "The previous Jetlink server is running again."
expect_in /etc/jetlink/server.env "# the previous settings"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_in "$UNITS/jetlink-server.service" "# 0.10.0's own unit"
check "the previous server was not started again" test "$(grep -c "systemctl restart jetlink-server" "$FAKE_LOG")" -ge 2
refute "the unit was left stopped" test -f "$FAKE_STATE/stopped-jetlink-server"
unset FAKE_SERVER_BROKEN

scenario "a new server that crashes after it said it serves is taken back out"
reset_box; jetson 39 2.1
run_installer curl '' --yes
: >"$FAKE_LOG"
FAKE_SERVER_CRASHLOOP=1 run_installer curl '' --update --ref v0.9.0
expect_rc 1
expect_in /var/log/jetlink-install.log "jetlink-server is serving"
expect_in /var/log/jetlink-install.log "the server did not stay up: systemd started it again 2 times"
expect_out "The previous Jetlink server is running again."
expect_no_out "Jetlink is installed and running"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_no_file /opt/jetlink/previous
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"

scenario "a build from before the serving line is up with a comma nothing on it serves"
reset_box; jetson 39 2.1
FAKE_SERVER_OLD=1 run_installer curl '' --yes
expect_rc 0
expect_in /var/log/jetlink-install.log "nothing on the comma is serving it yet"
expect_out "Jetlink is installed and running"

scenario "Ctrl-C during an update starts the server it stopped again"
reset_box; jetson 39 2.1
run_installer curl '' --yes
: >"$FAKE_LOG"
# a server that sleeps and has no awake lock stops for the update
interrupted INT bash /opt/jetlink/src/install.sh --update --ref v0.9.0
expect_rc 130
expect_out "Stopped by SIGINT."
expect_out "The previous Jetlink server is running again."
expect_before "systemctl stop jetlink-server" "releases/download/v0.9.0/jetlink-server-0.9.0-linux-aarch64.tar.gz"
refute "the unit was left stopped" test -f "$FAKE_STATE/stopped-jetlink-server"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_file /var/log/jetlink-install.log

scenario "a fresh install has no server to stop"
reset_box; jetson 39 2.1
run_installer curl '' --yes
expect_rc 0
expect_not_ran "systemctl stop jetlink-server"
expect_no_file /etc/jetlink/server.env.prev
# --yes takes the recommended wiring: always on, and the comma may shut it down
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=5600"

# a made password as the server makes it, in groups of four
made_password() { grep -qxE '[2-9a-z]{4}-[2-9a-z]{4}-[2-9a-z]{4}' <<<"$1"; }
# the one the finish screen shows
shown_password() { sed -n 's/^ *Password: \([^ ]*\) (change it with: sudo jetlink password)$/\1/p' "$OUT"; }

scenario "without a terminal the web page gets a password made for it, shown once at the end"
reset_box; jetson 39 2.1
run_installer curl '' --yes
expect_rc 0
expect_out "Serve the web page on port 5600 (with a new password, shown at the end)"
expect_ran "jetlink-server web-password --file /etc/jetlink/web-auth.json --generate"
# once the new server is up, so a run that fails leaves no password nobody saw
expect_before "systemctl restart jetlink-server" "jetlink-server web-password"
expect_out "Web page password set"
check "the password file is not root's alone" test "$(stat -c '%a %U' /etc/jetlink/web-auth.json)" = "600 root"
pw="$(shown_password)"
check "no made password at the end: '$pw'" made_password "$pw"
expect_in /etc/jetlink/web-auth.json "\"fake_password\": \"$pw\""
expect_not_in /var/log/jetlink-install.log "$pw"
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "(sign in with the web page's password)"

scenario "an update keeps the web page's password; one from before the sign-in gets one made"
cp /etc/jetlink/web-auth.json /tmp/web-auth.before
cli update
expect_rc 0
expect_not_ran "web-password"
check "the update changed the password file" cmp -s /tmp/web-auth.before /etc/jetlink/web-auth.json
expect_out "Serve the web page on port 5600 (with the password it has)"
expect_out "Sign in with its password, as before; sudo jetlink password sets a new one."
expect_no_out "Password:"
# an install from before the sign-in has no password file: its page shows the
# status only until the update makes one
rm /etc/jetlink/web-auth.json
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "(status only until it has a password: jetlink password)"
: >"$FAKE_LOG"
run_installer curl '' --update
expect_rc 0
expect_no_out "Which password"
expect_ran "jetlink-server web-password --file /etc/jetlink/web-auth.json --generate"
pw="$(shown_password)"
check "no made password at the end: '$pw'" made_password "$pw"
expect_in /etc/jetlink/web-auth.json "\"fake_password\": \"$pw\""

scenario "jetlink password sets a new one, typed twice or made, and says when the web page is off"
cp /etc/jetlink/web-auth.json /tmp/web-auth.before
printf 'short\nshort\nnew-secret-1\nnew-secret-2\nnew-secret-1\nnew-secret-1\n' | jetlink password >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "The password needs at least 8 characters."
expect_out "The two did not match; please type it again."
expect_out "The web page has a new password."
expect_out "Devices signed in to it are signed out, and sign in again with the new one."
expect_in /etc/jetlink/web-auth.json '"fake_password": "new-secret-1"'
check "the password file is not root's alone" test "$(stat -c '%a %U' /etc/jetlink/web-auth.json)" = "600 root"
expect_not_ran "new-secret"
expect_no_out "new-secret"
cli password --generate
expect_rc 0
pw="$(sed -n "s/^The web page's new password: //p" "$OUT")"
check "no made password shown: '$pw'" made_password "$pw"
expect_in /etc/jetlink/web-auth.json "\"fake_password\": \"$pw\""
# nothing typed, nothing changed
cp /etc/jetlink/web-auth.json /tmp/web-auth.before
jetlink password </dev/null >"$OUT" 2>&1; RC=$?
expect_rc 1
expect_out "Nothing changed."
check "the password file changed" cmp -s /tmp/web-auth.before /etc/jetlink/web-auth.json
# with the page off, there is nothing to sign in to
sed -i 's/^JETLINK_STATUS_PORT=.*/JETLINK_STATUS_PORT=0/' /etc/jetlink/server.env
cli password --generate
expect_rc 1
expect_out "The web page is off. To turn it on, give it a port in: jetlink setup"
check "the password file changed" cmp -s /tmp/web-auth.before /etc/jetlink/web-auth.json
sed -i 's/^JETLINK_STATUS_PORT=.*/JETLINK_STATUS_PORT=5600/' /etc/jetlink/server.env

scenario "jetlink setup offers to keep the web page's password; no sets a new one"
cp /etc/jetlink/web-auth.json /tmp/web-auth.before
# questions: power, the comma may shut it down, the desktop, the port (Enter
# for each), keep the password (y), go ahead
answers '\n\n\n\ny\ny\n'
JETLINK_INPUT=/tmp/answers jetlink setup >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "Keep the web page's password?"
expect_no_out "Which password should the web page ask for?"
expect_not_ran "web-password"
check "the password file changed" cmp -s /tmp/web-auth.before /etc/jetlink/web-auth.json
# the same, but no, and Enter for a new one made
: >"$FAKE_LOG"
answers '\n\n\n\nn\n\ny\n'
JETLINK_INPUT=/tmp/answers jetlink setup >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "Which password should the web page ask for?"
expect_ran "jetlink-server web-password --file /etc/jetlink/web-auth.json --generate"
refute "the password file is as it was" cmp -s /tmp/web-auth.before /etc/jetlink/web-auth.json
pw="$(shown_password)"
check "no made password at the end: '$pw'" made_password "$pw"

scenario "a first install that fails sets no password"
reset_box; jetson 39 2.1
FAKE_SERVER_CRASHLOOP=1 run_installer curl '' --yes
expect_rc 1
expect_not_ran "web-password"
expect_no_file /etc/jetlink/web-auth.json
expect_no_out "Password:"

scenario "jetlink setup --set changes answers with no terminal and no network, and keeps the server and the password"
reset_box; jetson 39 2.1
run_installer curl '' --yes
cp /etc/jetlink/web-auth.json /tmp/web-auth.before
head_before="$(git -C /opt/jetlink/src rev-parse HEAD)"
# no GitHub, no repository to fetch from, and a newer TensorRT that must not come
mv /tmp/repo /tmp/repo.away
unset FAKE_LATEST
: >"$FAKE_LOG"
FAKE_TRT10=10.16.3.1-1+cuda13.2 cli setup --set power=switched </dev/null
mv /tmp/repo.away /tmp/repo
expect_rc 0
expect_no_out "A few questions"
expect_no_out "Go ahead?"
expect_out "Keep the Jetlink server that is installed"
expect_out "Jetlink is installed and running"
refute "it used the network" grep -qE '^(curl|apt-get|apt-cache) ' "$FAKE_LOG"
check "the source moved" test "$(git -C /opt/jetlink/src rev-parse HEAD)" = "$head_before"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=switched"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=0"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF=""'
expect_no_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
# the other answers as they were, and the server restarted with them
expect_in /etc/jetlink/install.conf "JETLINK_DESKTOP_OFF=1"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=5600"
expect_ran "jetlink-server started: native, sleep 0"
expect_not_ran "web-password"
check "the password file changed" cmp -s /tmp/web-auth.before /etc/jetlink/web-auth.json
expect_out "Sign in with its password, as before"
# every answer that applies, as the web page sends them: the ones the install
# already has change nothing
: >"$FAKE_LOG"
cli setup --set power=switched --set comma_poweroff=no --set desktop=off </dev/null
expect_rc 0
expect_not_ran "systemctl set-default"
expect_no_out "Restart this computer once"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=switched"
expect_in /etc/jetlink/install.conf "JETLINK_DESKTOP_OFF=1"
# back to always on, with the sleep the Jetson can do; the power comes first
# whatever the order
cli setup --set comma_poweroff=yes --set power=always </dev/null
expect_rc 0
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF="--poweroff"'
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
# the desktop back, from the next restart
: >"$FAKE_LOG"
cli setup --set desktop=on </dev/null
expect_rc 0
expect_ran "systemctl set-default graphical.target"
expect_out "Restart this computer once to bring the desktop back: sudo reboot"
expect_in /etc/jetlink/install.conf "JETLINK_DESKTOP_OFF=0"

scenario "jetlink setup --set picks the power mode and the clocks, live, and an update keeps them"
reset_box; jetson 39 2.1
run_installer curl '' --yes
expect_in /etc/jetlink/install.conf "JETLINK_POWER_MODE=fastest"
expect_in /etc/jetlink/install.conf 'JETLINK_POWER_MODES=7W\ 15W\ 25W\ MAXN_SUPER'
expect_in /etc/jetlink/install.conf 'JETLINK_DEFAULTS=power=always\ comma_poweroff=yes\ desktop=off\ swap=on\ power_mode=fastest\ clocks=on'
expect_in /etc/jetlink/install.conf "JETLINK_CLOCKS=on"
: >"$FAKE_LOG"
cli setup --set power_mode=25W --set clocks=off </dev/null
expect_rc 0
expect_out "Switch to power mode 25W"
expect_out "Leave the clocks to NVIDIA's scaling"
expect_ran "nvpmodel -m 1"
expect_no_out "Restart this computer once"
expect_in /etc/jetlink/install.conf "JETLINK_POWER_MODE=25W"
expect_in /etc/jetlink/install.conf "JETLINK_CLOCKS=off"
# the server restarts with them: the clocks follow at once
expect_in /etc/jetlink/server.env 'JETLINK_CLOCK_FLAGS="--clocks off --power-mode 25W"'
expect_ran "systemctl restart jetlink-server"
# an update keeps both
: >"$FAKE_LOG"
run_installer curl '' --update
expect_rc 0
expect_not_ran "nvpmodel -m"
expect_in /etc/jetlink/install.conf "JETLINK_POWER_MODE=25W"
expect_in /etc/jetlink/server.env 'JETLINK_CLOCK_FLAGS="--clocks off --power-mode 25W"'
# back to the defaults
: >"$FAKE_LOG"
cli setup --set power_mode=fastest --set clocks=on </dev/null
expect_rc 0
expect_out "Hold the clocks at the power mode's top speed while serving"
expect_ran "nvpmodel -m 2"
expect_in /etc/jetlink/install.conf "JETLINK_POWER_MODE=fastest"
expect_in /etc/jetlink/server.env 'JETLINK_CLOCK_FLAGS="--clocks on --power-mode MAXN_SUPER"'
# the fastest by its name is the fastest
cli setup --set power_mode=MAXN_SUPER </dev/null
expect_rc 0
expect_in /etc/jetlink/install.conf "JETLINK_POWER_MODE=fastest"
# a mode this Jetson does not have, and a clocks answer that is not one
cli setup --set power_mode=60W </dev/null
expect_rc 1
expect_out "--set power_mode takes fastest or one of this Jetson's modes (7W, 15W, 25W, MAXN_SUPER), not '60W'."
cli setup --set clocks=auto </dev/null
expect_rc 1
expect_out "--set clocks takes on or off, not 'auto'."

scenario "--set with an unknown key, a bad value or the other computer's key fails and changes nothing"
cp /etc/jetlink/install.conf /tmp/install.conf.before
: >"$FAKE_LOG"
cli setup --set colour=blue </dev/null
expect_rc 1
expect_out "--set does not know 'colour'."
expect_out "On a Jetson: power=always|switched, comma_poweroff=yes|no, desktop=on|off, swap=on|off, power_mode=fastest|<a mode it has>, clocks=on|off. On a PC: autostart=yes|no."
cli setup --set power=sometimes </dev/null
expect_rc 1
expect_out "--set power takes always or switched, not 'sometimes'."
cli setup --set comma_poweroff=maybe </dev/null
expect_rc 1
expect_out "--set comma_poweroff takes yes or no, not 'maybe'."
cli setup --set power </dev/null
expect_rc 1
expect_out "--set takes KEY=VALUE, not 'power'."
cli setup --set autostart=no </dev/null
expect_rc 1
expect_out "--set autostart is for a PC: a Jetson always starts Jetlink."
run_installer curl '' --update --set power=always
expect_rc 1
expect_out "--set does not go with --update, --uninstall, --ref or --binary."
check "the answers changed" cmp -s /tmp/install.conf.before /etc/jetlink/install.conf
expect_not_ran "systemctl restart"
expect_not_ran "systemctl stop"
# on a PC: nothing to change before an install, a Jetson's key refused, its own taken
reset_box; pc 580.95.05
run_installer curl '' --set autostart=no
expect_rc 1
expect_out "Jetlink is not installed here, so --set has nothing to change."
expect_no_file /etc/jetlink
run_installer curl '' --yes
expect_rc 0
cli setup --set power=always </dev/null
expect_rc 1
expect_out "--set power is for a Jetson, and this computer is a PC."
: >"$FAKE_LOG"
cli setup --set autostart=no </dev/null
expect_rc 0
expect_out "Start Jetlink now (not at every boot)"
expect_in /etc/jetlink/install.conf "JETLINK_AUTOSTART=0"
expect_ran "systemctl disable jetlink-server"
expect_not_ran "pypi.nvidia.com"
expect_not_ran "releases/download"

scenario "a server that cannot use the GPU fails with advice, before anything changes"
reset_box; jetson 39 2.1
FAKE_GPU_BROKEN=1 run_installer curl '' --yes
expect_rc 1
expect_out "TensorRT cannot run on this computer."
expect_out "The Jetlink server cannot use the GPU: no CUDA driver: libcuda.so.1: cannot open shared object file"
expect_no_out "ort: not usable"
expect_no_file "$UNITS/jetlink-server.service"
expect_no_file /opt/jetlink/current

scenario "curl | bash installs the newest release, as GitHub's API names it"
reset_box; jetson 39 2.1
FAKE_LATEST=v0.9.0 run_installer curl '' --yes
expect_rc 0
# the API's answer, not the highest tag
expect_out "Getting Jetlink (v0.9.0)"
expect_ran "releases/download/v0.9.0/jetlink-server-0.9.0-linux-aarch64.tar.gz"
expect_file /opt/jetlink/src/.git
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=git"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.9.0"
expect_in /etc/jetlink/server.env "JETLINK_SERVER_VERSION=0.9.0"
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "v0.9.0 (follows releases)"

scenario "jetlink update moves to the next release and keeps the one before"
cli update
expect_rc 0
expect_out "Getting the newest Jetlink (v0.10.0)"
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_link /opt/jetlink/previous /opt/jetlink/0.9.0
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"

scenario "an install that saved main before 0.5.0 follows releases; no API, so the tags"
# what the installer wrote before 0.5.0: its default, main, and no version
sed -i -e 's/^JETLINK_REF=.*/JETLINK_REF=main/' -e '/^JETLINK_VERSION=/d' /etc/jetlink/install.conf
unset FAKE_LATEST
run_installer curl '' --update
expect_rc 0
expect_out "Jetlink now follows releases; for development builds, use --ref main."
expect_ran "releases/latest"
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_not_ran "0.11.0rc1"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"

scenario "--ref main pins development builds from the edge prerelease, and an update keeps them"
export FAKE_LATEST=v0.10.0
run_installer curl '' --update --ref main
expect_rc 0
expect_ran "releases/download/edge/jetlink-server-edge-linux-aarch64.tar.gz"
expect_in /etc/jetlink/install.conf "JETLINK_REF=main"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=main"
expect_link /opt/jetlink/current /opt/jetlink/0.11.0-dev.1
expect_link /opt/jetlink/previous /opt/jetlink/0.10.0
# older than the one before: gone
expect_no_file /opt/jetlink/0.9.0
: >"$FAKE_LOG"
run_installer curl '' --update
expect_rc 0
expect_no_out "now follows releases"
expect_ran "releases/download/edge/jetlink-server-edge-linux-aarch64.tar.gz"
expect_in /etc/jetlink/install.conf "JETLINK_REF=main"

scenario "--ref vX.Y.Z pins a release; --ref latest follows them again"
run_installer curl '' --update --ref v0.9.0
expect_rc 0
# a server that sleeps and has no awake lock (from before it) stops first
expect_before "systemctl stop jetlink-server" "releases/download/v0.9.0"
expect_not_ran "[held awake]"
expect_ran "releases/download/v0.9.0/jetlink-server-0.9.0-linux-aarch64.tar.gz"
expect_in /etc/jetlink/install.conf "JETLINK_REF=v0.9.0"
: >"$FAKE_LOG"
run_installer curl '' --update
expect_rc 0
expect_ran "releases/download/v0.9.0/"
: >"$FAKE_LOG"
run_installer curl '' --update --ref latest
expect_rc 0
expect_ran "releases/download/v0.10.0/"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_link /opt/jetlink/previous /opt/jetlink/0.9.0

scenario "no answer from GitHub: an update stays put, a first install stops"
unset FAKE_LATEST
export JETLINK_REPO_URL=file:///nonexistent
run_installer curl '' --update
expect_rc 0
expect_out "Could not look up the newest release; staying on v0.10.0."
expect_ran "releases/download/v0.10.0/"
reset_box; jetson 39 2.1
unset FAKE_LATEST
export JETLINK_REPO_URL=file:///nonexistent
run_installer curl '' --yes
expect_rc 1
expect_out "Could not find the newest Jetlink release."
expect_no_file /etc/jetlink
expect_not_ran "apt-get"

scenario "a download cut off part way is tried again"
reset_box; jetson 39 2.1
FAKE_DOWNLOAD_FAILS=2 run_installer curl '' --yes
expect_rc 0
expect_out "Downloading the Jetlink server (v0.10.0)"
expect_in /var/log/jetlink-install.log "the download was interrupted; trying again"
expect_in /etc/jetlink/server.env "JETLINK_SERVER_VERSION=0.10.0"

scenario "a damaged download is refused"
reset_box; jetson 39 2.1
FAKE_BAD_SUM=1 run_installer curl '' --yes
expect_rc 1
expect_out "its checksum does not match"
expect_no_file /opt/jetlink/current
expect_no_file "$UNITS/jetlink-server.service"

scenario "a branch with no ready-made server says how to build one"
reset_box; jetson 39 2.1
run_installer curl '' --yes --ref my-branch
expect_rc 1
expect_out "There is no ready-made Jetlink server for my-branch."
expect_out "scripts/build-linux.sh linux-aarch64"
expect_no_file /etc/jetlink
expect_not_ran "apt-get"

scenario "JetPack 7.2 follows the newest TensorRT, and refuses one too old"
reset_box; jetson 39 2.1
with_trt 10.16.2.10-1+cuda13.2 10
FAKE_TRT10=10.16.3.1-1+cuda13.2 run_installer curl '' --yes
expect_rc 0
expect_out "Updating TensorRT to 10.16.3.1"
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y install --only-upgrade --no-install-recommends libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10"
expect_not_ran "$(apt_install "libnvinfer10")"
expect_out "TensorRT 10.16.3.1"
# the TensorRT it had came without plugins: they come now, the same build
expect_ran "$(apt_install "libnvinfer-plugin10=10.16.3.1-1+cuda13.2")"
check "wanted the package list fetched once" test "$(grep -c "apt-get .* update" "$FAKE_LOG")" = 1
reset_box; jetson 39 2.1
with_trt 10.16.1.1-1+cuda13.2 10
FAKE_TRT10=10.16.1.1-1+cuda13.2 run_installer curl '' --yes
expect_rc 1
expect_out "Jetlink needs 10.16.2.10 or newer"
expect_no_file "$UNITS/jetlink-server.service"

scenario "JetPack 6.2 Jetson, switched power"
reset_box; jetson 36 4.3
# questions: power (2 = switched), keep the desktop, the web page's port
# (Enter), its password (Enter makes one), go ahead (Enter)
run_installer curl '2\nn\n\n\n\n'
expect_rc 0
expect_ran "jetlink-server web-password --file /etc/jetlink/web-auth.json --generate"
expect_out "Serve the web page on port 5600 (with a new password, shown at the end)"
expect_out "JetPack 6 (Jetson Linux 36.4.3)"
expect_not_ran "systemctl set-default"
expect_no_out "Turn off the desktop (from"
expect_in /etc/jetlink/install.conf "JETLINK_DESKTOP_OFF=0"
expect_ran "$(apt_install "libnvinfer10 libnvonnxparsers10")"
expect_out "TensorRT 10.3.0.30"
# JetPack 6 stays on its TensorRT 10.3
expect_not_ran "apt-cache policy"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-aarch64"
expect_ran "nvpmodel -m 2"
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_no_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
expect_no_file "$UNITS/jetlink-clocks.service"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_file "$UNITS/jetlink-server.service.d/20-jetson-gpu.conf"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF=""'
# the same boot changes as on JetPack 7.2
expect_ran "efibootmgr -t 1"
expect_in /etc/jetlink/install.conf "JETLINK_UEFI_TIMEOUT_PREV=5"
expect_quiet_added
expect_file "$EXTLINUX.jetlink-bak"

scenario "jetlink setup changes the answers: power, the desktop off, and the web page off"
# questions: power (1 = always on), the comma may shut it down, turn off the
# desktop, port 0, go ahead
answers '1\ny\ny\n0\ny\n'
JETLINK_INPUT=/tmp/answers jetlink setup >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "Does the Jetson's power stay on when the car is off?"
expect_ran "systemctl set-default multi-user.target"
expect_in /etc/jetlink/install.conf "JETLINK_DESKTOP_OFF=1"
expect_no_out "Serve the web page"
expect_no_out "Web page:"
expect_no_out "Keep the web page's password?"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=0"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF="--poweroff"'
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
jetlink status >/tmp/status.txt 2>&1
expect_not_in /tmp/status.txt "web page"
# the file an earlier installer wrote for "no" goes; one made by hand stays
echo "Written by the Jetlink installer: the comma may not power this computer off." >/mnt/data/jetlink/poweroff-dry-run
run_installer curl '' --update
expect_rc 0
expect_no_file /mnt/data/jetlink/poweroff-dry-run
touch /mnt/data/jetlink/poweroff-dry-run
run_installer curl '' --update
expect_rc 0
expect_file /mnt/data/jetlink/poweroff-dry-run

scenario "a power mode nvpmodel will not switch live needs no restart: Jetlink holds its limits"
reset_box; jetson 39 2.1
FAKE_PM_REBOOT=1 run_installer curl '' --yes
expect_rc 0
expect_out "Power mode set to MAXN_SUPER (Jetlink holds its clock limits)"
expect_in /etc/jetlink/server.env 'JETLINK_CLOCK_FLAGS="--clocks on --power-mode MAXN_SUPER"'
# the restart is the desktop's alone
expect_out "Restart this computer once to turn the desktop off: sudo reboot"

scenario "uninstall on a Jetson puts its boot back as it was, and takes the web page's password"
reset_box; jetson 39 2.1
run_installer curl '' --yes
expect_rc 0
expect_in "$FAKE_STATE/default-target" multi-user.target
expect_file /etc/jetlink/web-auth.json
# questions: remove?, delete the models?
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_out "The desktop comes back at the next start"
expect_in "$FAKE_STATE/default-target" graphical.target
expect_out "The firmware's boot menu wait is as it was"
expect_out "Kernel messages are on the console at boot again"
expect_in "$FAKE_STATE/uefi-timeout" 5
expect_extlinux_as_was
expect_no_file "$EXTLINUX.jetlink-bak"
expect_no_file "$UNITS/jetlink-clocks.service"
expect_no_file /etc/jetlink/web-auth.json

scenario "a Jetson that starts no desktop is not asked, and uninstall leaves it that way"
reset_box; jetson 39 2.1
echo multi-user.target >"$FAKE_STATE/default-target"
# questions: power (Enter), the comma may shut it down (Enter), the port
# (Enter), the password (Enter), go ahead
run_installer curl '\n\n\n\ny\n'
expect_rc 0
expect_no_out "Turn off the desktop?"
expect_not_ran "systemctl set-default"
expect_in /etc/jetlink/install.conf "JETLINK_DESKTOP_OFF=0"
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_not_ran "systemctl set-default"
expect_in "$FAKE_STATE/default-target" multi-user.target

scenario "boot settings of the user's own stay: a shorter firmware wait, their quiet, a wait changed since"
reset_box; jetson 39 2.1
echo 0 >"$FAKE_STATE/uefi-timeout"
extlinux "$(sed 's/^\( *APPEND .*\)$/\1 quiet splash/' "$EXTLINUX")"
run_installer curl '' --yes
expect_rc 0
expect_not_ran "efibootmgr -t"
expect_in /etc/jetlink/install.conf "JETLINK_UEFI_TIMEOUT_PREV=''"
expect_extlinux_as_was
expect_no_file "$EXTLINUX.jetlink-bak"
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_no_out "boot menu wait is as it was"
expect_no_out "Kernel messages are on the console"
expect_in "$FAKE_STATE/uefi-timeout" 0
expect_extlinux_as_was
# a wait the user set after the install is theirs too
reset_box; jetson 39 2.1
run_installer curl '' --yes
echo 3 >"$FAKE_STATE/uefi-timeout"
: >"$FAKE_LOG"
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_not_ran "efibootmgr -t"
expect_in "$FAKE_STATE/uefi-timeout" 3

scenario "a firmware with no boot menu wait set gets it taken away again"
reset_box; jetson 36 4.3
echo none >"$FAKE_STATE/uefi-timeout"
run_installer curl '' --yes
expect_rc 0
expect_in "$FAKE_STATE/uefi-timeout" 1
expect_in /etc/jetlink/install.conf "JETLINK_UEFI_TIMEOUT_PREV=none"
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_ran "efibootmgr -T"
expect_in "$FAKE_STATE/uefi-timeout" none

scenario "efibootmgr comes from apt; a firmware that keeps its wait, or no UEFI, changes nothing"
reset_box; jetson 39 2.1
rm -f "$FAKE_BIN/efibootmgr"
run_installer curl '' --yes
expect_rc 0
check "efibootmgr was not installed" grep -qE '^apt-get .* install --no-install-recommends .*efibootmgr' "$FAKE_LOG"
expect_ran "efibootmgr -t 1"
expect_in "$FAKE_STATE/uefi-timeout" 1
reset_box; jetson 39 2.1
FAKE_UEFI_LOCKED=1 run_installer curl '' --yes
expect_rc 0
expect_out "The firmware kept its boot menu wait"
expect_in /etc/jetlink/install.conf "JETLINK_UEFI_TIMEOUT_PREV=''"
# the rest of boot is changed all the same
expect_quiet_added
reset_box; jetson 39 2.1
rm -rf /tmp/efivars
run_installer curl '' --yes
expect_rc 0
# neither run nor installed
expect_not_ran "efibootmgr"

scenario "quiet goes on the entry that boots, and an entry the installer cannot read stays as it is"
reset_box; jetson 39 2.1
extlinux "TIMEOUT 30
DEFAULT jetlink

LABEL backup
      LINUX /boot/Image.backup
      APPEND \${cbootargs} root=/dev/nvme0n1p1 rw

LABEL jetlink
      MENU LABEL primary kernel
      LINUX /boot/Image
      APPEND \${cbootargs} root=/dev/nvme0n1p1 rw console=tty0"
run_installer curl '' --yes
expect_rc 0
expect_in "$EXTLINUX" "rw console=tty0 quiet"
expect_one_quiet
# without DEFAULT the boot loader starts the first entry
reset_box; jetson 39 2.1
extlinux "LABEL first
      LINUX /boot/Image
      APPEND \${cbootargs} root=/dev/nvme0n1p1 rw

LABEL second
      LINUX /boot/Image.backup
      APPEND \${cbootargs} root=/dev/nvme0n1p1 rw console=tty0"
run_installer curl '' --yes
expect_rc 0
expect_in "$EXTLINUX" "root=/dev/nvme0n1p1 rw quiet"
expect_one_quiet
# two APPEND lines in the entry that boots: which one counts is the boot
# loader's business, so the file is left alone
reset_box; jetson 39 2.1
extlinux "LABEL primary
      LINUX /boot/Image
      APPEND \${cbootargs} root=/dev/nvme0n1p1 rw
      APPEND console=tty0"
run_installer curl '' --yes
expect_rc 0
expect_out "has no boot entry Jetlink can add quiet to"
expect_extlinux_as_was
expect_no_file "$EXTLINUX.jetlink-bak"

scenario "PC with a driver too old for CUDA 13"
reset_box; pc 575.64.03
run_installer curl 'y\n'
expect_rc 0
expect_out "has NVIDIA driver 575.64.03 and needs 580 or newer"
expect_ran "ubuntu-drivers install nvidia:580-open"
expect_out "Restart the computer, then run the installer again"
expect_no_file /etc/jetlink/server.env

scenario "PC ready to go"
reset_box; pc 580.95.05
export FAKE_NO_CURL=1
# questions: start at boot, the web page's port (Enter), its password
# (Enter), go ahead
run_installer curl 'y\n\n\ny\n'
expect_rc 0
expect_out "NVIDIA driver 580.95.05"
expect_no_out "Does the Jetson's power stay on"
expect_out "Download NVIDIA TensorRT $PC_TRT into $PC_TRT_DIR"
check "install.sh's PC_TRT $PC_TRT is not the build build-linux.sh compiles against" \
  grep -qF "libnvinfer-headers-dev_${PC_TRT}-" "$SRC/scripts/build-linux.sh"
check "never installed libcurl4" grep -qE '^apt-get .* install --no-install-recommends .*libcurl4' "$FAKE_LOG"
# TensorRT is NVIDIA's wheel, unpacked: the libraries and the Linux builder
# resources, and nothing of NVIDIA's from apt
expect_ran "$PC_TRT_WHEEL"
for f in libnvinfer.so.11 libnvonnxparser.so.11 libnvinfer_plugin.so.11 libnvinfer_builder_resource_sm89.so.11.3.0 \
    libnvinfer_builder_resource_ptx.so.11.3.0 LICENSE.txt; do
  expect_file "$PC_TRT_DIR/$f"
done
expect_no_file "$PC_TRT_DIR/libnvinfer_builder_resource_win_sm89.so.11.3.0"
expect_no_file "$PC_TRT_DIR/__init__.py"
expect_no_file /opt/jetlink/tensorrt/.new
refute "installed TensorRT or CUDA from apt" grep -qE 'install .*(libnvinfer|libnvonnx|tensorrt|cuda)|^dpkg -i' "$FAKE_LOG"
check "wanted the package list fetched once" test "$(grep -c "apt-get .* update" "$FAKE_LOG")" = 1
expect_ran "jetlink-server backends --backend trt --tensorrt-libs $PC_TRT_DIR"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-x86_64.tar.gz"
expect_in /etc/jetlink/server.env "JETLINK_JETSON=0"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/var/lib/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-x86_64"
expect_no_file /etc/systemd/journald.conf.d/60-jetlink.conf
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_no_file "$UNITS/jetlink-clocks.service"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-gpu.conf"
expect_not_ran "efibootmgr"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF=""'
expect_out "Keep this computer plugged in and awake"
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "server         0.10.0 (TensorRT $PC_TRT)"
# the unit's command line, and a prepare, with the PC's TensorRT
jetlink run >/dev/null 2>&1
expect_ran "jetlink-server --usb --backend trt --cache /var/lib/jetlink --sleep-after 0 --status-port 5600 --web-auth /etc/jetlink/web-auth.json --tensorrt-libs $PC_TRT_DIR"
jetlink models prepare some-model >/dev/null 2>&1
expect_ran "jetlink-server models prepare --cache /var/lib/jetlink --tensorrt-libs $PC_TRT_DIR some-model"
systemctl start jetlink-server
# a server that never sleeps serves until the new one is downloaded and
# checked; TensorRT's directory is kept, and one from before it goes
mkdir -p /opt/jetlink/tensorrt/11.2.0.0 && echo old >/opt/jetlink/tensorrt/11.2.0.0/libnvinfer.so.11
: >"$FAKE_LOG"
cli update
expect_rc 0
expect_before "releases/download/v0.10.0" "systemctl stop jetlink-server"
expect_before "jetlink-server backends" "systemctl stop jetlink-server"
expect_not_ran "pypi.nvidia.com"
expect_file "$PC_TRT_DIR/libnvinfer.so.11"
expect_no_file /opt/jetlink/tensorrt/11.2.0.0

scenario "uninstall removes it and the PC's TensorRT, and keeps the models"
mkdir -p /var/lib/jetlink/models && echo x >/var/lib/jetlink/models/m.onnx
# questions: remove?, delete the models?
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_out "Jetlink is removed."
expect_no_out "sudo apt remove"
expect_no_file /etc/jetlink
expect_no_file "$PC_TRT_DIR"
expect_no_file /opt/jetlink
expect_no_file /usr/local/bin/jetlink
expect_no_file "$UNITS/jetlink-server.service"
expect_file /var/lib/jetlink/models/m.onnx
refute "removed a package" grep -qE '^apt-get .* (remove|purge)' "$FAKE_LOG"

scenario "Windows (WSL) is allowed, and marked untested"
reset_box; pc 580.95.05; wsl
run_installer curl '' --yes
expect_rc 0
expect_out "Windows (WSL) support is untested."
expect_out "usbipd"
# TensorRT as on any PC; a Linux driver, CUDA or TensorRT package inside WSL
# breaks the Windows driver's
expect_ran "$PC_TRT_WHEEL"
expect_file "$PC_TRT_DIR/libnvinfer.so.11"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""
refute "installed CUDA, TensorRT or a driver in WSL" grep -qE 'install .*(cuda|libnvinfer|tensorrt|nvidia-driver)|^dpkg -i' "$FAKE_LOG"
expect_in /etc/jetlink/install.conf "WSL"
reset_box; pc 575.64.03; wsl
run_installer curl '' --yes
expect_rc 1
expect_out "Update the NVIDIA driver in Windows"
expect_not_ran "ubuntu-drivers"
reset_box; pc 580.95.05; wsl
JETLINK_TEST_SYSTEMD_RUN=/nonexistent run_installer curl '' --yes
expect_rc 1
expect_out "systemd=true"

scenario "a damaged TensorRT download is refused, and nothing of it is left"
reset_box; pc 580.95.05
FAKE_BAD_WHEEL=1 run_installer curl '' --yes
expect_rc 1
expect_out "The TensorRT download is damaged: its checksum does not match."
expect_no_file /opt/jetlink/tensorrt/.new
expect_no_file "$PC_TRT_DIR"
expect_no_file "$UNITS/jetlink-server.service"

scenario "a PC short of room for TensorRT stops before downloading it"
reset_box; pc 580.95.05
FAKE_ROOT_FREE_GB=5 run_installer curl '' --yes
expect_rc 1
expect_out "Not enough free space for TensorRT: 5 GB on /opt/jetlink/tensorrt, and it needs 7 GB."
expect_not_ran "pypi.nvidia.com"
expect_no_file "$UNITS/jetlink-server.service"

scenario "a PC's TensorRT from apt, which an earlier installer put in, stays unused"
reset_box; pc 580.95.05
with_trt 11.3.0.99-1+cuda13.4 11
run_installer curl '' --yes
expect_rc 0
expect_out "TensorRT from apt is no longer used here; to remove it: sudo apt remove libnvinfer11 libnvonnxparsers11"
expect_ran "$PC_TRT_WHEEL"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""
refute "removed a package" grep -qE '^apt-get .* (remove|purge)' "$FAKE_LOG"
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_out "TensorRT from apt stays installed; to remove it: sudo apt remove libnvinfer11 libnvonnxparsers11"

# The other distributions: their own package manager for the base packages,
# NVIDIA's wheel for TensorRT as on Ubuntu, and the driver installed on
# Ubuntu's and Arch's families, or printed as the distribution's
# documentation gives it elsewhere.
scenario "Fedora: base packages from dnf, and the driver's steps printed"
reset_box; pc 575.64.03; distro fedora
run_installer curl '' --yes
expect_rc 1
expect_out "has NVIDIA driver 575.64.03 and needs 580 or newer"
expect_out "On Fedora Linux 42 (Workstation Edition):"
expect_out "rpmfusion-nonfree-release-42.noarch.rpm"
expect_out "sudo dnf install akmod-nvidia"
expect_not_ran "ubuntu-drivers"
expect_not_ran "dnf"
reset_box; pc 580.95.05; distro fedora
export FAKE_NO_CURL=1
run_installer curl '' --yes
expect_rc 0
expect_out "Jetlink is untested on Fedora Linux 42 (Workstation Edition)"
expect_ran "dnf -y install libcurl"
expect_not_ran "apt-get"
expect_ran "$PC_TRT_WHEEL"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""
expect_in /etc/jetlink/install.conf "Fedora"
run_installer checkout 'y\nn\n' --uninstall
expect_rc 0
expect_out "Jetlink is removed."
expect_no_out "sudo apt remove"

scenario "Arch and its derivatives: the driver from pacman, for the kernel that runs"
reset_box; pc 575.64.03; distro arch
FAKE_KERNEL=6.12.4-arch1-1 run_installer curl 'y\n'
expect_rc 0
expect_out "Install it now?"
expect_ran "pacman -S --needed --noconfirm nvidia-open"
expect_not_ran "nvidia-open-dkms"
expect_out "Restart the computer, then run the installer again"
expect_no_out "Secure Boot is on"
expect_no_file /etc/jetlink/server.env
reset_box; pc 575.64.03; distro arch
FAKE_KERNEL=6.12.4-1-lts run_installer curl '' --yes
expect_rc 0
expect_ran "pacman -S --needed --noconfirm nvidia-open-lts"
# another kernel builds the module itself, and Secure Boot would not load it
reset_box; pc 575.64.03; distro arch; secure_boot
FAKE_KERNEL=6.12.4-zen1-1-zen run_installer curl '' --yes
expect_rc 0
expect_ran "pacman -S --needed --noconfirm nvidia-open-dkms linux-zen-headers"
expect_out "Secure Boot is on"
# CachyOS runs its own kernel on Arch's repositories; Manjaro's driver comes
# from its own tool
reset_box; pc 575.64.03; distro cachyos
FAKE_KERNEL=6.12.4-2-cachyos run_installer curl '' --yes
expect_rc 0
expect_ran "pacman -S --needed --noconfirm nvidia-open-dkms linux-cachyos-headers"
reset_box; pc 575.64.03; distro manjaro
run_installer curl '' --yes
expect_rc 1
expect_out "sudo mhwd -a pci nonfree 0300"
expect_not_ran "pacman"
# a package list behind the mirror: the install is tried again with a refresh
reset_box; pc 580.95.05; distro arch
export FAKE_NO_CURL=1 FAKE_PACMAN_STALE=1
run_installer curl '' --yes
expect_rc 0
expect_ran "pacman -S --needed --noconfirm curl"
expect_ran "pacman -Sy --needed --noconfirm curl"
expect_ran "$PC_TRT_WHEEL"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""

scenario "openSUSE: base packages from zypper, and NVIDIA's repository for the driver"
reset_box; pc 575.64.03; distro tumbleweed
run_installer curl '' --yes
expect_rc 1
expect_out "sudo zypper addrepo https://download.nvidia.com/opensuse/tumbleweed NVIDIA"
expect_out "sudo zypper install-new-recommends --repo NVIDIA"
reset_box; pc 575.64.03; distro leap
run_installer curl '' --yes
expect_rc 1
expect_out "https://download.nvidia.com/opensuse/leap/15.6 NVIDIA"
reset_box; pc 580.95.05; distro tumbleweed
export FAKE_NO_CURL=1
run_installer curl '' --yes
expect_rc 0
expect_ran "zypper --non-interactive install libcurl4"
expect_not_ran "apt-get"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""

scenario "Debian and RHEL: the driver by hand; Ubuntu's derivatives: Ubuntu's tool"
reset_box; pc 575.64.03; distro debian
run_installer curl '' --yes
expect_rc 1
expect_out "cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
expect_out "sudo apt update && sudo apt install linux-headers-amd64 nvidia-open"
expect_not_ran "ubuntu-drivers"
reset_box; pc 575.64.03; distro rocky
run_installer curl '' --yes
expect_rc 1
expect_out "cuda/repos/rhel10/x86_64/cuda-rhel10.repo"
expect_out "sudo dnf install nvidia-open"
reset_box; pc 575.64.03; distro mint
run_installer curl 'y\n'
expect_rc 0
expect_ran "ubuntu-drivers install nvidia:580-open"
expect_out "Restart the computer, then run the installer again"
reset_box; pc 580.95.05; distro debian
export FAKE_NO_CURL=1
run_installer curl '' --yes
expect_rc 0
expect_out "Jetlink is untested on Debian GNU/Linux 12 (bookworm)"
expect_ran "$(apt_install libcurl4)"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""

scenario "a PC whose package manager the installer does not know installs when nothing is missing"
reset_box; pc 580.95.05; distro gentoo
run_installer curl '' --yes
expect_rc 0
expect_not_ran "apt-get"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""
reset_box; pc 580.95.05; distro gentoo
export FAKE_NO_CURL=1
run_installer curl '' --yes
expect_rc 1
expect_out "Jetlink needs libcurl4, and this system's package manager is not one the installer knows"
expect_no_file /etc/jetlink

scenario "an old glibc, or no systemd, stops before anything changes"
reset_box; pc 580.95.05
FAKE_GLIBC=2.31 run_installer curl '' --yes
expect_rc 1
expect_out "This system's glibc is 2.31, and the Jetlink server needs 2.35 or newer."
expect_no_file /etc/jetlink
reset_box; pc 580.95.05
JETLINK_TEST_SYSTEMD_RUN=/nonexistent run_installer curl '' --yes
expect_rc 1
expect_out "this system does not run systemd"
expect_no_out "systemd=true"

scenario "dry run changes nothing"
reset_box; jetson 39 2.1
run_installer curl '' --yes --dry-run
expect_rc 0
expect_out "Dry run: stopping here. Nothing was changed."
expect_no_file /etc/jetlink
expect_no_file /opt/jetlink
expect_not_ran "apt-get"

scenario "JetPack 5 is refused with a way forward"
reset_box; jetson 35 6.0
run_installer checkout '' --yes
expect_rc 1
expect_out "Flash JetPack 7.2.1"

scenario "a Jetson that cannot deep-sleep defaults to switched power"
reset_box; jetson 39 2.1
echo 's2idle' >/tmp/mem-sleep
run_installer curl '' --yes
expect_rc 0
expect_in /etc/jetlink/install.conf "JETLINK_POWER=switched"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"

scenario "an 8 GB Jetson gets the 3 GB of swap the largest model's build needs, even on a 64 GB card"
reset_box; jetson 39 2.1
# the free space a 64 GB SD card had left
JETLINK_TEST_FREE_GB=23 run_installer curl '' --yes
expect_rc 0
expect_out "Add 3 GB of swap, so the 1.7 GB models can be built"
expect_ran "fallocate -l 3G /mnt/data/jetlink-swapfile"
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP=on"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_GB=3"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_NEED_GB=3"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_SHORT_GB=0"

scenario "no swap on a disk too small for it, said plainly"
reset_box; jetson 39 2.1
JETLINK_TEST_FREE_GB=17 run_installer curl '' --yes
expect_rc 0
expect_out "Not enough disk space for 3 GB of swap, so the 1.7 GB models may fail to build."
expect_out "Free 1 GB, then add it with: jetlink setup --set swap=on"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_SHORT_GB=3"
refute "no swap should be added" grep -q swapfile /etc/fstab
expect_not_ran "fallocate"
expect_ran "nvpmodel -m 2"

scenario "swap the Jetson has counts, and only the rest is added"
reset_box; jetson 39 2.1
printf 'Filename\tType\tSize\tUsed\tPriority\n/swapfile file 2097148 0 -2\n' >/tmp/swaps-2g
JETLINK_TEST_SWAPS=/tmp/swaps-2g run_installer curl '' --yes
expect_rc 0
expect_out "Add 1 GB of swap, so the 1.7 GB models can be built"
expect_ran "fallocate -l 1G /mnt/data/jetlink-swapfile"
# enough of its own: nothing added
reset_box; jetson 39 2.1
printf 'Filename\tType\tSize\tUsed\tPriority\n/swapfile file 4194300 0 -2\n' >/tmp/swaps-4g
JETLINK_TEST_SWAPS=/tmp/swaps-4g run_installer curl '' --yes
expect_rc 0
expect_no_out "of swap, so the 1.7 GB models"
expect_not_ran "fallocate"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP=on"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_GB=0"

scenario "a 16 GB Jetson builds the largest model without swap"
reset_box; jetson 39 2.1
printf 'MemTotal:       15655000 kB\n' >/tmp/meminfo-16g
JETLINK_TEST_MEMINFO=/tmp/meminfo-16g run_installer curl '' --yes
expect_rc 0
expect_not_ran "fallocate"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_NEED_GB=0"

scenario "jetlink setup --set swap turns Jetlink's swap off and on"
reset_box; jetson 39 2.1
run_installer curl '' --yes
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
: >"$FAKE_LOG"
cli setup --set swap=off </dev/null
expect_rc 0
expect_out "Remove Jetlink's swap file"
expect_not_in /etc/fstab "jetlink-swapfile"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP=off"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_FILE="
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_GB=0"
# an update keeps it off
: >"$FAKE_LOG"
run_installer curl '' --update
expect_rc 0
expect_not_ran "fallocate"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP=off"
: >"$FAKE_LOG"
cli setup --set swap=on </dev/null
expect_rc 0
expect_ran "fallocate -l 3G /mnt/data/jetlink-swapfile"
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP=on"
# asked for with too little room: refused, saying why, and nothing changes
cli setup --set swap=off </dev/null
cp /etc/jetlink/install.conf /tmp/install.conf.before
: >"$FAKE_LOG"
JETLINK_TEST_FREE_GB=8 cli setup --set swap=on </dev/null
expect_rc 1
expect_out "Not enough free disk space for 3 GB of swap: 8 GB is free, and Jetlink keeps 6 GB free besides."
expect_out "Free 1 GB and try again."
expect_not_ran "fallocate"
check "the answers changed" cmp -s /tmp/install.conf.before /etc/jetlink/install.conf

scenario "no terminal and no --yes"
reset_box; jetson 39 2.1
run_installer checkout ''
expect_rc 1
expect_out "There is no terminal to ask questions in."

scenario "--binary installs a server built elsewhere"
reset_box; jetson 39 2.1
# from a checkout the installer has nothing to download
run_installer checkout '' --yes
expect_rc 1
expect_out "From a checkout, the installer installs a server you built"
run_installer checkout '' --yes --binary /tmp/releases/v0.10.0/jetlink-server-0.10.0-linux-x86_64.tar.gz
expect_rc 1
expect_out "is for another kind of computer; this one needs a linux-aarch64 build"
run_installer checkout '' --yes --binary /tmp/releases/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz
expect_rc 0
expect_out "Install the Jetlink server from /tmp/releases/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz"
expect_not_ran "releases/download"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=local"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=local"
# run from the checkout again, it keeps the server it has
: >"$FAKE_LOG"
run_installer checkout '' --update
expect_rc 0
expect_out "Keep the Jetlink server that is installed"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0

scenario "--ref from a checkout, which installs what is in it, says what to run instead"
reset_box; jetson 39 2.1
run_installer checkout '' --yes --ref v0.9.0
expect_rc 1
expect_out "--ref does nothing from a checkout"
expect_out "curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.9.0/install.sh | bash -s -- --update --ref v0.9.0"
expect_no_file /etc/jetlink

scenario "a tarball from a tree before the native server is refused"
reset_box; jetson 39 2.1
old=/tmp/dev/old-unit
rm -rf "$old" && mkdir -p "$old/bin" "$old/share/jetlink/systemd"
ln -s "$SRC/tests/installer/fake.sh" "$old/bin/jetlink-server"
echo 0.6.9 >"$old/VERSION"
printf '[Service]\nExecStart=/usr/local/lib/jetlink/run-server\n' >"$old/share/jetlink/systemd/jetlink-server.service"
tar -czf /tmp/dev/jetlink-server-0.6.9-linux-aarch64.tar.gz -C "$old" .
run_installer checkout '' --yes --binary /tmp/dev/jetlink-server-0.6.9-linux-aarch64.tar.gz
expect_rc 1
expect_out "does not start /opt/jetlink/current/bin/jetlink-server"
expect_out "scripts/build-linux.sh linux-aarch64"
expect_no_file /opt/jetlink/0.6.9
check "a stage was left" test -z "$(find /opt/jetlink -maxdepth 1 -name '.new-*' 2>/dev/null)"
expect_no_file "$UNITS/jetlink-server.service"

scenario "--binary over a release install leaves its source where it is"
reset_box; jetson 39 2.1
run_installer curl '' --yes
head_before="$(git -C /opt/jetlink/src rev-parse HEAD)"
: >"$FAKE_LOG"
bash /opt/jetlink/src/install.sh --update --binary /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz >"$OUT" 2>&1; RC=$?
expect_rc 0
check "the source moved" test "$(git -C /opt/jetlink/src rev-parse HEAD)" = "$head_before"
expect_not_ran "releases/latest"
expect_not_ran "releases/download"
expect_link /opt/jetlink/current /opt/jetlink/0.12.0-dev
expect_link /opt/jetlink/previous /opt/jetlink/0.10.0
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
# the unit that came with the binary, not the source's
expect_in "$UNITS/jetlink-server.service" "# the 0.12.0-dev build's unit"
# a release's build says which release it is
run_installer curl '' --update --binary /tmp/releases/v0.9.0/jetlink-server-0.9.0-linux-aarch64.tar.gz
expect_rc 0
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.9.0"
expect_not_in "$UNITS/jetlink-server.service" "0.12.0-dev"

scenario "while the newest release runs in Docker, a native install stays put unless --ref says"
reset_box; jetson 39 2.1
run_installer curl '' --yes
head_before="$(git -C /opt/jetlink/src rev-parse HEAD)"
: >"$FAKE_LOG"
FAKE_LATEST=v0.6.0 cli update
expect_rc 0
expect_out "runs the server in Docker. Nothing changed."
expect_out "curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.6.0/install.sh | bash -s -- --update --ref v0.6.0"
expect_not_ran "systemctl"
check "the source moved" test "$(git -C /opt/jetlink/src rev-parse HEAD)" = "$head_before"
# the installer run by itself, as curl | bash or jetlink setup runs it, keeps the server too
FAKE_LATEST=v0.6.0 run_installer curl '' --update
expect_rc 0
expect_out "v0.6.0 runs Jetlink in Docker, so the server installed here stays."
expect_out "Keep the Jetlink server that is installed"
expect_no_out "its own installer takes over"
expect_not_ran "releases/download"
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
expect_in "$UNITS/jetlink-server.service" "/opt/jetlink/current/bin/jetlink-server"
check "the source moved" test "$(git -C /opt/jetlink/src rev-parse HEAD)" = "$head_before"
# questions: power, the comma may shut it down, the desktop (Enter keeps the
# answer), the port, keep the web page's password, go ahead
answers '1\ny\n\n\ny\ny\n'
FAKE_LATEST=v0.6.0 JETLINK_INPUT=/tmp/answers jetlink setup >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "Keep the web page's password?"
expect_out "Keep the Jetlink server that is installed"
expect_in "$UNITS/jetlink-server.service" "/opt/jetlink/current/bin/jetlink-server"
# nor to a release older than the server here
bash /opt/jetlink/src/install.sh --update --binary /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz >"$OUT" 2>&1
: >"$FAKE_LOG"
cli update
expect_rc 0
expect_out "v0.10.0, the release this install follows, is older than the server here (0.12.0-dev)."
expect_not_ran "systemctl"
expect_link /opt/jetlink/current /opt/jetlink/0.12.0-dev
# --ref goes back, as asked
FAKE_LATEST=v0.6.0 FAKE_PUBLISHED=1 cli update --ref v0.6.0
expect_rc 0
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE_REF=ghcr.io/zoompilot/jetlink:0.6.0-cuda"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"

scenario "TensorRT already here without its plugins: they come, or it does without"
reset_box; jetson 36 4.3
with_trt 10.3.0.30-1+cuda12.5 10
run_installer curl '' --yes
expect_rc 0
expect_not_ran "$(apt_install "libnvinfer10")"
expect_out "Installing TensorRT's plugins"
expect_ran "$(apt_install "libnvinfer-plugin10=10.3.0.30-1+cuda12.5")"
# a package source without them: said, and not a failure
reset_box; jetson 36 4.3
with_trt 10.3.0.30-1+cuda12.5 10
FAKE_NO_PLUGIN=1 run_installer curl '' --yes
expect_rc 0
expect_out "Could not install TensorRT's plugins; Jetlink's models do not need them."
expect_out "Jetlink is installed and running"
# with them already there, nothing to do
reset_box; jetson 36 4.3
with_trt 10.3.0.30-1+cuda12.5 10
echo 10.3.0.30-1+cuda12.5 >"$FAKE_STATE/pkg-libnvinfer-plugin10"
run_installer curl '' --yes
expect_rc 0
expect_not_ran "install --no-install-recommends libnvinfer"

# From the Docker releases: each installed by its own installer, then moved by
# its own `jetlink update`, which runs this tree's installer.

scenario "a v0.4.3 JetPack 7.2 install moves out of Docker on jetlink update"
reset_box; jetson 39 2.1; with_docker
old_install v0.4.3 --ref v0.4.3
# what 0.4.x saved: main, its default, and no version
sed -i -e 's/^JETLINK_REF=.*/JETLINK_REF=main/' -e '/^JETLINK_VERSION=/d' /etc/jetlink/install.conf
# drop-ins of the user's: one runs docker, one does not
printf '[Service]\nExecStartPre=-/usr/bin/docker pull ghcr.io/zoompilot/jetlink:edge-cuda\n' >"$UNITS/jetlink-server.service.d/50-pull.conf"
printf '[Service]\nNice=-5\n' >"$UNITS/jetlink-server.service.d/60-nice.conf"
mkdir -p /mnt/data/jetlink/engines && echo plan >/mnt/data/jetlink/engines/abc.plan
echo '{"sha256": "abc"}' >/mnt/data/jetlink/last-loaded.json
cli update
expect_rc 0
expect_out "Jetlink now follows releases"
expect_out "Move Jetlink out of Docker"
expect_out "Jetlink is installed and running"
expect_out "It no longer runs in Docker"
# the Docker server serves until the native one is downloaded, has its
# TensorRT and has passed the GPU check
expect_before "$(apt_install "libnvinfer10 libnvonnxparsers10")" "systemctl stop jetlink-server"
expect_before "releases/download/v0.10.0/jetlink-server-0.10.0-linux-aarch64.tar.gz" "systemctl stop jetlink-server"
expect_before "jetlink-server backends" "systemctl stop jetlink-server"
expect_ran "docker rm -f jetlink"
# meanwhile it serves without sleeping, so it cannot suspend the Jetson under
# apt; the native server starts with the user's sleep
expect_out "Keeping this computer awake for the update"
expect_before "jetlink-server started: docker, sleep 0" "releases/download/v0.10.0"
expect_before "jetlink-server started: docker, sleep 0" "$(apt_install "libnvinfer10")"
expect_ran "jetlink-server started: native, sleep 120"
check "server.env.prev lost the sleep" grep -q '^JETLINK_SLEEP_AFTER=120' /etc/jetlink/server.env.prev
# its images go once the native server is up, and Docker stays
expect_before "jetlink-server started: native" "docker rmi"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.4.3-cuda"
refute "removed a package" grep -qE '^apt-get .* (remove|purge)' "$FAKE_LOG"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/mnt/data/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_STATUS_PORT=5600"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_not_in /etc/jetlink/server.env "JETLINK_GPU_ARGS"
expect_in /etc/jetlink/server.env.prev "JETLINK_IMAGE="
# the answers, all kept
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF="--poweroff"'
expect_in /etc/jetlink/install.conf "JETLINK_AUTOSTART=1"
expect_in /etc/jetlink/install.conf "JETLINK_SWAP_FILE=/mnt/data/jetlink-swapfile"
expect_in /etc/jetlink/install.conf "JETLINK_MASKED_UNITS=systemd-networkd-wait-online.service"
expect_in /etc/jetlink/install.conf "JETLINK_JOURNALD_CAPPED=1"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
# the Jetson as it was set up
expect_not_ran "nvpmodel -m"
expect_not_ran "fallocate"
check "the swap file is in fstab once" test "$(grep -c jetlink-swapfile /etc/fstab)" = 1
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
expect_file /mnt/data/jetlink/engines/abc.plan
expect_file /mnt/data/jetlink/last-loaded.json
expect_ran "systemctl enable jetlink-server"
# the Docker setup, saved; the native one in its place
expect_in /etc/jetlink/docker-era/systemd/jetlink-server.service "run-server"
expect_file /etc/jetlink/docker-era/systemd/jetlink-server.service.d/50-pull.conf
expect_file /etc/jetlink/docker-era/lib/run-server
expect_in /etc/jetlink/docker-era/enabled "jetlink-poweroff.path"
expect_not_in /etc/jetlink/docker-era/enabled "jetlink-poweroff.service"
expect_out "Your drop-in 50-pull.conf runs Docker, so it is set aside"
expect_no_file "$UNITS/jetlink-server.service.d/50-pull.conf"
expect_file "$UNITS/jetlink-server.service.d/60-nice.conf"
expect_in "$UNITS/jetlink-server.service.d/10-cache.conf" "RequiresMountsFor=/mnt/data/jetlink"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_no_file "$UNITS/jetlink-clocks.service"
expect_in "$UNITS/jetlink-server.service" "/opt/jetlink/current/bin/jetlink-server"
expect_no_file /usr/local/lib/jetlink
expect_no_file "$UNITS/jetlink-poweroff.path"
expect_no_file "$UNITS/jetlink-poweroff.service"
expect_ran "systemctl disable --now jetlink-poweroff.path"
expect_in /usr/local/bin/jetlink "SERVER=/opt/jetlink/current/bin/jetlink-server"
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "server         0.10.0"

scenario "a v0.5.0 JetPack 6 install moves out of Docker"
reset_box; jetson 36 4.3; with_docker
old_install v0.5.0
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
expect_ran "$(apt_install "libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10")"
expect_out "TensorRT 10.3.0.30"
expect_not_ran "apt-cache policy"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.5.0-jetpack6"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-aarch64"
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.10.0"
expect_no_file /usr/local/lib/jetlink

scenario "a v0.5.0 JetPack 7.2 install short of room on / deletes its images first"
reset_box; jetson 39 2.1; with_docker
old_install v0.5.0
FAKE_ROOT_FREE_GB=2 FAKE_IMAGE_GB=8 cli update
expect_rc 0
expect_out "deleting Jetlink's Docker images first"
expect_before "systemctl stop jetlink-server" "docker rmi"
expect_before "docker rmi" "$(apt_install "libnvinfer10 libnvonnxparsers10")"
check "an image is left" test ! -s "$FAKE_STATE/images"
expect_out "Jetlink is installed and running"
# and without room even then: nothing moved, and the way back said
reset_box; jetson 39 2.1; with_docker
old_install v0.5.0
FAKE_ROOT_FREE_GB=0 FAKE_IMAGE_GB=1 cli update
expect_rc 1
expect_out "Not enough free space for TensorRT: 1 GB on /, and it needs 6 GB."
expect_out "jetlink update --ref v0.6.0"
expect_not_ran "$(apt_install "libnvinfer10")"
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE="
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"

scenario "Docker's images on another disk are not deleted for room on /"
reset_box; jetson 39 2.1; with_docker
old_install v0.5.0
FAKE_ROOT_FREE_GB=2 FAKE_IMAGE_GB=8 FAKE_DOCKER_ROOT=/mnt/data/docker FAKE_OTHER_FS=/mnt/data cli update
expect_rc 1
expect_out "Jetlink's Docker images are not on the disk TensorRT goes on (/mnt/data/docker), so they stay."
expect_out "Not enough free space for TensorRT: 2 GB on /, and it needs 6 GB."
expect_not_ran "docker rmi"
expect_not_ran "systemctl stop jetlink-server"
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"

scenario "a v0.6.0 PC install moves out of Docker"
reset_box; pc 580.95.05; with_docker
old_install v0.6.0
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
expect_ran "$PC_TRT_WHEEL"
expect_in /etc/jetlink/server.env "JETLINK_TENSORRT=\"--tensorrt-libs $PC_TRT_DIR\""
expect_ran "releases/download/v0.10.0/jetlink-server-0.10.0-linux-x86_64.tar.gz"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.6.0-cuda"
refute "removed the toolkit" grep -qE '^apt-get .* (remove|purge)' "$FAKE_LOG"
expect_in /etc/jetlink/server.env "JETLINK_JETSON=0"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/var/lib/jetlink"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=linux-x86_64"
expect_in /etc/jetlink/install.conf "JETLINK_AUTOSTART=1"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_in /etc/jetlink/server.env 'JETLINK_POWEROFF=""'

scenario "a move from Docker deletes Jetlink's untagged images by ID, and nobody else's"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
# a pre-0.5.0 image whose tag a newer pull took, one built here, and other people's
{
  echo "ghcr.io/zoompilot/jetlink:<none> 5b1e0c7a9f00"
  echo "jetlink:<none> 77aa00bb11cc"
  echo "ubuntu:24.04"
  echo "someone/else:<none> 0000feedbeef"
  echo "<none>:<none> 1111deadbeef"
} >>"$FAKE_STATE/images"
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
grep '^docker rmi' "$FAKE_LOG" >/tmp/rmi.txt
check "the tagged image was not deleted by name" grep -qE ' ghcr\.io/zoompilot/jetlink:0\.6\.0-' /tmp/rmi.txt
expect_in /tmp/rmi.txt " 5b1e0c7a9f00"
expect_in /tmp/rmi.txt " 77aa00bb11cc"
# docker rmi refuses repo:<none>, and the rest are not Jetlink's
expect_not_in /tmp/rmi.txt "<none>"
expect_not_in /tmp/rmi.txt "ubuntu"
expect_not_in /tmp/rmi.txt "0000feedbeef"
expect_not_in /tmp/rmi.txt "1111deadbeef"
refute "a Jetlink image is left" grep -qE '^(jetlink|ghcr\.io/zoompilot/jetlink):' "$FAKE_STATE/images"
check "another's image went" test "$(LC_ALL=C sort "$FAKE_STATE/images" | tr '\n' '|')" = "<none>:<none> 1111deadbeef|someone/else:<none> 0000feedbeef|ubuntu:24.04|"

scenario "a failed move puts the Docker server back, and the next update finishes it"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
# the status page's own unit, from before it moved into the server
printf '[Service]\nExecStart=/usr/bin/python3 /usr/local/lib/jetlink/web/jetlink_web.py\n[Install]\nWantedBy=multi-user.target\n' \
  >"$UNITS/jetlink-web.service"
# and a unit of the bench's own, which is not the installer's to save or start
printf '[Service]\nExecStart=/usr/local/bin/jetlink-swift\n[Install]\nWantedBy=multi-user.target\n' \
  >"$UNITS/jetlink-swift.service"
# a failure while it serves without sleeping puts the sleep back
FAKE_BAD_SUM=1 cli update
expect_rc 1
expect_ran "jetlink-server started: docker, sleep 0"
check "not started again with its sleep" test "$(grep 'jetlink-server started' "$FAKE_LOG" | tail -n 1)" = "jetlink-server started: docker, sleep 120"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_out "The previous Jetlink server is running again."
: >"$FAKE_LOG"
export FAKE_SERVER_BROKEN=1
cli update
expect_rc 1
expect_ran "jetlink-server started: docker, sleep 0"
check "not started again with its sleep" test "$(grep 'jetlink-server started' "$FAKE_LOG" | tail -n 1)" = "jetlink-server started: docker, sleep 120"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_out "The previous Jetlink server is running again."
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_no_file "$UNITS/jetlink-server.service.d/20-jetson-clocks.conf"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE="
expect_in /etc/jetlink/install.conf "JETLINK_VERSION=v0.6.0"
expect_file /usr/local/lib/jetlink/run-server
expect_in /usr/local/bin/jetlink "docker run"
expect_file "$UNITS/jetlink-poweroff.path"
expect_file "$UNITS/jetlink-web.service"
expect_ran "systemctl enable --now jetlink-poweroff.path"
expect_ran "systemctl enable --now jetlink-web.service"
expect_out "jetlink-swift.service is not the installer's, and stays as it is."
expect_no_file /etc/jetlink/docker-era/systemd/jetlink-swift.service
expect_not_in /etc/jetlink/docker-era/enabled "jetlink-swift.service"
expect_not_ran "systemctl enable --now jetlink-swift.service"
expect_file "$UNITS/jetlink-swift.service"
expect_not_ran "docker rmi"
check "the Docker server was not started again" test "$(grep -c "systemctl restart jetlink-server" "$FAKE_LOG")" -ge 2
refute "the unit was left stopped" test -f "$FAKE_STATE/stopped-jetlink-server"
unset FAKE_SERVER_BROKEN
: >"$FAKE_LOG"
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_no_file "$UNITS/jetlink-web.service"
expect_ran "systemctl disable --now jetlink-web.service"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.6.0-cuda"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"

scenario "a move that crashes the new server puts Docker back; one that loads the model waits for it"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
echo '{"sha256": "abc", "frame_skip": 1}' >/mnt/data/jetlink/last-loaded.json
# the model it ran last crashes the new server as it loads
FAKE_PRELOAD=crash cli update
expect_rc 1
expect_out "Loading the model it ran last"
expect_in /var/log/jetlink-install.log "loading the model crashed the server, and systemd started it again"
expect_out "The previous Jetlink server is running again."
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_not_ran "docker rmi"
check "an image went" test -s "$FAKE_STATE/images"
# a rebuild that outlasts the wait is the server's to finish; it serves meanwhile
: >"$FAKE_LOG"
FAKE_PRELOAD=slow JETLINK_TEST_PRELOAD_S=0 cli update
expect_rc 0
expect_in /var/log/jetlink-install.log "still preparing the model after 0s; the server serves meanwhile"
expect_out "It is still preparing that model, and goes on in the background: jetlink logs"
expect_out "Jetlink is installed and running"
expect_ran "docker rmi ghcr.io/zoompilot/jetlink:0.6.0-cuda"
# a failed preparation is not a failed server: the comma sends the model again
: >"$FAKE_LOG"
FAKE_PRELOAD=failed run_installer curl '' --update
expect_rc 0
expect_in /var/log/jetlink-install.log "engine preparation failed"
expect_no_out "still preparing"
: >"$FAKE_LOG"
FAKE_PRELOAD=ready run_installer curl '' --update
expect_rc 0
expect_in /var/log/jetlink-install.log "engine ready: 0123456789abcdef"
expect_no_out "still preparing"

scenario "a move from Docker that crashes after it serves puts Docker back, images and all"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
FAKE_SERVER_CRASHLOOP=1 cli update
expect_rc 1
expect_in /var/log/jetlink-install.log "the server did not stay up"
expect_out "The previous Jetlink server is running again."
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE="
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_not_ran "docker rmi"
expect_no_out "Jetlink is installed and running"

scenario "a move stopped part way puts Docker back with its sleep; one killed keeps the sleep for the next"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
# the ssh session drops while the server downloads, with Docker held awake
interrupted HUP jetlink update
expect_rc 129
expect_ran "jetlink-server started: docker, sleep 0"
expect_in /var/log/jetlink-install.log "Stopped by SIGHUP."
expect_in /var/log/jetlink-install.log "The previous Jetlink server is running again."
check "not started again with its sleep" test "$(grep 'jetlink-server started' "$FAKE_LOG" | tail -n 1)" = "jetlink-server started: docker, sleep 120"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_not_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER_HELD"
expect_in "$UNITS/jetlink-server.service" "run-server"
# killed outright, nothing puts it back, but the sleep it held is kept beside the 0
: >"$FAKE_LOG"
interrupted KILL jetlink update
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER_HELD=120"
: >"$FAKE_LOG"
cli update
expect_rc 0
expect_out "Jetlink is installed and running"
expect_in /etc/jetlink/server.env.prev "JETLINK_SLEEP_AFTER=120"
expect_not_in /etc/jetlink/server.env.prev "JETLINK_SLEEP_AFTER_HELD"
expect_in /etc/jetlink/docker-era/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_not_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER_HELD"
expect_ran "jetlink-server started: docker, sleep 0"
expect_ran "jetlink-server started: native, sleep 120"

scenario "a Docker server that will not stop keeps serving, and no native one starts beside it"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
FAKE_DOCKER_STUCK=1 cli update
expect_rc 1
expect_out "The Docker server did not stop."
expect_before "docker rm -f jetlink" "docker ps -q --filter name=^/?jetlink$"
expect_not_ran "jetlink-server started: native"
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE="
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_out "The previous Jetlink server is running again."
expect_not_ran "docker rmi"

scenario "going back to v0.6.0 runs its own installer, and the curl line comes forward"
FAKE_PUBLISHED=1 cli update --ref v0.6.0
expect_rc 0
expect_out "Go back to v0.6.0, which runs Jetlink in Docker"
expect_out "v0.6.0 runs Jetlink in Docker; its own installer takes over from here."
expect_ran "docker pull ghcr.io/zoompilot/jetlink:0.6.0-cuda"
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE_REF=ghcr.io/zoompilot/jetlink:0.6.0-cuda"
# the answers the native install kept reach it
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/install.conf "JETLINK_REF=v0.6.0"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_file "$UNITS/jetlink-poweroff.path"
expect_file /opt/jetlink/0.10.0/bin/jetlink-server
: >"$FAKE_LOG"
run_installer curl '' --update --ref latest
expect_rc 0
expect_out "Move Jetlink out of Docker"
expect_not_in /etc/jetlink/server.env "JETLINK_IMAGE"
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
expect_in "$UNITS/jetlink-server.service" "/opt/jetlink/current/bin/jetlink-server"

scenario "the bench's round trip: the v0.6.0 curl line goes back, a --binary from the clone comes forward"
reset_box; jetson 39 2.1; with_docker
old_install v0.6.0
cli update
expect_rc 0
expect_link /opt/jetlink/current /opt/jetlink/0.10.0
# back, with that release's own installer
FAKE_LATEST=v0.6.0 FAKE_PUBLISHED=1 bash -s -- --update --ref v0.6.0 </releases/v0.6.0/install.sh >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_in "$UNITS/jetlink-server.service" "run-server"
expect_file /opt/jetlink/src/scripts/jetlink-run-server
# forward: a --binary with the clone still on v0.6.0 would install its Docker jetlink command
: >"$FAKE_LOG"
FAKE_LATEST=v0.6.0 run_installer curl '' --update --binary /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz
expect_rc 1
expect_out "/opt/jetlink/src holds Jetlink from before the native server"
expect_in "$UNITS/jetlink-server.service" "run-server"
# with the clone moved to the tree the server was built from
git -C /opt/jetlink/src fetch -q --depth 1 origin v0.10.0 && git -C /opt/jetlink/src reset -q --hard FETCH_HEAD
: >"$FAKE_LOG"
FAKE_LATEST=v0.6.0 bash /opt/jetlink/src/install.sh --update --binary /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "Move Jetlink out of Docker"
expect_link /opt/jetlink/current /opt/jetlink/0.12.0-dev
expect_link /opt/jetlink/previous /opt/jetlink/0.10.0
expect_in "$UNITS/jetlink-server.service" "/opt/jetlink/current/bin/jetlink-server"
expect_in /usr/local/bin/jetlink "SERVER=/opt/jetlink/current/bin/jetlink-server"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_ran "jetlink-server started: native, sleep 120"
# the way back pinned v0.6.0; kept, the pin would hold every later update there
expect_out "Jetlink follows releases again: it was pinned to v0.6.0, which runs the server in Docker."
expect_in /etc/jetlink/install.conf "JETLINK_REF=latest"
# while the newest release runs in Docker, an update does not follow it there
: >"$FAKE_LOG"
FAKE_LATEST=v0.6.0 cli update
expect_rc 0
expect_out "runs the server in Docker. Nothing changed."
expect_not_ran "systemctl"
expect_link /opt/jetlink/current /opt/jetlink/0.12.0-dev
# the build names itself, not the release it came over from
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "0.12.0-dev (a build between releases)"
expect_not_in /tmp/status.txt "v0.6.0 ("
# and once the newest release runs natively, the install follows it
FAKE_LATEST=v0.10.0 cli update
expect_rc 0
expect_out "v0.10.0, the release this install follows, is older than the server here (0.12.0-dev)."
# a --ref given with --binary still wins
bash /opt/jetlink/src/install.sh --update --binary /tmp/dev/jetlink-server-0.12.0-dev-linux-aarch64.tar.gz --ref v0.6.0 >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_no_out "follows releases again"
expect_in /etc/jetlink/install.conf "JETLINK_REF=v0.6.0"

scenario "uninstall after a move removes the Docker leftovers too"
echo "jetlink:local-cuda" >>"$FAKE_STATE/images"
mkdir -p /mnt/data/jetlink/engines && echo plan >/mnt/data/jetlink/engines/abc.plan
# questions: remove?, delete the old images?, delete the models?
run_installer checkout 'y\ny\nn\n' --uninstall
expect_rc 0
expect_out "Jetlink is removed."
expect_ran "docker rmi jetlink:local-cuda"
expect_no_file /etc/jetlink
expect_no_file /opt/jetlink
expect_out "sudo apt remove libnvinfer10 libnvonnxparsers10 libnvinfer-plugin10"
expect_file /mnt/data/jetlink/engines/abc.plan
show_on_failure

echo
if [ "$FAILED" -eq 0 ]; then
  echo "installer scenarios: $PASSED checks passed"
else
  echo "installer scenarios: $FAILED failed, $PASSED passed"
  exit 1
fi
