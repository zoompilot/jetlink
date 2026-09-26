#!/usr/bin/env bash
# The installer, end to end, inside a throwaway Ubuntu container: real files
# written, real units and scripts installed, and every system command the
# installer calls (apt, systemd, docker, nvpmodel, ...) replaced by fake.sh.
# run.sh starts the container; this runs in it, as root, with the source tree
# at /src.
#
# Each scenario is a computer (a JetPack 7.2 Jetson, a JetPack 6 one, a PC)
# plus the answers typed at the questions, and checks what the installer left
# behind, what it ran, and what it told the user.
set -uo pipefail

SRC=/src
FAKE_BIN=/tmp/fakebin
export FAKE_BIN FAKE_LOG=/tmp/fake.log FAKE_STATE=/tmp/fake-state
export JETLINK_TEST_DT_MODEL=/tmp/dt-model JETLINK_TEST_MEM_SLEEP=/tmp/mem-sleep
# the same questions on every machine: plenty of disk, and no swap yet
export JETLINK_TEST_FREE_GB=100 JETLINK_TEST_SWAPS=/tmp/swaps
# nothing waited on here is real, so there is nothing to wait for
export JETLINK_TEST_POLL_S=0
printf 'Filename\tType\tSize\tUsed\tPriority\n/dev/zram0 partition 1000000 0 5\n' >/tmp/swaps
PATH="$FAKE_BIN:$PATH"
OUT=/tmp/out.txt
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
expect_rc() { check "exit $RC, wanted $1" test "$RC" = "$1"; }

reset_box() {
  rm -rf /etc/jetlink /usr/local/lib/jetlink /usr/local/bin/jetlink /opt/jetlink /var/lib/jetlink /mnt/data \
    /etc/systemd/system/jetlink-* /etc/udev/rules.d/99-jetlink-usb-wakeup.rules \
    /etc/systemd/journald.conf.d/60-jetlink.conf "$FAKE_STATE" "$FAKE_LOG" "$FAKE_BIN" \
    /etc/nv_tegra_release /etc/nvpmodel.conf /tmp/dt-model /tmp/mem-sleep
  cp /tmp/fstab.orig /etc/fstab
  mkdir -p "$FAKE_BIN" /etc/systemd/system
  local c
  for c in uname apt-get systemctl journalctl nvpmodel ubuntu-drivers udevadm fallocate mkswap swapon \
      swapoff jetson_clocks curl gpg; do
    ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/$c"
  done
  unset FAKE_ARCH FAKE_SMI FAKE_GPU_OK FAKE_PUBLISHED FAKE_PM_REBOOT FAKE_NVIDIA_RUNTIME FAKE_NV_DOCKER_POLLS \
    FAKE_PULL_FAILS
}

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
  # no container toolkit: a JetPack 7.2.1 ISO install has none until one is installed
  export FAKE_ARCH=aarch64
}

pc() {  # pc DRIVER
  export FAKE_ARCH=x86_64 FAKE_SMI="NVIDIA GeForce RTX 4070 Laptop GPU, $1, 8.9"
  ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/nvidia-smi"
}

with_docker() { ln -sf "$SRC/tests/installer/fake.sh" "$FAKE_BIN/docker"; }

install() {  # install "answers" [installer args...]; answers "" means none (no terminal)
  local answers=$1
  shift
  if [ -n "$answers" ]; then
    printf '%b' "$answers" >/tmp/answers
    JETLINK_INPUT=/tmp/answers bash "$SRC/install.sh" "$@" >"$OUT" 2>&1
  else
    bash "$SRC/install.sh" "$@" >"$OUT" 2>&1 </dev/null
  fi
  RC=$?
}

scenario() {
  SCENARIO="$1"
  printf '  %s\n' "$1"
}

show_on_failure() {
  if [ "$FAILED" -gt "$1" ]; then
    echo "    --- installer output ---"
    sed 's/^/    | /' "$OUT"
    echo "    --- commands run ---"
    sed 's/^/    | /' "$FAKE_LOG" 2>/dev/null | head -80
  fi
}

cp /etc/fstab /tmp/fstab.orig 2>/dev/null || : >/tmp/fstab.orig
# what `curl | bash` clones: the tree under test, committed
rm -rf /tmp/repo
git init -q -b main /tmp/repo
# -R, not -a: the bind-mounted tree belongs to the CI runner's user, and a repo
# owned by someone else is "dubious ownership" to the root git below
cp -R "$SRC/." /tmp/repo/
git -C /tmp/repo add -A
git -C /tmp/repo -c user.name=test -c user.email=test@example.invalid commit -qm "tree under test"
# shellcheck disable=SC1091
echo "installer scenarios on $(. /etc/os-release; echo "$PRETTY_NAME")"

# ---------------------------------------------------------------------------
scenario "JetPack 7.2 Jetson, always-on power, fresh install"
reset_box; jetson 39 2.1; f=$FAILED
# questions: power (1 = always on), let the comma shut it down, go ahead
install '1\ny\ny\n'
expect_rc 0
expect_out "Orin Nano"
expect_out "JetPack 7 (Jetson Linux 39.2.1)"
expect_out "How is the Jetson powered in the car?"
expect_no_out "fastest power mode ("
expect_no_out "Add 8 GB of swap so"
expect_out "Jetlink is installed and running"
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin"
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y install nvidia-container-toolkit"
# JetPack's meta package would reinstall Docker in the background, mid-pull
refute "installed JetPack's nvidia-container" grep -qE "install nvidia-container( |$)" "$FAKE_LOG"
expect_ran "nvidia-ctk runtime configure --runtime=docker"
expect_ran "docker build --network host -f /src/docker/Dockerfile -t jetlink:local-cuda /src"
expect_ran "nvpmodel -m 2"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=cuda"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_in /etc/jetlink/server.env 'JETLINK_GPU_ARGS=--runtime\ nvidia\ --gpus\ all'
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=local"
expect_file /usr/local/lib/jetlink/run-server
expect_file /usr/local/bin/jetlink
expect_file /usr/local/lib/jetlink/wake-setup
expect_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
expect_in /etc/systemd/system/jetlink-server.service "ExecStart=/usr/local/lib/jetlink/run-server"
expect_in /etc/systemd/system/jetlink-server.service.d/10-cache.conf "RequiresMountsFor=/mnt/data/jetlink"
expect_in /etc/systemd/system/jetlink-poweroff.path "PathExists=/mnt/data/jetlink/poweroff"
expect_in /etc/systemd/system/jetlink-poweroff.service "ExecStart=/usr/local/lib/jetlink/poweroff /mnt/data/jetlink/poweroff"
expect_file /etc/systemd/journald.conf.d/60-jetlink.conf
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_file "$FAKE_STATE/masked-systemd-networkd-wait-online.service"
expect_ran "systemctl enable jetlink-server"
expect_ran "systemctl enable --now jetlink-poweroff.path"
# the launcher, from what was written
JETLINK_DRY_RUN=1 /usr/local/lib/jetlink/run-server >/tmp/cmd.txt 2>&1
expect_in /tmp/cmd.txt "--runtime nvidia --gpus all"
expect_in /tmp/cmd.txt "--sleep-after 120"
expect_in /tmp/cmd.txt "-v /sys/power:/sys/power"
# docker-default's AppArmor profile denies those writes, mounts or not
expect_in /tmp/cmd.txt "--security-opt apparmor=unconfined"
expect_in /tmp/cmd.txt "-v /mnt/data/jetlink:/var/cache/jetlink"
# the helper
jetlink status >/tmp/status.txt 2>&1
expect_in /tmp/status.txt "Jetlink is running"
expect_in /tmp/status.txt "always on: sleeps when the car is off; the comma can shut it down"
jetlink models list >/tmp/models.txt 2>&1
expect_in /tmp/models.txt "fake model list"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "update keeps the answers and asks nothing"
: >"$FAKE_LOG"; f=$FAILED
install '' --update
expect_rc 0
expect_no_out "A few questions"
expect_no_out "Go ahead?"
expect_out "Jetlink is installed and running"
expect_ran "docker build --network host"
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=120"
expect_not_ran "nvpmodel -m"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "a second run offers to keep the settings"
: >"$FAKE_LOG"; f=$FAILED
install 'y\n'
expect_rc 0
expect_out "Jetlink is already installed. Keep your current settings and update it?"
expect_no_out "How is the Jetson powered in the car?"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "a toolkit that only reaches the GPU through CDI"
reset_box; jetson 39 2.1; with_docker; f=$FAILED
export FAKE_GPU_OK="--device nvidia.com/gpu=all"
install '' --yes
expect_rc 0
expect_ran "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
expect_in /etc/jetlink/server.env 'JETLINK_GPU_ARGS=--device\ nvidia.com/gpu=all'
# --yes takes the recommended wiring: always on, and the comma may shut it down
expect_in /etc/jetlink/install.conf "JETLINK_POWER=always"
expect_in /etc/jetlink/install.conf "JETLINK_POWEROFF_WITH_COMMA=1"
expect_not_ran "install docker-ce"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "no GPU reachable at all fails with advice"
reset_box; jetson 39 2.1; with_docker; f=$FAILED
export FAKE_GPU_OK="--never"
install '' --yes
expect_rc 1
expect_out "Docker could not give the server access to the GPU."
expect_no_file /etc/systemd/system/jetlink-server.service
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "curl | bash from the repository, with a published image"
reset_box; jetson 39 2.1; with_docker; f=$FAILED
export FAKE_PUBLISHED=1 JETLINK_REPO_URL=file:///tmp/repo
bash </src/install.sh -s -- --yes >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_ran "docker pull ghcr.io/zoompilot/jetlink:edge-cuda"
expect_not_ran "docker build"
expect_file /opt/jetlink/src/.git
expect_in /etc/jetlink/install.conf "JETLINK_SOURCE=git"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE_REF=ghcr.io/zoompilot/jetlink:edge-cuda"
unset JETLINK_REPO_URL
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "JetPack still installing Docker when the installer starts"
reset_box; jetson 39 2.1; f=$FAILED
# nvidia-container's nv-install-docker is mid-run: wait it out, then use its Docker
export FAKE_NV_DOCKER_POLLS=3
install '' --yes
expect_rc 0
expect_out "Waiting for JetPack to finish installing Docker"
expect_not_ran "install docker-ce"
expect_ran "nvidia-ctk runtime configure --runtime=docker"
expect_ran "systemctl restart docker"
expect_out "Jetlink is installed and running"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "a download cut off part way is tried again"
reset_box; jetson 39 2.1; with_docker; f=$FAILED
export FAKE_PUBLISHED=1 FAKE_PULL_FAILS=2 JETLINK_REPO_URL=file:///tmp/repo
bash </src/install.sh -s -- --yes >"$OUT" 2>&1; RC=$?
expect_rc 0
expect_out "Downloading the Jetlink server"
expect_in /var/log/jetlink-install.log "the download was interrupted; trying again"
expect_in /etc/jetlink/server.env "JETLINK_IMAGE_REF=ghcr.io/zoompilot/jetlink:edge-cuda"
expect_not_ran "docker build"
unset JETLINK_REPO_URL
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "JetPack 6.2 Jetson, switched power"
reset_box; jetson 36 4.3; f=$FAILED
export FAKE_GPU_OK="--runtime nvidia"
# questions: power (2 = switched), then Enter to go ahead
install '2\n\n'
expect_rc 0
expect_out "JetPack 6 (Jetson Linux 36.4.3)"
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y install docker.io"
expect_ran "apt-get -o DPkg::Lock::Timeout=900 -y install nvidia-container-toolkit"
expect_ran "docker build --network host -f /src/docker/Dockerfile.jetpack6 -t jetlink:local-jetpack6 /src"
expect_in /etc/jetlink/server.env "JETLINK_FLAVOR=jetpack6"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
expect_in /etc/jetlink/server.env 'JETLINK_GPU_ARGS=--runtime\ nvidia'
expect_ran "nvpmodel -m 2"
expect_in /etc/fstab "/mnt/data/jetlink-swapfile none swap sw 0 0"
expect_no_file /etc/systemd/system/jetlink-poweroff.path
expect_no_file /etc/udev/rules.d/99-jetlink-usb-wakeup.rules
JETLINK_DRY_RUN=1 /usr/local/lib/jetlink/run-server >/tmp/cmd.txt 2>&1
if grep -q -- "--sleep-after" /tmp/cmd.txt; then fail "switched power should not sleep"; else ok; fi
refute "switched power needs no AppArmor exception" grep -qF -- "apparmor=unconfined" /tmp/cmd.txt
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "a power mode that needs a restart says so"
reset_box; jetson 39 2.1; with_docker; f=$FAILED
export FAKE_PM_REBOOT=1
install '' --yes
expect_rc 0
expect_out "Restart this computer once"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "PC with a driver too old for CUDA 13"
reset_box; pc 575.64.03; f=$FAILED
install 'y\n'
expect_rc 0
expect_out "has NVIDIA driver 575.64.03 and needs 580 or newer"
expect_ran "ubuntu-drivers install nvidia:580-open"
expect_out "Restart the computer, then run the installer again"
expect_no_file /etc/jetlink/server.env
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "PC ready to go"
reset_box; pc 580.95.05; f=$FAILED
# questions: start at boot, go ahead
install 'y\ny\n'
expect_rc 0
expect_out "NVIDIA driver 580.95.05"
expect_no_out "How is the Jetson powered"
expect_ran "install docker-ce docker-ce-cli containerd.io docker-buildx-plugin"
expect_ran "install nvidia-container-toolkit"
expect_in /etc/apt/sources.list.d/nvidia-container-toolkit.list "signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
expect_in /etc/jetlink/server.env "JETLINK_JETSON=0"
expect_in /etc/jetlink/server.env "JETLINK_CACHE_DIR=/var/lib/jetlink"
expect_no_file /etc/systemd/journald.conf.d/60-jetlink.conf
expect_out "Keep this computer plugged in and awake"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "uninstall removes it and keeps the models"
: >"$FAKE_LOG"; f=$FAILED
mkdir -p /var/lib/jetlink/models && echo x >/var/lib/jetlink/models/m.onnx
# questions: remove?, delete the image?, delete the models?
install 'y\ny\nn\n' --uninstall
expect_rc 0
expect_out "Jetlink is removed."
expect_no_file /etc/jetlink
expect_no_file /usr/local/bin/jetlink
expect_no_file /etc/systemd/system/jetlink-server.service
expect_file /var/lib/jetlink/models/m.onnx
expect_ran "docker rmi jetlink:local-cuda"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "dry run changes nothing"
reset_box; jetson 39 2.1; f=$FAILED
install '' --yes --dry-run
expect_rc 0
expect_out "Dry run: stopping here. Nothing was changed."
expect_no_file /etc/jetlink
expect_not_ran "apt-get"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "JetPack 5 is refused with a way forward"
reset_box; jetson 35 6.0; f=$FAILED
install '' --yes
expect_rc 1
expect_out "Flash JetPack 7.2.1"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "a Jetson that cannot deep-sleep defaults to switched power"
reset_box; jetson 39 2.1; with_docker; f=$FAILED
echo 's2idle' >/tmp/mem-sleep
install '' --yes
expect_rc 0
expect_in /etc/jetlink/install.conf "JETLINK_POWER=switched"
expect_in /etc/jetlink/server.env "JETLINK_SLEEP_AFTER=0"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "no swap on a disk too small for it, said plainly"
reset_box; jetson 39 2.1; with_docker; f=$FAILED
JETLINK_TEST_FREE_GB=20 install '' --yes
expect_rc 0
expect_out "Not enough disk space for 8 GB of swap"
refute "no swap should be added" grep -q swapfile /etc/fstab
expect_ran "nvpmodel -m 2"
show_on_failure "$f"

# ---------------------------------------------------------------------------
scenario "no terminal and no --yes"
reset_box; jetson 39 2.1; f=$FAILED
install ''
expect_rc 1
expect_out "There is no terminal to ask questions in."
show_on_failure "$f"

echo
if [ "$FAILED" -eq 0 ]; then
  echo "installer scenarios: $PASSED checks passed"
else
  echo "installer scenarios: $FAILED failed, $PASSED passed"
  exit 1
fi
