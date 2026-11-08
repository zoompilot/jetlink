#!/usr/bin/env bash
# One stand-in for every system command the installer touches, dispatched on
# the name it is called by: scenarios.sh symlinks each name to this file, and
# the release tarballs it makes carry it as bin/jetlink-server. Each call is
# appended to $FAKE_LOG, and the answers come from $FAKE_* variables and the
# files in $FAKE_STATE, so a scenario can say "TensorRT is missing" or "the
# server cannot reach the GPU". The Docker releases' installers run on it too,
# which is why Docker and the container toolkit are still here.
#
# Test code only: nothing here runs outside tests/installer.
set -u
name="$(basename "$0")"
state="${FAKE_STATE:-/tmp/fake-state}"
mkdir -p "$state"
# and whether the server's awake lock was held at the time
held=''
lock="${JETLINK_TEST_AWAKE_LOCK:-/nonexistent}"
if [ -e "$lock" ] && ! flock --exclusive --nonblock "$lock" true 2>/dev/null; then held=' [held awake]'; fi
printf '%s %s%s\n' "$name" "$*" "$held" >>"${FAKE_LOG:-/tmp/fake.log}"

# an installed package's version; fails when it is not installed
pkg() { cat "$state/pkg-$1" 2>/dev/null; }
# libcurl, installed under whatever name the package manager gives it
libcurl_in() { echo 8.5.0 >"$state/pkg-libcurl4"; }
# The images Docker has, a "REPOSITORY TAG ID" line each, from $state/images,
# where a line is "repo:tag", or "repo:tag id" for an ID the scenario names.
# An untagged image is repo:<none>, or <none>:<none> with no repository.
fake_images() {
  [ -f "$state/images" ] || return 0
  sort -u "$state/images" | while read -r ref id; do
    [ -n "$ref" ] || continue
    printf '%s %s %s\n' "${ref%:*}" "${ref##*:}" "$(fake_image_id "$ref" "$id")"
  done
}
# an image line's ID: the scenario's, else one made from its name
fake_image_id() {
  if [ -n "${2:-}" ]; then echo "$2"; else printf '%s' "$1" | cksum | awk '{ printf "%012x\n", $1 }'; fi
}
# what JetPack's repository offers
TRT10="${FAKE_TRT10:-10.16.2.10-1+cuda13.2}"

case "$name" in
  uname)
    case "${1:-}" in
      -m) echo "${FAKE_ARCH:-aarch64}" ;;
      -r) echo "${FAKE_KERNEL:-6.8.0-fake}" ;;
      -s|'') echo Linux ;;
      *) /bin/uname "$@" ;;
    esac ;;

  getconf)
    # the C library's version, Ubuntu 24.04's unless a scenario says
    case "${1:-}" in
      GNU_LIBC_VERSION) echo "glibc ${FAKE_GLIBC:-2.39}" ;;
      *) /usr/bin/getconf "$@" ;;
    esac ;;

  dnf|pacman|zypper)
    # the other distributions' package managers: installing libcurl, by
    # whatever name, makes it appear; a driver package installs silently.
    # FAKE_PACMAN_STALE: a pacman whose package list is behind the mirror
    # fails until it is refreshed (-Sy)
    if [ "$name" = pacman ] && [ "${FAKE_PACMAN_STALE:-0}" = 1 ] && [ "${1:-}" != -Sy ]; then
      echo "error: failed retrieving file 'curl-8.5.0-1-x86_64.pkg.tar.zst' from mirror : The requested URL returned error: 404" >&2
      exit 1
    fi
    for arg in "$@"; do
      case "$arg" in
        libcurl|libcurl4|curl) libcurl_in ;;
      esac
    done ;;

  apt-get)
    # installing a package makes its commands and libraries appear; an
    # upgrade leaves alone what is not installed
    upgrade=0
    [[ " $* " == *" --only-upgrade "* ]] && upgrade=1
    if [ "${FAKE_NO_PLUGIN:-0}" = 1 ] && [[ " $* " == *" libnvinfer-plugin"* ]]; then
      echo "E: Unable to locate package libnvinfer-plugin" >&2
      exit 100
    fi
    for arg in "$@"; do
      [ "$upgrade" = 1 ] && ! pkg "${arg%%=*}" >/dev/null && continue
      case "$arg" in
        docker.io|docker-ce) ln -sf "$0" "$FAKE_BIN/docker" ;;
        nvidia-container-toolkit) ln -sf "$0" "$FAKE_BIN/nvidia-ctk" ;;
        efibootmgr) ln -sf "$0" "$FAKE_BIN/efibootmgr" ;;
        libnvinfer10|libnvonnxparsers10|libnvinfer-plugin10) echo "$TRT10" >"$state/pkg-$arg" ;;
        libnvinfer*=*|libnvonnxparsers*=*) echo "${arg#*=}" >"$state/pkg-${arg%%=*}" ;;
        libcurl4) libcurl_in ;;
      esac
    done ;;

  apt-cache)
    case "${1:-}" in
      policy) printf '%s:\n  Installed: %s\n  Candidate: %s\n' "$2" "$(pkg "$2" || echo '(none)')" "$TRT10" ;;
    esac ;;

  dpkg)
    case "${1:-}" in
      --print-architecture) if [ "${FAKE_ARCH:-aarch64}" = x86_64 ]; then echo amd64; else echo arm64; fi ;;
    esac ;;

  dpkg-query)
    # -W -f FORMAT PACKAGE...
    fmt="$3" rc=0
    shift 3
    for p in "$@"; do
      if v="$(pkg "$p")"; then
        printf '%s' "$v"
        [[ $fmt == *'\n'* ]] && echo
      else
        echo "dpkg-query: no packages found matching $p" >&2
        rc=1
      fi
    done
    exit "$rc" ;;

  ldconfig)
    # the loader's cache: libcurl unless a scenario takes it away, and
    # TensorRT's libraries once their packages are in
    echo "fake libs found in cache"
    if [ "${FAKE_NO_CURL:-0}" = 0 ] || pkg libcurl4 >/dev/null; then
      printf '\tlibcurl.so.4 (libc6) => /usr/lib/libcurl.so.4\n'
    fi
    for m in 10 11; do
      pkg "libnvinfer$m" >/dev/null && printf '\tlibnvinfer.so.%s (libc6) => /usr/lib/libnvinfer.so.%s\n' "$m" "$m"
      pkg "libnvonnxparsers$m" >/dev/null && printf '\tlibnvonnxparser.so.%s (libc6) => /usr/lib/libnvonnxparser.so.%s\n' "$m" "$m"
      pkg "libnvinfer-plugin$m" >/dev/null && printf '\tlibnvinfer_plugin.so.%s (libc6) => /usr/lib/libnvinfer_plugin.so.%s\n' "$m" "$m"
    done ;;

  df)
    # every filesystem is / but FAKE_OTHER_FS, a disk of its own: each has
    # FAKE_ROOT_FREE_GB free, plus what deleted images freed
    free=$((${FAKE_ROOT_FREE_GB:-100} + $(cat "$state/freed-gb" 2>/dev/null || echo 0)))
    dev=/dev/fake mnt=/ path="${*: -1}"
    if [ -n "${FAKE_OTHER_FS:-}" ] && { [ "$path" = "$FAKE_OTHER_FS" ] || [[ $path == "$FAKE_OTHER_FS"/* ]]; }; then
      dev=/dev/other mnt=$FAKE_OTHER_FS
    fi
    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n%s 999999999 0 %s 1%% %s\n' \
      "$dev" $((free * 1048576)) "$mnt" ;;

  systemctl)
    case "${1:-}" in
      is-active)
        quiet=0; [[ " $* " == *" --quiet "* ]] && quiet=1
        [ -f "$state/stopped-${*: -1}" ] && { [ "$quiet" = 1 ] || echo inactive; exit 3; }
        [ "$quiet" = 1 ] || echo active ;;
      stop) touch "$state/stopped-${*: -1}" ;;
      start|restart)
        rm -f "$state/stopped-${*: -1}"
        if [ "${*: -1}" = jetlink-server ]; then
          # which server came up, and whether it may sleep
          kind=native
          grep -qs run-server /etc/systemd/system/jetlink-server.service && kind=docker
          # a native one that crashes once it serves, or loading the model
          rm -f "$state/crashing"
          if [ "$kind" = native ] && { [ "${FAKE_SERVER_CRASHLOOP:-0}" = 1 ] || [ "${FAKE_PRELOAD:-}" = crash ]; }; then
            touch "$state/crashing"
          fi
          printf 'jetlink-server started: %s, sleep %s\n' "$kind" \
            "$(sed -n 's/^JETLINK_SLEEP_AFTER=//p' /etc/jetlink/server.env 2>/dev/null | tail -n 1)" \
            >>"${FAKE_LOG:-/tmp/fake.log}"
        fi ;;
      is-enabled)
        u="${*: -1}"
        if [ -f "$state/masked-$u" ]; then echo masked
        # a unit with no [Install] section, like the poweroff flag's service
        elif [ -f "/etc/systemd/system/$u" ] && ! grep -q '^\[Install\]' "/etc/systemd/system/$u"; then echo static
        else echo enabled; fi ;;
      show)
        # -p NRestarts --value UNIT: systemd's restarts of it since it was started
        if [[ " $* " == *" NRestarts "* ]]; then
          if [ "${*: -1}" = jetlink-server ] && [ -f "$state/crashing" ]; then echo 2; else echo 0; fi
        fi ;;
      list-unit-files)
        u=systemd-networkd-wait-online.service
        [ "$u" = "${*: -1}" ] && echo "$u enabled enabled" ;;
      # what the computer starts: the desktop (stock JetPack's) until set
      get-default) cat "$state/default-target" 2>/dev/null || echo graphical.target ;;
      set-default) echo "$2" >"$state/default-target" ;;
      mask) for u in "${@:2}"; do touch "$state/masked-$u"; done ;;
      unmask) for u in "${@:2}"; do rm -f "$state/masked-$u"; done ;;
    esac ;;

  journalctl)
    # the server's log, as it writes it, and systemd's lines about the unit
    if [ "${FAKE_SERVER_BROKEN:-0}" = 1 ]; then
      for n in 1 2 3; do
        echo "ERROR io.zoompilot.jetlink.main: no TensorRT 10: libnvinfer.so.10: cannot open shared object file"
        echo "jetlink-server.service: Scheduled restart job, restart counter is at $n."
      done
    else
      plan=0123456789abcdef.trt10.16.2.10-Orin-sm87.plan
      echo "INFO io.zoompilot.jetlink.main: backend trt 10.16.2.10 on Orin-sm87, cache /mnt/data/jetlink"
      # the model it ran last, which it starts loading before it serves;
      # FAKE_PRELOAD says how that goes: ready, failed, crash, or slow
      [ -z "${FAKE_PRELOAD:-}" ] || echo "INFO io.zoompilot.jetlink.engine: preloading the engine loaded last: $plan"
      if [ "${FAKE_SERVER_OLD:-0}" = 1 ]; then
        # a build from before the serving line, a comma on the bus that
        # nothing on it serves yet
        echo "WARNING io.zoompilot.jetlink.server: the comma's gadget is on the bus, but nothing on the comma is serving it yet"
      else
        echo "INFO io.zoompilot.jetlink.main: jetlink-server is serving"
        echo "WARNING io.zoompilot.jetlink.server: waiting for a jetlink gadget at 1209:0001"
      fi
      case "${FAKE_PRELOAD:-}" in
        ready) echo "INFO io.zoompilot.jetlink.engine: engine ready: $plan" ;;
        failed) echo "ERROR io.zoompilot.jetlink.engine: engine preparation failed: failed(\"artifact invalid and the model is not on disk\")" ;;
      esac
      if [ -f "$state/crashing" ]; then
        echo "jetlink-server.service: Main process exited, code=dumped, status=11/SEGV"
        echo "jetlink-server.service: Scheduled restart job, restart counter is at 1."
      fi
    fi ;;

  jetlink-server)
    # the server from a release tarball: bin/ in the version's directory
    home="$(dirname "$(dirname "$0")")"
    case "${1:-}" in
      --version) cat "$home/VERSION" ;;
      backends)
        if [ "${FAKE_GPU_BROKEN:-0}" = 1 ]; then
          echo "trt: not usable: no CUDA driver: libcuda.so.1: cannot open shared object file: No such file or directory"
          echo "ort: not usable: libonnxruntime.so: cannot open shared object file: No such file or directory"
          exit 1
        fi
        echo "trt: usable: TensorRT 10.16.2.10 on Orin-sm87"
        echo "ort: not usable: libonnxruntime.so: cannot open shared object file: No such file or directory" ;;
      web-password)
        # --file FILE [--generate]: the password from stdin's first line, or
        # one made and printed, as the real one makes it. The file holds it in
        # the clear, so a scenario can see which reached it, and a nonce, so
        # every write differs, as a new salt makes the real one's.
        file='' generate=0
        shift
        while [ $# -gt 0 ]; do
          case "$1" in
            --file) file=$2; shift ;;
            --generate) generate=1 ;;
          esac
          shift
        done
        [ -n "$file" ] || { echo "web-password needs --file" >&2; exit 1; }
        if [ "$generate" = 1 ]; then
          chars=23456789abcdefghjkmnpqrstuvwxyz pw=''
          for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
            pw="$pw${chars:RANDOM % ${#chars}:1}"
            [ $((i % 4)) != 0 ] || [ "$i" = 12 ] || pw="$pw-"
          done
        else
          IFS= read -r pw || [ -n "$pw" ] || { echo "no password on standard input" >&2; exit 1; }
          if [ ${#pw} -lt 8 ]; then echo "The password needs at least 8 characters." >&2; exit 1; fi
          if [ ${#pw} -gt 128 ]; then echo "The password can have at most 128 characters." >&2; exit 1; fi
        fi
        mkdir -p "$(dirname "$file")"
        (umask 077 && printf '{"fake_password": "%s", "nonce": "%s-%s"}\n' "$pw" "$RANDOM" "$RANDOM" >"$file.new")
        mv -f "$file.new" "$file"
        [ "$generate" = 0 ] || echo "$pw" ;;
      *) echo "serving" ;;
    esac ;;

  docker)
    case "${1:-}" in
      --version) echo "Docker version 29.1.0, build fake" ;;
      info)
        if [[ " $* " == *DockerRootDir* ]]; then
          echo "${FAKE_DOCKER_ROOT:-/var/lib/docker}"
        elif [ -f "$state/nvidia-runtime" ]; then
          echo '{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}'
        else echo '{"runc":{"path":"runc"}}'; fi ;;
      manifest) [ "${FAKE_PUBLISHED:-0}" = 1 ] || { echo "no such manifest" >&2; exit 1; } ;;
      pull) echo "$2" >>"$state/images" ;;
      build)
        prev=''
        for a in "$@"; do [ "$prev" = -t ] && echo "$a" >>"$state/images"; prev=$a; done ;;
      rmi)
        # each image deleted frees FAKE_IMAGE_GB on /; by repo:tag or by ID,
        # and repo:<none> is no reference at all, as Docker says
        shift
        rc=0
        for a in "$@"; do
          case "$a" in *"<none>"*)
            echo "Error response from daemon: invalid reference format" >&2
            rc=1
            continue ;;
          esac
          found=0
          : >"$state/images.new"
          while read -r ref id; do
            [ -n "$ref" ] || continue
            if [ "$ref" = "$a" ] || [ "$(fake_image_id "$ref" "$id")" = "$a" ]; then found=1; continue; fi
            echo "$ref${id:+ $id}" >>"$state/images.new"
          done < <(cat "$state/images" 2>/dev/null)
          mv "$state/images.new" "$state/images"
          [ "$found" = 1 ] || continue
          echo $(($(cat "$state/freed-gb" 2>/dev/null || echo 0) + ${FAKE_IMAGE_GB:-5})) >"$state/freed-gb"
        done
        exit "$rc" ;;
      rm|stop) ;;
      ps)
        # a container that outlived docker rm -f
        [ "${FAKE_DOCKER_STUCK:-0}" = 1 ] && echo 0123456789ab ;;
      image)
        case "${2:-}" in
          inspect) echo "sha256:$(printf '%064d' 7)" ;;
          ls)
            if [[ " $* " == *'{{.ID}}'* ]]; then fake_images
            else fake_images | awk '{ print $1 ":" $2 }'; fi ;;
        esac ;;
      run)
        # the Docker installers' GPU probe; every way in works
        case " $* " in
          *" --entrypoint python3 "*) echo "Orin (compute 8.7), TensorRT 10.16.2.10" ;;
        esac ;;
    esac ;;

  nvidia-ctk)
    case "${1:-} ${2:-}" in
      "runtime configure") touch "$state/nvidia-runtime" ;;
    esac ;;

  nvpmodel)
    case "${1:-}" in
      -q) echo "NV Power Mode: $(cat "$state/pm" 2>/dev/null || echo 15W)"; echo 0 ;;
      -m) read -r _ || true
          # the mode with that ID in nvpmodel.conf, unless it needs a restart
          if [ "${FAKE_PM_REBOOT:-0}" = 1 ]; then echo "reboot required"
          else sed -n "s/.*POWER_MODEL ID=$2 NAME=\([^ >]*\).*/\1/p" /etc/nvpmodel.conf >"$state/pm"; fi ;;
    esac ;;

  efibootmgr)
    # the firmware's boot menu wait in $state/uefi-timeout: JetPack's 5 s
    # until -t or -T changes it, and `none` for no Timeout variable.
    # FAKE_UEFI_LOCKED: a firmware that keeps its variables as they are
    if [ "${FAKE_UEFI_LOCKED:-0}" = 0 ]; then
      case "${1:-}" in
        -t) echo "$2" >"$state/uefi-timeout" ;;
        -T) echo none >"$state/uefi-timeout" ;;
      esac
    fi
    t="$(cat "$state/uefi-timeout" 2>/dev/null || echo 5)"
    echo "BootCurrent: 0001"
    [ "$t" = none ] || echo "Timeout: $t seconds"
    echo "BootOrder: 0001,0000"
    echo "Boot0000* Enter Setup"
    echo "Boot0001* UEFI PNY CS1030 1TB SSD" ;;

  nvidia-smi)
    [ -n "${FAKE_SMI:-}" ] || exit 9
    echo "$FAKE_SMI" ;;

  ubuntu-drivers|udevadm|fallocate|mkswap|swapon|swapoff|jetson_clocks) ;;

  curl)
    # the network the installer needs, answered from here; anything else fails
    out='' url=''
    while [ $# -gt 0 ]; do
      case "$1" in
        -o) out=$2; shift ;;
        http*) url=$1 ;;
      esac
      shift
    done
    case "$url" in
      https://github.com) ;;
      https://api.github.com/repos/zoompilot/jetlink/releases/latest)
        # the newest release, FAKE_LATEST; without one, GitHub's rate limit
        [ -n "${FAKE_LATEST:-}" ] || { echo "curl: (22) The requested URL returned error: 403" >&2; exit 22; }
        printf '{\n  "html_url": "https://github.com/zoompilot/jetlink/releases/tag/%s",\n  "tag_name": "%s",\n  "prerelease": false\n}\n' \
          "$FAKE_LATEST" "$FAKE_LATEST" ;;
      https://github.com/zoompilot/jetlink/releases/download/*)
        # the assets scenarios.sh made; the first FAKE_DOWNLOAD_FAILS server
        # downloads are cut off part way
        file="/tmp/releases/${url#*/releases/download/}"
        [ -f "$file" ] || { echo "curl: (22) The requested URL returned error: 404" >&2; exit 22; }
        case "$file" in
          *.tar.gz)
            if [ "${FAKE_DOWNLOAD_HANG:-0}" = 1 ]; then
              # until the scenario's signal cuts it off; the sleep is a backstop
              touch "$state/download-hanging"
              sleep 30
              exit 18
            fi
            left="$(cat "$state/download-fails" 2>/dev/null || echo "${FAKE_DOWNLOAD_FAILS:-0}")"
            if [ "$left" -gt 0 ]; then
              echo $((left - 1)) >"$state/download-fails"
              echo "curl: (18) transfer closed with outstanding read data remaining" >&2
              exit 18
            fi ;;
          *.sha256)
            if [ "${FAKE_BAD_SUM:-0}" = 1 ]; then
              printf '%064d  %s\n' 0 "${file##*/}" >"$out"
              exit 0
            fi ;;
        esac
        cp "$file" "$out" ;;
      https://pypi.nvidia.com/*.whl)
        # NVIDIA's wheel index: the stand-in scenarios.sh made, or a damaged one
        file="/tmp/pypi/${url##*/}"
        [ -f "$file" ] || { echo "curl: (22) The requested URL returned error: 404" >&2; exit 22; }
        if [ "${FAKE_BAD_WHEEL:-0}" = 1 ]; then echo "not the wheel" >"$out"; else cp "$file" "$out"; fi ;;
      *download.docker.com*|*nvidia.github.io*) echo "deb https://example.invalid/fake stable main" ;;
      *) echo "fake curl: no route for $url" >&2; exit 22 ;;
    esac ;;

  gpg) cat >/dev/null ;;

  *) echo "fake.sh: no stand-in for $name" >&2; exit 127 ;;
esac
exit 0
