#!/usr/bin/env bash
# One stand-in for every system command the installer touches, dispatched on
# the name it is called by (scenarios.sh symlinks each name to this file). Each
# call is appended to $FAKE_LOG, and the answers come from $FAKE_* variables,
# so a scenario can say "Docker is missing" or "only CDI reaches the GPU".
#
# Test code only: nothing here runs outside tests/installer.
set -u
name="$(basename "$0")"
state="${FAKE_STATE:-/tmp/fake-state}"
mkdir -p "$state"
printf '%s %s\n' "$name" "$*" >>"${FAKE_LOG:-/tmp/fake.log}"

case "$name" in
  uname)
    case "${1:-}" in
      -m) echo "${FAKE_ARCH:-aarch64}" ;;
      -s|'') echo Linux ;;
      *) /bin/uname "$@" ;;
    esac ;;

  apt-get)
    # installing Docker or the toolkit makes their commands appear
    for pkg in "$@"; do
      case "$pkg" in
        docker.io|docker-ce) ln -sf "$0" "$FAKE_BIN/docker" ;;
        nvidia-container-toolkit) ln -sf "$0" "$FAKE_BIN/nvidia-ctk" ;;
      esac
    done ;;

  systemctl)
    case "${1:-}" in
      is-active)
        quiet=0; [[ " $* " == *" --quiet "* ]] && quiet=1
        [ -f "$state/stopped-${*: -1}" ] && { [ "$quiet" = 1 ] || echo inactive; exit 3; }
        [ "$quiet" = 1 ] || echo active ;;
      stop) touch "$state/stopped-${*: -1}" ;;
      start|restart) rm -f "$state/stopped-${*: -1}" ;;
      is-enabled) if [ -f "$state/masked-${*: -1}" ]; then echo masked; else echo enabled; fi ;;
      list-unit-files)
        u=systemd-networkd-wait-online.service
        [ "$u" = "${*: -1}" ] && echo "$u enabled enabled" ;;
      mask) for u in "${@:2}"; do touch "$state/masked-$u"; done ;;
      unmask) for u in "${@:2}"; do rm -f "$state/masked-$u"; done ;;
      show)
        case " $* " in
          *" nv-install-docker.service "*)
            # JetPack's own Docker install: running for FAKE_NV_DOCKER_POLLS
            # looks, and Docker is there once it has finished
            left="$(cat "$state/nv-docker" 2>/dev/null || echo "${FAKE_NV_DOCKER_POLLS:-0}")"
            if [ "$left" -gt 0 ]; then
              echo $((left - 1)) >"$state/nv-docker"
              echo activating
            else
              [ -f "$state/nv-docker" ] && ln -sf "$0" "$FAKE_BIN/docker"
              echo inactive
            fi ;;
          *) echo "${FAKE_RESTARTS:-0}" ;;
        esac ;;
    esac ;;

  journalctl)
    if [ "${FAKE_SERVER_BROKEN:-0}" = 1 ]; then
      echo "ImportError: libnvinfer.so.10: cannot open shared object file"
    else
      echo "backend trt 10.16.2.10 on Orin-sm87, cache /var/cache/jetlink"
      echo "waiting for a jetlink gadget at 1209:0001"
    fi ;;

  docker)
    case "${1:-}" in
      --version) echo "Docker version 29.1.0, build fake" ;;
      info) if [ -f "$state/nvidia-runtime" ]; then
              echo '{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}'
            else echo '{"runc":{"path":"runc"}}'; fi ;;
      manifest)
        # the first FAKE_MANIFEST_HANGS requests hang on a dead connection
        left="$(cat "$state/manifest-hangs" 2>/dev/null || echo "${FAKE_MANIFEST_HANGS:-0}")"
        if [ "$left" -gt 0 ]; then
          echo $((left - 1)) >"$state/manifest-hangs"
          sleep 30
        fi
        [ "${FAKE_PUBLISHED:-0}" = 1 ] || { echo "no such manifest" >&2; exit 1; } ;;
      pull)
        # the first FAKE_PULL_FAILS pulls are cut off part way
        left="$(cat "$state/pull-fails" 2>/dev/null || echo "${FAKE_PULL_FAILS:-0}")"
        if [ "$left" -gt 0 ]; then
          echo $((left - 1)) >"$state/pull-fails"
          echo "failed to copy: failed to send write: EOF" >&2
          exit 1
        fi ;;
      build|rm|rmi|stop) ;;
      image)
        case "${2:-}" in
          inspect) echo "sha256:$(printf '%064d' 7)" ;;
          ls) echo "jetlink:local-cuda" ;;
        esac ;;
      run)
        case " $* " in
          *" -m jetlink.registry "*) echo "fake model list" ;;
          *" --entrypoint python3 "*)
            # the GPU probe: only the way this scenario says works, and a
            # CDI-only toolkit only once its device list has been generated
            ok="${FAKE_GPU_OK:---runtime nvidia --gpus all}"
            if [[ " $* " == *" --network host $ok --entrypoint "* ]] \
                && { [ "$ok" != "--device nvidia.com/gpu=all" ] || [ -f "$state/cdi" ]; }; then
              echo "Orin (compute 8.7), TensorRT 10.16.2.10"
              exit 0
            fi
            echo "could not select device driver" >&2
            exit 125 ;;
        esac ;;
    esac ;;

  nvidia-ctk)
    case "${1:-} ${2:-}" in
      "runtime configure") touch "$state/nvidia-runtime" ;;
      "cdi generate") touch "$state/cdi" ;;
    esac ;;

  nvpmodel)
    case "${1:-}" in
      -q) echo "NV Power Mode: $(cat "$state/pm" 2>/dev/null || echo 15W)"; echo 0 ;;
      -m) read -r _ || true
          if [ "${FAKE_PM_REBOOT:-0}" = 1 ]; then echo "reboot required"; else echo MAXN_SUPER >"$state/pm"; fi ;;
    esac ;;

  nvidia-smi)
    [ -n "${FAKE_SMI:-}" ] || exit 9
    echo "$FAKE_SMI" ;;

  ubuntu-drivers|udevadm|fallocate|mkswap|swapon|swapoff|jetson_clocks) ;;

  curl)
    # the network the installer needs, answered from here; anything else fails
    case " $* " in
      *" https://github.com "*) ;;
      *download.docker.com*|*nvidia.github.io*) echo "deb https://example.invalid/fake stable main" ;;
      *) echo "fake curl: no route for $*" >&2; exit 22 ;;
    esac ;;

  gpg) cat >/dev/null ;;

  *) echo "fake.sh: no stand-in for $name" >&2; exit 127 ;;
esac
exit 0
