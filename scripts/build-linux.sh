#!/usr/bin/env bash
# Builds jetlink for Linux: the server tarball the installer takes, and the
# TensorRT shim against the real headers. One entry point for CI (plain
# ubuntu-22.04 and ubuntu-22.04-arm runners), local Docker or podman, and a
# device.
#
#   scripts/build-linux.sh [--container] <linux-aarch64|linux-x86_64> [step...]
#
# Steps, run in the order given (default: server):
#   headers    fetch and unpack the flavor's pinned TensorRT + CUDA headers
#   ort        fetch onnxruntime's pinned CPU tarball for the tests and print
#              its directory: include/ for -Xcc -I, lib/ for LD_LIBRARY_PATH
#   shim       compile CTrt/jl_trt.cpp against them, and check it links
#              nothing of NVIDIA's (everything is dlopened at run time)
#   server     dist/jetlink-server-<version>-<flavor>.tar.gz and .sha256
#
# linux-aarch64 is TensorRT 10 for the Jetson, compiled against 10.3 (JetPack
# 6.2's, the oldest it runs on); linux-x86_64 is TensorRT 11.3 for PCs.
#
# The version is jetlink/__init__.py's __version__ when HEAD is tagged
# v<__version__>, else <__version__>-dev.<short sha>; $JETLINK_VERSION wins.
#
# A step runs where its tools are: `server` in $JETLINK_SWIFT_IMAGE
# (swift:6.3.3-jammy, CI's) unless already inside it, the C and C++ steps in
# ubuntu:22.04 unless this is a Linux host of the flavor's architecture.
# --container sends every step but the fetches to a container. Downloads are
# cached in $JETLINK_BUILD_CACHE, by default ${XDG_CACHE_HOME:-~/.cache}/jetlink-build:
# NVIDIA's headers never enter the repo. Objects go to build/, tarballs to dist/.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE=${JETLINK_BUILD_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/jetlink-build}
CTRT=$ROOT/JetlinkKit/Sources/CTrt
SWIFT_IMAGE=${JETLINK_SWIFT_IMAGE:-swift:6.3.3-jammy}
# 6.3.3 from swift:6.3.3-jammy
SWIFT_VERSION=${SWIFT_IMAGE#*:}
SWIFT_VERSION=${SWIFT_VERSION%%-*}
C_BASE=ubuntu:22.04
C_IMAGE=jetlink-build-c:22.04

die() {
  echo "build-linux: $*" >&2
  exit 1
}

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

# --- the header bundles ------------------------------------------------------------
# Only headers: TensorRT's, its ONNX parser's, and the CUDA runtime and driver
# API headers TensorRT's include (cuda.h lives in cudart-dev; cuda-driver-dev
# holds only a stub library). Pinned by URL and sha256.

JETSON=https://repo.download.nvidia.com/jetson/common/pool/main
CUDA_X86=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
# the release fixture-pins.txt names, as everything else does
ORT_VERSION=$(sed -n 's/^onnxruntime==//p' "$ROOT/JetlinkKit/Scripts/fixture-pins.txt")
# jetlink's own copy: microsoft/onnxruntime and its release downloads went 404
# on 2026-11-25. The headers and libraries of the NuGet package
# Microsoft.ML.OnnxRuntime, repacked in the layout of the official tarball.
ORT_RELEASES=https://github.com/zoompilot/jetlink/releases/download/onnxruntime-$ORT_VERSION

# Sets BUNDLE (its cache directory's name) and DEBS ("url sha256" each), and
# ORT: onnxruntime's tarball ("url sha256"), whose C headers COrt
# compiles against; the server opens the library at run time.
pins() {
  case $FLAVOR in
  aarch64)
    ORT="$ORT_RELEASES/onnxruntime-linux-aarch64-$ORT_VERSION.tgz 648e7019bf746e86105c2cf934673c4bb5e2f6d4dc2ecc8367159bd447fed232"
    BUNDLE=trt10.3.0.30-cuda12.6-aarch64
    DEBS=(
      "$JETSON/t/tensorrt/libnvinfer-headers-dev_10.3.0.30-1+cuda12.5_arm64.deb 40a4fa566218f71176144a0eafa5aef8cf0af7ee9211b20487f1970d5344bbb8"
      "$JETSON/t/tensorrt/libnvonnxparsers-dev_10.3.0.30-1+cuda12.5_arm64.deb 8a6675f3d444dead0e0430fb4e7e6b546af9d9a72414ee8964b18edf1fb28756"
      "$JETSON/c/cuda-cudart/cuda-cudart-dev-12-6_12.6.68-1_arm64.deb 2ddcaeec93f2c508533f52bea7ad3f339d223d8724537e4695cf14a034b65193"
      "$JETSON/c/cuda-nvcc/cuda-crt-12-6_12.6.68-1_arm64.deb b95d26f0e63f113e6e2f5b8578841644de549293679d65c70e112164d2080311"
    )
    ;;
  x86_64)
    ORT="$ORT_RELEASES/onnxruntime-linux-x64-$ORT_VERSION.tgz 8378a201a9081f5f203242589f8897aa4bcfc0f5f1da05253b682f80ac4cb884"
    # 11.x keeps NvOnnxParser.h in libnvonnxparsers-dev, not the headers package
    BUNDLE=trt11.3.0.99-cuda13.4-x86_64
    DEBS=(
      "$CUDA_X86/libnvinfer-headers-dev_11.3.0.99-1+cuda13.4_amd64.deb 6383e49c3753e6ed8d3bdb95c0c6607d92d259b4e923407e817b5b4e0c483324"
      "$CUDA_X86/libnvonnxparsers-dev_11.3.0.99-1+cuda13.4_amd64.deb dc8c998f95b3fd9deae65d353650562fa12fefa9f43267facaa9aa057384b0ec"
      "$CUDA_X86/cuda-cudart-dev-13-4_13.4.92-1_amd64.deb 71bf3d001f2560a4710bdaba9297fcd3f775a50e3b320ec1c5fcc880d458b0df"
      "$CUDA_X86/cuda-crt-13-4_13.4.92-1_amd64.deb ec05400c48dafc7d2186f34c3b000596513895aa8d3c2d84446385c6d51f6d49"
    )
    ;;
  esac
}

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# $1, a URL, into $CACHE/debs unless it is there already, held to its pinned
# sha256 $2. Prints where it is.
fetch_pinned() {
  local file
  file=$CACHE/debs/$(basename "$1")
  mkdir -p "$CACHE/debs"
  if [[ ! -f $file || $(sha256 "$file") != "$2" ]]; then
    echo "headers: fetching $(basename "$1")" >&2
    curl -fsSL --retry 3 -o "$file.part" "$1"
    mv "$file.part" "$file"
  fi
  [[ $(sha256 "$file") == "$2" ]] || die "$(basename "$1") does not match its pinned sha256"
  echo "$file"
}

# A deb's files into $2: dpkg-deb where there is one, else bsdtar (macOS),
# which reads the ar archive a deb is.
unpack_deb() {
  if command -v dpkg-deb >/dev/null; then
    dpkg-deb -x "$1" "$2"
  else
    tar -xOf "$1" 'data.tar.*' | tar -xJf - -C "$2"
  fi
}

ort_dir() {
  echo "$CACHE/$(basename "${ORT% *}" .tgz)"
}

# onnxruntime's headers under include/onnxruntime/, where COrt looks for them,
# and its library under lib/, which the tests open.
ort_headers() {
  local url=${ORT% *} file dir
  dir=$(ort_dir)
  [[ -f $dir/.complete && $(cat "$dir/.complete") == "$ORT lib" ]] && return
  file=$(fetch_pinned "$url" "${ORT#* }")
  rm -rf "$dir"
  mkdir -p "$dir/include/onnxruntime"
  tar -xzf "$file" -C "$dir/include/onnxruntime" --strip-components 2 "$(basename "$url" .tgz)/include"
  tar -xzf "$file" -C "$dir" --strip-components 1 "$(basename "$url" .tgz)/lib"
  echo "$ORT lib" >"$dir/.complete"
}

step_ort() {
  ort_headers
  ort_dir
}

step_headers() {
  local dir=$CACHE/$BUNDLE stamp entry file
  ort_headers
  stamp=$(printf '%s\n' "${DEBS[@]}")
  if [[ -f $dir/.complete && $(cat "$dir/.complete") == "$stamp" ]]; then
    echo "headers: $dir/include"
    return
  fi
  rm -rf "$dir" "$dir.tmp"
  mkdir -p "$dir.tmp" "$dir/include"
  for entry in "${DEBS[@]}"; do
    file=$(fetch_pinned "${entry% *}" "${entry#* }")
    unpack_deb "$file" "$dir.tmp"
  done
  # one flat include dir: TensorRT's from usr/include/<triplet>, CUDA's from
  # usr/local/cuda-*/targets/<triplet>/include
  cp "$dir.tmp"/usr/include/*-linux-gnu/*.h "$dir/include/"
  cp -R "$dir.tmp"/usr/local/cuda-*/targets/*/include/. "$dir/include/"
  rm -rf "$dir.tmp"
  [[ -f $dir/include/NvInfer.h && -f $dir/include/NvOnnxParser.h && -f $dir/include/cuda.h &&
    -f $dir/include/cuda_runtime_api.h ]] || die "the $FLAVOR bundle is missing a header"
  echo "$stamp" >"$dir/.complete"
  echo "headers: $dir/include"
}

# --- the shim -----------------------------------------------------------------------

step_shim() {
  local out=$ROOT/build/linux-$FLAVOR inc=$CACHE/$BUNDLE/include cxx=${CXX:-g++}
  [[ -f $CACHE/$BUNDLE/.complete ]] || step_headers
  mkdir -p "$out"
  echo "shim: $cxx $($cxx -dumpversion) against $BUNDLE"
  "$cxx" -std=c++17 -O2 -fPIC -Wall -Wextra -Werror -I"$CTRT/include" -isystem "$inc" \
    -c "$CTRT/jl_trt.cpp" -o "$out/jl_trt.o"
  # Everything of NVIDIA's comes through dlopen: a reference here would make
  # the binary need TensorRT or CUDA just to start.
  local leaks
  leaks=$(nm -u "$out/jl_trt.o" | awk '{print $NF}' | grep -E '^cu[A-Z]|nvinfer|nvonnx|createInfer|getInferLib|initLibNvInfer' || true)
  [[ -z $leaks ]] || die "jl_trt.o references NVIDIA symbols directly: $leaks"
  echo "shim: $out/jl_trt.o, no NVIDIA symbol referenced"
}

# --- the server ---------------------------------------------------------------------

version() {
  if [[ -n ${JETLINK_VERSION:-} ]]; then
    echo "$JETLINK_VERSION"
    return
  fi
  local base tag sha
  base=$(sed -n "s/^__version__ = '\(.*\)'$/\1/p" "$ROOT/jetlink/__init__.py")
  [[ -n $base ]] || die "no __version__ in jetlink/__init__.py"
  tag=$(git -C "$ROOT" -c safe.directory='*' describe --exact-match --tags HEAD 2>/dev/null || true)
  if [[ $tag == "v$base" ]]; then
    echo "$base"
    return
  fi
  sha=$(git -C "$ROOT" -c safe.directory='*' rev-parse --short HEAD 2>/dev/null) ||
    die "no git here to name the build: set JETLINK_VERSION"
  echo "$base-dev.$sha"
}

step_server() {
  [[ -f $CACHE/$BUNDLE/.complete ]] || step_headers
  local version name stage bin scratch=$ROOT/build/swift-linux-$FLAVOR ort
  ort=$(ort_dir)
  ort_headers
  version=$(version)
  name=jetlink-server-$version-linux-$FLAVOR
  # its own scratch path, so a Mac's JetlinkKit/.build is never touched
  JETLINK_TENSORRT=$CACHE/$BUNDLE/include swift build --package-path "$ROOT/JetlinkKit" --scratch-path "$scratch" \
    -c release --static-swift-stdlib --product jetlink-server -Xcc -I"$ort/include"
  # SwiftPM keeps release/ as a link to the configuration's bin directory
  bin=$scratch/release
  stage=$ROOT/dist/$name
  rm -rf "$stage"
  mkdir -p "$stage/bin" "$stage/share/jetlink/systemd" "$stage/share/jetlink/udev"
  # stripped: the symbol table is a third of the binary
  strip -o "$stage/bin/jetlink-server" "$bin/jetlink-server"
  # SwiftPM looks for a target's resources in a bundle beside the executable
  cp -R "$bin"/*.resources "$stage/bin/"
  # the installer's own unit and rules, as they are in this tree
  cp "$ROOT/scripts/jetlink-server.service" "$stage/share/jetlink/systemd/"
  cp "$ROOT"/scripts/99-jetlink-*.rules "$stage/share/jetlink/udev/"
  cp "$ROOT/LICENSE" "$stage/"
  echo "$version" >"$stage/VERSION"
  # no onnxruntime: a TensorRT host needs none
  tar -C "$ROOT/dist" -czf "$ROOT/dist/$name.tar.gz" "$name"
  (cd "$ROOT/dist" && echo "$(sha256 "$name.tar.gz")  $name.tar.gz" >"$name.tar.gz.sha256")
  rm -rf "$stage"
  echo "server: dist/$name.tar.gz"
}

# --- where each step runs -------------------------------------------------------------

in_swift_container() {
  [[ ${JETLINK_IN_CONTAINER:-} == 1 ]] && return 0
  [[ -f /.dockerenv || -f /run/.containerenv ]] && command -v swift >/dev/null &&
    swift --version 2>/dev/null | grep -q "Swift version $SWIFT_VERSION"
}

native_c() {
  local arch
  arch=$(uname -m)
  [[ ${JETLINK_IN_CONTAINER:-} == 1 ]] ||
    { [[ $(uname -s) == Linux && ${arch/arm64/aarch64} == "$FLAVOR" ]] && command -v "${CXX:-g++}" >/dev/null; }
}

# The image a step needs, or "" to run it here.
image_for() {
  case $1 in
  headers | ort) echo "" ;;
  server) in_swift_container && echo "" || echo "$SWIFT_IMAGE" ;;
  *) [[ $FORCE_CONTAINER == 0 ]] && native_c && echo "" || echo "$C_IMAGE" ;;
  esac
}

container() {
  local image=$1 engine platform=linux/arm64 swift=()
  shift
  engine=$(command -v docker || command -v podman) || die "$* needs docker or podman, or a matching host"
  [[ $FLAVOR == x86_64 ]] && platform=linux/amd64
  if [[ $image == "$C_IMAGE" ]]; then
    # the compilers installed once, not at every run
    image=$C_IMAGE-$FLAVOR
    if ! "$engine" image inspect "$image" >/dev/null 2>&1; then
      printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends g++ binutils >/dev/null && rm -rf /var/lib/apt/lists/*\n' \
        "$C_BASE" | "$engine" build -q --platform "$platform" -t "$image" - >/dev/null
    fi
  else
    # Swift needs no packages, so it runs as the caller and leaves files the
    # caller owns; the version comes from here, where git is
    swift=(--user "$(id -u):$(id -g)" -e HOME=/tmp/home -e JETLINK_VERSION="$(version)")
  fi
  mkdir -p "$CACHE"
  # ${a[@]+...}: bash 3.2, the Mac's, calls an empty array unbound
  "$engine" run --rm --platform "$platform" ${swift[@]+"${swift[@]}"} -v "$ROOT:/src" -v "$CACHE:/cache" \
    -e JETLINK_BUILD_CACHE=/cache -e JETLINK_IN_CONTAINER=1 -w /src "$image" \
    scripts/build-linux.sh "linux-$FLAVOR" "$@"
}

FORCE_CONTAINER=0
if [[ ${1:-} == --container ]]; then
  FORCE_CONTAINER=1
  shift
fi
case ${1:-} in
linux-aarch64 | aarch64) FLAVOR=aarch64 ;;
linux-x86_64 | x86_64) FLAVOR=x86_64 ;;
*) usage ;;
esac
shift
pins
STEPS=("$@")
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=(server)
for step in "${STEPS[@]}"; do
  case $step in
  headers | ort | shim | server) ;;
  *) usage ;;
  esac
done
# Consecutive steps for the same container go in one run of it.
group=()
group_image=
flush() {
  if [[ ${#group[@]} -gt 0 ]]; then
    container "$group_image" "${group[@]}"
  fi
  group=()
}
for step in "${STEPS[@]}"; do
  image=$(image_for "$step")
  if [[ -z $image ]]; then
    flush
    "step_${step//-/_}"
    continue
  fi
  # the headers come down here, where curl is, before the container needs them
  step_headers >/dev/null
  [[ $image == "$group_image" ]] || flush
  group_image=$image
  group+=("$step")
done
flush
