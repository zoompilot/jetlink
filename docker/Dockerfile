# The server image for NVIDIA GPUs on CUDA 13, one Dockerfile for two platforms:
#
#   linux/amd64  laptops and desktops, NVIDIA driver 580 or newer
#   linux/arm64  Jetson Orin on JetPack 7.2 or newer
#
# JetPack 7.2 moved Orin onto the same arm64 (SBSA) CUDA stack Arm servers run,
# so one base image serves both. A Jetson still on JetPack 6 needs
# docker/Dockerfile.jetpack6. install.sh picks the right one.
#
# Only TensorRT comes from a different place on each:
#
#   amd64  PyPI's tensorrt-cu13 wheels. They carry their own copies of the CUDA
#          libraries they load, which is why the base is `-base` and not
#          `-runtime`: -runtime would add a second 2.3 GB copy nothing loads.
#   arm64  JetPack's own TensorRT build, from NVIDIA's Jetson apt repository.
#          TensorRT 11 does not support Jetson, and 10.16.2 is the build JetPack
#          7.2 ships, so this is the one known to run on Orin under its driver.
ARG CUDA_VERSION=13.2.1
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04

# Set by BuildKit. The legacy builder leaves it empty, and the install step
# falls back to the architecture dpkg reports.
ARG TARGETARCH

# amd64: pinned exactly because `tensorrt-cu13` on PyPI is a stub sdist whose
# setup.py fetches the real wheels while pip runs, so the version resolved here
# is the only record of what went in.
ARG TENSORRT_PIP_VERSION=11.3.0.99
# arm64: the JetPack release's repository and the TensorRT build in it.
ARG JETPACK_REPO=r39.2
ARG TENSORRT_JETPACK_VERSION=10.16.2.10-1+cuda13.2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    JETLINK_CACHE=/var/cache/jetlink \
    PATH=/opt/venv/bin:$PATH

# libusb-1.0-0 explicitly: pip's libusb1 is a ctypes wrapper and ships no
# library. Without it the server logs "could not open the gadget" forever,
# which reads as a cable fault. --system-site-packages because on arm64
# TensorRT's Python bindings are an apt package in the system interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-venv ca-certificates curl gnupg libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv --system-site-packages /opt/venv

# On its own, and before the source: this is the multi-gigabyte layer, and
# editing a Python file must not rebuild it.
RUN set -eux; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$arch" in \
      amd64) \
        python -m pip install --no-cache-dir "tensorrt-cu13==${TENSORRT_PIP_VERSION}" ;; \
      arm64) \
        curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
          | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson.gpg; \
        echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/common ${JETPACK_REPO} main" \
          > /etc/apt/sources.list.d/nvidia-jetson.list; \
        apt-get update; \
        v="${TENSORRT_JETPACK_VERSION}"; \
        apt-get install -y --no-install-recommends \
          python3-libnvinfer="$v" libnvinfer10="$v" libnvinfer-plugin10="$v" \
          libnvinfer-vc-plugin10="$v" libnvonnxparsers10="$v"; \
        rm -rf /var/lib/apt/lists/* ;; \
      *) \
        echo "no TensorRT for $arch: build on amd64 or on a Jetson" >&2; exit 1 ;; \
    esac

WORKDIR /opt/jetlink
COPY pyproject.toml README.md LICENSE ./
COPY jetlink ./jetlink
COPY scripts ./scripts
# TensorRT is already in place from the layer above; this adds its neighbours.
RUN python -m pip install --no-cache-dir ".[server-cu13,usb,nvml]"

EXPOSE 5599
ENTRYPOINT ["jetlink-server", "--backend", "trt"]
CMD ["--transport", "tcp"]
