# jetlink inference server for Jetson (JetPack 6.1 / L4T r36.4).
#
# l4t-jetpack already carries CUDA 12.6, cuDNN and TensorRT 10.3. Do not
# "upgrade" the Python: its TensorRT bindings are built for 3.10 only.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libusb-1.0-0 explicitly: pip's libusb1 is a ctypes wrapper and ships no library.
# Without it the server logs "could not open the gadget" forever, reading as a cable fault.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      python3-dev \
      libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --ignore-installed: apt's python3-numpy 1.21.5 otherwise reports "already satisfied"
RUN pip3 install --no-cache-dir --ignore-installed \
      "numpy>=1.24,<2" \
      "onnx>=1.16" \
      "cuda-python>=12.6,<13" \
      "libusb1>=3.0"

WORKDIR /opt/jetlink
COPY jetlink /opt/jetlink/jetlink
COPY scripts /opt/jetlink/scripts
ENV PYTHONPATH=/opt/jetlink

# mount /mnt/data/jetlink here: the rootfs is 3.7 GB and a single plan is ~770 MB
ENV JETLINK_CACHE=/mnt/data/jetlink
VOLUME /mnt/data/jetlink

EXPOSE 5599
ENTRYPOINT ["python3", "-m", "jetlink.server.main"]
CMD ["--transport", "tcp"]
