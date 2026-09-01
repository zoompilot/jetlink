# jetlink inference server for Jetson (JetPack 6.1 / L4T r36.4).
#
# Based on l4t-jetpack, which already carries CUDA 12.6, cuDNN and TensorRT
# 10.3 with Python 3.10 bindings. Do not "upgrade" to a newer Python: the
# TensorRT bindings in this image are built for 3.10 only, and the server
# deliberately has no tinygrad dependency, so 3.10 is enough.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

# --ignore-installed: apt's python3-numpy 1.21.5 is in /usr/lib/python3/dist-packages
# and pip otherwise reports "already satisfied" and leaves it in place.
RUN pip3 install --no-cache-dir --ignore-installed \
      "numpy>=1.24,<2" \
      "onnx>=1.16" \
      "cuda-python>=12.6,<13"

WORKDIR /opt/jetlink
COPY jetlink /opt/jetlink/jetlink
COPY scripts /opt/jetlink/scripts
ENV PYTHONPATH=/opt/jetlink

# Engine and model cache. Mount /mnt/data/jetlink here: the Jetson's rootfs is
# a 3.7 GB partition and a single plan is ~770 MB.
ENV JETLINK_CACHE=/mnt/data/jetlink
VOLUME /mnt/data/jetlink

EXPOSE 5599
ENTRYPOINT ["python3", "-m", "jetlink.server.main"]
CMD ["--transport", "tcp"]
