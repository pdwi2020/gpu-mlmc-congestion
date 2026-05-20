# Reproducibility image for the GPU-MLMC congestion paper.
#
# CPU-only by default; works on any Linux/macOS Docker host. For GPU runs,
# use --gpus all and set ACCELERATOR=cuda. PyCUDA is NOT installed (driver-
# coupled); the framework auto-falls back to a PyTorch CUDA shim when
# pycuda is absent.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first for layer caching.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt && \
    pip install scapy dpkt

# Optional: install torch CPU build for the parallel_mc shim.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch || \
    echo "[warn] torch wheel install failed; proceed without GPU shim"

COPY . /workspace
RUN pip install -e . || echo "[warn] editable install failed; using sys.path"

ENV PYTHONPATH=/workspace/src:/workspace

CMD ["bash", "scripts/reproduce_all.sh"]
