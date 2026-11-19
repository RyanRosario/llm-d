# P0 feasibility image: upstream vLLM + CRIU (with the NVIDIA CUDA plugin) + cuda-checkpoint.
#
# Builds CRIU from source because the Ubuntu/PPA packages do not ship cuda_plugin.so.
# `make criu cuda_plugin` skips the man pages (asciidoc/xmlto) and the Python lib, which
# the pod does not need. PREFIX=/usr matches CR_PLUGIN_DEFAULT (/usr/lib/criu/) so no
# --libdir flag is needed at runtime.
#
#   docker build -t <registry>/vllm-criu:v0.30.0-criu4.3.1 docker/scripts/snapshot/criu/p0

ARG VLLM_IMAGE=docker.io/vllm/vllm-openai:v0.30.0
FROM ${VLLM_IMAGE}

ARG CRIU_VERSION=v4.3.1
# NVIDIA/cuda-checkpoint main as of 2026-11-18. The binary is a ~6 KiB shim that dlopens
# libcuda.so, so the real C/R implementation comes from the node's driver (>= 570 required).
ARG CUDA_CHECKPOINT_REF=00d5cce84c628088d6caa203fc4af40c1538b6f7

# hadolint ignore=DL3008
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git pkg-config curl ca-certificates \
        libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler python3-protobuf \
        libbsd-dev libcap-dev libnl-3-dev libnet1-dev libaio-dev libgnutls28-dev \
        libnftables-dev libdrm-dev uuid-dev \
        iproute2 nftables bc jq \
    && git clone --depth 1 --branch "${CRIU_VERSION}" https://github.com/checkpoint-restore/criu.git /tmp/criu \
    && make -C /tmp/criu -j"$(nproc)" criu cuda_plugin \
    && make -C /tmp/criu install-criu install-cuda_plugin PREFIX=/usr \
    && rm -rf /tmp/criu \
    && curl -fsSL -o /usr/local/bin/cuda-checkpoint \
        "https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/${CUDA_CHECKPOINT_REF}/bin/x86_64_Linux/cuda-checkpoint" \
    && chmod 0755 /usr/local/bin/cuda-checkpoint \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY criu-snapshot.sh criu-action.sh /usr/local/bin/
# cuda-checkpoint links libcuda.so.1, which only exists once the GPU driver is injected at
# runtime, so it cannot be executed here; just confirm it is an ELF binary.
RUN chmod 0755 /usr/local/bin/criu-snapshot.sh /usr/local/bin/criu-action.sh \
    && criu --version \
    && test -f /usr/lib/criu/cuda_plugin.so \
    && head -c 4 /usr/local/bin/cuda-checkpoint | grep -q ELF
