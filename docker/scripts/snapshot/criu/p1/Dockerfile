# P1 image: the P0 vLLM+CRIU image plus the gcloud CLI (snapshot transfer to/from GCS) and the
# two-pod entrypoint. Build P0 first, or point BASE_IMAGE at the pushed P0 tag.
#
#   docker build --build-arg BASE_IMAGE=<registry>/vllm-criu:v0.30.0-criu4.3.1 \
#     -t <registry>/vllm-criu:v0.30.0-criu4.3.1-p1 -f docker/scripts/snapshot/criu/p1/Dockerfile docker/scripts/snapshot/criu
ARG BASE_IMAGE=vllm-criu:v0.30.0-criu4.3.1
FROM ${BASE_IMAGE}

# hadolint ignore=DL3008
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-transport-https gnupg \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends google-cloud-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Refresh the P0 driver too, so a P1 build picks up driver changes without rebuilding P0.
COPY p0/criu-snapshot.sh p0/criu-action.sh p1/criu-entrypoint.sh /usr/local/bin/
RUN chmod 0755 /usr/local/bin/criu-snapshot.sh /usr/local/bin/criu-action.sh /usr/local/bin/criu-entrypoint.sh \
    && gcloud --version | head -1

ENTRYPOINT ["/usr/local/bin/criu-entrypoint.sh"]
