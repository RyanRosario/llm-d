#!/usr/bin/env bash
# P1: two-pod CRIU snapshot entrypoint (container `command`).
#
# Pod 1 (no images for this KEY in the bucket): cold start vLLM, sleep, criu dump, collect the
# runtime files the images refer to, upload everything, wake, serve.
# Pod 2 (images exist): download, put the runtime files back, criu restore, wake, serve.
# Mirrors the gVisor guide's flow: replicas=1 creates the snapshot, scaling out restores from it.
#
# Required env: SNAPSHOT_BUCKET. Optional: MODEL, KEY_SALT, everything criu-snapshot.sh reads.
# The sibling CRIU P0 driver (criu-snapshot.sh) does the actual cold/dump/restore/wake steps.
set -euo pipefail

: "${SNAPSHOT_BUCKET:?SNAPSHOT_BUCKET (gs bucket name) is required}"
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export IMG_DIR="${IMG_DIR:-/checkpoints/images}"
export TIMINGS="${TIMINGS:-/checkpoints/timings.csv}"
export CRIU_VERBOSITY="${CRIU_VERBOSITY:-1}"
export SAVE_AUX=0  # P1 collects runtime files itself (collect_runtime_files), not the P0 tar

log() { echo "[$(date -u +%FT%T.%3NZ)] [criu-entrypoint] $*"; }
now() { date +%s.%N; }
phase() {
  local d; d=$(echo "$3 - $2" | bc)
  printf 'PHASE %-22s %8.3f s\n' "$1" "${d}"
  [[ -s "${TIMINGS}" ]] || echo "ts,phase,seconds" > "${TIMINGS}"
  printf '%s,%s,%.3f\n' "$(date -u +%FT%T.%3NZ)" "$1" "${d}" >> "${TIMINGS}"
}

# ---- snapshot key: same idea as GKE's distilled pod-spec hash -------------------------------
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | tr -d ' ')
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr -d ' ')
criu_ver=$(criu --version | head -1)
vllm_ver=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)
KEY=$(printf '%s|%s|%s|%s|%s|%s|%s|%s' "${MODEL}" "${MAX_MODEL_LEN:-8192}" "${GPU_MEM_UTIL:-0.90}" \
      "${VLLM_EXTRA_ARGS:-}" "${driver}" "${gpu}" "${criu_ver}" "${vllm_ver}${KEY_SALT:-}" | sha256sum | cut -c1-16)
STORE="gs://${SNAPSHOT_BUCKET}/criu/${KEY}"
log "key=${KEY} model=${MODEL} gpu=${gpu} driver=${driver} ${criu_ver} vllm=${vllm_ver}"
log "store=${STORE}"
mkdir -p "$(dirname "${IMG_DIR}")"

# Regular files the dumped tree has open or mapped under writable, pod-local paths. CRIU images
# do not carry file contents; restore reopens these by path, so a fresh pod needs them.
collect_runtime_files() {
  local pid list="${IMG_DIR}/runtime-files.txt"
  : > "${list}"
  for pid in $(criu-snapshot.sh _tree_pids); do
    { awk '{print $6}' "/proc/${pid}/maps" 2>/dev/null; readlink "/proc/${pid}"/fd/* 2>/dev/null; } \
      | grep -E '^/(tmp|dev/shm|root|checkpoints|var/tmp)/' | grep -v ' (deleted)$'
  done | sort -u | while read -r f; do [[ -f "${f}" && ! -S "${f}" ]] && echo "${f}"; done > "${list}"
  log "runtime files referenced by the tree: $(wc -l < "${list}")"
  tar -C / -cf "${IMG_DIR}/aux.tar" -T <(sed 's#^/##' "${list}") 2>/dev/null || true
  du -sh "${IMG_DIR}/aux.tar"
}

serve_forever() {
  local pid; pid=$(cat /checkpoints/vllm.pid)
  log "serving; supervising PID ${pid}"
  trap 'log "TERM received"; kill -TERM -- -"$(cat /checkpoints/vllm.pid)" 2>/dev/null; exit 0' TERM INT
  while kill -0 "${pid}" 2>/dev/null; do sleep 5; done
  log "vLLM exited"; exit 1
}

if gcloud storage ls "${STORE}/inventory.img" >/dev/null 2>&1; then
  # ------------------------------------------------------------------ RESTORE (pod 2+)
  log "snapshot found, restoring"
  t0=$(now)
  mkdir -p "${IMG_DIR}"
  gcloud storage rsync -r "${STORE}" "${IMG_DIR}" >/dev/null
  phase download "${t0}" "$(now)"
  du -sh "${IMG_DIR}"
  [[ -f "${IMG_DIR}/aux.tar" ]] && tar -C / -xf "${IMG_DIR}/aux.tar" && log "runtime files restored"
  # Log file must exist for fd 1/2 of the restored tree (opened O_APPEND, so size may differ).
  touch /checkpoints/vllm.log
  # Checkpoint GPU 0 -> this node's GPU 0 (UUIDs differ between nodes).
  export CRIU_RESTORE_ARGS="--plugin-option=cuda_plugin.device-map=auto ${CRIU_RESTORE_ARGS:-}"
  criu-snapshot.sh restore
  criu-snapshot.sh wake
  criu-snapshot.sh verify || true
  cat "${TIMINGS}"
  serve_forever
else
  # ------------------------------------------------------------------ COLD (pod 1)
  log "no snapshot for key ${KEY}, cold start"
  # Push PIDs high so the restored tree never collides with a fresh pod's early PIDs.
  echo 20000 > /proc/sys/kernel/ns_last_pid 2>/dev/null || log "WARN: could not set ns_last_pid"
  criu-snapshot.sh cold
  criu-snapshot.sh dump
  collect_runtime_files
  # Resume serving right away; upload in the background so pod 1's readiness is not delayed.
  criu-snapshot.sh wake
  criu-snapshot.sh verify || true
  (
    t0=$(now)
    log "uploading $(du -sh "${IMG_DIR}" | cut -f1) to ${STORE}"
    gcloud storage rsync -r "${IMG_DIR}" "${STORE}" >/dev/null 2>"${IMG_DIR}/upload.err" \
      && { phase upload "${t0}" "$(now)"; echo "Snapshot upload complete."; cat "${TIMINGS}"; } \
      || { log "ERROR: upload failed: $(tail -3 "${IMG_DIR}/upload.err")"; }
  ) &
  serve_forever
fi
