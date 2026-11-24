#!/usr/bin/env bash
# P0 feasibility: checkpoint and restore a slept vLLM server with CRIU + cuda-checkpoint
# inside one privileged pod, timing each phase.
#
# Usage: criu-snapshot.sh check|cold|dump|kill|restore|wake|verify|status|cycle
#
#   check    print driver/CRIU/plugin versions and run `criu check`
#   cold     start vLLM (setsid, detached) and wait for /health          -> cold_start_to_health
#   dump     POST /sleep?level=1, purge MODEL_CACHE_DIR, criu dump       -> sleep_l1, criu_dump
#            (--leave-running: the original server keeps serving; link remaps saved by criu-action.sh)
#   kill     SIGKILL the vLLM process tree (so restore can reuse its PIDs)
#   restore  criu restore from IMG_DIR                                    -> criu_restore
#   wake     POST /wake_up and wait for /health                           -> wake_up
#   verify   one /v1/completions request
#   status   show vLLM PIDs, /is_sleeping and GPU memory
#   cycle    cold, dump, wake, verify, kill, restore, wake, verify
#
# Knobs: MODEL (Qwen/Qwen3-8B), PORT (8000), IMG_DIR (/checkpoints/images), TIMINGS
#        (/checkpoints/timings.csv), MODEL_CACHE_DIR (unset = no purge), SAVE_AUX (0; tar
#        /dev/shm and /tmp next to the images, needed once restore happens in another pod),
#        VLLM_EXTRA_ARGS, CRIU_EXTRA_ARGS, CRIU_RESTORE_ARGS, CRIU_VERBOSITY (4), MAX_MODEL_LEN (8192),
#        GPU_MEM_UTIL (0.90).
#
# The echo'd marker lines deliberately match docker/scripts/snapshot/vllm/wrapper.py so
# guides/pod-snapshot/scripts/snapshot_timings.py (branch pod-snapshot-cicd) can parse a
# CRIU run without changes.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
IMG_DIR="${IMG_DIR:-/checkpoints/images}"
TIMINGS="${TIMINGS:-/checkpoints/timings.csv}"
SAVE_AUX="${SAVE_AUX:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VLLM_LOG="${VLLM_LOG:-/checkpoints/vllm.log}"
PIDFILE="${PIDFILE:-/checkpoints/vllm.pid}"
BASE="http://127.0.0.1:${PORT}"
ACTION_SCRIPT="${ACTION_SCRIPT:-/usr/local/bin/criu-action.sh}"
CRIU_VERBOSITY="${CRIU_VERBOSITY:-4}"  # -v4 writes ~8 MB per dump/restore log; use 1 for timing runs

now() { date +%s.%N; }
ts() { date -u +%FT%T.%3NZ; }
log() { echo "[$(ts)] [criu-snapshot] $*"; }
phase() { # phase <name> <start> <end>
  local d
  d=$(echo "$3 - $2" | bc)
  printf 'PHASE %-22s %8.3f s\n' "$1" "$d"
  mkdir -p "$(dirname "${TIMINGS}")"
  [[ -s "${TIMINGS}" ]] || echo "ts,phase,seconds" > "${TIMINGS}"
  printf '%s,%s,%.3f\n' "$(ts)" "$1" "$d" >> "${TIMINGS}"
}
api() { curl -fsS -X "$1" "${BASE}$2" "${@:3}"; }
wait_health() { # wait_health <timeout_s>
  local deadline=$(( $(date +%s) + ${1:-1800} ))
  until curl -fsS "${BASE}/health" >/dev/null 2>&1; do
    (( $(date +%s) < deadline )) || { echo "ERROR: /health not ready after ${1:-1800}s" >&2; return 1; }
    sleep 1
  done
}
vllm_pid() { [[ -f "${PIDFILE}" ]] && cat "${PIDFILE}"; }
tree_pids() { # all PIDs in the vLLM session (setsid makes the root a session leader)
  local root; root=$(vllm_pid) || return 0
  ps -eo pid=,sid= | awk -v s="${root}" '$2 == s { print $1 }'
}

cmd_check() {
  log "node driver / GPU:"; nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
  log "cuda-checkpoint: $(cuda-checkpoint --help 2>&1 | head -1)"
  log "criu: $(criu --version | head -1)"
  ls -l /usr/lib/criu/cuda_plugin.so
  log "kernel: $(uname -r); io_uring_disabled=$(cat /proc/sys/kernel/io_uring_disabled 2>/dev/null || echo n/a)"
  log "capabilities: $(grep CapEff /proc/self/status)"
  log "criu check:"; criu check --all || log "WARN: criu check reported problems (see above)"
}

cmd_cold() {
  if pid=$(vllm_pid) && kill -0 "${pid}" 2>/dev/null; then
    log "vLLM already running as PID ${pid}"; return 0
  fi
  # O_APPEND matters: CRIU skips the size check on restore only for O_WRONLY|O_APPEND files,
  # and the log keeps growing after a --leave-running dump.
  mkdir -p "$(dirname "${VLLM_LOG}")"; : > "${VLLM_LOG}"
  log "starting vLLM (${MODEL}) on :${PORT}"
  local t0; t0=$(now)
  # UV_USE_IO_URING=0: uvloop's bundled libuv (>= 1.45) opens an io_uring ring in the API
  # server, which CRIU cannot dump ("Unknown shit 600 (anon_inode:[io_uring])").
  # shellcheck disable=SC2086
  setsid env VLLM_SERVER_DEV_MODE=1 VLLM_HOST_IP=127.0.0.1 PYTHONUNBUFFERED=1 UV_USE_IO_URING=0 \
    vllm serve "${MODEL}" --port "${PORT}" --enable-sleep-mode \
      --safetensors-load-strategy eager \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" --max-model-len "${MAX_MODEL_LEN}" \
      --disable-access-log-for-endpoints=/health,/metrics,/v1/models \
      ${VLLM_EXTRA_ARGS:-} \
      < /dev/null >> "${VLLM_LOG}" 2>&1 &
  echo $! > "${PIDFILE}"
  wait_health 1800
  phase cold_start_to_health "${t0}" "$(now)"
  log "vLLM ready, PID $(vllm_pid); tree: $(tree_pids | tr '\n' ' ')"
}

cmd_dump() {
  local pid; pid=$(vllm_pid) || { echo "ERROR: no PIDFILE; run cold first" >&2; exit 1; }
  kill -0 "${pid}" || { echo "ERROR: PID ${pid} is not running" >&2; exit 1; }

  echo "Executing engine.sleep(level=1) to release physical VRAM..."
  local t0; t0=$(now)
  api POST "/sleep?level=1"
  until [[ "$(api GET /is_sleeping)" == *true* ]]; do sleep 0.5; done
  phase sleep_l1 "${t0}" "$(now)"
  nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory used after sleep: /'

  if [[ -n "${MODEL_CACHE_DIR:-}" ]]; then
    local d; d=$(eval echo "${MODEL_CACHE_DIR}")
    log "purging ${d}"; rm -rf "${d:?}"/*
  fi
  if [[ "${SAVE_AUX}" == "1" ]]; then
    mkdir -p "${IMG_DIR}"
    tar -C / -cf "${IMG_DIR}/aux.tar" dev/shm tmp 2>/dev/null || true
    du -sh "${IMG_DIR}/aux.tar"
  fi

  rm -rf "${IMG_DIR}"; mkdir -p "${IMG_DIR}"
  echo "Triggering snapshot checkpoint..."
  local t1; t1=$(now)
  # --leave-running is required: only on this path does the CUDA plugin restore/unlock the
  # GPU contexts after the dump (with --leave-stopped they stay "checkpointed" and the first
  # CUDA call after SIGCONT blocks forever). The cost is that CRIU then deletes the
  # link_remap.* files it made for unlinked-but-mapped files (Python's multiprocessing
  # semaphore in /dev/shm), which restore needs, so the post-dump action script saves them.
  # shellcheck disable=SC2086
  IMG_DIR="${IMG_DIR}" criu dump --tree "${pid}" --images-dir "${IMG_DIR}" \
    --leave-running --tcp-close --file-locks --link-remap \
    --manage-cgroups=ignore --timeout 120 \
    --action-script "${ACTION_SCRIPT}" \
    --log-file dump.log -v"${CRIU_VERBOSITY}" ${CRIU_EXTRA_ARGS:-} \
    || { echo "ERROR: criu dump failed; tail of ${IMG_DIR}/dump.log:" >&2; tail -40 "${IMG_DIR}/dump.log" >&2; exit 1; }
  phase criu_dump "${t1}" "$(now)"
  echo "Snapshot checkpoint created successfully."
  du -sh "${IMG_DIR}"; du -sh "${IMG_DIR}"/pages-*.img 2>/dev/null | sort -h | tail -5 || true
}

cmd_kill() {
  local pids; pids=$(tree_pids | tr '\n' ' ')
  [[ -n "${pids}" ]] || { log "nothing to kill"; return 0; }
  log "killing vLLM tree: ${pids}"
  # shellcheck disable=SC2086
  kill -9 ${pids} 2>/dev/null || true
  for _ in $(seq 1 30); do [[ -z "$(tree_pids)" ]] && break; sleep 0.5; done
  # Leftover ZMQ ipc socket files would make bind() fail on restore.
  find /tmp -maxdepth 1 -type s -delete 2>/dev/null || true
  log "remaining tree: '$(tree_pids | tr '\n' ' ')'"
}

cmd_restore() {
  [[ -f "${IMG_DIR}/inventory.img" ]] || { echo "ERROR: no images in ${IMG_DIR}" >&2; exit 1; }
  [[ -z "$(tree_pids)" ]] || { echo "ERROR: vLLM tree still running; run kill first" >&2; exit 1; }
  if [[ "${SAVE_AUX}" == "1" && -f "${IMG_DIR}/aux.tar" ]]; then
    tar -C / -xf "${IMG_DIR}/aux.tar" 2>/dev/null || true
  fi
  # Put back the link_remap.* files saved by the post-dump action script (CRIU unlinks them
  # again after a successful restore, so a set of images is single-use unless re-copied).
  if [[ -d "${IMG_DIR}/remaps" ]]; then
    cp -a "${IMG_DIR}/remaps/." / && log "restored $(find "${IMG_DIR}/remaps" -type f | wc -l) link remap(s)"
  fi
  log "criu restore from ${IMG_DIR}"
  rm -f "${PIDFILE}"  # CRIU opens --pidfile O_EXCL and fails the whole restore if it exists
  local t0; t0=$(now)
  # shellcheck disable=SC2086
  criu restore --images-dir "${IMG_DIR}" --restore-detached \
    --tcp-close --file-locks --link-remap --manage-cgroups=ignore \
    --pidfile "${PIDFILE}" --log-file restore.log -v"${CRIU_VERBOSITY}" ${CRIU_EXTRA_ARGS:-} ${CRIU_RESTORE_ARGS:-} \
    || { echo "ERROR: criu restore failed; tail of ${IMG_DIR}/restore.log:" >&2; tail -40 "${IMG_DIR}/restore.log" >&2; exit 1; }
  phase criu_restore "${t0}" "$(now)"
  echo "Process restored from snapshot checkpoint. Resuming engine..."
  log "restored PID $(vllm_pid); tree: $(tree_pids | tr '\n' ' ')"
}

cmd_wake() {
  echo "Executing engine.wake_up() to restore VRAM..."
  local t0; t0=$(now)
  api POST /wake_up
  until [[ "$(api GET /is_sleeping)" == *false* ]]; do sleep 0.5; done
  wait_health 300
  phase wake_up "${t0}" "$(now)"
  nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory used after wake: /'
}

cmd_verify() {
  local t0; t0=$(now)
  api POST /v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"How are you today?\",\"max_tokens\":16,\"temperature\":0}" \
    | jq -c '{text: .choices[0].text, usage: .usage}'
  phase first_completion "${t0}" "$(now)"
}

cmd_status() {
  log "PIDFILE=$(vllm_pid 2>/dev/null || echo none); tree: $(tree_pids | tr '\n' ' ')"
  curl -fsS "${BASE}/is_sleeping" 2>/dev/null || echo "(server not answering)"
  nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=memory.used,memory.total --format=csv,noheader
  [[ -f "${TIMINGS}" ]] && cat "${TIMINGS}"
}

cmd_cycle() {
  # The server is asleep after dump and blocks requests until woken, so wake before verify.
  cmd_cold; cmd_dump; cmd_wake; cmd_verify; cmd_kill; cmd_restore; cmd_wake; cmd_verify
  echo; log "cycle complete"; cat "${TIMINGS}"
}

case "${1:-}" in
  check|cold|dump|kill|restore|wake|verify|status|cycle) "cmd_$1" ;;
  _tree_pids) tree_pids ;;  # used by the P1 entrypoint
  *) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
