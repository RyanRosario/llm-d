# CRIU P0: can CRIU + cuda-checkpoint snapshot a slept vLLM on a GKE runc node?

Feasibility spike for the platform-agnostic snapshot provider promised in the
[Pod Snapshots guide](../../../../../guides/pod-snapshot/README.md). Everything happens inside **one
privileged pod** on a plain (non-gVisor) GPU node: cold start, `sleep(level=1)`, `criu dump`, kill,
`criu restore`, `wake_up`, serve. No storage plumbing, no llm-d code changes. Cross-pod restore and the
GCS transfer (the numbers comparable to the gVisor benchmark) are P1.

## Files

- `Dockerfile` — upstream `vllm/vllm-openai` + CRIU 4.3.1 built with `cuda_plugin.so` + NVIDIA `cuda-checkpoint`.
- `criu-snapshot.sh` — phase driver (`check|cold|dump|kill|restore|wake|verify|status|cycle`); writes `/checkpoints/timings.csv`.
- `criu-action.sh` — CRIU `--action-script`; saves the `link_remap.*` files on `post-dump` (see findings).
- `pod.yaml` — privileged pod pinned to the `a100-pool` node pool of the `fma-e2e` cluster.

## Result (2026-11-19): feasible

**This experiment proved feasibility, but the measures here are not yet aligned with the gVisor implementation.**

`Qwen/Qwen3-8B`, vLLM `v0.30.0`, 1× A100-SXM4-40GB (`a2-highgpu-2g`, Spot), GKE `fma-e2e` runc node:
COS kernel `6.12.110+`, containerd `2.2.7`, driver `580.178.04`, CRIU `4.3.1`. Same pod, `emptyDir` images.
Restored server returns the same completion as the original.

| Phase | Cold (creates images) | Restore from images |
| :--- | ---: | ---: |
| Start → `/health` (weights already on disk; 274.9s on the first run that also downloaded them) | 88.0s | — |
| `sleep(level=1)` — 34.8 GiB freed, 15.4 GiB backed up to host RAM, 1.1 GiB left on GPU | 11.3s | — |
| **`criu dump`** (3 processes, GPU contexts via `cuda_plugin`; 25 GB of images) | 85.7s | — |
| **`criu restore`** | — | **8.3s** |
| `wake_up` (weights back to VRAM, 36.8 GiB used) | 2.7s | **2.6s** |
| First `/v1/completions` | 0.25s | 0.37s |
| **Images → serving** | | **≈ 11s** (vs 88s warm cold start, 275s true cold start) |

`criu dump` and `criu restore` are the phases that differ in mechanism from gVisor; the cross-node
total that compares with the guide's 18.0s (image pull, GCS transfer, container start) is P1.

## Findings (each one is a line in the future `CRIUSnapshotProvider`)

| # | Symptom | Cause | Fix in this directory |
| :--- | :--- | :--- | :--- |
| 1 | `dump`: `Unknown shit 600 (anon_inode:[io_uring])` | uvloop's bundled libuv (≥ 1.45) opens an io_uring in the API server | `UV_USE_IO_URING=0` on the vLLM process |
| 2 | `restore`: `File checkpoints/vllm.log has bad size` | stdout log grew after the dump; CRIU skips the size check only for `O_WRONLY\|O_APPEND` files | open the log with `>>` |
| 3 | `restore`: `Can't link dev/shm/link_remap.N -> dev/shm/sem.*` | `--leave-running` deletes CRIU's link remaps (unlinked-but-mapped Python `multiprocessing` semaphore) right after the dump | `criu-action.sh` copies them to `images/remaps/` on `post-dump`; `restore` copies them back. They are consumed by a successful restore, so re-copy every time |
| 4 | `wake_up` hangs forever after a `--leave-stopped` dump | the CUDA plugin leaves the processes in state `checkpointed`; it only restores/unlocks them on the `--leave-running` path | use `--leave-running` (+ #3). `cuda-checkpoint --action restore` then `--action unlock --pid <pid>` rescues a stuck process |
| 5 | `restore`: `pidfile: Can't open ... File exists` after the whole tree was already restored | CRIU opens `--pidfile` with `O_EXCL` | `rm -f` the pidfile before `restore` |
| 6 | PID 1 `sleep infinity` would leave zombies holding the PIDs restore needs | no init reaping | `pod.yaml` runs a bash `wait` loop as PID 1 |
| — | `nvidia-smi` shows both A100s although the pod requests one | `privileged: true` bypasses the device plugin's `/dev/nvidia*` filtering | P1: replace with `CAP_SYS_ADMIN, CAP_CHECKPOINT_RESTORE, CAP_SYS_PTRACE` + `seccomp Unconfined`; pin `CUDA_VISIBLE_DEVICES` meanwhile |

Not hit: CUDA VMM (`cuMemCreate`/`cuMemMap` from sleep mode) checkpoints fine with driver 580; ZMQ ipc sockets, the `:8000`
listening socket and `/dev/shm` ring buffers restore without extra flags in the same pod (`find /tmp -type s -delete` after `kill`
is enough). `criu check --all` only warns about `statmount`.

## Run

```bash
IMAGE=us-central1-docker.pkg.dev/ryanrosario-gke-dev/vllm/vllm-criu:v0.30.0-criu4.3.1
docker build -t "${IMAGE}" docker/scripts/snapshot/criu/p0 && docker push "${IMAGE}"

kubectl create namespace criu-p0 --dry-run=client -o yaml | kubectl apply -f -
sed "s#IMAGE#${IMAGE}#" docker/scripts/snapshot/criu/p0/pod.yaml | kubectl apply -n criu-p0 -f -
kubectl wait -n criu-p0 --for=condition=Ready pod/criu-p0 --timeout=600s

x() { kubectl exec -n criu-p0 criu-p0 -- env MODEL="${MODEL:-Qwen/Qwen3-8B}" criu-snapshot.sh "$@"; }
x check            # driver >= 570, cuda_plugin.so present, criu check
x cycle            # cold -> dump -> verify -> kill -> restore -> wake -> verify, prints timings
x status
```

Logs: `/checkpoints/vllm.log`, `/checkpoints/images/{dump,restore}.log` inside the pod.

## What P0 decides

| Risk | Signal |
| :--- | :--- |
| CUDA VMM (`cuMemCreate`/`cuMemMap` used by sleep mode) vs `cuda-checkpoint` | `dump` fails in the cuda plugin → retry with `VLLM_EXTRA_ARGS=` without `--enable-sleep-mode` and skip `/sleep` |
| PID reuse | `restore.log`: `Can't fork` / `pid … already in use` |
| ZMQ ipc sockets, `/dev/shm` | `restore.log`: `bind()`/`EADDRINUSE` or missing shmem file |
| Size and time | `du -sh /checkpoints/images`, `criu_dump` and `criu_restore` rows in `timings.csv` |
