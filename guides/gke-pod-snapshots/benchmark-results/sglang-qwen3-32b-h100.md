# Qwen/Qwen3-32B Snapshot & Restore Benchmark on SGLang (1×H100)

The benchmark compares initial cold start (pod 1 creating the snapshot on node 1) against fast horizontal scale-out (pod 2 restoring from the snapshot onto a second `a3-highgpu-1g` node, TP=1).

> [!NOTE]
> This guide's value metric is **pod startup latency**, so these figures were collected from Kubernetes pod
> lifecycle events and SGLang logs rather than request-throughput benchmarks
> ([`llmdbenchmark`](https://github.com/llm-d/llm-d-benchmark) / `inference-perf`). Because snapshot restoration
> resumes the initialized SGLang process and its captured CUDA graphs in GPU memory, steady-state serving
> behavior is expected to match a standard deployment.

**Reference configuration:** `a3-highgpu-1g` (1× NVIDIA H100 80GB, driver `580.126.20`),
GKE `v1.36.4-gke.1247000`, gVisor sandbox, GKE Image Streaming enabled (`--enable-image-streaming`);
`Qwen/Qwen3-32B` on SGLang `v0.5.19` at `--mem-fraction-static 0.90`, `--context-length 8192`,
`--enable-memory-saver`, and `--enable-weights-cpu-backup` (CUDA graphs enabled).

## Comparing Cold Start to Snapshot Restore

Treat these as one measured data point, not a specification. Phase times measure from **Pod scheduled** (`PodScheduled=True`) through
**serving-ready** (for cold start: when model load, KV cache allocation, CUDA graph capture, and server warmup complete right before `release_memory_occupation`;
for snapshot restore: when Pod `Ready=True` and `/health` returns HTTP `200`). Both include container startup, and both
exclude **node provisioning time** (`Pod created → Pod scheduled`, which varies by cloud capacity). Download speed, model size,
and GPU will move the figures.

| Metric | Without snapshots (Cold start) | Restore from snapshot |
| :--- | ---: | ---: |
| Pod scheduled → serving-ready | 4m 58s | **23.0s** (`19.0s` excl. `4.0s` image pull) |
| Weight loads from disk | Yes (61.07 GiB) | **No** |
| Speedup | — | **13.0×** (`15.7×` excl. image pull) |

<details>
<summary><b><i>Click</i></b> to view the per-phase breakdown</summary>

### Cold start — first pod, creates the snapshot

| Phase | Measured | Source / Notes |
| :--- | ---: | :--- |
| Node provisioning (`Pod created → Pod scheduled`) | varies (excluded) | Excluded to isolate pod startup latency from cloud VM provisioning |
| Image pull | 2.9s | kubelet `Pulling` → `Pulled` (`Successfully pulled image "docker.io/lmsysorg/sglang:v0.5.19" in 2.859s`; image size `15.10 GB`, backed by GKE Image Streaming) |
| **Pod scheduled → serving-ready** (weights loaded, KV cache allocated, CUDA graphs captured, server warmup) | **4m 58s** | `PodScheduled=True` (`01:01:14Z`) → `[Control Plane] Sleep signal received...` (`01:06:12Z`), measured directly as `298.0s` (`291.76s` from launcher start through server warmup plus `5.0s` image pull/container start); the first `126.0s` is container start plus streaming Python, CUDA, and SGLang libraries (`91.0s` in `pid=1` + `20.2s` in subprocesses) up to the SGLang log `Load weight begin.` |
| ↳ Model loading — download + load into VRAM | 100.3s | `Load weight begin.` → `Load weight end. elapsed=100.34 s, type=Qwen3ForCausalLM, avail mem=17.40 GB, mem usage=61.07 GB.` |
| ↳ Engine init (KV cache, CUDA graph capture, radix cache, warmup) | 71.7s | `Load weight end.` → `Froze Python garbage collection in 0.09s.`; includes KV cache allocation (`0.41s`, `9.54 GB` for `39,078` tokens), CUDA graph capture (`61.61s` total: `55.92s` prefill costing `2.84 GB` + `5.69s` decode costing `0.13 GB`), and server warmup (`7.29s`) |
| `release_memory_occupation(tags=["weights", "kv_cache"])` | 11.8s | `[Control Plane] Sleep signal received...` → `Released GPU memory occupation for tags=['weights', 'kv_cache'] in 11.77s.`; offloads `61.07 GiB` of weights to host RAM and discards `9.54 GiB` of KV cache |
| Checkpoint + upload to GCS | 34.9s | `Triggering snapshot checkpoint...` → `gVisor checkpoint completed successfully (barrier unblocked in 34.89s, status=b'r')` and `PodSnapshot` `Ready=True`; includes `0.01s` to purge `/root/.cache/huggingface/hub`; snapshot `pages.img` is `66.64 GiB` (`71,552,172,032` bytes) across `componentCount: 2134` objects |
| **Total — Pod scheduled → snapshot `Ready`** (`PodSnapshot` condition `Ready=True`) | **5m 45s** | `298.0s + 11.8s + 34.9s = 344.7s` (`345.0s` wall-clock `01:01:14Z` → `01:06:59Z`); excludes node provisioning (Pod 1 itself reaches `Ready=True` `5.0s` later at `5m 50s` after `2.41s` `resume_memory_occupation`) |

### Restore — every subsequent pod

| Phase | Measured | Source / Notes |
| :--- | ---: | :--- |
| Node provisioning (`Pod created → Pod scheduled`) | varies (excluded) | GCE VM provisioning time (excluded to isolate pod restore latency) |
| Image pull | 4.0s | kubelet `Pulling` (`01:07:09Z`) → `Pulled` (`01:07:13Z`, `Successfully pulled image "docker.io/lmsysorg/sglang:v0.5.19" in 4.01s`); time to fetch the image onto the second node before the container starts |
| Pod scheduled → process restored (rootfs mount, sandbox create, checkpoint stream from GCS) | 16.0s | `PodScheduled=True` (`01:07:07Z`) → Pod condition `PodRestored=True` (`01:07:23Z`); container `startedAt` lands at `11.0s` (`01:07:18Z`, including the `4.0s` image pull) into this window, so the remaining `5.0s` is GKE streaming the checkpoint from GCS and restoring the process image |
| Process restored → Pod `Ready` | 7.0s | `PodRestored=True` (`01:07:23Z`) → Pod condition `Ready=True` (`01:07:30Z`); includes `2.50s` for `resume_memory_occupation(tags=["weights", "kv_cache"])` to copy weights back into GPU VRAM and set `ServerStatus.Up` |
| **Total — Pod scheduled → Pod `Ready`** (serving-ready: readiness probe `GET /health` returns HTTP `200`) | **23.0s** | `PodScheduled=True` → `Ready=True` (`16.0s + 7.0s = 23.0s`, or `22.682s` via `time kubectl rollout status`; `19.0s` when the container image is already cached on the node); excludes node provisioning; a **13.0×** speedup versus the `4m 58s` cold start (and **15.2×** versus the `5m 50s` cold start + snapshot creation) |

**Pod scheduled → process restored** measures from `PodScheduled=True` to the Pod condition `PodRestored=True`. The container's
`state.running.startedAt` falls `11.0s` in (`7.0s` when the image is already cached on the node), once the kubelet has pulled the image (`4.01s`), created and started the container, and the gVisor sandbox is
running; GKE then streams the checkpoint from GCS and restores the process image (`Snapshot checkpoint created / process restored from checkpoint`),
which accounts for the remaining `5.0s`.

**Process restored → Pod `Ready`** measures from `PodRestored=True` to the Pod condition `Ready=True`, i.e. the readiness probe
(`GET /health` returning HTTP `200`) succeeds. It includes `resume_memory_occupation(tags=["weights", "kv_cache"])` (`2.50s`), which copies weights from host RAM
back into VRAM and sets `ServerStatus.Up`, so a pod reporting `Ready` is serving requests, not merely restored.

</details>
