# Qwen/Qwen3-32B Snapshot & Restore Benchmark on vLLM (1×H100)

The benchmark runs on a single H100 80GB GPU under gVisor, with one model server pod (TP=1).

> [!NOTE]
> This guide's value metric is **pod startup latency**, not request throughput, so these figures were
> collected from pod lifecycle events and model server logs rather than from
> [`llmdbenchmark`](https://github.com/llm-d/llm-d-benchmark) / `inference-perf`. Steady-state serving
> performance after a restore is unchanged from a normal vLLM deployment — the snapshot restores the
> same process, not a degraded one.

**Reference configuration:** `a3-highgpu-1g` (1× NVIDIA H100 80GB HBM3, driver `580.126.20`),
GKE `v1.36.4-gke.1082000`, gVisor sandbox, Spot provisioning, 300 GB boot disk;
`Qwen/Qwen3-32B` on vLLM `v0.25.0` at `--gpu-memory-utilization 0.95` and `--enforce-eager`.

> [!NOTE]
> The servers run with `--gpu-memory-utilization=0.95` because the vLLM 0.9 default leaves too little KV
> headroom for Qwen3-32B's 40960-token context on an 80&nbsp;GB H100 — at `0.90` the engine cannot start
> at all. If you swap models, check the `Available KV cache memory` line during startup before
> assuming the shipped value fits.

The single-GPU A3 shapes (`a3-highgpu-1g`, `-2g`, `-4g`) are only offered as Spot or Flex-start VMs —
see [GPU machine types](https://cloud.google.com/compute/docs/gpus). That is a property of those
machine types, not a requirement of this guide, and it needed no change to the manifests: GKE injects
the `sandbox.gke.io/runtime` and `nvidia.com/gpu` tolerations automatically, and the node pool carries
no Spot taint.

## Comparing Cold Start to Snapshot Restore

Treat these as one measured data point, not a specification. Download speed, model size, and GPU will
move the figures.

| Metric | Cold start (creates snapshot) | Restore from snapshot |
| :--------------------- | ----------------------------: | --------------------: |
| Pod created → `Ready`  | ~17.5 min                     | **33s**               |
| Weight loads from disk | 1                             | **0**                 |
| Speedup                | —                             | **6.3×**              |

<details>
<summary><b><i>Click</i></b> to view the per-phase breakdown</summary>

### Cold start — first pod, creates the snapshot

| Phase | Measured |
| :--- | ---: |
| Image pull (8.82 GB) | 68s |
| Weight download from HuggingFace | 87s |
| Model load into VRAM (61.03 GiB) | 128s |
| **Pod start → checkpoint triggered** | **3m 27s** |
| `engine.sleep(level=1)` freed | 77.07 GiB (61.68 GiB moved to host RAM) |
| Checkpoint + upload to GCS | **13m 56s** |
| **Total to first `Ready`** | **~17.5 min** |
| Snapshot size in GCS | **64.56 GiB** |

The checkpoint upload dominates: 80% of the cold start is spent writing pages to GCS, during which
the pod is frozen and emits no logs.

### Restore — every subsequent pod

| Phase | Measured |
| :--- | ---: |
| Pod deleted → new pod `Ready` | **33s** |
| `engine.wake_up()` | 2.68s |
| Safetensors loads performed | **none** |

### Throughput of the snapshot path

| Direction | Rate |
| :--- | ---: |
| Checkpoint write to GCS | 82.9 MB/s |
| Restore read from GCS | ~2 GB/s |

Upload ran at 82.9 MB/s here and 82.0 MB/s on a smaller A100 / 8B run — within 1% across different
GPUs and a 3.5× difference in snapshot size, which suggests a fixed ceiling of the snapshot write
path rather than a property of the workload.

As a planning rule, **snapshot GB ÷ 5 ≈ minutes of checkpoint freeze**.

</details>
