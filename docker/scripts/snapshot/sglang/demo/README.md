# GKE Pod Snapshots (vLLM & SGLang) — Demo Runbook

This runbook outlines what to present, the exact commands to run, and the measured benchmark results for a live demo of **vLLM and SGLang Pod Snapshots on GKE** (`Qwen/Qwen3-32B` on a single NVIDIA H100 80GB GPU with GKE Sandbox / gVisor).

---

## Demo Goals

1. **Show the dramatic startup speedup across both engines:** Scaling out a vLLM or SGLang replica from a GCS pod snapshot in **~20–23s** instead of a **~4.5–5m** cold start (**13×–16× faster**).
2. **Highlight the non-invasive snapshot architecture:** How the snapshot launchers and wrappers hook into vLLM's FastAPI lifespan (`patch_vllm_lifespan` -> `engine.sleep(level=1)` / `engine.wake_up()`) and SGLang's startup callbacks (`sglang_warmup_and_hold` + `sglang_snapshot_callback` -> `/release_memory_occupation` / `/resume_memory_occupation`) using stock upstream container images via ConfigMap injection.

---

## Quick Start (Automated Script)

By default, [`run_demo.sh`](run_demo.sh) runs **both** `vllm` and `sglang` sequentially. Pass `--engine vllm` (`--vllm`) or `--engine sglang` (`--sglang`) to run a single engine:

```bash
# Run both vLLM and SGLang (default: --engine both)
./docker/scripts/snapshot/sglang/demo/run_demo.sh preflight              # Step 0: Verify cluster, GPU gVisor node pool, CRDs & GCS bucket
./docker/scripts/snapshot/sglang/demo/run_demo.sh setup                  # Step 1: Configure GCS IAM (vLLM & SGLang KSAs), namespace, secret & router
./docker/scripts/snapshot/sglang/demo/run_demo.sh reset                  # Clear snapshots, GCS bucket, node image & page cache (deploy runs this first)
./docker/scripts/snapshot/sglang/demo/run_demo.sh deploy                 # Step 2 (Act I): Clean cold start & snapshot creation on Pod 1
./docker/scripts/snapshot/sglang/demo/run_demo.sh --stop-after-snapshot  # Run setup -> reset -> deploy, then stop once the PodSnapshot is Ready
./docker/scripts/snapshot/sglang/demo/run_demo.sh --from-restore         # Start demo from restore phase (verifies Pod 1 & snapshot, then runs scale -> verify -> report)
./docker/scripts/snapshot/sglang/demo/run_demo.sh scale                  # Step 3 (Act II): Scale to 2 replicas & restore Pod 2 from GCS
./docker/scripts/snapshot/sglang/demo/run_demo.sh verify                 # Step 4 (Act III): Send live inference request to Pod 1 & restored Pod 2
./docker/scripts/snapshot/sglang/demo/run_demo.sh report                 # Re-print side-by-side timing & speedup tables anytime
./docker/scripts/snapshot/sglang/demo/run_demo.sh cleanup                # Step 5: Tear down demo resources

# Run only vLLM or only SGLang
./docker/scripts/snapshot/sglang/demo/run_demo.sh --vllm
./docker/scripts/snapshot/sglang/demo/run_demo.sh --sglang
```

---

## Part 1: Architectural Walkthrough (~2 Minutes)

Walk through the following files to explain how vLLM and SGLang snapshotting work under the hood:

1. **Zero-Image-Build Injection (`kustomization.yaml` & `patch.yaml`):**
   - [`docker/scripts/snapshot/kustomization.yaml`](../../kustomization.yaml) packages the snapshot runtime (`snapshot-scripts`, `snapshot-scripts-vllm`, and `snapshot-scripts-sglang`) as ConfigMaps.
   - [`guides/pod-snapshot/modelserver/gpu/vllm/gke/patch.yaml`](../../../../../guides/pod-snapshot/modelserver/gpu/vllm/gke/patch.yaml) mounts `snapshot-scripts` and `snapshot-scripts-vllm` into the stock `vllm/vllm-openai` container and sets the entrypoint to:
     ```yaml
     command: ["python3", "-m", "docker.scripts.snapshot.vllm.launcher"]
     ```
   - [`guides/pod-snapshot/modelserver/gpu/sglang/gke/patch.yaml`](../../../../../guides/pod-snapshot/modelserver/gpu/sglang/gke/patch.yaml) mounts `snapshot-scripts` and `snapshot-scripts-sglang` into the stock `lmsysorg/sglang` container and sets the entrypoint to:
     ```yaml
     command: ["python3", "-m", "docker.scripts.snapshot.sglang.launcher"]
     ```

2. **Hooking vLLM's Startup Lifecycle (`vllm/launcher.py` & `vllm/wrapper.py`):**
   - [`docker/scripts/snapshot/vllm/launcher.py`](../../vllm/launcher.py) hooks `build_app` before delegating CLI invocation to `vllm serve`.
   - [`docker/scripts/snapshot/vllm/wrapper.py`](../../vllm/wrapper.py) (`patch_vllm_lifespan`) wraps the FastAPI application lifespan context manager:
     1. **Warmup & CUDA Graphs:** Runs standard vLLM engine initialization (loads weights with `--safetensors-load-strategy eager`, profiles memory, compiles `torch.compile`, and captures CUDA graphs).
     2. **Sleep (VRAM Release):** Calls `await engine.sleep(level=1)` (enabled by `--enable-sleep-mode`) to offload model weights to host CPU RAM and discard the KV cache.
     3. **Disk Cache Purge & Checkpoint Trigger:** Purges `MODEL_CACHE_DIR` (`~/.cache/huggingface/hub`) and writes to `/proc/gvisor/checkpoint` via `GKESnapshotProvider.trigger()`.
     4. **Wake Up (VRAM Restore):** Upon resume or restore, calls `await engine.wake_up()` to re-allocate physical VRAM maps and yields control to Uvicorn to begin serving traffic.

3. **Hooking SGLang's Startup Lifecycle (`sglang/launcher.py` & `sglang/wrapper.py`):**
   - [`docker/scripts/snapshot/sglang/launcher.py`](../launcher.py) passes `sglang_warmup_and_hold` (`execute_warmup_func`) and `sglang_snapshot_callback` (`launch_callback`) to `sglang.srt.entrypoints.http_server.launch_server`.
   - [`docker/scripts/snapshot/sglang/wrapper.py`](../wrapper.py) orchestrates the lifecycle while holding `tokenizer_manager.server_status = ServerStatus.Starting` (so `/health` returns `503` until GPU memory is restored):
     1. **Warmup & Hold:** Runs `_execute_server_warmup` to capture CUDA graphs and warm up the engine, then holds `server_status` at `ServerStatus.Starting`.
     2. **Sleep (VRAM Release):** Calls `POST /release_memory_occupation` (enabled by `--enable-memory-saver` and `--enable-weights-cpu-backup`) to offload model weights to host CPU RAM and discard the KV cache.
     3. **Disk Cache Purge & Checkpoint Trigger:** Purges `MODEL_CACHE_DIR` (`~/.cache/huggingface/hub`) and triggers `/proc/gvisor/checkpoint` via `GKESnapshotProvider.trigger()`.
     4. **Wake Up (VRAM Restore):** Calls `POST /resume_memory_occupation` to copy weights back into GPU VRAM and restores `tokenizer_manager.server_status = ServerStatus.Up`.

---

## Part 2: Pre-Demo Environment & Cluster Preflight Check

Set your environment variables before starting the demo:

```bash
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
export GUIDE_NAME=pod-snapshot
export NAMESPACE=llm-d-pod-snapshot
export PROJECT_ID=ryanrosario-gke-dev
export REGION=us-central1
export ZONE=us-central1-a
export CLUSTER_NAME=ryanrosario-snapshots
export NODE_POOL_NAME=gpu-h100-pool
export GCS_BUCKET=ryanrosario-gke-dev-sglang-pod-snapshots
export MODEL=Qwen/Qwen3-32B
export CURL_TEST_IMAGE=cfmanteiga/alpine-bash-curl-jq:latest

source ${REPO_ROOT}/guides/env.sh
```

Verify that your GKE cluster is running, has a GPU + GKE Sandbox (`gvisor`) node pool, has the Pod Snapshots CRDs installed, and has a hierarchical-namespace GCS bucket:

```bash
./docker/scripts/snapshot/sglang/demo/run_demo.sh preflight
```

*(If any of the cluster, GPU gVisor node pool, or GCS bucket do not exist yet, create them automatically with:)*
```bash
./docker/scripts/snapshot/sglang/demo/run_demo.sh provision
```

> [!IMPORTANT]
> **Grant GCS Access to Both ServiceAccounts:** The vLLM overlay creates `pod-snapshot-nvidia-gpu-vllm-sa` and the SGLang overlay creates `pod-snapshot-nvidia-gpu-sglang-sa`. Ensure both KSAs are bound to your snapshot GCS bucket:
> ```bash
> PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
>
> for SA in pod-snapshot-nvidia-gpu-vllm-sa pod-snapshot-nvidia-gpu-sglang-sa; do
>   gcloud storage buckets add-iam-policy-binding gs://${GCS_BUCKET} \
>     --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${SA}" \
>     --role="projects/${PROJECT_ID}/roles/podSnapshotGcsReadWriter"
> done
> ```

Ensure the namespace, HuggingFace secret, and CRDs are created:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml

kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

if [[ -n "${HF_TOKEN:-}" ]]; then
  kubectl create secret generic llm-d-hf-token \
    --from-literal="HF_TOKEN=${HF_TOKEN}" \
    --namespace "${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -
fi
```

*(Optional)* Deploy or upgrade the standalone router if demoing through the inference router:

```bash
export ROUTER_BASE_VALUES="-f ${REPO_ROOT}/guides/recipes/router/base.values.yaml"
export ROUTER_VALUES="-f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml"

helm upgrade --install ${GUIDE_NAME} \
  ${ROUTER_STANDALONE_CHART} \
  ${ROUTER_BASE_VALUES} \
  ${ROUTER_VALUES} \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

---

## Part 3: Act I — Initial Deployment, Cold Start & Snapshot Creation

### 1. Apply the Shared GKE Snapshot CRs (`PodSnapshotStorageConfig` & `PodSnapshotPolicy`)

```bash
kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/snapshot/gke \
  | sed -e "s/gcs-bucket-placeholder/${GCS_BUCKET}/g" \
        -e "s/pod-snapshot-storage-config/${NAMESPACE}-pod-snapshot-storage-config/g" \
  | kubectl apply -n ${NAMESPACE} -f -
```

### 2. Deploy the Model Server Overlay (vLLM or SGLang)

**For vLLM:**
```bash
kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm/gke \
  | kubectl apply -n ${NAMESPACE} -f -

time kubectl rollout status deployment/pod-snapshot-nvidia-gpu-vllm-decode -n ${NAMESPACE} --timeout=2400s
time kubectl wait --for=condition=Ready podsnapshots --all -n ${NAMESPACE} --timeout=600s
```

**For SGLang:**
```bash
kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/sglang/gke \
  | kubectl apply -n ${NAMESPACE} -f -

time kubectl rollout status deployment/pod-snapshot-nvidia-gpu-sglang-decode -n ${NAMESPACE} --timeout=2400s
time kubectl wait --for=condition=Ready podsnapshots --all -n ${NAMESPACE} --timeout=600s
```

### 3. Verify the `PodSnapshot` Resource is Ready

```bash
kubectl get podsnapshots -n ${NAMESPACE}
```

Confirm `STATUS` is `AllSnapshotsAvailable` (`Ready=True`).

---

## Part 4: Act II — Live Scale-Out & Fast Restoration (The "Aha!" Moment)

With a ready snapshot in GCS, scale the deployment from `1` to `2` replicas live:

### 1. Scale Out to 2 Replicas

**For vLLM:**
```bash
kubectl scale deployment pod-snapshot-nvidia-gpu-vllm-decode -n ${NAMESPACE} --replicas=2
kubectl rollout status deployment/pod-snapshot-nvidia-gpu-vllm-decode -n ${NAMESPACE} --timeout=600s
```

**For SGLang:**
```bash
kubectl scale deployment pod-snapshot-nvidia-gpu-sglang-decode -n ${NAMESPACE} --replicas=2
kubectl rollout status deployment/pod-snapshot-nvidia-gpu-sglang-decode -n ${NAMESPACE} --timeout=600s
```

### 2. Verify Nodes & `GKEPodSnapshotting` Events

```bash
kubectl get pods -l llm-d.ai/guide=${GUIDE_NAME} -n ${NAMESPACE} -o wide
kubectl get events -n ${NAMESPACE} --field-selector reason=GKEPodSnapshotting --sort-by=.lastTimestamp
```

Expected event:
```text
Normal   GKEPodSnapshotting   pod/pod-snapshot-nvidia-gpu-...   Successfully restored the pod from PodSnapshot ...
```

### 3. Print the Timing & Speedup Report

```bash
./docker/scripts/snapshot/sglang/demo/run_demo.sh report --vllm
./docker/scripts/snapshot/sglang/demo/run_demo.sh report --sglang
```

---

## Part 5: Act III — Live Inference Verification

Prove that both Pod 1 (checkpointed & resumed) and Pod 2 (restored from snapshot) have their weights and KV cache in GPU VRAM and serve live inference requests:

```bash
./docker/scripts/snapshot/sglang/demo/run_demo.sh verify --vllm
./docker/scripts/snapshot/sglang/demo/run_demo.sh verify --sglang
```

Or query via the Standalone Router Service:

```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')

kubectl run curl-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'curl -sS -X POST "http://${IP}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"prompt\": \"How are you today?\", \"max_tokens\": 64}" | jq .'
```

---

## Part 6: Measured Benchmark Results (`Qwen/Qwen3-32B` on 1×H100)

*(Full reports also saved at [`guides/pod-snapshot/benchmark-results/vllm-qwen3-32b-h100.md`](../../../../../guides/pod-snapshot/benchmark-results/vllm-qwen3-32b-h100.md) and [`guides/pod-snapshot/benchmark-results/sglang-qwen3-32b-h100.md`](../../../../../guides/pod-snapshot/benchmark-results/sglang-qwen3-32b-h100.md))*

### vLLM (`Qwen/Qwen3-32B` on 1×H100)

| Metric | Without snapshots (Cold start) | Restore from snapshot |
| :--- | ---: | ---: |
| Pod scheduled → serving-ready | 4m 37s | **18.0s** |
| Weight loads from disk | Yes (61.03 GiB) | **No** |
| Speedup | — | **15.4×** |

| Phase | Cold Start (Pod 1) | Restore (Pod 2) |
| :--- | ---: | ---: |
| Image pull | 4.0s | 4.0s |
| **Pod scheduled → serving-ready** | **4m 37s** (`277.4s`) | **18.0s** (`14.0s` restore + `4.0s` wake/ready) |
| ↳ Model loading (download + safetensors load) | 129.8s (`43.41s` safetensors read) | — (bypassed) |
| ↳ Engine init (profile, KV cache, CUDA graphs, warmup) | 72.4s (`9.0s` CUDA graphs) | — (bypassed) |
| `engine.sleep(level=1)` | 17.4s | — |
| Checkpoint + upload to GCS (`65.73 GiB`) | 33.2s | — |
| **Total — Pod scheduled → snapshot `Ready`** | **5m 28s** (`328.0s`) | — |

### SGLang (`Qwen/Qwen3-32B` on 1×H100)

| Metric | Without snapshots (Cold start) | Restore from snapshot |
| :--- | ---: | ---: |
| Pod scheduled → serving-ready | 4m 58s (or 4m 45s on v0.5.20) | **23.0s** (`19.0s` excl. `4.0s` image pull; **20.0s** on v0.5.20) |
| Weight loads from disk | Yes (61.07 GiB) | **No** |
| Speedup | — | **13.0×–14.3×** (`15.7×` excl. image pull) |

#### Cold start — first pod, creates the snapshot

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

#### Restore — every subsequent pod

| Phase | Measured | Source / Notes |
| :--- | ---: | :--- |
| Node provisioning (`Pod created → Pod scheduled`) | varies (excluded) | GCE VM provisioning time (excluded to isolate pod restore latency) |
| Image pull | 4.0s | kubelet `Pulling` (`01:07:09Z`) → `Pulled` (`01:07:13Z`, `Successfully pulled image "docker.io/lmsysorg/sglang:v0.5.19" in 4.01s`); time to fetch the image onto the second node before the container starts |
| Pod scheduled → process restored (rootfs mount, sandbox create, checkpoint stream from GCS) | 16.0s | `PodScheduled=True` (`01:07:07Z`) → Pod condition `PodRestored=True` (`01:07:23Z`); container `startedAt` lands at `11.0s` (`01:07:18Z`, including the `4.0s` image pull) into this window, so the remaining `5.0s` is GKE streaming the checkpoint from GCS and restoring the process image |
| Process restored → Pod `Ready` | 7.0s | `PodRestored=True` (`01:07:23Z`) → Pod condition `Ready=True` (`01:07:30Z`); includes `2.50s` for `resume_memory_occupation(tags=["weights", "kv_cache"])` to copy weights back into GPU VRAM and set `ServerStatus.Up` |
| **Total — Pod scheduled → Pod `Ready`** (serving-ready: readiness probe `GET /health` returns HTTP `200`) | **23.0s** | `PodScheduled=True` → `Ready=True` (`16.0s + 7.0s = 23.0s`, or `22.682s` via `time kubectl rollout status`; `19.0s` when the container image is already cached on the node); excludes node provisioning; a **13.0×** speedup versus the `4m 58s` cold start (and **15.2×** versus the `5m 50s` cold start + snapshot creation) |

---

## Part 7: Post-Demo Cleanup

```bash
./docker/scripts/snapshot/sglang/demo/run_demo.sh cleanup
```

Or manually:

```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE} --ignore-not-found

kubectl delete podsnapshots --all -n ${NAMESPACE} --ignore-not-found=true

kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm/gke \
  | kubectl delete -n ${NAMESPACE} --ignore-not-found=true -f -

kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/sglang/gke \
  | kubectl delete -n ${NAMESPACE} --ignore-not-found=true -f -

kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/snapshot/gke \
  | sed -e "s/gcs-bucket-placeholder/${GCS_BUCKET}/g" \
        -e "s/pod-snapshot-storage-config/${NAMESPACE}-pod-snapshot-storage-config/g" \
  | kubectl delete -n ${NAMESPACE} --ignore-not-found=true -f -

kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
```
