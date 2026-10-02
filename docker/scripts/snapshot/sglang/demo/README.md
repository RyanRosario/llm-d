# SGLang GKE Pod Snapshots — Demo Runbook

This runbook outlines what to present, the exact commands to run, and the measured benchmark results for a live demo of **SGLang Pod Snapshots on GKE** (`Qwen/Qwen3-32B` on a single NVIDIA H100 80GB GPU with GKE Sandbox / gVisor).

---

## Demo Goals

1. **Show the dramatic startup speedup:** Scaling out an SGLang replica from a GCS pod snapshot in **23.0s** (`19.0s` excluding the `4.0s` cold image pull) instead of a **4m 58s** cold start (**13.0×–15.7× faster**, or **15.2× faster** than the **5m 50s** end-to-end cold start + snapshot creation).
2. **Highlight the non-invasive SGLang architecture:** How the snapshot wrapper hooks into SGLang's startup lifecycle (`_wait_and_warmup`) using a stock `lmsysorg/sglang` image via ConfigMap injection.

---

## Quick Start (Automated Script)

You can run the entire demo or individual stages using [`run_demo.sh`](run_demo.sh) from the repository root:

```bash
./docker/scripts/snapshot/sglang/demo/run_demo.sh preflight  # Step 0: Verify cluster, GPU gVisor node pool, CRDs & GCS bucket
./docker/scripts/snapshot/sglang/demo/run_demo.sh setup      # Step 1: Configure GCS IAM, namespace, secret & router
./docker/scripts/snapshot/sglang/demo/run_demo.sh deploy     # Step 2 (Act I): Clean cold start & snapshot creation on Pod 1
./docker/scripts/snapshot/sglang/demo/run_demo.sh scale      # Step 3 (Act II): Scale to 2 replicas & restore Pod 2 from GCS
./docker/scripts/snapshot/sglang/demo/run_demo.sh verify     # Step 4 (Act III): Send live inference request to restored Pod 2
./docker/scripts/snapshot/sglang/demo/run_demo.sh report     # Re-print side-by-side timing & speedup tables anytime
./docker/scripts/snapshot/sglang/demo/run_demo.sh cleanup    # Step 5: Tear down demo resources
```

---

## Part 1: Architectural Walkthrough (~2 Minutes)

Walk through the following files to explain how SGLang snapshotting works under the hood:

1. **Zero-Image-Build Injection (`kustomization.yaml` & `patch-sglang.yaml`):**
   - [`docker/scripts/snapshot/kustomization.yaml`](../../kustomization.yaml) packages the snapshot runtime (`snapshot-scripts` and `snapshot-scripts-sglang`) as ConfigMaps.
   - [`guides/gke-pod-snapshots/modelserver/gpu/gke/sglang/patch-sglang.yaml`](../../../../../guides/gke-pod-snapshots/modelserver/gpu/gke/sglang/patch-sglang.yaml) mounts those ConfigMaps into the stock `lmsysorg/sglang` container and sets the entrypoint to:
     ```yaml
     command: ["python3", "-m", "docker.scripts.snapshot.sglang.launcher"]
     ```

2. **Hooking SGLang's Startup Lifecycle (`launcher.py` & `wrapper.py`):**
   - [`docker/scripts/snapshot/sglang/launcher.py`](../launcher.py) patches SGLang before calling `sglang.srt.entrypoints.http_server.launch_server`.
   - [`docker/scripts/snapshot/sglang/wrapper.py`](../wrapper.py) (`patch_sglang_wait_and_warmup`) intercepts `http_server._wait_and_warmup`:
     1. **Warmup & CUDA Graphs:** Runs standard SGLang warmup (`_execute_server_warmup`) to capture CUDA graphs and initialize memory pools, then freezes Python GC.
     2. **Sleep (VRAM Release):** Calls `tokenizer_manager.release_memory_occupation(tags=["weights", "kv_cache"])` (enabled by SGLang flags `--enable-memory-saver` and `--enable-weights-cpu-backup`) to offload model weights to host CPU RAM and discard the KV cache.
     3. **Disk Cache Purge & Checkpoint Trigger:** Purges `MODEL_CACHE_DIR` (`~/.cache/huggingface/hub`) so 60+ GiB of model weights are not duplicated on disk and in RAM, then writes to `/proc/gvisor/checkpoint` via `GKESnapshotProvider.trigger()`.
     4. **Wake Up (VRAM Restore):** When the process resumes (either after checkpointing completes with `postCheckpoint: resume` or when a new pod restores from GCS), calls `tokenizer_manager.resume_memory_occupation(tags=["weights", "kv_cache"])` to copy weights back into GPU VRAM and marks `tokenizer_manager.server_status = ServerStatus.Up`.

3. **SGLang Readiness Probe Nuance (`patch-sglang.yaml`):**
   - Unlike vLLM, SGLang starts Uvicorn and responds `200 OK` on `/v1/models` *before* the Scheduler finishes loading weights or `_wait_and_warmup` completes.
   - Both `startupProbe` and `readinessProbe` in [`patch-sglang.yaml`](../../../../../guides/gke-pod-snapshots/modelserver/gpu/gke/sglang/patch-sglang.yaml) target `/health` (which returns `503` until `ServerStatus.Up`), preventing Kubernetes from routing traffic before GPU memory is restored.

---

## Part 2: Pre-Demo Environment & Cluster Preflight Check

Set your environment variables before starting the demo:

```bash
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
export GUIDE_NAME=gke-pod-snapshots
export NAMESPACE=llm-d-gke-pod-snapshots
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
> **Grant GCS Access to the SGLang ServiceAccount:** The SGLang overlay uses `namePrefix: gke-pod-snapshots-nvidia-gpu-sglang-`, which creates a Kubernetes ServiceAccount named `gke-pod-snapshots-nvidia-gpu-sglang-sa` (distinct from the `...-vllm-sa` account). Ensure this KSA is bound to your snapshot GCS bucket:
> ```bash
> PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
>
> gcloud storage buckets add-iam-policy-binding gs://${GCS_BUCKET} \
>   --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/gke-pod-snapshots-nvidia-gpu-sglang-sa" \
>   --role="projects/${PROJECT_ID}/roles/podSnapshotGcsReadWriter"
> ```

Ensure the namespace, HuggingFace secret, and CRDs are created:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml

kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Only needed if llm-d-hf-token does not already exist in ${NAMESPACE}:
if [[ -n "${HF_TOKEN:-}" ]]; then
  kubectl create secret generic llm-d-hf-token \
    --from-literal="HF_TOKEN=${HF_TOKEN}" \
    --namespace "${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -
fi
```

*(Optional)* Deploy or upgrade the standalone router if demoing through the inference router (`helm upgrade --install` is idempotent if the release already exists):

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

> **Demo Tip:** Because downloading `Qwen/Qwen3-32B` and capturing CUDA graphs takes ~4m 58s on a completely clean cold start (+ ~47s to sleep and upload the snapshot), deploy Pod 1 **before** the live demo begins so the snapshot is already `Ready`, or start near the end of Pod 1's initialization.

### 1. Clear All Caches & Deploy the SGLang Kustomize Overlay

Delete any previous SGLang deployment and stale `PodSnapshot` objects, empty the GCS snapshot bucket, and clear the cached container image (`crictl rmi`) and kernel page cache (`drop_caches`) on each GPU node so Pod 1 performs a completely clean cold start:

```bash
kubectl delete deployment gke-pod-snapshots-nvidia-gpu-sglang-decode sglang-decode -n ${NAMESPACE} --ignore-not-found=true --wait=true
kubectl delete podsnapshots --all -n ${NAMESPACE} --ignore-not-found=true --wait=true
gcloud storage rm -r "gs://${GCS_BUCKET}/**" 2>/dev/null || true

# Clear container image cache & kernel page cache on all GPU nodes
for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=${NODE_POOL_NAME} -o jsonpath='{.items[*].metadata.name}'); do
  kubectl run "cache-cleaner-${node##*-}" -n "${NAMESPACE}" --rm -i --restart=Never \
    --image=alpine:3.20 \
    --overrides="{
      \"spec\": {
        \"nodeName\": \"${node}\",
        \"hostPID\": true,
        \"tolerations\": [{\"operator\": \"Exists\"}],
        \"containers\": [{
          \"name\": \"cleaner\",
          \"image\": \"alpine:3.20\",
          \"securityContext\": {\"privileged\": true},
          \"command\": [\"nsenter\", \"-t\", \"1\", \"-m\", \"-u\", \"-i\", \"-n\", \"--\", \"/bin/sh\", \"-c\", \"crictl rmi docker.io/lmsysorg/sglang:v0.5.19 2>/dev/null || true; sync; echo 3 > /proc/sys/vm/drop_caches; echo Cleared caches on ${node}\"]
        }]
      }
    }"
done

kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/gke/sglang/ \
  | sed "s/gcs-bucket-placeholder/${GCS_BUCKET}/g" \
  | kubectl apply -n ${NAMESPACE} -f -
```

Wait for the rollout and `PodSnapshot` creation to complete:

```bash
time kubectl rollout status deployment/gke-pod-snapshots-nvidia-gpu-sglang-decode -n ${NAMESPACE} --timeout=2400s
time kubectl wait --for=condition=Ready podsnapshots --all -n ${NAMESPACE} --timeout=600s
```

### 2. Inspect the Cold-Start & Checkpoint Logs on Pod 1

Show the logs from the first pod to highlight the full sleep -> checkpoint -> wake lifecycle:

```bash
POD1=$(kubectl get pods -l llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')
POD1_IP=$(kubectl get pod ${POD1} -n ${NAMESPACE} -o jsonpath='{.status.podIP}')
echo "Pod 1 (Cold Start): ${POD1} (${POD1_IP})"
kubectl logs ${POD1} -n ${NAMESPACE} | grep -E "sglang\.snapshot|snapshot\.providers|\[Control Plane\]"
```

Point out these key log lines:
```text
[2026-09-30 23:20:46] INFO [sglang.snapshot.wrapper:88] [Control Plane] Entering patched _wait_and_warmup (pid=1, provider=GKESnapshotProvider, proc_path='/proc/gvisor/checkpoint', cache_dir='~/.cache/huggingface/hub', elapsed_since_start=214.70s).
[2026-09-30 23:20:46] INFO [sglang.snapshot.wrapper:129] [Control Plane] Starting SGLang server warmup (capturing CUDA graphs & pre-allocating VRAM)...
[2026-09-30 23:20:53] INFO [sglang.snapshot.wrapper:133] [Control Plane] Server warmup completed in 7.34s (cold-start elapsed_since_start=222.04s).
[2026-09-30 23:20:53] INFO [sglang.snapshot.wrapper:144] [Control Plane] Froze Python garbage collection in 0.08s.
[2026-09-30 23:20:53] INFO [sglang.snapshot.wrapper:159] [Control Plane] Sleep signal received. Releasing GPU memory occupation for tags=['weights', 'kv_cache']...
[2026-09-30 23:21:04] INFO [sglang.snapshot.wrapper:167] [Control Plane] Released GPU memory occupation for tags=['weights', 'kv_cache'] in 11.33s.
[2026-09-30 23:21:04] INFO [sglang.snapshot.wrapper:174] [Control Plane] Triggering snapshot checkpoint via GKESnapshotProvider...
[2026-09-30 23:21:04] INFO [snapshot.providers:106] Purged 4 cache entry/entries from '/root/.cache/huggingface/hub' in 0.01s.
[2026-09-30 23:21:04] INFO [snapshot.providers:139] Writing trigger byte to '/proc/gvisor/checkpoint' to initiate GKE Pod Snapshot...
[2026-09-30 23:21:04] INFO [snapshot.providers:145] Waiting on '/proc/gvisor/checkpoint' barrier until checkpoint/restore completes...
[2026-09-30 23:21:45] INFO [snapshot.providers:161] gVisor checkpoint completed successfully (barrier unblocked in 40.10s, status=b'r').
[2026-09-30 23:21:45] INFO [sglang.snapshot.wrapper:187] [Control Plane] Wake signal received. Resuming GPU memory occupation for tags=['weights', 'kv_cache']...
[2026-09-30 23:21:47] INFO [sglang.snapshot.wrapper:197] [Control Plane] Resumed GPU memory occupation for tags=['weights', 'kv_cache'] in 2.41s.
[2026-09-30 23:21:47] INFO [sglang.snapshot.wrapper:206] [Control Plane] Set tokenizer_manager.server_status = ServerStatus.Up (wake-to-ready=2.41s). The server is fired up and ready to roll!
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

```bash
kubectl scale deployment gke-pod-snapshots-nvidia-gpu-sglang-decode -n ${NAMESPACE} --replicas=2
kubectl rollout status deployment/gke-pod-snapshots-nvidia-gpu-sglang-decode -n ${NAMESPACE} --timeout=600s
```

### 2. Verify Nodes & `GKEPodSnapshotting` Events

```bash
kubectl get pods -l llm-d.ai/guide=${GUIDE_NAME} -n ${NAMESPACE} -o wide
kubectl get events -n ${NAMESPACE} --field-selector reason=GKEPodSnapshotting --sort-by=.lastTimestamp
```

Expected event:
```text
Normal   GKEPodSnapshotting   pod/gke-pod-snapshots-nvidia-gpu-sglang-decode-...   Successfully restored the pod from PodSnapshot ...
```

### 3. Show the Restored Pod's Logs & Timing Report

Find the newly created pod and inspect its logs:

```bash
POD2=$(kubectl get pods -l llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
POD2_IP=$(kubectl get pod ${POD2} -n ${NAMESPACE} -o jsonpath='{.status.podIP}')
echo "Pod 2 (Restored): ${POD2} (${POD2_IP})"
kubectl logs ${POD2} -n ${NAMESPACE} | grep -E "sglang\.snapshot|snapshot\.providers|\[Control Plane\]"
```

Highlight that the restored pod **completely bypassed weight downloads, safetensors loading, and CUDA graph compilation**, resuming execution directly inside `patched_wait_and_warmup` right after `snapshot_provider.trigger()`:

```text
[2026-09-30 23:33:34] INFO [snapshot.providers:161] gVisor checkpoint completed successfully (barrier unblocked in 749.99s, status=b'r').
[2026-09-30 23:33:34] INFO [sglang.snapshot.wrapper:177] [Control Plane] Snapshot checkpoint created / process restored from checkpoint (barrier elapsed=749.99s).
[2026-09-30 23:33:34] INFO [sglang.snapshot.wrapper:187] [Control Plane] Wake signal received. Resuming GPU memory occupation for tags=['weights', 'kv_cache']...
[2026-09-30 23:33:37] INFO [sglang.snapshot.wrapper:197] [Control Plane] Resumed GPU memory occupation for tags=['weights', 'kv_cache'] in 2.43s.
[2026-09-30 23:33:37] INFO [sglang.snapshot.wrapper:206] [Control Plane] Set tokenizer_manager.server_status = ServerStatus.Up (wake-to-ready=2.43s). The server is fired up and ready to roll!
```

Print the full timing and speedup comparison report:

```bash
./docker/scripts/snapshot/sglang/demo/run_demo.sh report
```

---

## Part 5: Act III — Live Inference Verification

Prove that both Pod 1 (checkpointed & resumed) and Pod 2 (restored from snapshot) have their weights and KV cache in GPU VRAM and serve live inference requests.

### Option A: Query Both Pod 1 and Pod 2 Directly

```bash
POD1=$(kubectl get pods -n "${NAMESPACE}" -l llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')
POD2=$(kubectl get pods -n "${NAMESPACE}" -l llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
POD1_IP=$(kubectl get pod "${POD1}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')
POD2_IP=$(kubectl get pod "${POD2}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')

echo "Pod 1 (Cold Start): ${POD1} (${POD1_IP})"
echo "Pod 2 (Restored):   ${POD2} (${POD2_IP})"

kubectl run curl-test -n "${NAMESPACE}" --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  -- /bin/sh -c "
    echo '=== Pod 1 (${POD1} @ ${POD1_IP}:8000) ===' &&
    time curl -sS http://${POD1_IP}:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\": \"${MODEL}\", \"prompt\": \"Pod snapshots on GKE allow SGLang to\", \"max_tokens\": 32, \"temperature\": 0}' | jq . &&
    echo '' &&
    echo '=== Pod 2 (${POD2} @ ${POD2_IP}:8000) ===' &&
    time curl -sS http://${POD2_IP}:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\": \"${MODEL}\", \"prompt\": \"Pod snapshots on GKE allow SGLang to\", \"max_tokens\": 32, \"temperature\": 0}' | jq .
  "
```

### Option B: Query via the Standalone Router Service

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

*(Full report also saved at [`guides/gke-pod-snapshots/benchmark-results/sglang-qwen3-32b-h100.md`](../../../../../guides/gke-pod-snapshots/benchmark-results/sglang-qwen3-32b-h100.md))*

### Comparing Cold Start to Snapshot Restore

| Metric | Without snapshots (Cold start) | Restore from snapshot |
| :--- | ---: | ---: |
| Pod scheduled → serving-ready | 4m 58s | **23.0s** (`19.0s` excl. `4.0s` image pull) |
| Weight loads from disk | Yes (61.07 GiB) | **No** |
| Speedup | — | **13.0×** (`15.7×` excl. image pull) |

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

---

## Part 7: Post-Demo Cleanup

```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE} --ignore-not-found

kubectl delete podsnapshots --all -n ${NAMESPACE} --ignore-not-found=true

kubectl kustomize ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/gke/sglang/ \
  | kubectl delete -n ${NAMESPACE} --ignore-not-found=true -f -

kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
```
