# SGLang GKE Pod Snapshots — Plain `kubectl` Demo & Timing Guide (No Kustomize / No Helm)

This guide uses **only basic `kubectl` commands** and a single self-contained manifest ([`sglang-snapshot.yaml`](sglang-snapshot.yaml))—no `kustomize` and no `helm` required—with `time` on each step.

---

## Step 0: Set Variables & Check Cluster Nodes

```bash
export NAMESPACE=llm-d-gke-pod-snapshots
export MODEL=Qwen/Qwen3-32B
export GCS_BUCKET=ryanrosario-gke-dev-sglang-pod-snapshots
export NODE_POOL_NAME=gpu-h100-pool
```

Verify your cluster and GPU gVisor nodes are `Ready`:

```bash
kubectl get nodes -L sandbox.gke.io/runtime,cloud.google.com/gke-accelerator
kubectl get crd podsnapshots.podsnapshot.gke.io podsnapshotpolicies.podsnapshot.gke.io podsnapshotstorageconfigs.podsnapshot.gke.io
```

---

## Step 1: Create Namespace & Snapshot Script ConfigMaps (`kubectl create configmap`)

Create the namespace (if not already present) and package the Python snapshot scripts into two ConfigMaps directly from disk:

```bash
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap snapshot-scripts \
  --from-file=docker/scripts/snapshot/__init__.py \
  --from-file=docker/scripts/snapshot/launcher.py \
  --from-file=docker/scripts/snapshot/providers.py \
  -n ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap snapshot-scripts-sglang \
  --from-file=docker/scripts/snapshot/sglang/__init__.py \
  --from-file=docker/scripts/snapshot/sglang/launcher.py \
  --from-file=docker/scripts/snapshot/sglang/wrapper.py \
  -n ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
```

*(If `llm-d-hf-token` does not already exist in `${NAMESPACE}`, create it:)*
```bash
kubectl get secret llm-d-hf-token -n ${NAMESPACE}
# Or create it if missing:
# kubectl create secret generic llm-d-hf-token --from-literal="HF_TOKEN=${HF_TOKEN}" -n ${NAMESPACE}
```

---

## Step 2: Clean Any Previous Run & Clear All Caches

Delete any previous SGLang deployments and `PodSnapshot` objects, empty the GCS snapshot bucket, and clear the container image cache (`crictl rmi`) and kernel page cache (`drop_caches`) on each GPU node before timing the cold start:

```bash
kubectl delete -f docker/scripts/snapshot/sglang/demo/sglang-snapshot.yaml --ignore-not-found=true --wait=true
kubectl delete deployment gke-pod-snapshots-nvidia-gpu-sglang-decode -n ${NAMESPACE} --ignore-not-found=true --wait=true
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
```

---

## Step 3: Deploy Pod 1 & Time the Cold Start + Snapshot Creation

Apply the plain manifest [`sglang-snapshot.yaml`](sglang-snapshot.yaml) (which creates the `ServiceAccount`, `PodSnapshotStorageConfig`, `PodSnapshotPolicy`, `Deployment`, and `Service`):

```bash
kubectl apply -f docker/scripts/snapshot/sglang/demo/sglang-snapshot.yaml
```

### Time 3a: Total Cold Start + Snapshot + Wake-Up (`PodScheduled -> Ready`)

```bash
time kubectl rollout status deployment/sglang-decode -n ${NAMESPACE} --timeout=2400s
```

### Time 3b: Verify `PodSnapshot` Upload Completion (`AllSnapshotsAvailable`)

```bash
time kubectl wait --for=condition=Ready podsnapshots --all -n ${NAMESPACE} --timeout=600s
kubectl get podsnapshots -n ${NAMESPACE}
```

### Time 3c: View Per-Step Timings Logged by Pod 1

```bash
POD1=$(kubectl get pods -l app=sglang-snapshot-demo -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')
POD1_IP=$(kubectl get pod ${POD1} -n ${NAMESPACE} -o jsonpath='{.status.podIP}')
echo "Pod 1 (Cold Start): ${POD1} (${POD1_IP})"

kubectl logs ${POD1} -n ${NAMESPACE} | grep -E "Engine startup timings|sglang\.snapshot|snapshot\.providers|\[Control Plane\]"
```

This prints the exact duration of every internal phase on Pod 1:
- `Engine startup timings (s): load_weight=..., kv_cache_allocation=..., cuda_graph={prefill=..., decode=...}`
- `Entering patched _wait_and_warmup (... elapsed_since_start=...s)`
- `Server warmup completed in ...s (cold-start elapsed_since_start=...s)`
- `Froze Python garbage collection in ...s`
- `Released GPU memory occupation for tags=['weights', 'kv_cache'] in ...s` (Sleep)
- `Purged ... cache entry/entries from '/root/.cache/huggingface/hub' in ...s` (Cache purge)
- `gVisor checkpoint completed successfully (barrier unblocked in ...s, status=b'r')` (Snapshot + GCS upload)
- `Resumed GPU memory occupation for tags=['weights', 'kv_cache'] in ...s` (Wake)
- `Set tokenizer_manager.server_status = ServerStatus.Up (wake-to-ready=...s)`

To inspect Pod 1's Kubernetes condition timestamps (`PodScheduled` and `Ready`):
```bash
kubectl get pod ${POD1} -n ${NAMESPACE} -o custom-columns="NAME:.metadata.name,NODE:.spec.nodeName,SCHEDULED:.status.conditions[?(@.type=='PodScheduled')].lastTransitionTime,READY:.status.conditions[?(@.type=='Ready')].lastTransitionTime"
```

---

## Step 4: Scale to 2 Replicas & Time the Snapshot Restore on Pod 2

Scale `sglang-decode` from `1` to `2` replicas and time how long Pod 2 takes to restore from GCS and pass its `/health` readiness probe:

### Time 4a: End-to-End Scale-Out & Restore (`kubectl scale` -> `Ready`)

```bash
kubectl scale deployment sglang-decode -n ${NAMESPACE} --replicas=2 && \
  time kubectl rollout status deployment/sglang-decode -n ${NAMESPACE} --timeout=600s
```

### Time 4b: Inspect Pod 2's Kubernetes Restore Timestamps (`PodScheduled`, `Started`, `PodRestored`, `Ready`)

```bash
POD2=$(kubectl get pods -l app=sglang-snapshot-demo -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
POD2_IP=$(kubectl get pod ${POD2} -n ${NAMESPACE} -o jsonpath='{.status.podIP}')
echo "Pod 2 (Restored): ${POD2} (${POD2_IP})"

kubectl get pod ${POD2} -n ${NAMESPACE} -o custom-columns="NAME:.metadata.name,NODE:.spec.nodeName,SCHEDULED:.status.conditions[?(@.type=='PodScheduled')].lastTransitionTime,STARTED:.status.containerStatuses[0].state.running.startedAt,RESTORED:.status.conditions[?(@.type=='PodRestored')].lastTransitionTime,READY:.status.conditions[?(@.type=='Ready')].lastTransitionTime"
```

Check the `GKEPodSnapshotting` event confirming Pod 2 restored from the `PodSnapshot`:
```bash
kubectl get events -n ${NAMESPACE} --field-selector reason=GKEPodSnapshotting --sort-by=.lastTimestamp
```

### Time 4c: View Per-Step Wake-Up Timings Logged by Pod 2

```bash
kubectl logs ${POD2} -n ${NAMESPACE} | grep -E "sglang\.snapshot|snapshot\.providers|\[Control Plane\]"
```

This shows:
- `Resumed GPU memory occupation for tags=['weights', 'kv_cache'] in ...s` (copying weights from host RAM back into GPU VRAM)
- `Set tokenizer_manager.server_status = ServerStatus.Up (wake-to-ready=...s)`

---

## Step 5: Send Live Inference Requests to Both Pod 1 and Pod 2

```bash
POD1=$(kubectl get pods -n "${NAMESPACE}" -l app=sglang-snapshot-demo \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')
POD2=$(kubectl get pods -n "${NAMESPACE}" -l app=sglang-snapshot-demo \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
POD1_IP=$(kubectl get pod "${POD1}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')
POD2_IP=$(kubectl get pod "${POD2}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')

echo "Pod 1 (Cold Start): ${POD1} (${POD1_IP})"
echo "Pod 2 (Restored):   ${POD2} (${POD2_IP})"

kubectl run curl-test -n "${NAMESPACE}" --rm -i --restart=Never \
  --image=cfmanteiga/alpine-bash-curl-jq:latest \
  -- /bin/sh -c "
    echo '=== Pod 1 (${POD1} @ ${POD1_IP}:8000) ===' &&
    time curl -sS http://${POD1_IP}:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\": \"${MODEL:-Qwen/Qwen3-32B}\", \"prompt\": \"Pod snapshots on GKE allow SGLang to\", \"max_tokens\": 32, \"temperature\": 0}' | jq . &&
    echo '' &&
    echo '=== Pod 2 (${POD2} @ ${POD2_IP}:8000) ===' &&
    time curl -sS http://${POD2_IP}:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\": \"${MODEL:-Qwen/Qwen3-32B}\", \"prompt\": \"Pod snapshots on GKE allow SGLang to\", \"max_tokens\": 32, \"temperature\": 0}' | jq .
  "
```

---

## Measured Timings (`Qwen/Qwen3-32B` on 1×H100)

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

## Step 6: Cleanup

```bash
kubectl delete podsnapshots --all -n ${NAMESPACE} --ignore-not-found=true
kubectl delete -f docker/scripts/snapshot/sglang/demo/sglang-snapshot.yaml --ignore-not-found=true
kubectl delete configmap snapshot-scripts snapshot-scripts-sglang -n ${NAMESPACE} --ignore-not-found=true
```
