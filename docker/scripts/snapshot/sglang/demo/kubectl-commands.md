run_demo.sh --stop-after-snapshot
run_demo.sh --from-restore

# GKE Pod Snapshots (vLLM & SGLang) — Native Commands by Phase

Copy-pasteable native commands (`kubectl`, `gcloud`, `helm`) matching every phase of [`run_demo.sh`](run_demo.sh).

---

## 0. Environment Variables (Run First in Your Shell)

```bash
export REPO_ROOT="$(realpath "$(git rev-parse --show-toplevel)")"
export GUIDE_NAME="pod-snapshot"
export NAMESPACE="llm-d-pod-snapshot"
export PROJECT_ID="ryanrosario-gke-dev"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER_NAME="ryanrosario-snapshots"
export GCS_BUCKET="ryanrosario-gke-dev-sglang-pod-snapshots"
export NODE_POOL_NAME="gpu-h100-pool"
export GPU_MACHINE_TYPE="a3-highgpu-1g"
export GPU_ACCELERATOR="nvidia-h100-80gb"
export GPU_COUNT="1"
export MAX_NODES="2"
export DISK_SIZE="200GB"
export MODEL="Qwen/Qwen3-32B"
export CURL_TEST_IMAGE="cfmanteiga/alpine-bash-curl-jq:latest"

export SNAPSHOT_OVERLAY_DIR="${REPO_ROOT}/guides/${GUIDE_NAME}/snapshot/gke"
export DEMO_PHASES="${REPO_ROOT}/docker/scripts/snapshot/sglang/demo/demo_phases.py"

# Select engine for single-engine native commands ("vllm" or "sglang"):
export ENGINE="sglang"
if [[ "${ENGINE}" == "vllm" ]]; then
  export OVERLAY_DIR="${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm/gke"
  export DEPLOYMENT="pod-snapshot-nvidia-gpu-vllm-decode"
  export POD_SELECTOR="llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type notin (sglang)"
else
  export OVERLAY_DIR="${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/sglang/gke"
  export DEPLOYMENT="pod-snapshot-nvidia-gpu-sglang-decode"
  export POD_SELECTOR="llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang"
fi

source "${REPO_ROOT}/guides/env.sh"
```

---

## Phase 0: Preflight Check (`preflight`)

```bash
# 1. Verify GKE cluster is RUNNING and configure kubectl credentials
gcloud container clusters describe "${CLUSTER_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status)"

gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}"

# 2. Verify GPU + gVisor node pool and registered nodes
gcloud container node-pools list \
  --cluster="${CLUSTER_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}"

kubectl get nodes -L sandbox.gke.io/runtime,cloud.google.com/gke-accelerator

# 3. Verify RuntimeClass 'gvisor' and Pod Snapshots CRDs
kubectl get runtimeclass gvisor
kubectl get crd \
  podsnapshots.podsnapshot.gke.io \
  podsnapshotpolicies.podsnapshot.gke.io \
  podsnapshotstorageconfigs.podsnapshot.gke.io

# 4. Verify Hierarchical Namespace is enabled on the GCS bucket
gcloud storage buckets describe "gs://${GCS_BUCKET}" \
  --raw \
  --format="value(hierarchicalNamespace.enabled)"
```

---

## Phase 0b (Optional): Provision Cluster, Node Pool & GCS Bucket (`provision`)

Only needed if the cluster, GPU gVisor node pool, or GCS bucket do not exist yet:

```bash
# 1. Create GKE cluster with Pod Snapshots & Image Streaming
gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=e2-standard-16 \
  --num-nodes=1 \
  --release-channel=rapid \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --enable-image-streaming \
  --enable-pod-snapshots

gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# 2. Create GPU + gVisor node pool (add --spot if using Spot VMs)
gcloud container node-pools create "${NODE_POOL_NAME}" \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type="${GPU_MACHINE_TYPE}" \
  --disk-size="${DISK_SIZE}" \
  --image-type=cos_containerd \
  --workload-metadata=GKE_METADATA \
  --enable-image-streaming \
  --sandbox type=gvisor \
  --accelerator="type=${GPU_ACCELERATOR},count=${GPU_COUNT},gpu-driver-version=latest" \
  --num-nodes=1 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes="${MAX_NODES}"

# 3. Create Hierarchical Namespace GCS bucket & IAM role
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"

gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --enable-hierarchical-namespace \
  --soft-delete-duration=0 \
  --uniform-bucket-level-access

gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@container-engine-robot.iam.gserviceaccount.com" \
  --role="roles/storage.objectUser"

gcloud iam roles create podSnapshotGcsReadWriter \
  --project="${PROJECT_ID}" \
  --title="Pod Snapshot GCS Read/Writer" \
  --permissions="storage.buckets.get,storage.objects.get,storage.objects.list,storage.objects.create,storage.objects.delete,storage.folders.create"
```

---

## Phase 1: Setup (`setup`)

```bash
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"

# 1. Bind both vLLM and SGLang Kubernetes ServiceAccounts to the GCS snapshot bucket
for SA in pod-snapshot-nvidia-gpu-vllm-sa pod-snapshot-nvidia-gpu-sglang-sa; do
  gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${SA}" \
    --role="projects/${PROJECT_ID}/roles/podSnapshotGcsReadWriter"
done

# 2. Apply Gateway API Inference Extension CRDs & create namespace
kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# 3. Create HuggingFace token secret (skip if llm-d-hf-token already exists in ${NAMESPACE})
kubectl get secret llm-d-hf-token -n "${NAMESPACE}" || \
  kubectl create secret generic llm-d-hf-token \
    --from-literal="HF_TOKEN=${HF_TOKEN}" \
    --namespace "${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

# 4. Install/upgrade standalone inference router
helm upgrade --install "${GUIDE_NAME}" \
  "${ROUTER_STANDALONE_CHART}" \
  -f "${REPO_ROOT}/guides/recipes/router/base.values.yaml" \
  -f "${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml" \
  -n "${NAMESPACE}" --version "${ROUTER_CHART_VERSION}"
```

---

## Phase 2a: Reset for a Clean Cold Start (`reset`)

```bash
# 1. Delete existing model server Deployments & wait for pods to terminate
kubectl delete deployment \
  pod-snapshot-nvidia-gpu-vllm-decode \
  pod-snapshot-nvidia-gpu-sglang-decode \
  -n "${NAMESPACE}" --ignore-not-found=true --wait=true --timeout=120s
kubectl wait --for=delete pod -l "llm-d.ai/guide=${GUIDE_NAME}" -n "${NAMESPACE}" --timeout=300s 2>/dev/null || true

# 2. Delete PodSnapshots and empty GCS bucket
kubectl delete podsnapshots --all -n "${NAMESPACE}" --wait=true --timeout=300s
gcloud storage rm --recursive "gs://${GCS_BUCKET}/**" 2>/dev/null || true

# 3. Clear cached container images and page cache on each GPU node
for node in $(kubectl get nodes -l "cloud.google.com/gke-nodepool=${NODE_POOL_NAME}" -o jsonpath='{.items[*].metadata.name}'); do
  kubectl run "cache-cleaner-${node##*-}" -n "${NAMESPACE}" --rm --attach --restart=Never \
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
          \"command\": [\"nsenter\", \"-t\", \"1\", \"-m\", \"-u\", \"-i\", \"-n\", \"--\", \"/bin/sh\", \"-c\", \"export PATH=/home/kubernetes/bin:\$PATH CONTAINER_RUNTIME_ENDPOINT=unix:///run/containerd/containerd.sock IMAGE_SERVICE_ENDPOINT=unix:///run/containerd/containerd.sock; crictl rmi docker.io/vllm/vllm-openai:v0.19.0 docker.io/lmsysorg/sglang:v0.5.10.post1 2>/dev/null || true; sync; echo 3 > /proc/sys/vm/drop_caches; echo Cleared caches on ${node}\"]
        }]
      }
    }"
done

# 4. Clear old namespace events
kubectl delete events --all -n "${NAMESPACE}"
```

---

## Phase 2b (Act I): Deploy Pod 1, Cold Start & Create Snapshot (`deploy` / `--stop-after-snapshot`)

```bash
# 1. Apply the shared GKE snapshot CRs (PodSnapshotStorageConfig & PodSnapshotPolicy)
kubectl kustomize "${SNAPSHOT_OVERLAY_DIR}" \
  | sed -e "s/gcs-bucket-placeholder/${GCS_BUCKET}/g" \
        -e "s/pod-snapshot-storage-config/${NAMESPACE}-pod-snapshot-storage-config/g" \
  | kubectl apply -n "${NAMESPACE}" -f -

# 2. Apply the selected engine's Kustomize overlay (vLLM or SGLang)
kubectl kustomize "${OVERLAY_DIR}" \
  | kubectl apply -n "${NAMESPACE}" -f -

# 3. Watch Pod 1 cold start and snapshot creation (either with plain kubectl or demo_phases.py)
time kubectl rollout status "deployment/${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=2400s
# Or with live phase milestones:
# python3 "${DEMO_PHASES}" watch --namespace "${NAMESPACE}" --selector "${POD_SELECTOR}" --deployment "${DEPLOYMENT}" --engine "${ENGINE}" --timeout 2400 --label "Pod 1"

# 4. Wait for PodSnapshot upload to reach Ready=True (AllSnapshotsAvailable) in GCS
time kubectl wait --for=condition=Ready podsnapshots --all -n "${NAMESPACE}" --timeout=600s
kubectl get podsnapshots -n "${NAMESPACE}"

# 5. Inspect Pod 1's snapshot wrapper & provider log lines
export POD1="$(kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')"
export POD1_IP="$(kubectl get pod "${POD1}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"
echo "Pod 1 (Cold Start): ${POD1} (${POD1_IP})"
kubectl logs "${POD1}" -n "${NAMESPACE}" | grep -E "snapshot|release_memory_occupation|resume_memory_occupation|sleep|wake_up|ServerStatus"
```

---

## Phase 3 (Act II — Restore Start Point): Scale Out (`1 → 2`) & Restore Pod 2 (`--from-restore` / `scale`)

If starting the live demo from the restore phase, run from here:

```bash
# 1. Confirm Pod 1 is Ready and PodSnapshot is AllSnapshotsAvailable
kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" -o wide
kubectl get podsnapshots -n "${NAMESPACE}"

# (If a previous demo run left the deployment at 2 replicas, scale back to 1 first:)
# kubectl scale deployment "${DEPLOYMENT}" -n "${NAMESPACE}" --replicas=1
# kubectl rollout status "deployment/${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=300s

# 2. Scale Deployment from 1 -> 2 replicas and time the fast restore (~20-23s)
kubectl scale deployment "${DEPLOYMENT}" -n "${NAMESPACE}" --replicas=2
time kubectl rollout status "deployment/${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=600s
# Or with live phase milestones:
# python3 "${DEMO_PHASES}" watch --namespace "${NAMESPACE}" --selector "${POD_SELECTOR}" --deployment "${DEPLOYMENT}" --engine "${ENGINE}" --timeout 600 --label "Pod 2"

# 3. Show both running pods and the GKEPodSnapshotting restore event
kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" -o wide
kubectl get events -n "${NAMESPACE}" --field-selector reason=GKEPodSnapshotting --sort-by=.lastTimestamp

# 4. Inspect Pod 2 logs (note: no weight download and no CUDA graph capture — resumes straight from checkpoint)
export POD2="$(kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')"
export POD2_IP="$(kubectl get pod "${POD2}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"
echo "Pod 2 (Restored): ${POD2} (${POD2_IP})"
kubectl logs "${POD2}" -n "${NAMESPACE}" --tail=25
```

---

## Phase 4 (Act III): Verify Live Inference Against Pod 1 & Pod 2 (`verify`)

```bash
export POD1="$(kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')"
export POD2="$(kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')"
export POD1_IP="$(kubectl get pod "${POD1}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"
export POD2_IP="$(kubectl get pod "${POD2}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"

echo "Pod 1 (Cold Start): ${POD1} (${POD1_IP}:8000)"
echo "Pod 2 (Restored):   ${POD2} (${POD2_IP}:8000)"

kubectl run curl-test -n "${NAMESPACE}" --rm -i --restart=Never \
  --image="${CURL_TEST_IMAGE}" \
  -- /bin/sh -c "
    echo '=== Pod 1 (${POD1} @ ${POD1_IP}:8000) ===' &&
    time curl -fsS http://${POD1_IP}:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\": \"${MODEL}\", \"prompt\": \"Pod snapshots on GKE allow ${ENGINE} to\", \"max_tokens\": 32, \"temperature\": 0}' | jq . &&
    echo '' &&
    echo '=== Pod 2 (${POD2} @ ${POD2_IP}:8000) ===' &&
    time curl -fsS http://${POD2_IP}:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\": \"${MODEL}\", \"prompt\": \"Pod snapshots on GKE allow ${ENGINE} to\", \"max_tokens\": 32, \"temperature\": 0}' | jq .
  "
```

---

## Phase 5: Print Side-by-Side Timing & Speedup Report (`report`)

```bash
# Option A: Full per-phase breakdown using demo_phases.py
python3 "${DEMO_PHASES}" report --namespace "${NAMESPACE}" --selector "${POD_SELECTOR}" --engine "${ENGINE}"

# Option B: Pure kubectl jsonpath inspection of Pod 1 vs. Pod 2 condition timestamps
kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp \
  -o custom-columns="NAME:.metadata.name,NODE:.spec.nodeName,CREATED:.metadata.creationTimestamp,SCHEDULED:.status.conditions[?(@.type=='PodScheduled')].lastTransitionTime,RESTORED:.status.conditions[?(@.type=='PodRestored')].lastTransitionTime,READY:.status.conditions[?(@.type=='Ready')].lastTransitionTime"
```

---

## Phase 6: Cleanup (`cleanup`)

```bash
helm uninstall "${GUIDE_NAME}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete podsnapshots --all -n "${NAMESPACE}" --ignore-not-found=true
kubectl kustomize "${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm/gke" | kubectl delete -n "${NAMESPACE}" --ignore-not-found=true -f -
kubectl kustomize "${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/sglang/gke" | kubectl delete -n "${NAMESPACE}" --ignore-not-found=true -f -
kubectl kustomize "${SNAPSHOT_OVERLAY_DIR}" \
  | sed -e "s/gcs-bucket-placeholder/${GCS_BUCKET}/g" \
        -e "s/pod-snapshot-storage-config/${NAMESPACE}-pod-snapshot-storage-config/g" \
  | kubectl delete -n "${NAMESPACE}" --ignore-not-found=true -f -
kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
```
