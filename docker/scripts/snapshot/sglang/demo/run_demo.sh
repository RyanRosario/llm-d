#!/usr/bin/env bash
# Interactive step-by-step demo script for SGLang Pod Snapshots on GKE.
# Reports detailed per-step durations from both Kubernetes lifecycle conditions
# and the SGLang snapshot wrapper/launcher/provider logs.
#
# Usage:
#   ./docker/scripts/snapshot/sglang/demo/run_demo.sh [preflight|provision|setup|deploy|scale|verify|report|cleanup|all]

set -euo pipefail

REPO_ROOT="$(realpath "$(git rev-parse --show-toplevel)")"
GUIDE_NAME="${GUIDE_NAME:-gke-pod-snapshots}"
NAMESPACE="${NAMESPACE:-llm-d-gke-pod-snapshots}"
PROJECT_ID="${PROJECT_ID:-ryanrosario-gke-dev}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
CLUSTER_NAME="${CLUSTER_NAME:-ryanrosario-snapshots}"
GCS_BUCKET="${GCS_BUCKET:-ryanrosario-gke-dev-sglang-pod-snapshots}"
NODE_POOL_NAME="${NODE_POOL_NAME:-gpu-h100-pool}"
GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-a3-highgpu-1g}"
GPU_ACCELERATOR="${GPU_ACCELERATOR:-nvidia-h100-80gb}"
GPU_COUNT="${GPU_COUNT:-1}"
MAX_NODES="${MAX_NODES:-2}"
DISK_SIZE="${DISK_SIZE:-200GB}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
CURL_TEST_IMAGE="${CURL_TEST_IMAGE:-cfmanteiga/alpine-bash-curl-jq:latest}"
ACTION="${1:-all}"

# Source common guide variables (GAIE_URL, ROUTER_STANDALONE_CHART, ROUTER_CHART_VERSION, etc.)
# shellcheck disable=SC1091
source "${REPO_ROOT}/guides/env.sh"

banner() {
  echo ""
  echo "================================================================================"
  echo "  $*"
  echo "================================================================================"
}

pause_step() {
  if [[ "${INTERACTIVE:-1}" == "1" ]]; then
    read -r -p $'\n>>> Press [Enter] to continue to the next step (or Ctrl+C to stop)... '
  fi
}

require_var() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: Environment variable ${var_name} must be set." >&2
    exit 1
  fi
}

# Print a comprehensive timing summary for a pod by parsing both K8s Pod/PodSnapshot
# conditions and the SGLang snapshot wrapper/launcher/provider logs.
print_pod_timing_report() {
  local pod_name="$1"
  local role_label="$2"

  echo ""
  echo "--------------------------------------------------------------------------------"
  echo "  Timing Breakdown: ${role_label} (${pod_name})"
  echo "--------------------------------------------------------------------------------"

  local pod_json logs_text snap_json
  pod_json="$(kubectl get pod "${pod_name}" -n "${NAMESPACE}" -o json)"
  logs_text="$(kubectl logs "${pod_name}" -n "${NAMESPACE}" 2>/dev/null | grep -v -E '"GET /(metrics|health|v1/models) HTTP' || true)"
  snap_json="$(kubectl get podsnapshots -n "${NAMESPACE}" -o json 2>/dev/null || echo '{"items":[]}')"

  POD_JSON="${pod_json}" LOGS_TEXT="${logs_text}" SNAP_JSON="${snap_json}" python3 - "${role_label}" <<'PYEOF'
import datetime
import json
import os
import re
import sys

role_label = sys.argv[1]
pod_json_str = os.environ["POD_JSON"]
logs_text = os.environ["LOGS_TEXT"]
snap_json_str = os.environ["SNAP_JSON"]
pod = json.loads(pod_json_str)
snaps = json.loads(snap_json_str).get("items", [])

def parse_ts(ts_str):
    if not ts_str:
        return None
    return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

def fmt_dur(seconds):
    if seconds is None:
        return "N/A"
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = seconds - mins * 60
        return f"{seconds:.2f}s ({mins}m {secs:.1f}s)"
    return f"{seconds:.2f}s"

conditions = {
    c["type"]: parse_ts(c.get("lastTransitionTime"))
    for c in pod.get("status", {}).get("conditions", [])
    if c.get("status") == "True"
}
created_ts = parse_ts(pod.get("metadata", {}).get("creationTimestamp"))
scheduled_ts = conditions.get("PodScheduled")
restored_ts = conditions.get("PodRestored")
ready_ts = conditions.get("Ready")

container_started_ts = None
for cs in pod.get("status", {}).get("containerStatuses", []):
    running = cs.get("state", {}).get("running", {})
    if running.get("startedAt"):
        container_started_ts = parse_ts(running["startedAt"])
        break

rows = []

# 1. Kubernetes Pod Lifecycle Timings (Total PodScheduled -> Pod Ready first)
if scheduled_ts and ready_ts:
    rows.append(("K8s: Total PodScheduled -> Pod Ready", fmt_dur((ready_ts - scheduled_ts).total_seconds())))

# PodSnapshot readiness timing if available
if snaps:
    latest_snap = sorted(snaps, key=lambda s: s.get("metadata", {}).get("creationTimestamp", ""))[-1]
    snap_created = parse_ts(latest_snap.get("metadata", {}).get("creationTimestamp"))
    snap_ready = None
    for c in latest_snap.get("status", {}).get("conditions", []):
        if c.get("type") == "Ready" and c.get("status") == "True":
            snap_ready = parse_ts(c.get("lastTransitionTime"))
    if snap_created and snap_ready and restored_ts is None:
        rows.append(("K8s: PodSnapshot AwaitingCheckpoint -> Ready (GCS upload)", fmt_dur((snap_ready - snap_created).total_seconds())))

if scheduled_ts and restored_ts:
    rows.append(("K8s: PodScheduled -> PodRestored (GCS stream & restore)", fmt_dur((restored_ts - scheduled_ts).total_seconds())))
if restored_ts and ready_ts:
    rows.append(("K8s: PodRestored -> Pod Ready (wake-up + /health probe)", fmt_dur((ready_ts - restored_ts).total_seconds())))
if scheduled_ts and container_started_ts:
    rows.append(("K8s: PodScheduled -> Container started", fmt_dur((container_started_ts - scheduled_ts).total_seconds())))
if created_ts and scheduled_ts:
    rows.append(("K8s: Pod created -> PodScheduled (node provisioning)", fmt_dur((scheduled_ts - created_ts).total_seconds())))

# 2. Wrapper / Launcher / Provider Internal Step Timings from Logs
if restored_ts is not None:
    # On a restored pod, the process resumes directly after os.read('/proc/gvisor/checkpoint').
    # Pre-checkpoint phases were bypassed, and t0 before os.read was captured when Pod 1 took the snapshot.
    log_patterns = [
        ("Wrapper: Wake (resume_memory_occupation -> VRAM)", r"Resumed GPU memory occupation .* in ([0-9.]+)s"),
        ("Wrapper: Total Wake -> ServerStatus.Up (/health 200)", r"Set tokenizer_manager\.server_status = .* \(wake-to-ready=([0-9.]+)s\)"),
        ("Wrapper: launch_callback execution", r"launch_callback completed in ([0-9.]+)s"),
    ]
else:
    log_patterns = [
        ("Wrapper: Cumulative cold-start to warmup complete", r"Server warmup completed in [0-9.]+s \(cold-start elapsed_since_start=([0-9.]+)s\)"),
        ("Launcher: Hook _wait_and_warmup (first import)", r"Hooked SGLang _wait_and_warmup in ([0-9.]+)s"),
        ("Launcher: Prepare CLI server args", r"Prepared SGLang server arguments in ([0-9.]+)s"),
        ("SGLang: Engine init & weight load (to _wait_and_warmup)", r"Entering patched _wait_and_warmup .*elapsed_since_start=([0-9.]+)s"),
        ("Wrapper: Wait for checkpoint engine weights", r"Model weights are ready in GPUs \(waited ([0-9.]+)s\)"),
        ("Wrapper: Server warmup (CUDA graphs & VRAM pre-alloc)", r"Server warmup completed in ([0-9.]+)s"),
        ("Wrapper: Freeze Python GC", r"Froze Python garbage collection in ([0-9.]+)s"),
        ("Wrapper: Sleep (release_memory_occupation -> CPU RAM)", r"Released GPU memory occupation .* in ([0-9.]+)s"),
        ("Provider: Purge local HuggingFace weight cache", r"Purged \d+ cache entry/entries .* in ([0-9.]+)s"),
        ("Provider: gVisor checkpoint & GCS transfer barrier", r"gVisor checkpoint completed successfully \(barrier unblocked in ([0-9.]+)s"),
        ("Wrapper: Wake (resume_memory_occupation -> VRAM)", r"Resumed GPU memory occupation .* in ([0-9.]+)s"),
        ("Wrapper: Total Wake -> ServerStatus.Up (/health 200)", r"Set tokenizer_manager\.server_status = .* \(wake-to-ready=([0-9.]+)s\)"),
        ("Wrapper: launch_callback execution", r"launch_callback completed in ([0-9.]+)s"),
    ]

for label, pattern in log_patterns:
    matches = re.findall(pattern, logs_text)
    if matches:
        val = float(matches[-1])
        rows.append((label, fmt_dur(val)))

print(f"{'Phase / Step':<58} | {'Duration':>20}")
print("-" * 81)
for phase, dur in rows:
    print(f"{phase:<58} | {dur:>20}")
print("-" * 81)
PYEOF

  echo ""
  echo "--- Snapshot Wrapper / Provider Log Lines (${pod_name}) ---"
  grep -E "sglang\.snapshot|snapshot\.providers|\[Control Plane\]" <<<"${logs_text}" || echo "(No snapshot wrapper logs found yet)"
}

# Compare Cold Start (Pod 1) vs Restored Pod (Pod 2)
print_comparison_report() {
  local pods
  mapfile -t pods < <(kubectl get pods -l "llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')

  if [[ "${#pods[@]}" -eq 0 ]]; then
    echo "No SGLang pods found in namespace ${NAMESPACE}."
    return
  fi

  if [[ "${#pods[@]}" -ge 2 ]]; then
    local last_idx=$(( ${#pods[@]} - 1 ))
    local restored_pod="${pods[$last_idx]}"

    local pod1_json pod2_json
    pod1_json="$(kubectl get pod "${pods[0]}" -n "${NAMESPACE}" -o json)"
    pod2_json="$(kubectl get pod "${restored_pod}" -n "${NAMESPACE}" -o json)"

    python3 - "${pod1_json}" "${pod2_json}" <<'PYEOF'
import datetime
import json
import sys

p1 = json.loads(sys.argv[1])
p2 = json.loads(sys.argv[2])

def get_sched_to_ready(pod):
    conds = {
        c["type"]: datetime.datetime.fromisoformat(c["lastTransitionTime"].replace("Z", "+00:00"))
        for c in pod.get("status", {}).get("conditions", [])
        if c.get("status") == "True" and c.get("lastTransitionTime")
    }
    if "PodScheduled" in conds and "Ready" in conds:
        return (conds["Ready"] - conds["PodScheduled"]).total_seconds()
    return None

t1 = get_sched_to_ready(p1)
t2 = get_sched_to_ready(p2)
if t1 and t2 and t2 > 0:
    speedup = t1 / t2
    print("")
    print("================================================================================")
    print("  HEADLINE COMPARISON (PodScheduled -> Pod Ready):")
    print(f"    Pod 1 (Cold Start + Snapshot): {t1:.2f}s ({int(t1 // 60)}m {t1 % 60:.1f}s)")
    print(f"    Pod 2 (Snapshot Restore):      {t2:.2f}s")
    print(f"    Speedup:                       {speedup:.1f}x faster")
    print("================================================================================")
PYEOF
  fi

  print_pod_timing_report "${pods[0]}" "Pod 1 — Cold Start & Snapshot Creation"

  if [[ "${#pods[@]}" -ge 2 ]]; then
    local last_idx=$(( ${#pods[@]} - 1 ))
    local restored_pod="${pods[$last_idx]}"
    print_pod_timing_report "${restored_pod}" "Pod 2 — Restored from GCS PodSnapshot"
  fi
}

step_preflight() {
  local t_start=$SECONDS
  banner "Step 0 (Preflight): Verify Running GKE Cluster, GPU gVisor Node Pool & GCS Bucket"

  # 1. Try to infer PROJECT_ID / REGION / CLUSTER_NAME from current kubectl context if not explicitly set
  local current_ctx=""
  current_ctx="$(kubectl config current-context 2>/dev/null || true)"
  if [[ "${current_ctx}" =~ ^gke_([^_]+)_([^_]+)_(.+)$ ]]; then
    PROJECT_ID="${PROJECT_ID:-${BASH_REMATCH[1]}}"
    REGION="${REGION:-${BASH_REMATCH[2]}}"
    CLUSTER_NAME="${CLUSTER_NAME:-${BASH_REMATCH[3]}}"
  fi

  # 2. Verify GKE cluster status via gcloud (if CLUSTER_NAME & REGION are known)
  if [[ -n "${CLUSTER_NAME:-}" && -n "${REGION:-}" && -n "${PROJECT_ID:-}" ]]; then
    echo "Checking GKE cluster '${CLUSTER_NAME}' in '${REGION}' (project '${PROJECT_ID}')..."
    local cluster_status=""
    cluster_status="$(gcloud container clusters describe "${CLUSTER_NAME}" \
      --location="${REGION}" \
      --project="${PROJECT_ID}" \
      --format="value(status)" 2>/dev/null || true)"

    if [[ -z "${cluster_status}" ]]; then
      echo "ERROR: GKE cluster '${CLUSTER_NAME}' was not found in location '${REGION}' (project '${PROJECT_ID}')." >&2
      echo "Run './docker/scripts/snapshot/sglang/demo/run_demo.sh provision' to create the cluster, node pool, and bucket." >&2
      exit 1
    elif [[ "${cluster_status}" != "RUNNING" && "${cluster_status}" != "RECONCILING" ]]; then
      echo "ERROR: GKE cluster '${CLUSTER_NAME}' is in status '${cluster_status}' (expected RUNNING)." >&2
      exit 1
    fi
    echo "  [OK] GKE cluster '${CLUSTER_NAME}' status: ${cluster_status}"

    # Ensure kubectl context points to this cluster
    if [[ "${current_ctx}" != "gke_${PROJECT_ID}_${REGION}_${CLUSTER_NAME}" ]]; then
      echo "Fetching kubectl credentials for '${CLUSTER_NAME}'..."
      gcloud container clusters get-credentials "${CLUSTER_NAME}" --location="${REGION}" --project="${PROJECT_ID}"
    fi

    # 3. Check GKE node pools for GPU + gVisor sandbox
    echo "Checking GKE node pools on '${CLUSTER_NAME}'..."
    local np_json
    np_json="$(gcloud container node-pools list \
      --cluster="${CLUSTER_NAME}" \
      --location="${REGION}" \
      --project="${PROJECT_ID}" \
      --format=json)"

    python3 - "${np_json}" <<'PYEOF'
import json
import sys

pools = json.loads(sys.argv[1])
if not pools:
    print("ERROR: No node pools found on cluster.", file=sys.stderr)
    sys.exit(1)

gpu_gvisor_pools = []
for np in pools:
    name = np.get("name")
    status = np.get("status")
    cfg = np.get("config", {})
    sandbox = cfg.get("sandboxConfig", {}).get("type", "none")
    accels = cfg.get("accelerators", [])
    gpu_info = ", ".join(f"{a.get('acceleratorCount')}x {a.get('acceleratorType')}" for a in accels) or "no-gpu"
    print(f"  - NodePool '{name}': status={status}, machineType={cfg.get('machineType')}, sandbox={sandbox}, accelerators={gpu_info}")
    if status in ("RUNNING", "RECONCILING") and sandbox == "GVISOR" and accels:
        gpu_gvisor_pools.append(name)

if not gpu_gvisor_pools:
    print(
        "\nERROR: No RUNNING GPU node pool with GKE Sandbox (type=GVISOR) and NVIDIA accelerators found!",
        file=sys.stderr,
    )
    print(
        "Run './docker/scripts/snapshot/sglang/demo/run_demo.sh provision' to create a GPU gVisor node pool.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"  [OK] Found running GPU + gVisor node pool(s): {', '.join(gpu_gvisor_pools)}")
PYEOF
  else
    echo "Checking active kubectl cluster connection (context: ${current_ctx:-none})..."
    if ! kubectl cluster-info >/dev/null 2>&1; then
      echo "ERROR: kubectl cannot reach a running Kubernetes cluster." >&2
      echo "Set PROJECT_ID, REGION,ZONE, and CLUSTER_NAME and run '$0 provision' or connect kubectl to your GKE cluster." >&2
      exit 1
    fi
    echo "  [OK] Connected to Kubernetes cluster via context '${current_ctx}'."
  fi

  # 4. Verify RuntimeClass 'gvisor' and Pod Snapshots CRDs exist on the cluster
  echo "Checking 'gvisor' RuntimeClass and GKE Pod Snapshots CRDs..."
  if ! kubectl get runtimeclass gvisor >/dev/null 2>&1; then
    echo "ERROR: RuntimeClass 'gvisor' not found on the cluster. Ensure a GKE Sandbox (gVisor) node pool exists." >&2
    exit 1
  fi
  echo "  [OK] RuntimeClass 'gvisor' is present."

  for crd in \
    podsnapshots.podsnapshot.gke.io \
    podsnapshotpolicies.podsnapshot.gke.io \
    podsnapshotstorageconfigs.podsnapshot.gke.io; do
    if ! kubectl get crd "${crd}" >/dev/null 2>&1; then
      echo "ERROR: Required CRD '${crd}' is not installed on the cluster. Ensure the cluster was created with --enable-pod-snapshots." >&2
      exit 1
    fi
  done
  echo "  [OK] GKE Pod Snapshots CRDs (PodSnapshot, PodSnapshotPolicy, PodSnapshotStorageConfig) are installed."

  # 5. Show current Kubernetes nodes (including any active gVisor GPU nodes)
  echo "Checking currently registered Kubernetes nodes..."
  kubectl get nodes -L sandbox.gke.io/runtime,cloud.google.com/gke-accelerator
  local gvisor_node_count
  gvisor_node_count="$(kubectl get nodes -l sandbox.gke.io/runtime=gvisor --no-headers 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${gvisor_node_count}" -eq 0 ]]; then
    if [[ -z "${CLUSTER_NAME:-}" ]]; then
      echo "WARNING: 0 active nodes with label 'sandbox.gke.io/runtime=gvisor' right now (if node pool autoscaling is at 0 nodes, GKE will scale up when the pod is scheduled; otherwise ensure a gVisor GPU node pool exists)."
    else
      echo "  [INFO] 0 active gVisor nodes right now; GKE node pool autoscaler will provision a node when the pod is scheduled."
    fi
  else
    echo "  [OK] ${gvisor_node_count} ready gVisor GPU node(s) currently registered in the cluster."
  fi

  # 6. Verify GCS bucket if GCS_BUCKET is set
  if [[ -n "${GCS_BUCKET:-}" ]]; then
    echo "Checking GCS bucket 'gs://${GCS_BUCKET}'..."
    local hns_enabled=""
    if ! hns_enabled="$(gcloud storage buckets describe "gs://${GCS_BUCKET}" --raw --format="value(hierarchicalNamespace.enabled)" 2>/dev/null)"; then
      echo "ERROR: GCS bucket 'gs://${GCS_BUCKET}' does not exist or is not accessible." >&2
      echo "Run '$0 provision' to create a hierarchical-namespace GCS bucket." >&2
      exit 1
    fi
    if [[ "${hns_enabled}" != "True" && "${hns_enabled}" != "true" ]]; then
      echo "ERROR: GCS bucket 'gs://${GCS_BUCKET}' does not have Hierarchical Namespace enabled (hierarchicalNamespace.enabled=${hns_enabled})." >&2
      echo "GKE Pod Snapshots requires a bucket created with --enable-hierarchical-namespace." >&2
      exit 1
    fi
    echo "  [OK] GCS bucket 'gs://${GCS_BUCKET}' exists with Hierarchical Namespace enabled."
  fi

  echo ">>> Step 0 (preflight) passed in $(( SECONDS - t_start ))s."
}

step_provision() {
  require_var PROJECT_ID
  require_var REGION
  require_var ZONE
  require_var CLUSTER_NAME
  require_var GCS_BUCKET

  local t_start=$SECONDS
  banner "Provisioning Missing GKE Cluster, GPU gVisor Node Pool & GCS Bucket"

  # 1. Create GKE Cluster if missing
  if gcloud container clusters describe "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  [Skip] GKE cluster '${CLUSTER_NAME}' already exists."
  else
    local t_cl=$SECONDS
    echo "Creating GKE cluster '${CLUSTER_NAME}' with Pod Snapshots enabled..."
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
    echo ">>> Created GKE cluster '${CLUSTER_NAME}' in $(( SECONDS - t_cl ))s."
  fi

  gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}"

  # 2. Create GPU + gVisor Node Pool if missing
  if gcloud container node-pools describe "${NODE_POOL_NAME}" --cluster="${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  [Skip] Node pool '${NODE_POOL_NAME}' already exists on '${CLUSTER_NAME}'."
  else
    local t_np=$SECONDS
    echo "Creating GPU + gVisor node pool '${NODE_POOL_NAME}' (${GPU_MACHINE_TYPE}, ${GPU_COUNT}x ${GPU_ACCELERATOR})..."
    local extra_np_flags=()
    if [[ "${SPOT:-0}" == "1" ]]; then
      extra_np_flags+=("--spot")
    fi
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
      --max-nodes="${MAX_NODES}" \
      "${extra_np_flags[@]}"
    echo ">>> Created GPU gVisor node pool '${NODE_POOL_NAME}' in $(( SECONDS - t_np ))s."
  fi

  # 3. Create Hierarchical Namespace GCS Bucket & IAM role if missing
  local project_number
  project_number="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"

  if gcloud storage buckets describe "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
    echo "  [Skip] GCS bucket 'gs://${GCS_BUCKET}' already exists."
  else
    echo "Creating hierarchical-namespace GCS bucket 'gs://${GCS_BUCKET}' in '${REGION}'..."
    gcloud storage buckets create "gs://${GCS_BUCKET}" \
      --project="${PROJECT_ID}" \
      --location="${REGION}" \
      --enable-hierarchical-namespace \
      --soft-delete-duration=0 \
      --uniform-bucket-level-access
  fi

  echo "Ensuring GKE Service Agent has roles/storage.objectUser on gs://${GCS_BUCKET}..."
  gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="serviceAccount:service-${project_number}@container-engine-robot.iam.gserviceaccount.com" \
    --role="roles/storage.objectUser" >/dev/null

  if ! gcloud iam roles describe podSnapshotGcsReadWriter --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Creating custom IAM role 'podSnapshotGcsReadWriter' in project '${PROJECT_ID}'..."
    gcloud iam roles create podSnapshotGcsReadWriter \
      --project="${PROJECT_ID}" \
      --title="Pod Snapshot GCS Read/Writer" \
      --permissions="storage.buckets.get,storage.objects.get,storage.objects.list,storage.objects.create,storage.objects.delete,storage.folders.create"
  fi

  echo ">>> Provisioning completed in $(( SECONDS - t_start ))s."
}

step_setup() {
  require_var PROJECT_ID
  require_var GCS_BUCKET

  step_preflight

  local t_start=$SECONDS
  banner "Step 1: Configure GCS IAM for SGLang ServiceAccount & Setup Namespace/Router"

  local project_number
  project_number="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"

  echo "Binding Workload Identity KSA (gke-pod-snapshots-nvidia-gpu-sglang-sa) to gs://${GCS_BUCKET}..."
  gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="principal://iam.googleapis.com/projects/${project_number}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/gke-pod-snapshots-nvidia-gpu-sglang-sa" \
    --role="projects/${PROJECT_ID}/roles/podSnapshotGcsReadWriter"

  echo "Applying Gateway API Inference Extension CRDs..."
  kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml"

  echo "Creating namespace ${NAMESPACE}..."
  kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Creating/updating HuggingFace token secret 'llm-d-hf-token'..."
    kubectl create secret generic llm-d-hf-token \
      --from-literal="HF_TOKEN=${HF_TOKEN}" \
      --namespace "${NAMESPACE}" \
      --dry-run=client -o yaml | kubectl apply -f -
  elif kubectl get secret llm-d-hf-token -n "${NAMESPACE}" >/dev/null 2>&1; then
    echo "  [OK] Existing secret 'llm-d-hf-token' found in namespace '${NAMESPACE}'; reusing it."
  else
    echo "ERROR: HF_TOKEN is not set and secret 'llm-d-hf-token' does not exist in namespace '${NAMESPACE}'." >&2
    exit 1
  fi

  echo "Installing/upgrading standalone inference router..."
  helm upgrade --install "${GUIDE_NAME}" \
    "${ROUTER_STANDALONE_CHART}" \
    -f "${REPO_ROOT}/guides/recipes/router/base.values.yaml" \
    -f "${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml" \
    -n "${NAMESPACE}" --version "${ROUTER_CHART_VERSION}"

  echo ">>> Step 1 (setup) finished in $(( SECONDS - t_start ))s."
}

step_deploy() {
  require_var GCS_BUCKET

  local t_start=$SECONDS
  banner "Step 2 (Act I): Deploy SGLang Model Server & Create Initial PodSnapshot (Cold Start)"

  # Clean up any existing deployment pods and stale PodSnapshots so Pod 1 performs
  # a true cold start with the latest ConfigMap scripts and avoids racing with
  # terminating pods from a previous run.
  if [[ "${REUSE_SNAPSHOT:-0}" != "1" ]]; then
    echo "Cleaning up any pre-existing SGLang deployments, stale PodSnapshots, GCS bucket objects, and node caches for a fresh cold start..."
    kubectl delete deployment gke-pod-snapshots-nvidia-gpu-sglang-decode sglang-decode -n "${NAMESPACE}" --ignore-not-found=true --wait=true
    kubectl wait --for=delete pod -l "llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang" -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true
    kubectl delete podsnapshots --all -n "${NAMESPACE}" --ignore-not-found=true --wait=true
    gcloud storage rm -r "gs://${GCS_BUCKET}/**" 2>/dev/null || true

    for node in $(kubectl get nodes -l "cloud.google.com/gke-nodepool=${NODE_POOL_NAME}" -o jsonpath='{.items[*].metadata.name}'); do
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
  fi

  kubectl kustomize "${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/gke/sglang/" \
    | sed "s/gcs-bucket-placeholder/${GCS_BUCKET}/g" \
    | kubectl apply -n "${NAMESPACE}" -f -

  local t_pod_wait=$SECONDS
  echo "Waiting for Pod 1 to finish cold start, create snapshot, and reach Ready..."
  kubectl rollout status deployment/gke-pod-snapshots-nvidia-gpu-sglang-decode -n "${NAMESPACE}" --timeout=2400s
  echo ">>> Pod 1 reached Ready in $(( SECONDS - t_pod_wait ))s wall-clock."

  local t_snap_wait=$SECONDS
  echo "Waiting for PodSnapshot to reach Ready=True (AllSnapshotsAvailable) in GCS..."
  kubectl wait --for=condition=Ready podsnapshots --all -n "${NAMESPACE}" --timeout=600s
  echo ">>> PodSnapshot reached Ready in $(( SECONDS - t_snap_wait ))s after Pod Ready."

  echo ""
  echo "--- PodSnapshot Status ---"
  kubectl get podsnapshots -n "${NAMESPACE}"

  local pod1
  pod1="$(kubectl get pods -l "llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')"
  print_pod_timing_report "${pod1}" "Pod 1 — Cold Start & Snapshot Creation"

  echo ">>> Step 2 (deploy) finished in $(( SECONDS - t_start ))s total."
}

step_scale() {
  local t_start=$SECONDS
  banner "Step 3 (Act II): Scale Out to 2 Replicas & Restore from GCS Snapshot"

  kubectl scale deployment gke-pod-snapshots-nvidia-gpu-sglang-decode -n "${NAMESPACE}" --replicas=2

  echo "Waiting for restored replica to become Ready..."
  kubectl rollout status deployment/gke-pod-snapshots-nvidia-gpu-sglang-decode -n "${NAMESPACE}" --timeout=600s
  echo ">>> Scale-out to Ready completed in $(( SECONDS - t_start ))s wall-clock."

  echo ""
  echo "--- Pods in ${NAMESPACE} ---"
  kubectl get pods -l "llm-d.ai/guide=${GUIDE_NAME}" -n "${NAMESPACE}" -o wide

  echo ""
  echo "--- GKEPodSnapshotting Events ---"
  kubectl get events -n "${NAMESPACE}" --field-selector reason=GKEPodSnapshotting --sort-by=.lastTimestamp

  print_comparison_report
}

step_verify() {
  local t_start=$SECONDS
  banner "Step 4 (Act III): Verify Live Inference Against Pod 1 and Pod 2"

  local pod1 pod2 pod1_ip pod2_ip
  pod1="$(kubectl get pods -l "llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')"
  pod2="$(kubectl get pods -l "llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')"
  pod1_ip="$(kubectl get pod "${pod1}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"
  pod2_ip="$(kubectl get pod "${pod2}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"

  echo "Pod 1 (Cold Start): ${pod1} (${pod1_ip}:8000)"
  echo "Pod 2 (Restored):   ${pod2} (${pod2_ip}:8000)"
  echo "Sending test completion requests to both Pod 1 and Pod 2..."

  kubectl run curl-test -n "${NAMESPACE}" --rm -i --restart=Never \
    --image="${CURL_TEST_IMAGE}" \
    -- /bin/sh -c "
      echo '=== Pod 1 (${pod1} @ ${pod1_ip}:8000) ===' &&
      time curl -sS http://${pod1_ip}:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"${MODEL}\", \"prompt\": \"Pod snapshots on GKE allow SGLang to\", \"max_tokens\": 32, \"temperature\": 0}' | jq . &&
      echo '' &&
      echo '=== Pod 2 (${pod2} @ ${pod2_ip}:8000) ===' &&
      time curl -sS http://${pod2_ip}:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"${MODEL}\", \"prompt\": \"Pod snapshots on GKE allow SGLang to\", \"max_tokens\": 32, \"temperature\": 0}' | jq .
    "

  echo ">>> Step 4 (verify) finished in $(( SECONDS - t_start ))s (including test pod scheduling)."
}

step_cleanup() {
  local t_start=$SECONDS
  banner "Step 5: Cleanup Demo Resources"

  helm uninstall "${GUIDE_NAME}" -n "${NAMESPACE}" --ignore-not-found
  kubectl delete podsnapshots --all -n "${NAMESPACE}" --ignore-not-found=true
  kubectl kustomize "${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/gke/sglang/" \
    | kubectl delete -n "${NAMESPACE}" --ignore-not-found=true -f -
  kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true

  echo ">>> Step 5 (cleanup) finished in $(( SECONDS - t_start ))s."
}

case "${ACTION}" in
  preflight)
    step_preflight
    ;;
  provision)
    step_provision
    ;;
  setup)
    step_setup
    ;;
  deploy)
    step_deploy
    ;;
  scale)
    step_scale
    ;;
  verify)
    step_verify
    ;;
  report)
    print_comparison_report
    ;;
  cleanup)
    step_cleanup
    ;;
  all)
    step_setup
    pause_step
    step_deploy
    pause_step
    step_scale
    pause_step
    step_verify
    ;;
  *)
    echo "Usage: $0 [preflight|provision|setup|deploy|scale|verify|report|cleanup|all]" >&2
    exit 1
    ;;
esac
