#!/usr/bin/env bash
# Interactive step-by-step demo script for SGLang Pod Snapshots on GKE.
# Reports detailed per-step durations from both Kubernetes lifecycle conditions
# and the SGLang snapshot wrapper/launcher/provider logs.
#
# Usage:
#   ./docker/scripts/snapshot/sglang/demo/run_demo.sh [preflight|provision|setup|reset|deploy|scale|verify|report|cleanup|all]
#
# The script clears the previous run's state before the demo, not after it: `all` runs setup,
# reset, deploy, scale, verify, and a standalone `deploy` runs reset first. The reset deletes the
# SGLang Deployment and its pods, leftover helper pods, every PodSnapshot and event in NAMESPACE,
# and every object and folder in GCS_BUCKET. It also removes the SGLang image from each
# NODE_POOL_NAME node and drops the node's page cache, so that Pod 1 is a true cold start. Each
# deletion is verified; the script stops at the first failure.
#
# Output:
#   - Every command is printed as `$ command` before it runs.
#   - While a model server pod starts, each startup phase is printed as it happens (by
#     demo_phases.py watch): 1. container start -> SGLang hooked, 2. subprocesses hooked,
#     3. model weights, 4. KV cache, CUDA graph capture, warmup, 5. snapshot (release,
#     checkpoint, resume), or the restore for a pod that starts from the snapshot.
#   - deploy, scale, verify, report and all end with a timing table (demo_phases.py report): for
#     each pod, numbered steps and lettered sub-steps that add up, with the cumulative time.
#   - The milestones worth pointing out (image pulls, model weights, KV cache, CUDA graphs, the
#     snapshot and the restore, Step 3, AllSnapshotsAvailable) are shown in bold pink.
#   - If every step succeeds, the last two lines are "Success / OK" and "PROCESS COMPLETE".
#
# Required environment variables: (all optional; PROJECT_ID, CLUSTER_NAME, GCS_BUCKET, MODEL and
# the other settings below `set -euo pipefail` have defaults)
# - HF_TOKEN: Hugging Face token for the llm-d-hf-token secret (setup; not needed if it exists).
# - REUSE_SNAPSHOT: 1 skips the reset; Pod 1 may restore from the existing snapshot.
# - CLEAR_NODE_CACHES: 0 keeps the image and page cache on the nodes (no privileged cleaner pod).
# - INTERACTIVE: 0 does not pause between steps or before the reset.
# - FORCE_COLOR: 1 uses colors even when stdout is not a terminal (e.g. `| tee demo.log`).
# - NO_COLOR: 1 never uses colors.
# - SPOT: 1 creates the GPU node pool with Spot VMs (provision).
# - GAIE_URL: set by guides/env.sh.
# - ROUTER_CHART_VERSION: set by guides/env.sh.
# - ROUTER_STANDALONE_CHART: set by guides/env.sh.
# - SECONDS: bash built-in, the seconds since the script started (step timings).
# - BASH_REMATCH: bash built-in, the groups of the last =~ match.

set -euo pipefail

# Commands are shown on fd 3, a copy of stdout, so that they are visible even inside $(...).
exec 3>&1

# Colors (commands in bold cyan, highlights in bold pink) only on a terminal, or with FORCE_COLOR.
if [[ -z "${NO_COLOR:-}" && ( -t 1 || -n "${FORCE_COLOR:-}" ) ]]; then
  DEMO_COLOR=1
  CYAN=$'\033[1;36m'
  PINK=$'\033[1;38;5;205m'
  RESET=$'\033[0m'
else
  DEMO_COLOR=0
  CYAN=''
  PINK=''
  RESET=''
fi
# demo_phases.py colors the same way.
export DEMO_COLOR

REPO_ROOT="$(realpath "$(git rev-parse --show-toplevel)")"
DEMO_PHASES="${REPO_ROOT}/docker/scripts/snapshot/sglang/demo/demo_phases.py"
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
NODE_CLEANER_IMAGE="${NODE_CLEANER_IMAGE:-alpine:3.20}"
CLEAR_NODE_CACHES="${CLEAR_NODE_CACHES:-1}"
REUSE_SNAPSHOT="${REUSE_SNAPSHOT:-0}"
ACTION="${1:-all}"

OVERLAY_DIR="${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/gke/sglang"
DEPLOYMENT="gke-pod-snapshots-nvidia-gpu-sglang-decode"
POD_SELECTOR="llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/engine-type=sglang"
# Set by step_reset, so `all` does not reset twice.
RESET_DONE=0
# For the timing table: the wall-clock time of each step ("label=seconds") and the latency of
# each test request ("Pod N=seconds").
STEP_TIMES=()
VERIFY_LATENCIES=()

# Allow-lists for values that reach gcloud, kubectl and the privileged node cleaner.
BUCKET_RE='^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$'
IMAGE_RE='^[a-z0-9][a-z0-9._-]*(:[0-9]+)?(/[a-z0-9][a-z0-9._-]*)*(:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?(@sha256:[a-f0-9]{64})?$'
K8S_NAME_RE='^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$'
IP_RE='^[0-9A-Fa-f.:]+$'

# Source common guide variables (GAIE_URL, ROUTER_STANDALONE_CHART, ROUTER_CHART_VERSION, etc.)
# shellcheck disable=SC1091
source "${REPO_ROOT}/guides/env.sh"

banner() {
  echo ""
  echo "================================================================================"
  echo "  $*"
  echo "================================================================================"
}

# A banner in bold pink, for the steps the audience should notice.
pink_banner() {
  printf '%s' "${PINK}"
  banner "$@"
  printf '%s' "${RESET}"
}

# Copies stdin to stdout with every WORD in bold pink: some_command | pink_word WORD.
pink_word() {
  if [[ "${DEMO_COLOR}" == "1" ]]; then
    sed "s/$1/${PINK}&${RESET}/g"
  else
    cat
  fi
}

pause_step() {
  local prompt="${1:-Press [Enter] to continue to the next step (or Ctrl+C to stop)...}"
  if [[ "${INTERACTIVE:-1}" == "1" ]]; then
    read -r -p $'\n>>> '"${prompt} "
  fi
}

require_var() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: Environment variable ${var_name} must be set." >&2
    exit 1
  fi
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prints the arguments as one command line; arguments with special characters are single-quoted.
quote_cmd() {
  local line="" arg sq="'" esc="'\\''"
  for arg in "$@"; do
    if [[ "${arg}" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then
      line+="${arg} "
    else
      line+="'${arg//${sq}/${esc}}' "
    fi
  done
  printf '%s' "${line% }"
}

# Prints a command line, given as text, before it runs (bold cyan on a terminal).
show_line() {
  printf '\n%s$ %s%s\n' "${CYAN}" "$*" "${RESET}" >&3
}

# Prints a command, given as arguments, before it runs.
show_cmd() {
  show_line "$(quote_cmd "$@")"
}

# Prints a command, then runs it.
run() {
  show_cmd "$@"
  "$@"
}

# Records the wall-clock time of a step for the timing table: record_step LABEL START_SECONDS.
record_step() {
  STEP_TIMES+=("$1=$(( SECONDS - $2 ))")
}

# Waits for the Deployment rollout like `kubectl rollout status` (same exit code), and meanwhile
# prints each startup phase of the new pod(s), labeled LABEL.
wait_for_rollout() {
  local label="$1" timeout_s="$2"
  show_cmd kubectl rollout status "deployment/${DEPLOYMENT}" -n "${NAMESPACE}" "--timeout=${timeout_s}s"
  echo "  While it waits, $(basename "${DEMO_PHASES}") prints each startup phase of ${label} from"
  echo "  kubectl get pods, kubectl get events and kubectl logs --timestamps (polled every 5s):"
  python3 "${DEMO_PHASES}" watch --namespace "${NAMESPACE}" --selector "${POD_SELECTOR}" \
    --deployment "${DEPLOYMENT}" --timeout "${timeout_s}" --label "${label}"
}

# Prints the timing table: the pod lifecycle and the SGLang phases of Pod 1 (cold start) and
# Pod 2 (restore), the wall-clock time of each step of this run, and the test request latencies.
print_timing_table() {
  local args=(report --namespace "${NAMESPACE}" --selector "${POD_SELECTOR}") entry
  for entry in ${STEP_TIMES[@]+"${STEP_TIMES[@]}"}; do
    args+=(--step "${entry}")
  done
  for entry in ${VERIFY_LATENCIES[@]+"${VERIFY_LATENCIES[@]}"}; do
    args+=(--latency "${entry}")
  done
  run python3 "${DEMO_PHASES}" "${args[@]}"
}

step_preflight() {
  local t_start=$SECONDS
  banner "Step 0 (Preflight): Verify Running GKE Cluster, GPU gVisor Node Pool & GCS Bucket"

  # 1. Try to infer PROJECT_ID / REGION / CLUSTER_NAME from current kubectl context if not explicitly set
  local current_ctx=""
  show_cmd kubectl config current-context
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
    show_cmd gcloud container clusters describe "${CLUSTER_NAME}" --location="${REGION}" \
      --project="${PROJECT_ID}" --format="value(status)"
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
      run gcloud container clusters get-credentials "${CLUSTER_NAME}" --location="${REGION}" --project="${PROJECT_ID}"
    fi

    # 3. Check GKE node pools for GPU + gVisor sandbox
    echo "Checking GKE node pools on '${CLUSTER_NAME}'..."
    local np_json
    show_cmd gcloud container node-pools list --cluster="${CLUSTER_NAME}" --location="${REGION}" \
      --project="${PROJECT_ID}" --format=json
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
    show_cmd kubectl cluster-info
    if ! kubectl cluster-info >/dev/null 2>&1; then
      echo "ERROR: kubectl cannot reach a running Kubernetes cluster." >&2
      echo "Set PROJECT_ID, REGION,ZONE, and CLUSTER_NAME and run '$0 provision' or connect kubectl to your GKE cluster." >&2
      exit 1
    fi
    echo "  [OK] Connected to Kubernetes cluster via context '${current_ctx}'."
  fi

  # 4. Verify RuntimeClass 'gvisor' and Pod Snapshots CRDs exist on the cluster
  echo "Checking 'gvisor' RuntimeClass and GKE Pod Snapshots CRDs..."
  show_cmd kubectl get runtimeclass gvisor
  if ! kubectl get runtimeclass gvisor >/dev/null 2>&1; then
    echo "ERROR: RuntimeClass 'gvisor' not found on the cluster. Ensure a GKE Sandbox (gVisor) node pool exists." >&2
    exit 1
  fi
  echo "  [OK] RuntimeClass 'gvisor' is present."

  for crd in \
    podsnapshots.podsnapshot.gke.io \
    podsnapshotpolicies.podsnapshot.gke.io \
    podsnapshotstorageconfigs.podsnapshot.gke.io; do
    show_cmd kubectl get crd "${crd}"
    if ! kubectl get crd "${crd}" >/dev/null 2>&1; then
      echo "ERROR: Required CRD '${crd}' is not installed on the cluster. Ensure the cluster was created with --enable-pod-snapshots." >&2
      exit 1
    fi
  done
  echo "  [OK] GKE Pod Snapshots CRDs (PodSnapshot, PodSnapshotPolicy, PodSnapshotStorageConfig) are installed."

  # 5. Show current Kubernetes nodes (including any active gVisor GPU nodes)
  echo "Checking currently registered Kubernetes nodes..."
  run kubectl get nodes -L sandbox.gke.io/runtime,cloud.google.com/gke-accelerator
  local gvisor_node_count
  show_line "kubectl get nodes -l sandbox.gke.io/runtime=gvisor --no-headers | wc -l"
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
    show_cmd gcloud storage buckets describe "gs://${GCS_BUCKET}" --raw --format="value(hierarchicalNamespace.enabled)"
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

  record_step "Preflight: cluster, node pool, CRDs, bucket" "${t_start}"
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
  show_cmd gcloud container clusters describe "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}"
  if gcloud container clusters describe "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  [Skip] GKE cluster '${CLUSTER_NAME}' already exists."
  else
    local t_cl=$SECONDS
    echo "Creating GKE cluster '${CLUSTER_NAME}' with Pod Snapshots enabled..."
    run gcloud container clusters create "${CLUSTER_NAME}" \
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

  run gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}"

  # 2. Create GPU + gVisor Node Pool if missing
  show_cmd gcloud container node-pools describe "${NODE_POOL_NAME}" --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" --project="${PROJECT_ID}"
  if gcloud container node-pools describe "${NODE_POOL_NAME}" --cluster="${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  [Skip] Node pool '${NODE_POOL_NAME}' already exists on '${CLUSTER_NAME}'."
  else
    local t_np=$SECONDS
    echo "Creating GPU + gVisor node pool '${NODE_POOL_NAME}' (${GPU_MACHINE_TYPE}, ${GPU_COUNT}x ${GPU_ACCELERATOR})..."
    local extra_np_flags=()
    if [[ "${SPOT:-0}" == "1" ]]; then
      extra_np_flags+=("--spot")
    fi
    run gcloud container node-pools create "${NODE_POOL_NAME}" \
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
      ${extra_np_flags[@]+"${extra_np_flags[@]}"}
    echo ">>> Created GPU gVisor node pool '${NODE_POOL_NAME}' in $(( SECONDS - t_np ))s."
  fi

  # 3. Create Hierarchical Namespace GCS Bucket & IAM role if missing
  local project_number
  show_cmd gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)"
  project_number="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"

  show_cmd gcloud storage buckets describe "gs://${GCS_BUCKET}"
  if gcloud storage buckets describe "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
    echo "  [Skip] GCS bucket 'gs://${GCS_BUCKET}' already exists."
  else
    echo "Creating hierarchical-namespace GCS bucket 'gs://${GCS_BUCKET}' in '${REGION}'..."
    run gcloud storage buckets create "gs://${GCS_BUCKET}" \
      --project="${PROJECT_ID}" \
      --location="${REGION}" \
      --enable-hierarchical-namespace \
      --soft-delete-duration=0 \
      --uniform-bucket-level-access
  fi

  echo "Ensuring GKE Service Agent has roles/storage.objectUser on gs://${GCS_BUCKET}..."
  run gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="serviceAccount:service-${project_number}@container-engine-robot.iam.gserviceaccount.com" \
    --role="roles/storage.objectUser" >/dev/null

  show_cmd gcloud iam roles describe podSnapshotGcsReadWriter --project="${PROJECT_ID}"
  if ! gcloud iam roles describe podSnapshotGcsReadWriter --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Creating custom IAM role 'podSnapshotGcsReadWriter' in project '${PROJECT_ID}'..."
    run gcloud iam roles create podSnapshotGcsReadWriter \
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
  show_cmd gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)"
  project_number="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"

  echo "Binding Workload Identity KSA (gke-pod-snapshots-nvidia-gpu-sglang-sa) to gs://${GCS_BUCKET}..."
  run gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="principal://iam.googleapis.com/projects/${project_number}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/gke-pod-snapshots-nvidia-gpu-sglang-sa" \
    --role="projects/${PROJECT_ID}/roles/podSnapshotGcsReadWriter"

  echo "Applying Gateway API Inference Extension CRDs..."
  run kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml"

  echo "Creating namespace ${NAMESPACE}..."
  show_line "kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -"
  kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Creating/updating HuggingFace token secret 'llm-d-hf-token'..."
    # The token itself is never printed.
    show_line "kubectl create secret generic llm-d-hf-token --from-literal=HF_TOKEN=<redacted> --namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -"
    kubectl create secret generic llm-d-hf-token \
      --from-literal="HF_TOKEN=${HF_TOKEN}" \
      --namespace "${NAMESPACE}" \
      --dry-run=client -o yaml | kubectl apply -f -
  else
    show_cmd kubectl get secret llm-d-hf-token -n "${NAMESPACE}"
    if kubectl get secret llm-d-hf-token -n "${NAMESPACE}" >/dev/null 2>&1; then
      echo "  [OK] Existing secret 'llm-d-hf-token' found in namespace '${NAMESPACE}'; reusing it."
    else
      echo "ERROR: HF_TOKEN is not set and secret 'llm-d-hf-token' does not exist in namespace '${NAMESPACE}'." >&2
      exit 1
    fi
  fi

  echo "Installing/upgrading standalone inference router..."
  run helm upgrade --install "${GUIDE_NAME}" \
    "${ROUTER_STANDALONE_CHART}" \
    -f "${REPO_ROOT}/guides/recipes/router/base.values.yaml" \
    -f "${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml" \
    -n "${NAMESPACE}" --version "${ROUTER_CHART_VERSION}"

  record_step "Setup: IAM, CRDs, namespace, secret, router" "${t_start}"
  echo ">>> Step 1 (setup) finished in $(( SECONDS - t_start ))s."
}

# Prints the distinct images of the model server Deployment rendered from OVERLAY_DIR, one per line.
modelserver_images() {
  show_line "kubectl kustomize ${OVERLAY_DIR} | python3 -c '<print the image(s) of the rendered Deployment>'"
  kubectl kustomize "${OVERLAY_DIR}" | python3 -c '
import re
import sys

docs = [d for d in re.split(r"(?m)^---\s*$", sys.stdin.read())
        if re.search(r"(?m)^kind: Deployment\s*$", d)]
if len(docs) != 1:
    sys.exit(f"expected 1 Deployment in the rendered overlay, found {len(docs)}")
images = re.findall(r"(?m)^\s*(?:- )?image:\s*\"?([^\"\s]+)\"?\s*$", docs[0])
if not images:
    sys.exit("found no image in the rendered Deployment")
print("\n".join(dict.fromkeys(images)))
'
}

# Waits until `kubectl get KIND [-l SELECTOR]` in NAMESPACE returns nothing; fails after TIMEOUT_S.
wait_until_gone() {
  local kind="$1" selector="$2" timeout_s="$3" left
  local deadline=$(( SECONDS + timeout_s ))
  local args=(get "${kind}" -n "${NAMESPACE}" -o name)
  if [[ -n "${selector}" ]]; then
    args+=(-l "${selector}")
  fi
  show_line "$(quote_cmd kubectl "${args[@]}")    # every 5s until nothing is left (up to ${timeout_s}s)"
  while true; do
    left="$(kubectl "${args[@]}")" || return 1
    if [[ -z "${left}" ]]; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "Still present after ${timeout_s}s: ${left//$'\n'/ }" >&2
      return 1
    fi
    sleep 5
  done
}

# Prints the top-level objects and folders of gs://GCS_BUCKET, one per line; nothing if empty.
gcs_top_level() {
  local out err rc=0
  show_cmd gcloud storage ls "gs://${GCS_BUCKET}"
  err="$(mktemp)"
  out="$(gcloud storage ls "gs://${GCS_BUCKET}" 2>"${err}")" || rc=$?
  if (( rc != 0 )) && ! grep -q "matched no objects" "${err}"; then
    cat "${err}" >&2
    rm -f "${err}"
    return 1
  fi
  rm -f "${err}"
  if [[ -n "${out}" ]]; then
    printf '%s\n' "${out}"
  fi
}

# Deletes every object, object version and folder in gs://GCS_BUCKET, but never the bucket, and
# verifies that it is empty. Retries, since objects can change or disappear while being deleted.
clear_bucket() {
  local listing entries=() entry attempt
  for attempt in 1 2 3 4; do
    listing="$(gcs_top_level)" || die "could not list gs://${GCS_BUCKET}"
    if [[ -z "${listing}" ]]; then
      echo "  [OK] gs://${GCS_BUCKET} is empty."
      return 0
    fi
    if (( attempt == 4 )); then
      break
    fi
    mapfile -t entries <<<"${listing}"
    for entry in "${entries[@]}"; do
      # rm --recursive on the bucket URL itself would delete the bucket.
      [[ "${entry}" == "gs://${GCS_BUCKET}/"?* ]] || die "refusing to delete unexpected entry '${entry}'"
    done
    echo "Deleting ${#entries[@]} top-level object(s)/folder(s) from gs://${GCS_BUCKET}..."
    run gcloud storage rm --recursive "${entries[@]}" \
      || echo "WARNING: gcloud storage rm reported errors; checking the bucket again..." >&2
  done
  die "gs://${GCS_BUCKET} is still not empty: ${listing//$'\n'/ }"
}

# TODO(security): runs a privileged, hostPID pod on every NODE_POOL_NAME node that enters the
# host's namespaces to remove the model server image(s) from containerd and drop the page cache,
# so Pod 1 is a true cold start. Demo clusters only; CLEAR_NODE_CACHES=0 skips it. Node names and
# images are validated and passed as positional parameters, never spliced into the script.
clear_node_caches() {
  local listing nodes=() node overrides script i=0
  show_cmd kubectl get nodes -l "cloud.google.com/gke-nodepool=${NODE_POOL_NAME}" -o 'jsonpath={.items[*].metadata.name}'
  listing="$(kubectl get nodes -l "cloud.google.com/gke-nodepool=${NODE_POOL_NAME}" \
    -o jsonpath='{.items[*].metadata.name}')" || die "could not list the ${NODE_POOL_NAME} nodes"
  read -r -a nodes <<<"${listing}"
  if (( ${#nodes[@]} == 0 )); then
    echo "  [OK] ${NODE_POOL_NAME} has no nodes; the autoscaler will provision a fresh one."
    return 0
  fi
  # COS keeps crictl in /home/kubernetes/bin, which is not on the default PATH.
  script="$(cat <<'SH'
node="$1"; shift
export PATH="/home/kubernetes/bin:$PATH"
export CONTAINER_RUNTIME_ENDPOINT=unix:///run/containerd/containerd.sock
export IMAGE_SERVICE_ENDPOINT=unix:///run/containerd/containerd.sock
cached() { crictl inspecti -o json "$1" 2>/dev/null | grep -q '"id"'; }
if ! crictl images >/dev/null; then
  echo "  [FAIL] crictl cannot list the images on $node" >&2
  exit 1
fi
rc=0
for image in "$@"; do
  if ! cached "$image"; then
    echo "  [OK] $image is not cached on $node"
  elif crictl rmi "$image" >/dev/null && ! cached "$image"; then
    echo "  [OK] removed $image from $node"
  else
    echo "  [FAIL] could not remove $image from $node" >&2
    rc=1
  fi
done
if sync && echo 3 >/proc/sys/vm/drop_caches; then
  echo "  [OK] dropped the page cache on $node"
else
  echo "  [FAIL] could not drop the page cache on $node" >&2
  rc=1
fi
exit "$rc"
SH
)"
  for node in "${nodes[@]}"; do
    [[ "${node}" =~ ${K8S_NAME_RE} ]] || die "unexpected node name '${node}'"
    overrides="$(python3 - "${node}" "${NODE_CLEANER_IMAGE}" "${script}" "$@" <<'PY'
import json
import sys

node, image, script, *images = sys.argv[1:]
print(json.dumps({
    "apiVersion": "v1",
    "spec": {
        "nodeName": node,
        "hostPID": True,
        "automountServiceAccountToken": False,
        "tolerations": [{"operator": "Exists"}],
        "containers": [{
            "name": "cleaner",
            "image": image,
            "securityContext": {"privileged": True},
            # The node name and images are positional parameters of sh -c, never script text.
            "command": ["nsenter", "-t", "1", "-m", "-u", "-i", "-n", "--",
                        "/bin/sh", "-c", script, "sh", node, *images],
        }],
    },
}))
PY
)" || die "could not build the cleaner pod spec for ${node}"
    echo "Clearing ${node}..."
    show_line "kubectl run cache-cleaner-${i}-${node##*-} -n ${NAMESPACE} --rm --attach --restart=Never --pod-running-timeout=5m --image=${NODE_CLEANER_IMAGE} --overrides='<privileged hostPID pod on ${node}: nsenter -t 1 -m -u -i -n -- /bin/sh -c \"crictl rmi ${*}; sync; echo 3 > /proc/sys/vm/drop_caches\">'"
    kubectl run "cache-cleaner-${i}-${node##*-}" -n "${NAMESPACE}" --rm --attach --restart=Never \
      --pod-running-timeout=5m --image="${NODE_CLEANER_IMAGE}" --overrides="${overrides}" \
      || die "could not clear the caches on ${node} (CLEAR_NODE_CACHES=0 skips this step)"
    i=$(( i + 1 ))
  done
}

step_reset() {
  require_var GCS_BUCKET
  [[ "${GCS_BUCKET}" =~ ${BUCKET_RE} ]] || die "unexpected GCS_BUCKET '${GCS_BUCKET}'"
  [[ "${NODE_CLEANER_IMAGE}" =~ ${IMAGE_RE} ]] || die "unexpected NODE_CLEANER_IMAGE '${NODE_CLEANER_IMAGE}'"

  banner "Reset: Clear PodSnapshots, GCS Objects, Cached Image & Page Cache for a Fresh Cold Start"

  local listing images=() image helper_pods=()
  listing="$(modelserver_images)" || die "could not read the model server image(s) from ${OVERLAY_DIR}"
  mapfile -t images <<<"${listing}"
  for image in "${images[@]}"; do
    [[ "${image}" =~ ${IMAGE_RE} ]] || die "unexpected image reference '${image}' in ${OVERLAY_DIR}"
  done

  echo "This deletes:"
  echo "  - Deployment ${DEPLOYMENT} (and the legacy sglang-decode) and their pods in ${NAMESPACE}"
  echo "  - leftover curl-test and cache-cleaner pods from interrupted runs"
  echo "  - every PodSnapshot in ${NAMESPACE}"
  echo "  - every object, object version and folder in gs://${GCS_BUCKET}"
  if [[ "${CLEAR_NODE_CACHES}" == "1" ]]; then
    echo "  - ${images[*]} and the page cache on every ${NODE_POOL_NAME} node (privileged pod)"
  fi
  echo "  - every event in ${NAMESPACE}"
  pause_step "Press [Enter] to delete all of the above (or Ctrl+C to stop)..."

  local t_start=$SECONDS
  show_line "kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -"
  kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

  echo "Deleting the SGLang Deployment and waiting for its pods to terminate..."
  run kubectl delete deployment "${DEPLOYMENT}" sglang-decode -n "${NAMESPACE}" \
    --ignore-not-found=true --wait=true --timeout=120s || die "could not delete the SGLang Deployment"
  wait_until_gone pods "${POD_SELECTOR}" 300 || die "could not confirm that the SGLang pods are gone"

  show_cmd kubectl get pods -n "${NAMESPACE}" -o name
  listing="$(kubectl get pods -n "${NAMESPACE}" -o name)" || die "could not list the pods in ${NAMESPACE}"
  mapfile -t helper_pods < <(grep -E '^pod/(curl-test|cache-cleaner-[a-z0-9-]+)$' <<<"${listing}" || true)
  if (( ${#helper_pods[@]} > 0 )); then
    echo "Deleting leftover helper pods: ${helper_pods[*]}"
    run kubectl delete -n "${NAMESPACE}" --ignore-not-found=true --wait=true --timeout=120s "${helper_pods[@]}" \
      || die "could not delete the leftover helper pods"
  fi

  echo "Deleting PodSnapshots..."
  run kubectl delete podsnapshots --all -n "${NAMESPACE}" --wait=true --timeout=300s \
    || die "could not delete the PodSnapshots in ${NAMESPACE}"
  wait_until_gone podsnapshots "" 60 || die "could not confirm that the PodSnapshots are gone"

  echo "Emptying gs://${GCS_BUCKET}..."
  clear_bucket

  if [[ "${CLEAR_NODE_CACHES}" == "1" ]]; then
    echo "Removing the image(s) and dropping the page cache on the ${NODE_POOL_NAME} nodes..."
    clear_node_caches "${images[@]}"
  else
    echo "WARNING: CLEAR_NODE_CACHES=0: the nodes keep the image and page cache; Pod 1 may start warm." >&2
  fi

  show_cmd kubectl delete events --all -n "${NAMESPACE}"
  kubectl delete events --all -n "${NAMESPACE}" >/dev/null \
    || echo "WARNING: could not delete the old events; step 3 may list events from earlier runs." >&2

  RESET_DONE=1
  record_step "Reset: snapshots, bucket, node caches" "${t_start}"
  echo ">>> Reset finished in $(( SECONDS - t_start ))s."
}

step_deploy() {
  require_var GCS_BUCKET
  [[ "${GCS_BUCKET}" =~ ${BUCKET_RE} ]] || die "unexpected GCS_BUCKET '${GCS_BUCKET}'"

  # Pod 1 must cold start; `all` has already reset by this point.
  if [[ "${REUSE_SNAPSHOT}" == "1" ]]; then
    echo "REUSE_SNAPSHOT=1: skipping the reset; Pod 1 may restore from the existing snapshot instead of cold starting."
  elif [[ "${RESET_DONE}" != "1" ]]; then
    step_reset
  fi

  local t_start=$SECONDS
  banner "Step 2 (Act I): Deploy SGLang Model Server & Create Initial PodSnapshot (Cold Start)"

  show_line "kubectl kustomize ${OVERLAY_DIR} | sed s/gcs-bucket-placeholder/${GCS_BUCKET}/g | kubectl apply -n ${NAMESPACE} -f -"
  kubectl kustomize "${OVERLAY_DIR}" \
    | sed "s/gcs-bucket-placeholder/${GCS_BUCKET}/g" \
    | kubectl apply -n "${NAMESPACE}" -f -

  local t_pod_wait=$SECONDS
  echo "Waiting for Pod 1 to cold start, create the snapshot, and become Ready (about 6 minutes)..."
  wait_for_rollout "Pod 1" 2400
  echo ">>> Pod 1 reached Ready in $(( SECONDS - t_pod_wait ))s wall-clock."

  local t_snap_wait=$SECONDS
  echo "Waiting for PodSnapshot to reach Ready=True (${PINK}AllSnapshotsAvailable${RESET}) in GCS..."
  run kubectl wait --for=condition=Ready podsnapshots --all -n "${NAMESPACE}" --timeout=600s
  echo ">>> PodSnapshot reached Ready in $(( SECONDS - t_snap_wait ))s after Pod Ready."

  echo ""
  echo "--- PodSnapshot Status ---"
  run kubectl get podsnapshots -n "${NAMESPACE}" | pink_word AllSnapshotsAvailable

  record_step "Deploy: Pod 1 cold start, snapshot Ready" "${t_start}"
  echo ">>> Step 2 (deploy) finished in $(( SECONDS - t_start ))s total."
}

step_scale() {
  local t_start=$SECONDS
  pink_banner "Step 3 (Act II): Scale Out to 2 Replicas & Restore from GCS Snapshot"

  run kubectl scale deployment "${DEPLOYMENT}" -n "${NAMESPACE}" --replicas=2

  echo "Waiting for Pod 2 to restore from the snapshot and become Ready..."
  wait_for_rollout "Pod 2" 600
  echo ">>> Scale-out to Ready completed in $(( SECONDS - t_start ))s wall-clock."

  echo ""
  echo "--- Pods in ${NAMESPACE} ---"
  run kubectl get pods -l "llm-d.ai/guide=${GUIDE_NAME}" -n "${NAMESPACE}" -o wide

  echo ""
  echo "--- GKEPodSnapshotting Events ---"
  run kubectl get events -n "${NAMESPACE}" --field-selector reason=GKEPodSnapshotting --sort-by=.lastTimestamp

  record_step "Scale: Pod 2 restore from the snapshot" "${t_start}"
}

step_verify() {
  local t_start=$SECONDS
  banner "Step 4 (Act III): Verify Live Inference Against Pod 1 and Pod 2"

  local pod1 pod2 pod1_ip pod2_ip out rc=0 line script
  show_cmd kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp \
    -o 'jsonpath={.items[0].metadata.name} {.items[-1].metadata.name}'
  pod1="$(kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')"
  pod2="$(kubectl get pods -l "${POD_SELECTOR}" -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')"
  show_cmd kubectl get pod "${pod1}" "${pod2}" -n "${NAMESPACE}" -o 'jsonpath={.status.podIP}'
  pod1_ip="$(kubectl get pod "${pod1}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"
  pod2_ip="$(kubectl get pod "${pod2}" -n "${NAMESPACE}" -o jsonpath='{.status.podIP}')"
  [[ "${pod1_ip}" =~ ${IP_RE} && "${pod2_ip}" =~ ${IP_RE} ]] \
    || die "unexpected pod IPs '${pod1_ip}' and '${pod2_ip}'"

  echo "Pod 1 (Cold Start): ${pod1} (${pod1_ip}:8000)"
  echo "Pod 2 (Restored):   ${pod2} (${pod2_ip}:8000)"
  echo "Sending test completion requests to both Pod 1 and Pod 2..."

  # Runs in the test pod. The model and the pod IPs are positional parameters, never script text.
  # Fails if a request fails or a reply has no completion text.
  script="$(cat <<'SH'
model="$1"
shift
rc=0
while [ "$#" -ge 2 ]; do
  name="$1"
  ip="$2"
  shift 2
  case "$ip" in
    *:*) url="http://[$ip]:8000/v1/completions" ;;
    *) url="http://$ip:8000/v1/completions" ;;
  esac
  body=$(jq -cn --arg model "$model" \
    '{model: $model, prompt: "Pod snapshots on GKE allow SGLang to", max_tokens: 32, temperature: 0}')
  echo ""
  echo "=== $name ($ip:8000) ==="
  echo "\$ curl -fsS --max-time 300 -H 'Content-Type: application/json' -d '$body' $url"
  # Timed with /proc/uptime: the time_total of this image's curl (7.55.0) is often 2^32 us too high.
  t0=$(cut -d' ' -f1 /proc/uptime)
  if ! out=$(curl -fsS --max-time 300 -H 'Content-Type: application/json' -d "$body" "$url"); then
    echo "FAILED: the completion request to $name failed." >&2
    rc=1
    continue
  fi
  t1=$(cut -d' ' -f1 /proc/uptime)
  secs=$(awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "%.2f", t1 - t0 }')
  printf '%s\n' "$out" | jq .
  if ! printf '%s\n' "$out" | jq -e '.choices[0].text | type == "string"' >/dev/null; then
    echo "FAILED: $name returned no completion text." >&2
    rc=1
    continue
  fi
  echo "Request to $name completed in ${secs}s"
done
exit "$rc"
SH
)"
  show_line "kubectl run curl-test -n ${NAMESPACE} --rm -i --restart=Never --image=${CURL_TEST_IMAGE} -- /bin/sh -c '<for each pod: curl /v1/completions and check the reply>' sh ${MODEL} 'Pod 1' ${pod1_ip} 'Pod 2' ${pod2_ip}"
  out="$(mktemp)"
  kubectl run curl-test -n "${NAMESPACE}" --rm -i --restart=Never \
    --image="${CURL_TEST_IMAGE}" \
    -- /bin/sh -c "${script}" sh "${MODEL}" "Pod 1" "${pod1_ip}" "Pod 2" "${pod2_ip}" | tee "${out}" || rc=$?
  while IFS= read -r line; do
    if [[ "${line}" =~ ^Request\ to\ (Pod\ [12])\ completed\ in\ ([0-9.]+)s$ ]]; then
      VERIFY_LATENCIES+=("${BASH_REMATCH[1]}=${BASH_REMATCH[2]}")
    fi
  done <"${out}"
  rm -f "${out}"
  (( rc == 0 )) || die "the test requests failed (kubectl run exited with code ${rc})"
  (( ${#VERIFY_LATENCIES[@]} == 2 )) || die "expected a completion from both Pod 1 and Pod 2"

  record_step "Verify: test pod, 2 completion requests" "${t_start}"
  echo ">>> Step 4 (verify) finished in $(( SECONDS - t_start ))s (including test pod scheduling)."
}

step_cleanup() {
  local t_start=$SECONDS
  banner "Step 5: Cleanup Demo Resources"

  run helm uninstall "${GUIDE_NAME}" -n "${NAMESPACE}" --ignore-not-found
  run kubectl delete podsnapshots --all -n "${NAMESPACE}" --ignore-not-found=true
  show_line "kubectl kustomize ${OVERLAY_DIR} | kubectl delete -n ${NAMESPACE} --ignore-not-found=true -f -"
  kubectl kustomize "${OVERLAY_DIR}" \
    | kubectl delete -n "${NAMESPACE}" --ignore-not-found=true -f -
  run kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true

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
  reset)
    step_reset
    ;;
  deploy)
    step_deploy
    print_timing_table
    ;;
  scale)
    step_scale
    print_timing_table
    ;;
  verify)
    step_verify
    print_timing_table
    ;;
  report)
    print_timing_table
    ;;
  cleanup)
    step_cleanup
    ;;
  all)
    step_setup
    # step_reset pauses before deleting, which doubles as the pause after setup.
    if [[ "${REUSE_SNAPSHOT}" != "1" ]]; then
      step_reset
    fi
    pause_step
    step_deploy
    pause_step
    step_scale
    pause_step
    step_verify
    print_timing_table
    ;;
  *)
    echo "Usage: $0 [preflight|provision|setup|reset|deploy|scale|verify|report|cleanup|all]" >&2
    exit 1
    ;;
esac

# Reached only if every step succeeded: any failure exits above (set -e, die).
echo "Success / OK"
echo "PROCESS COMPLETE"
