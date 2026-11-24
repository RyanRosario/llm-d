# CRIU P1: checkpoint on pod 1, restore on pod 2 (different node) via GCS

Second step after [P0](../p0/README.md). Same CRIU + CUDA plugin mechanism, but now set up like the
[gVisor Pod Snapshots guide](../../../../../guides/pod-snapshot/README.md): a Deployment at `replicas: 1`
creates the snapshot, scaling to `2` restores it on another node, images travel through GCS. This is the
experiment whose numbers can sit next to the gVisor benchmark's.

## Files

- `criu-entrypoint.sh` — container entrypoint. Computes a snapshot key (model, args, driver, GPU, CRIU and vLLM
  versions), then either **cold → dump → upload → wake → serve** (no images for the key) or
  **download → restore → wake → serve** (images found). Reuses the P0 driver for the individual steps.
- `Dockerfile` — P0 image + gcloud CLI + entrypoint.
- `deployment.yaml` — ServiceAccount + Deployment (privileged, 1 GPU, pod anti-affinity so pod 2 lands on a second node).

## What changed from P0 for cross-pod restore

| Concern | P0 (same pod) | P1 |
| :--- | :--- | :--- |
| Images location | emptyDir | `gs://<bucket>/criu/<key>/` via `gcloud storage rsync` (Workload Identity) |
| Files the tree has open/mapped under `/tmp`, `/dev/shm`, `/root`, `/checkpoints` | already there | `collect_runtime_files` lists them from `/proc/<pid>/{maps,fd}` and ships `aux.tar`; restore untars it first |
| Link remaps | copied back from `images/remaps/` | same (they ride along in the bucket) |
| PID collision with pod 2's own early PIDs | not an issue | pod 1 sets `ns_last_pid=20000` before launching vLLM |
| GPU identity | same GPU | `--plugin-option=cuda_plugin.device-map=auto` (checkpoint GPU 0 → restore node's GPU 0) |
| Serving | script exits | entrypoint supervises the vLLM PID; readiness on `GET /health` |

## Run

```bash
IMAGE=us-central1-docker.pkg.dev/ryanrosario-gke-dev/vllm/vllm-criu:v0.30.0-criu4.3.1-p1
GCS_BUCKET=ryanrosario-gke-dev-pod-snapshots
PROJECT_NUMBER=$(gcloud projects describe ryanrosario-gke-dev --format='value(projectNumber)')

docker build --build-arg BASE_IMAGE=us-central1-docker.pkg.dev/ryanrosario-gke-dev/vllm/vllm-criu:v0.30.0-criu4.3.1 \
  -t "${IMAGE}" -f docker/scripts/snapshot/criu/p1/Dockerfile docker/scripts/snapshot/criu && docker push "${IMAGE}"

kubectl create namespace criu-p1 --dry-run=client -o yaml | kubectl apply -f -
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/ryanrosario-gke-dev.svc.id.goog/subject/ns/criu-p1/sa/criu-runner" \
  --role="projects/ryanrosario-gke-dev/roles/podSnapshotGcsReadWriter"

sed "s#IMAGE#${IMAGE}#; s#BUCKET#${GCS_BUCKET}#" docker/scripts/snapshot/criu/p1/deployment.yaml | kubectl apply -n criu-p1 -f -
kubectl logs -n criu-p1 -l app=criu-p1 -f            # wait for "Snapshot upload complete."
kubectl scale deployment criu-p1 -n criu-p1 --replicas=2
kubectl get pods -n criu-p1 -w                        # pod 2: Pending -> Running -> Ready
```

Timings: `PHASE` lines in each pod's log (cold: `cold_start_to_health`, `sleep_l1`, `criu_dump`, `wake_up`, `upload`;
restore: `download`, `criu_restore`, `wake_up`) plus pod conditions for the Kubernetes-level numbers:

```bash
kubectl get pod -n criu-p1 <pod2> -o jsonpath='{range .status.conditions[*]}{.type}{"\t"}{.lastTransitionTime}{"\n"}{end}'
```
