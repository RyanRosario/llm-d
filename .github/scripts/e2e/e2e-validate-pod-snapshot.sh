#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# e2e-validate-pod-snapshot.sh
# -----------------------------------------------------------------------------
# Validates the pod-snapshot guide beyond the default smoke test.
#
# The smoke loop only proves the guide's router serves chat and text
# completions. This script checks the snapshot/restore lifecycle itself:
# guides/pod-snapshot/scripts/check_thresholds.py reads Pod 1 (cold start +
# snapshot) and Pod 2 (restore from the snapshot), computes the composite
# alerting metrics from the GKE Pod Snapshots CI/CD Plan, prints a PASS/FAIL
# table, and exits non-zero if any metric violates its threshold.
#
# Expects the deploy from guides/pod-snapshot/scripts/nightly-deploy-gke.sh.
# -----------------------------------------------------------------------------

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -n, --namespace NAMESPACE   Kubernetes namespace (default: llm-d)
  -m, --model MODEL_ID        Model the smoke loop discovered (logged only)
  -h, --help                  Show help
EOF
}

NAMESPACE="llm-d"
MODEL_ID=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    -m|--model)     MODEL_ID="$2"; shift 2 ;;
    -h|--help)      show_help; exit 0 ;;
    # Fail, not exit 0 like --help: a mistyped flag in a workflow's
    # e2e_validate_args must not pass the validation step without running it.
    *) echo "Unknown option: $1" >&2; show_help >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/guides/pod-snapshot/scripts"
SELECTOR="llm-d.ai/guide=pod-snapshot"
# The reusable workflow uploads this directory as a run artifact.
RESULTS_DIR="${RESULTS_DIR:-/tmp/pod-logs-pod-snapshot}"
mkdir -p "${RESULTS_DIR}"

echo "Namespace:    ${NAMESPACE}"
echo "Router model: ${MODEL_ID:-not given}"
echo "Results dir:  ${RESULTS_DIR}"

echo "=== Pod snapshot per-phase timing report ==="
# Informational: the full timing table for on-call, plus the per-phase CSV
# (same format as the demo's results.csv). A failure here must not skip the
# threshold checks below, which decide pass/fail.
python3 "${SCRIPTS_DIR}/snapshot_timings.py" report \
  --namespace "${NAMESPACE}" \
  --selector "${SELECTOR}" \
  --csv "${RESULTS_DIR}/results.csv" \
  || echo "Warning: snapshot_timings.py report failed; continuing to the threshold checks." >&2

echo "=== Checking pod-snapshot alerting thresholds ==="
exec python3 "${SCRIPTS_DIR}/check_thresholds.py" \
  --namespace "${NAMESPACE}" \
  --selector "${SELECTOR}"
