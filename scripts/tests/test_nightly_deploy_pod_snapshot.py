"""Contract tests for the pod-snapshot GKE nightly deploy helper.

The script is run against fake kubectl/helm/gcloud binaries that only log their
argv (``kubectl kustomize`` is passed to the real kubectl, and manifests piped
to ``kubectl apply -f -`` are kept), so the tests check which commands the
guide.yaml emits produce for CI, what the CI-rendered manifests look like, and
how the GCS bucket reset behaves.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "guides/pod-snapshot"
SCRIPT = GUIDE / "scripts/nightly-deploy-gke.sh"
NS = "nightly-test"
BUCKET = "nightly-test-snapshots"

FAKE_KUBECTL = """#!/usr/bin/env bash
echo "kubectl $*" >> "${FAKE_CALLS}"
# Rendering is real: the script builds the guide's snapshot and model server overlays.
if [[ "${1:-}" == "kustomize" ]]; then exec "${REAL_KUBECTL}" "$@"; fi
if [[ "${1:-}" == "apply" && "$*" == *"-f -"* ]]; then
  cat >> "${FAKE_STDIN_APPLIED}"
  echo "---" >> "${FAKE_STDIN_APPLIED}"
  exit 0
fi
if [[ "${1:-}" == "apply" && "$*" == *"https://"* && "${FAKE_CRD_APPLY_FAIL:-0}" == "1" ]]; then
  exit 1
fi
exit 0
"""

FAKE_HELM = """#!/usr/bin/env bash
echo "helm $*" >> "${FAKE_CALLS}"
exit 0
"""

# `gcloud storage ls` prints the first line of FAKE_GCS_LISTINGS (one listing per
# line, entries separated by ';') and consumes it, so a test scripts what the
# bucket looks like on each successive call. An exhausted file means empty.
FAKE_GCLOUD = """#!/usr/bin/env bash
echo "gcloud $*" >> "${FAKE_CALLS}"
if [[ "${1:-} ${2:-}" == "storage ls" ]]; then
  if [[ -s "${FAKE_GCS_LISTINGS}" ]]; then
    head -n 1 "${FAKE_GCS_LISTINGS}" | tr ';' '\\n' | sed '/^$/d'
    sed -i '1d' "${FAKE_GCS_LISTINGS}"
  fi
  exit 0
fi
exit 0
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path: Path, *, listings: list[str] = (), **extra_env: str) -> subprocess.CompletedProcess[str]:
    real_kubectl = shutil.which("kubectl")
    if real_kubectl is None or Path(real_kubectl).parent == tmp_path / "bin":
        pytest.skip("kubectl is required to render the kustomize overlays")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    _write_executable(fake_bin / "kubectl", FAKE_KUBECTL)
    _write_executable(fake_bin / "helm", FAKE_HELM)
    _write_executable(fake_bin / "gcloud", FAKE_GCLOUD)
    (tmp_path / "calls.log").write_text("")
    (tmp_path / "stdin-applied.yaml").write_text("")
    (tmp_path / "gcs-listings.txt").write_text("".join(line + "\n" for line in listings))
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.update({
        "NAMESPACE": NS,
        "MODEL_SERVER": "vllm",
        "GCS_BUCKET": BUCKET,
        "OUTPUT_DIR": str(tmp_path / "out"),
        "CRD_RETRY_DELAY": "0",
        "FAKE_CALLS": str(tmp_path / "calls.log"),
        "FAKE_STDIN_APPLIED": str(tmp_path / "stdin-applied.yaml"),
        "FAKE_GCS_LISTINGS": str(tmp_path / "gcs-listings.txt"),
        "REAL_KUBECTL": real_kubectl,
    })
    env.pop("REPO_ROOT", None)
    env.update(extra_env)
    return subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)


def _calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls.log").read_text().splitlines()


def _index(calls: list[str], prefix: str) -> int:
    matches = [i for i, c in enumerate(calls) if c.startswith(prefix)]
    assert len(matches) == 1, (prefix, calls)
    return matches[0]


def _applied(tmp_path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all((tmp_path / "stdin-applied.yaml").read_text()) if d]


def _one(docs: list[dict], kind: str) -> dict:
    found = [d for d in docs if d["kind"] == kind]
    assert len(found) == 1, (kind, [d["kind"] for d in docs])
    return found[0]


@pytest.mark.parametrize("model_server", ["vllm", "sglang"])
def test_happy_path_deploys_the_guide_from_guide_yaml(tmp_path: Path, model_server: str) -> None:
    result = _run(tmp_path, MODEL_SERVER=model_server)
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    selector = "llm-d.ai/guide=pod-snapshot"

    # The reusable hands the script a fresh namespace, so the only reset is the
    # GCS bucket: one listing, nothing to delete, no Deployment/PodSnapshot
    # deletes before the deploy.
    bucket_ls = _index(calls, f"gcloud storage ls gs://{BUCKET}")
    assert not any(c.startswith("gcloud storage rm") for c in calls)
    assert not any(c.startswith("kubectl delete") for c in calls)

    # prerequisites.gaie, URL resolved through guides/env.sh.
    crd = _index(calls, "kubectl apply -f https://")
    assert "gateway-api-inference-extension" in calls[crd]

    # The router: deploy.standalone with the guide's values, release pod-snapshot.
    router = _index(calls, "helm install pod-snapshot ")
    assert f"-n {NS}" in calls[router]
    assert "guides/recipes/router/base.values.yaml" in calls[router]
    assert "guides/pod-snapshot/router/pod-snapshot.values.yaml" in calls[router]

    # deploy.snapshot and deploy.modelserver are both rendered with kustomize
    # and piped to kubectl apply -n NS -f -.
    applied = _applied(tmp_path)
    storage = _one(applied, "PodSnapshotStorageConfig")
    policy = _one(applied, "PodSnapshotPolicy")
    deployment = _one(applied, "Deployment")
    apply_calls = [i for i, c in enumerate(calls) if c == f"kubectl apply -n {NS} -f -"]
    assert len(apply_calls) == 2
    snapshot_apply, ms_apply = apply_calls

    # The cluster-scoped storage config is prefixed with the namespace and
    # points at the real bucket, not the README placeholder.
    assert storage["metadata"]["name"] == f"{NS}-pod-snapshot-storage-config"
    assert BUCKET in yaml.safe_dump(storage)
    assert "gcs-bucket-placeholder" not in yaml.safe_dump(applied)
    assert policy["spec"]["storageConfigName"] == f"{NS}-pod-snapshot-storage-config"

    # The model server: one replica of the requested engine, labelled for the
    # selector the waits, the validator and the smoke test use.
    assert deployment["metadata"]["name"] == f"pod-snapshot-nvidia-gpu-{model_server}-decode"
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["template"]["metadata"]["labels"]["llm-d.ai/guide"] == "pod-snapshot"
    assert deployment["spec"]["template"]["spec"]["runtimeClassName"] == "gvisor"

    # verify.tests.snapshot: Pod 1 Ready AND a PodSnapshot Ready (readiness
    # alone does not prove a checkpoint was written), then verify.tests.restore:
    # scale to 2 and wait for Pod 2, then the rollout.
    pod1_wait = _index(calls, f"kubectl wait --for=condition=ready pod -l {selector} -n {NS} --timeout=2400s")
    snap_wait = _index(calls, f"kubectl wait --for=condition=Ready podsnapshots --all -n {NS}")
    scale = _index(calls, f"kubectl scale deployment -l {selector} -n {NS} --replicas=2")
    pod2_wait = _index(calls, f"kubectl wait --for=condition=ready pod -l {selector} -n {NS} --timeout=600s")
    rollout = _index(calls, f"kubectl rollout status deployment/{deployment['metadata']['name']} -n {NS}")
    assert bucket_ls < crd < router < snapshot_apply < ms_apply < pod1_wait < snap_wait < scale < pod2_wait < rollout

    # No step marked skip_in: [ci] (git clone, namespace, HF token) is run.
    assert not any("git clone" in c for c in calls)
    assert not any(c.startswith("kubectl create namespace") for c in calls)
    assert not any("llm-d-hf-token" in c for c in calls)


def test_a_non_empty_bucket_is_emptied_before_the_deploy(tmp_path: Path) -> None:
    first = f"gs://{BUCKET}/snap-a/;gs://{BUCKET}/snap-b"
    result = _run(tmp_path, listings=[first, ""])
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    rm = _index(calls, "gcloud storage rm --recursive ")
    assert calls[rm] == f"gcloud storage rm --recursive gs://{BUCKET}/snap-a/ gs://{BUCKET}/snap-b"
    assert len([c for c in calls if c.startswith("gcloud storage ls")]) == 2
    assert rm < _index(calls, "kubectl apply -f https://")


def test_bucket_reset_refuses_entries_outside_the_bucket(tmp_path: Path) -> None:
    result = _run(tmp_path, listings=["gs://some-other-bucket/snap"])
    assert result.returncode != 0
    assert "refusing to delete unexpected entry" in result.stderr
    assert not any(c.startswith("gcloud storage rm") for c in _calls(tmp_path))
    assert not any(c.startswith("helm ") for c in _calls(tmp_path))


def test_bucket_reset_gives_up_when_the_bucket_stays_non_empty(tmp_path: Path) -> None:
    sticky = f"gs://{BUCKET}/stuck"
    result = _run(tmp_path, listings=[sticky] * 6)
    assert result.returncode != 0
    assert "is still not empty" in result.stderr
    calls = _calls(tmp_path)
    assert len([c for c in calls if c.startswith("gcloud storage ls")]) == 4
    assert len([c for c in calls if c.startswith("gcloud storage rm")]) == 3
    assert not any(c.startswith("kubectl apply") for c in calls)


def test_crd_install_gives_up_after_three_attempts(tmp_path: Path) -> None:
    result = _run(tmp_path, FAKE_CRD_APPLY_FAIL="1")
    assert result.returncode != 0
    assert "CRD install failed after 3 attempts" in result.stderr
    assert len([c for c in _calls(tmp_path) if c.startswith("kubectl apply -f https://")]) == 3
    assert not any(c.startswith("helm ") for c in _calls(tmp_path))


@pytest.mark.parametrize("bad", ["", "triton", "VLLM"])
def test_rejects_unknown_model_servers_before_touching_anything(tmp_path: Path, bad: str) -> None:
    result = _run(tmp_path, MODEL_SERVER=bad)
    assert result.returncode != 0
    assert "MODEL_SERVER" in result.stderr
    assert _calls(tmp_path) == []


def test_rejects_a_malformed_bucket_name_before_touching_anything(tmp_path: Path) -> None:
    result = _run(tmp_path, GCS_BUCKET="gs://Not_A_Bucket")
    assert result.returncode != 0
    assert "unexpected GCS_BUCKET" in result.stderr
    assert _calls(tmp_path) == []
