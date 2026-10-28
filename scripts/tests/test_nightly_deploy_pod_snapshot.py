"""Contract tests for the pod-snapshot GKE nightly deploy helper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "guides/pod-snapshot/scripts/nightly-deploy-gke.sh"
NS = "nightly-test"
BUCKET = "nightly-test-snapshots"

FAKE_KUBECTL = """#!/usr/bin/env bash
echo "kubectl $*" >> "${FAKE_CALLS}"
"""

FAKE_HELM = """#!/usr/bin/env bash
echo "helm $*" >> "${FAKE_CALLS}"
"""

FAKE_GCLOUD = """#!/usr/bin/env bash
echo "gcloud $*" >> "${FAKE_CALLS}"
if [[ "${1:-} ${2:-}" == "storage ls" && -n "${FAKE_GCS_LS:-}" ]]; then
  printf '%s\\n' "${FAKE_GCS_LS}"
fi
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    _write_executable(fake_bin / "kubectl", FAKE_KUBECTL)
    _write_executable(fake_bin / "helm", FAKE_HELM)
    _write_executable(fake_bin / "gcloud", FAKE_GCLOUD)
    (tmp_path / "calls.log").write_text("")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.update({
        "NAMESPACE": NS,
        "MODEL_SERVER": "vllm",
        "GCS_BUCKET": BUCKET,
        "OUTPUT_DIR": str(tmp_path / "out"),
        "CRD_RETRY_DELAY": "0",
        "FAKE_CALLS": str(tmp_path / "calls.log"),
    })
    env.pop("REPO_ROOT", None)
    env.update(extra_env)
    return subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)


def _calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls.log").read_text().splitlines()


@pytest.mark.parametrize("model_server", ["vllm", "sglang"])
def test_happy_path_deploys_without_skip_in_ci_steps(tmp_path: Path, model_server: str) -> None:
    result = _run(tmp_path, MODEL_SERVER=model_server)
    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)

    assert calls[:2] == [
        f"gcloud storage rm -r gs://{BUCKET}/**",
        f"gcloud storage ls gs://{BUCKET}",
    ]
    assert any(c.startswith("helm install pod-snapshot ") and f"-n {NS}" in c for c in calls)
    assert any(f"modelserver/gpu/{model_server}/gke/" in c for c in calls)
    assert f"kubectl rollout status deployment/pod-snapshot-nvidia-gpu-{model_server}-decode -n {NS} --timeout=600s" in calls

    # No step marked skip_in: [ci] (git clone, namespace, HF token) is run.
    assert not any("git clone" in c for c in calls)
    assert not any(c.startswith("kubectl create namespace") for c in calls)
    assert not any("llm-d-hf-token" in c for c in calls)


def test_bucket_reset_fails_when_the_bucket_stays_non_empty(tmp_path: Path) -> None:
    result = _run(tmp_path, FAKE_GCS_LS=f"gs://{BUCKET}/stuck")
    assert result.returncode != 0
    assert "is still not empty" in result.stderr
    assert _calls(tmp_path) == [
        f"gcloud storage rm -r gs://{BUCKET}/**",
        f"gcloud storage ls gs://{BUCKET}",
    ]


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
