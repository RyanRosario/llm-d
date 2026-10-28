"""Unit tests for the pod-snapshot nightly validator.

Covers guides/pod-snapshot/scripts/check_thresholds.py (the alerting checks and
their threshold table) and the parts of snapshot_timings.py it relies on: the
log parser, the 1..4f metric hierarchy, pod selection and the CSV. Everything
runs on synthetic pods and `kubectl logs --timestamps` text; no cluster.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "guides/pod-snapshot/scripts"
VERIFY_DIR = ROOT / ".github/scripts/nightly-e2e-verification"
for p in (str(SCRIPTS), str(VERIFY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import check_thresholds as ct  # noqa: E402
import snapshot_timings as st  # noqa: E402

T0 = dt.datetime(2026, 10, 25, 3, 0, 0, tzinfo=dt.timezone.utc)
SELECTOR = "llm-d.ai/guide=pod-snapshot"


def at(seconds: float) -> dt.datetime:
    return T0 + dt.timedelta(seconds=seconds)


def iso(seconds: float) -> str:
    return at(seconds).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(lines) -> str:
    """`kubectl logs --timestamps` text: one (offset seconds, message) per line."""
    return "".join(f"{at(s).strftime('%Y-%m-%dT%H:%M:%S.%f')}Z {text}\n" for s, text in lines)


# --- Synthetic logs, one representative line per regex the metrics depend on -----------------

VLLM_COLD = [
    (90, "Loading model from scratch..."),
    (160, "Model loading took 61.03 GiB memory and 70.10 seconds"),
    (167, "Available KV cache memory: 10.5 GiB"),
    (200, "Graph capturing finished in 19 secs, took 1.07 GiB"),
    (207, "Executing engine.sleep(level=1) to release physical VRAM..."),
    (215, "It took 8.00 seconds to fall asleep."),
    (250, "Snapshot checkpoint created successfully."),
    (261, "It took 2.65 seconds to wake up tags {'kv_cache', 'weights'}."),
]
VLLM_RESTORE = [  # offsets relative to Pod 2's creation
    (10, "Snapshot checkpoint created successfully."),
    (16.5, "It took 2.75 seconds to wake up tags {'kv_cache', 'weights'}."),
]
SGLANG_COLD = [
    (90, "Load weight begin. avail mem=78.50 GB"),
    (170, "Load weight end. elapsed=80.00 s"),
    (175, "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 39206, K size: 4.79 GB"),
    (194, "Capture target prefill CUDA graph end. elapsed=60.18 s"),
    (200, "Capture target decode CUDA graph end. bs=[1, 2, 4], elapsed=6.03 s"),
    (207, "Holding server_status at Starting until the snapshot completes."),
    (215, 'INFO:     127.0.0.1:41234 - "POST /release_memory_occupation HTTP/1.1" 200 OK'),
    (250, "Snapshot checkpoint created successfully."),
    (261, "Restored server_status to ServerStatus.Up."),
]
SGLANG_RESTORE = [
    (10, "Snapshot checkpoint created successfully."),
    (16.5, "Restored server_status to ServerStatus.Up."),
]
LOGS = {"vllm": (VLLM_COLD, VLLM_RESTORE), "sglang": (SGLANG_COLD, SGLANG_RESTORE)}


# --- Synthetic pods ----------------------------------------------------------------------------


def k8s_pod(name, *, created, scheduled, sandbox, started, ready, restored=None):
    """A minimal `kubectl get pod -o json` item."""
    conds = [
        {"type": "PodScheduled", "status": "True", "lastTransitionTime": iso(scheduled)},
        {"type": "PodReadyToStartContainers", "status": "True", "lastTransitionTime": iso(sandbox)},
        {"type": "Ready", "status": "True", "lastTransitionTime": iso(ready)},
    ]
    if restored is not None:
        conds.append({"type": "PodRestored", "status": "True", "lastTransitionTime": iso(restored)})
    return {
        "metadata": {"name": name, "creationTimestamp": iso(created)},
        "spec": {"containers": [{"name": "vllm"}]},
        "status": {"conditions": conds,
                   "containerStatuses": [{"state": {"running": {"startedAt": iso(started)}}}]},
    }


def pull_events(name, pulling, pulled):
    return [
        {"involvedObject": {"kind": "Pod", "name": name}, "reason": "Pulling", "firstTimestamp": iso(pulling)},
        {"involvedObject": {"kind": "Pod", "name": name}, "reason": "Pulled", "firstTimestamp": iso(pulled)},
    ]


class FakeKubectl:
    """Serves canned `get` JSON and per-pod logs; records every call."""

    def __init__(self, pods=(), events=(), snaps=(), logs=None, fail_pods=False):
        self.namespace = "ns"
        self.calls = []
        self._get = {"pods": None if fail_pods else {"items": list(pods)},
                     "events": {"items": list(events)},
                     "podsnapshots": {"items": list(snaps)}}
        self._logs = logs or {}

    def run(self, *args):
        self.calls.append(args)
        if args[0] == "logs":
            return self._logs.get(args[1], "")
        return None

    def json(self, *args):
        self.calls.append(args)
        res = self._get.get(args[1])
        if res is None:
            raise RuntimeError(f"kubectl {' '.join(args)} -n {self.namespace} failed")
        return res


def two_pods(engine):
    """Pod 1 (cold start + snapshot, created at T0) and Pod 2 (restore, created at T0+1000s).

    Pod 1: 1 = 55s (5 wait, 25 sandbox, 20 pull, 5 start), 2 = 152s, 4 = 54s, 3 = 39s -> Ready at 300s.
    Pod 2: 1 = 6s (1, 3, 1, 1), 4e = 4s, 4f = 6.5s, 3 = 3.5s -> Ready at 20s.
    """
    cold, rest = LOGS[engine]
    p1 = k8s_pod("ms-1", created=0, scheduled=5, sandbox=30, started=55, ready=300)
    p2 = k8s_pod("ms-2", created=1000, scheduled=1001, sandbox=1004, started=1006, ready=1020,
                 restored=1010)
    kube = FakeKubectl(
        pods=[p2, p1],  # newest first, as the API often returns them
        events=pull_events("ms-1", 30, 50) + pull_events("ms-2", 1004, 1005),
        snaps=[{"status": {"conditions": [{"type": "Ready", "status": "True", "lastTransitionTime": iso(290)}]}}],
        logs={"ms-1": log(cold), "ms-2": log([(1000 + s, t) for s, t in rest])},
    )
    return kube


def metrics_pod(*, created=0, scheduled=5, sandbox=30, pulled=50, started=55, ready=300, restored=None, log=None):
    """A pod_times dict built directly, for metric and evaluate() tests."""
    return {
        "name": "pod",
        "created": at(created), "scheduled": at(scheduled), "sandbox": at(sandbox),
        "pulling_ts": at(sandbox), "pulled_ts": at(pulled), "started": at(started),
        "ready": at(ready), "restored": at(restored) if restored is not None else None,
        "log": log or {},
    }


COLD_MARKERS = {
    "weights_begin_ts": 90,
    "weights_end_ts": 160,
    "kv_ts": 167,
    "graph_end_ts": 200,
    "warmup_done_ts": 207,
    "released_ts": 215,
    "checkpoint_ts": 250,
    "up_ts": 261,
}


def cold_log(shift=0, after=None, **overrides):
    """A parsed cold-start log. `shift` moves every marker; `after=(key, secs)` moves only the
    markers from `key` on (e.g. a slow phase); `overrides` set or (with None) drop single markers."""
    ts = dict(COLD_MARKERS)
    if after:
        key, secs = after
        keys = list(COLD_MARKERS)
        for k in keys[keys.index(key):]:
            ts[k] += secs
    ts = {k: v + shift for k, v in ts.items()}
    ts.update(overrides)
    return {k: at(v) for k, v in ts.items() if v is not None}


def restore_log(created=1000, restore=10, up=16.5):
    return {"restored": True, "restore_ts": at(created + restore), "up_ts": at(created + up)}


# --- Log parsing -------------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_cold_start_log_yields_every_phase_timestamp(engine):
    v = st.log_values(log(LOGS[engine][0]))
    assert "restored" not in v
    expected = {
        "weights_begin_ts": 90,
        "weights_end_ts": 160 if engine == "vllm" else 170,
        "kv_ts": 167 if engine == "vllm" else 175,
        "graph_end_ts": 200,
        "warmup_done_ts": 207,
        "released_ts": 215,
        "checkpoint_ts": 250,
        "up_ts": 261,
    }
    assert v == {k: at(secs) for k, secs in expected.items()}


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_restore_log_is_recognised_by_a_checkpoint_line_without_cold_start_markers(engine):
    v = st.log_values(log(LOGS[engine][1]))
    assert v == {"restored": True, "restore_ts": at(10), "up_ts": at(16.5)}


def test_lines_without_a_timestamp_or_from_probes_are_ignored():
    text = "no timestamp here Snapshot checkpoint created successfully\n" + log([
        (5, '"GET /health HTTP/1.1" 503 - Snapshot checkpoint created successfully'),
    ])
    assert st.log_values(text) == {}


# --- Metrics -----------------------------------------------------------------------------------


def test_cold_metrics_are_contiguous_and_add_up():
    m = st.compute_cold_metrics(metrics_pod(log=cold_log()))
    assert m["1"] == 55 and (m["1a"], m["1b"], m["1c"], m["1d"]) == (5, 25, 20, 5)
    assert m["2"] == 152 and (m["2a"], m["2b"], m["2c"], m["2d"], m["2e"], m["2f"], m["2g"]) == (0, 0, 35, 70, 7, 33, 7)
    assert m["4"] == 54 and (m["4a"], m["4b"], m["4c"], m["4d"], m["4e"], m["4f"]) == (0, 0, 8, 0, 35, 11)
    assert m["3"] == 39
    assert m["1"] + m["2"] + m["3"] + m["4"] == 300
    for step, subs in (("1", "abcd"), ("2", "abcdefg"), ("4", "abcdef")):
        assert sum(m[step + s] for s in subs) == pytest.approx(m[step])


def test_restore_metrics_are_contiguous_and_add_up():
    m = st.compute_restore_metrics(metrics_pod(created=1000, scheduled=1001, sandbox=1004, pulled=1005,
                                               started=1006, ready=1020, restored=1010, log=restore_log()))
    assert (m["1a"], m["1b"], m["1c"], m["1d"]) == (1, 3, 1, 1)
    assert (m["4e"], m["4f"], m["3"]) == (4, 6.5, 3.5)
    assert m["1"] + m["4e"] + m["4f"] + m["3"] == 20


def test_missing_log_markers_collapse_to_zero_length_steps_not_negative_ones():
    m = st.compute_cold_metrics(metrics_pod(log=cold_log(kv_ts=None, graph_end_ts=None,
                                                         released_ts=None, checkpoint_ts=None)))
    assert all(v >= 0 for v in m.values())
    assert m["1"] + m["2"] + m["3"] + m["4"] == 300


def test_out_of_order_markers_are_clamped():
    m = st.compute_cold_metrics(metrics_pod(log=cold_log(kv_ts=150)))
    assert m["2e"] == 0 and m["2f"] == 40


# --- evaluate() and the threshold table --------------------------------------------------------


def names_of(checks, passed):
    return [c.name for c in checks if c.passed is passed]


def good_pods():
    p1 = metrics_pod(log=cold_log())
    p2 = metrics_pod(created=1000, scheduled=1001, sandbox=1004, pulled=1005, started=1006, ready=1020,
                     restored=1010, log=restore_log())
    return p1, p2


def test_thresholds_table_is_the_agreed_one():
    # Deliberate changes to the alerting thresholds should show up as a test change.
    assert ct.THRESHOLDS == {
        "vllm": {"max_e2e_cold_start_s": 1800.0, "max_cold_start_s": 350.0,
                 "max_snapshot_overhead_s": 65.0, "min_restore_speedup": 11.0},
        "sglang": {"max_e2e_cold_start_s": 1800.0, "max_cold_start_s": 350.0,
                   "max_snapshot_overhead_s": 85.0, "min_restore_speedup": 11.0},
    }
    assert ct.MAX_POD2_APP_COLD_START_S == 0.0


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_a_healthy_run_passes_every_check(engine):
    checks = ct.evaluate(*good_pods(), engine)
    assert names_of(checks, False) == []
    assert names_of(checks, True) == [
        "Pod 1 cold started", "Pod 2 present",
        "Pod 1 phase markers",
        "E2E Cold Start (Rows 1+2+3)", "Cold Start, no node provisioning (1b+1c+1d+2)",
        "Total Snapshot Overhead Time (Row 4)",
        "Pod 2 phase markers",
        "Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)",
        "Application Cold Start on Pod 2 (Row 2)",
    ]
    detail = {c.name: c.detail for c in checks}
    # ECS = 55 + 152 + 39; CS = 25 + 20 + 5 + 152; TSOT = 54; WSR = 3 + 1 + 4 + 6.5 + 3.5 = 18.
    assert detail["Pod 1 phase markers"] == "all 8 present"
    assert detail["Pod 2 phase markers"] == "all 2 present"
    assert detail["E2E Cold Start (Rows 1+2+3)"].startswith("246.00s <= 1800s")
    assert detail["Cold Start, no node provisioning (1b+1c+1d+2)"].startswith("202.00s <= 350s")
    assert detail["Total Snapshot Overhead Time (Row 4)"].startswith("0 < 54.00s <= ")
    assert detail["Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)"].startswith("13.67x >= 11x (WSR 18.00s)")


@pytest.mark.parametrize("marker", st.COLD_MARKERS)
def test_missing_pod_1_phase_marker_fails_check(marker):
    p1, p2 = good_pods()
    p1["log"] = cold_log(**{marker: None})
    checks = ct.evaluate(p1, p2, "vllm")
    assert "Pod 1 phase markers" in names_of(checks, False)
    detail = next(c.detail for c in checks if c.name == "Pod 1 phase markers")
    assert detail == f"missing: {marker}"


@pytest.mark.parametrize("marker", st.RESTORE_MARKERS)
def test_missing_pod_2_phase_marker_fails_check(marker):
    p1, p2 = good_pods()
    p2["log"].pop(marker)
    checks = ct.evaluate(p1, p2, "vllm")
    assert "Pod 2 phase markers" in names_of(checks, False)
    detail = next(c.detail for c in checks if c.name == "Pod 2 phase markers")
    assert detail == f"missing: {marker}"


def test_slow_e2e_cold_start_fails_only_that_check():
    p1, p2 = good_pods()
    p1["scheduled"] = at(1600)  # a 1600s wait for a GPU node: ECS = 1600 + ... > 1800
    p1["sandbox"] = p1["pulling_ts"] = at(1625)
    p1["pulled_ts"], p1["started"] = at(1645), at(1650)
    p1["log"] = cold_log(shift=1595)
    p1["ready"] = at(1895)
    checks = ct.evaluate(p1, p2, "vllm")
    # ECS = 1650 + 152 + 39 = 1841 > 1800; CS is unchanged and the speedup only grows.
    assert names_of(checks, False) == ["E2E Cold Start (Rows 1+2+3)"]
    assert "1841.00s <= 1800s" in next(c.detail for c in checks if not c.passed)


def test_slow_application_cold_start_fails_cold_start_but_not_the_node_wait():
    p1, p2 = good_pods()
    # Weights took 300s instead of 70s: CS = 25 + 20 + 5 + 382 > 350, but ECS still < 1800.
    p1["log"] = cold_log(after=("weights_end_ts", 230))
    p1["ready"] = at(530)
    failed = names_of(ct.evaluate(p1, p2, "vllm"), False)
    assert "Cold Start, no node provisioning (1b+1c+1d+2)" in failed
    assert "E2E Cold Start (Rows 1+2+3)" not in failed


@pytest.mark.parametrize("engine,overhead,ok", [
    ("vllm", 54, True), ("vllm", 65, True), ("vllm", 66, False),
    ("sglang", 85, True), ("sglang", 86, False),
])
def test_snapshot_overhead_is_checked_against_the_engine_threshold(engine, overhead, ok):
    p1, p2 = good_pods()
    # Stretch the checkpoint (4e) so row 4 = overhead; keep everything after it contiguous.
    p1["log"] = cold_log(after=("checkpoint_ts", overhead - 54))
    p1["ready"] = at(300 + overhead - 54)
    failed = names_of(ct.evaluate(p1, p2, engine), False)
    assert ("Total Snapshot Overhead Time (Row 4)" in failed) is (not ok)


def test_zero_snapshot_overhead_fails_because_no_snapshot_was_taken():
    p1, p2 = good_pods()
    p1["log"] = cold_log(released_ts=None, checkpoint_ts=None, up_ts=None)
    failed = names_of(ct.evaluate(p1, p2, "vllm"), False)
    assert "Pod 1 phase markers" in failed
    assert "Total Snapshot Overhead Time (Row 4)" in failed


def test_slow_restore_fails_the_speedup_check():
    p1, p2 = good_pods()
    p2["log"] = restore_log(restore=10, up=30)  # 4f = 20s, 3 = 4s -> WSR = 3+1+4+20+4 = 32s -> 7.7x
    p2["ready"] = at(1034)
    checks = ct.evaluate(p1, p2, "vllm")
    assert names_of(checks, False) == ["Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)"]
    assert "7.69x >= 11x (WSR 32.00s)" in next(c.detail for c in checks if not c.passed)


def test_pod_2_that_cold_started_instead_of_restoring_fails():
    p1, _ = good_pods()
    p2 = metrics_pod(created=1000, scheduled=1005, sandbox=1030, pulled=1050, started=1055, ready=1300,
                     log=cold_log(shift=1000))
    checks = ct.evaluate(p1, p2, "vllm")
    failed = names_of(checks, False)
    assert failed == ["Application Cold Start on Pod 2 (Row 2)"]
    assert "152.00s <= 0s" in next(c.detail for c in checks if not c.passed)
    # No restore metrics, so no speedup check at all rather than a misleading pass.
    assert "Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)" not in [c.name for c in checks]


def test_pod_2_unrestored_with_empty_logs_fails_instead_of_false_passing():
    p1, _ = good_pods()
    p2 = metrics_pod(created=1000, scheduled=1001, sandbox=1004, pulled=1005, started=1006, ready=1020,
                     restored=None, log={})
    checks = ct.evaluate(p1, p2, "vllm")
    failed = names_of(checks, False)
    assert failed == ["Application Cold Start on Pod 2 (Row 2)"]
    assert "Pod 2 did not restore" in next(c.detail for c in checks if not c.passed)
    assert "Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)" not in [c.name for c in checks]


def test_pod_1_restored_from_a_stale_snapshot_fails_and_skips_the_cold_start_metrics():
    _, p2 = good_pods()
    p1 = metrics_pod(restored=10, log=restore_log(created=0))
    checks = ct.evaluate(p1, p2, "vllm")
    assert names_of(checks, False) == ["Pod 1 cold started"]
    assert "E2E Cold Start (Rows 1+2+3)" not in [c.name for c in checks]


def test_missing_pod_2_fails_presence_and_skips_restore_checks():
    p1, _ = good_pods()
    checks = ct.evaluate(p1, None, "vllm")
    assert names_of(checks, False) == ["Pod 2 present"]
    assert "Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)" not in [c.name for c in checks]


# --- Pod selection and the end-to-end path through load_pods -----------------------------------


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_load_pods_picks_oldest_as_pod_1_and_reads_the_cluster_once(engine):
    kube = two_pods(engine)
    p1, p2, snap_ready = st.load_pods(kube, SELECTOR)
    assert (p1["name"], p2["name"]) == ("ms-1", "ms-2")
    assert not st.is_restored(p1) and st.is_restored(p2)
    assert snap_ready == at(290)
    gets = [c[1] for c in kube.calls if c[0] == "get"]
    assert sorted(gets) == ["events", "pods", "podsnapshots"]
    assert [c[1] for c in kube.calls if c[0] == "logs"] == ["ms-1", "ms-2"]
    assert kube.calls[0] == ("get", "pods", "-l", SELECTOR)


def test_load_pods_ignores_terminating_pods():
    kube = two_pods("vllm")
    dying = k8s_pod("ms-0", created=-500, scheduled=-499, sandbox=-490, started=-480, ready=-300)
    dying["metadata"]["deletionTimestamp"] = iso(-10)
    kube._get["pods"]["items"].append(dying)
    p1, p2, _ = st.load_pods(kube, SELECTOR)
    assert (p1["name"], p2["name"]) == ("ms-1", "ms-2")


def test_load_pods_raises_when_pods_cannot_be_listed():
    with pytest.raises(RuntimeError, match="kubectl get pods"):
        st.load_pods(FakeKubectl(fail_pods=True), SELECTOR)


def test_fewer_than_two_pods():
    kube = two_pods("vllm")
    kube._get["pods"]["items"] = kube._get["pods"]["items"][1:]  # only ms-1
    p1, p2, _ = st.load_pods(kube, SELECTOR)
    assert p1["name"] == "ms-1" and p2 is None
    kube._get["pods"]["items"] = []
    assert st.load_pods(kube, SELECTOR)[:2] == (None, None)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_parsed_pods_pass_the_thresholds_end_to_end(engine):
    p1, p2, _ = st.load_pods(two_pods(engine), SELECTOR)
    checks = ct.evaluate(p1, p2, engine)
    assert names_of(checks, False) == [], [(c.name, c.detail) for c in checks]


# --- Report and CSV ----------------------------------------------------------------------------


def test_csv_has_five_columns_and_the_composite_metrics(tmp_path):
    p1, p2, snap_ready = st.load_pods(two_pods("vllm"), SELECTOR)
    cold_m, rest_m = st.compute_cold_metrics(p1), st.compute_restore_metrics(p2)
    rows = st.csv_rows(cold_m, rest_m, p1, snap_ready, "vllm")
    text = st.write_csv(rows, str(tmp_path / "results.csv"))
    lines = text.splitlines()
    assert lines[0] == "Seq,ModelServer,Mode,MetricName,MetricValue"
    assert (tmp_path / "results.csv").read_text() == text
    assert all(line.startswith("1,vLLM,") for line in lines[1:])
    by_name = {}
    for _model_server, mode, name, val in rows:
        by_name[(mode, name)] = val
    assert by_name[("ColdStart", "E2E Cold Start (including node provisioning)")] == pytest.approx(246)
    assert by_name[("ColdStart", "Cold Start (excluding node provisioning)")] == pytest.approx(202)
    assert by_name[("ColdStart", "Total Snapshot Overhead Time")] == pytest.approx(54)
    assert by_name[("Restore", "Warm Start (from restoration)")] == pytest.approx(18)
    assert by_name[("ColdStart", "Total: pod created -> Ready")] == 300
    assert by_name[("ColdStart", "PodSnapshot Ready")] == 290
    assert set(mode for mode, _ in by_name) == {"ColdStart", "Restore"}
    assert len([n for m, n in by_name if m == "ColdStart" and n[:2] in ("1.", "2.", "3.", "4.")]) == 4


def test_report_prints_both_pods_and_the_composites(capsys):
    p1, p2, snap_ready = st.load_pods(two_pods("sglang"), SELECTOR)
    st.print_report(p1, p2, snap_ready, engine="sglang")
    out = capsys.readouterr().out
    assert out.startswith("Seq,ModelServer,Mode,MetricName,MetricValue\n")
    assert "1,SGLang,ColdStart,E2E Cold Start (including node provisioning),246.00" in out
    assert "1,SGLang,Restore,Warm Start (from restoration),18.00" in out
    assert "1,SGLang,ColdStart,Total Snapshot Overhead Time,54.00" in out


# --- main(): the report must never mask the checks --------------------------------------------


def test_main_runs_the_checks_even_if_the_report_crashes(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(st, "load_pods", lambda kube, selector: (*good_pods(), at(290)))
    monkeypatch.setattr(st, "print_report", lambda *_a, **_k: 1 / 0)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    rc = ct.main(["--namespace", "ns", "--engine", "vllm", "--csv", str(tmp_path / "r.csv")])
    captured = capsys.readouterr()
    assert rc == 0
    assert "the timing report failed (ZeroDivisionError" in captured.err
    assert "PASS: 9 check(s) passed" in captured.out
    assert summary.read_text().startswith("### Pod Snapshot alerting thresholds (vllm)")
    assert summary.read_text().count("| PASS |") == 9


def test_main_passes_the_csv_path_and_engine_to_the_report(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(st, "load_pods", lambda kube, selector: (*good_pods(), None))
    monkeypatch.setattr(st, "print_report", lambda p1, p2, snap, engine, csv_path: seen.update(engine=engine, csv=csv_path))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert ct.main(["--namespace", "ns", "--engine", "sglang", "--csv", str(tmp_path / "r.csv")]) == 0
    assert seen == {"engine": "sglang", "csv": str(tmp_path / "r.csv")}


def test_main_exits_1_on_a_failed_check(monkeypatch, capsys):
    p1, _ = good_pods()
    monkeypatch.setattr(st, "load_pods", lambda kube, selector: (p1, None, None))
    monkeypatch.setattr(st, "print_report", lambda *_a, **_k: None)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert ct.main(["--namespace", "ns", "--engine", "vllm"]) == 1
    assert "FAIL: 1/" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--namespace", "ns"], ["--namespace", "ns", "--engine", "triton"]])
def test_main_rejects_missing_or_unknown_engine(argv):
    with pytest.raises(SystemExit) as exc:
        ct.main(argv)
    assert exc.value.code == 2


def test_main_exits_1_when_the_cluster_cannot_be_read(monkeypatch, capsys):
    monkeypatch.setattr(st, "Kubectl", lambda _namespace: FakeKubectl(fail_pods=True))
    assert ct.main(["--namespace", "ns", "--engine", "vllm"]) == 1
    assert "kubectl get pods" in capsys.readouterr().err
