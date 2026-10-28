#!/usr/bin/env python3
"""Startup, snapshot and restore timings for GKE Pod Snapshots (vLLM and SGLang).

Used by the nightly through check_thresholds.py (from .github/scripts/e2e/e2e-validate-pod-snapshot.sh),
which reads the cluster once with load_pods, prints the timing table for on-call with print_report
(also writing the per-phase metrics as a 5-column CSV: Seq, ModelServer, Mode, MetricName,
MetricValue; Seq is always 1 in the nightly), and then evaluates the alerting thresholds with the
compute_*_metrics helpers. `snapshot_timings.py report` prints the same table by hand.

Provenance: this file started as a trimmed copy of the demo's demo_phases.py (the
pod-snapshot-sglang-demo branch, docker/scripts/snapshot/sglang/demo/demo_phases.py), which also
has a live `watch` mode, terminal colors, per-milestone narration, a cold-start cache and a
multi-iteration results.csv/.R writer. THIS FILE IS CANONICAL FOR THE NIGHTLY: the two are not
kept in sync, and when the snapshot hooks change their log lines the regexes below are updated
here against the images the nightly runs. The CSV header and metric names match the demo's
results.csv so the two can still be compared.

The phases of a cold start are:
  1. Container start -> engine hooked (vLLM API server / SGLang HTTP server)
  2. Subprocesses initialized (vLLM EngineCore / SGLang scheduler + detokenizer)
  3. Model weights downloaded and loaded into GPU VRAM
  4. KV cache allocation, torch.compile / CUDA graph capture, warmup
  5. Snapshot (sleep / release VRAM to CPU RAM, gVisor checkpoint to GCS, wake / resume VRAM)

The helper only reads from the cluster: it runs kubectl get and kubectl logs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

KUBECTL_TIMEOUT_S = 60

NUM = r"([0-9]+(?:\.[0-9]+)?)"
# SGLang regexes
HOOK_RE = re.compile(r"Hooked SGLang _wait_and_warmup in " + NUM + "s")
HOOKING_RE = re.compile(r"Hooking SGLang HTTP server startup")
WEIGHTS_BEGIN_RE = re.compile(r"Load weight begin\. avail mem=" + NUM + " GB")
WEIGHTS_END_RE = re.compile(r"Load weight end\. elapsed=" + NUM + " s")
KV_RE = re.compile(r"KV Cache is allocated\..*#tokens: ([0-9]+)")
GRAPH_END_RE = re.compile(
    r"Capture (?:target )?(?:(\w+) )?cuda graph end\. (?:elapsed=|Time elapsed: )" + NUM + " ?s",
    re.IGNORECASE,
)
WARMUP_DONE_RE = re.compile(
    r"Server warmup completed in " + NUM + r"s(?: \(cold-start elapsed_since_start=" + NUM + r"s\))?"
)
HOLD_RE = re.compile(r"server_status = ServerStatus\.Starting|Holding server_status at Starting")
GC_RE = re.compile(r"Froze Python garbage collection in " + NUM + "s")
IDLE_RE = re.compile(r"The SGLang scheduler is idle \(waited " + NUM + r"s\)")
RELEASE_REQ_RE = re.compile(r"Executing POST /release_memory_occupation")
RELEASE_HTTP_OK_RE = re.compile(r'"POST /release_memory_occupation HTTP/1\.[01]"\s+200')
RELEASED_RE = re.compile(r"Released GPU memory occupation .* in " + NUM + "s")
PURGED_RE = re.compile(r"Purged [0-9]+ cache entry/entries .* in " + NUM + "s")
CHECKPOINT_RE = re.compile(r"gVisor checkpoint completed successfully \(barrier unblocked in " + NUM + "s")
CHECKPOINT_OK_RE = re.compile(r"Snapshot checkpoint created successfully")
RESUME_REQ_RE = re.compile(r"Executing POST /resume_memory_occupation")
RESUME_HTTP_OK_RE = re.compile(r'"POST /resume_memory_occupation HTTP/1\.[01]"\s+200')
RESUMED_RE = re.compile(r"Resumed GPU memory occupation .* in " + NUM + "s")
RESUME_OK_RE = re.compile(r"SGLang memory occupation resumed successfully")
UP_RE = re.compile(
    r"(?:server_status = ServerStatus\.Up|Restored server_status to \S+) \(wake-to-ready=" + NUM + r"s\)"
)
STATUS_RESTORED_RE = re.compile(r"Restored server_status to ")

# vLLM regexes
VLLM_HOOKING_RE = re.compile(r"Hooking vLLM HTTP server startup")
VLLM_HOOK_RE = re.compile(r"Hooked vLLM build_app \(([^)]+)\) in " + NUM + "s")
VLLM_PATCHED_RE = re.compile(r"Successfully patched vLLM FastAPI lifespan context")
VLLM_APISERVER_RE = re.compile(r"vLLM API server version (\S+)")
VLLM_ENGINECORE_RE = re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\)|Initializing an? LLM engine")
VLLM_WEIGHTS_BEGIN_RE = re.compile(r"(?:Starting to load model (\S+)\.\.\.|Loading model from scratch\.\.\.)")
VLLM_WEIGHTS_END_RE = re.compile(r"Model loading took " + NUM + r" Gi?B(?: memory)? and " + NUM + r" seconds")
VLLM_KV_GIB_RE = re.compile(r"Available KV cache memory: " + NUM + r" GiB")
VLLM_KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
VLLM_COMPILE_RE = re.compile(r"torch\.compile takes " + NUM + r" s in total")
VLLM_GRAPH_END_RE = re.compile(r"Graph captur(?:e|ing) finished in " + NUM + r" secs, took " + NUM + r" GiB")
VLLM_WARMUP_DONE_RE = re.compile(
    r"init engine \(profile, create kv cache, warmup model\) took " + NUM + r" seconds"
)
VLLM_SLEEP_REQ_RE = re.compile(r"Executing engine\.sleep\(level=1\) to release physical VRAM")
VLLM_SLEEP_DONE_RE = re.compile(
    r"(?:It took " + NUM + r" seconds to fall asleep|Released physical VRAM via engine\.sleep\(level=1\) in " + NUM + r"s)"
)
VLLM_TRIGGER_RE = re.compile(r"Triggering snapshot checkpoint")
VLLM_RESTORED_RE = re.compile(r"Process restored from snapshot checkpoint\. Resuming engine")
VLLM_WAKE_REQ_RE = re.compile(r"Executing engine\.wake_up\(\) to restore VRAM")
VLLM_WAKE_DONE_RE = re.compile(
    r"(?:It took " + NUM + r" seconds to wake up|Restored physical VRAM via engine\.wake_up\(\) in " + NUM + r"s)"
)
VLLM_APP_UP_RE = re.compile(r"(?:Application startup complete\.|Uvicorn running on http)")

TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)$")
# Terminal escape sequences and other control characters, which pod names must not inject.
CONTROL_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]|[\x00-\x1f\x7f-\x9f]")


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_ts(value):
    """Parses a Kubernetes RFC 3339 timestamp (nanoseconds allowed); returns None if invalid."""
    if not isinstance(value, str):
        return None
    m = TS_RE.match(value.strip())
    if not m:
        return None
    base, frac, zone = m.groups()
    text = base + ("." + frac[:6].ljust(6, "0") if frac else "") + ("+00:00" if zone == "Z" else zone)
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def clean(text, limit=220):
    """Removes control characters from cluster-provided text and shortens it for one output line."""
    text = CONTROL_RE.sub(" ", str(text))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def fmt_dur(seconds, decimals=2):
    if seconds is None:
        return "-"
    seconds = round(max(0.0, seconds), decimals)
    if seconds < 60:
        return f"{seconds:.{decimals}f}s"
    minutes, rest = divmod(seconds, 60)
    width = 2 + (decimals + 1 if decimals else 0)
    rest_text = f"{rest:0{width}.{decimals}f}s"
    if minutes < 60:
        return f"{int(minutes)}m {rest_text}"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {rest_text}"


def delta(start, end):
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def infer_engine(pod=None, selector=""):
    """Infers 'vllm' or 'sglang' from pod labels, pod name, or selector string."""
    if pod:
        meta = pod.get("metadata", {})
        labels = meta.get("labels", {})
        eng = labels.get("llm-d.ai/engine-type", "").lower()
        if eng in ("vllm", "sglang"):
            return eng
        name = meta.get("name", "").lower()
        if "vllm" in name:
            return "vllm"
        if "sglang" in name:
            return "sglang"
    sel = (selector or "").lower()
    if "notin (sglang)" in sel or "engine-type=vllm" in sel:
        return "vllm"
    if "sglang" in sel:
        return "sglang"
    return ""


def engine_display(engine):
    if engine == "vllm":
        return "vLLM"
    if engine == "sglang":
        return "SGLang"
    return "Model Server"


class Kubectl:
    """Runs read-only kubectl commands in one namespace; returns None when a command fails."""

    def __init__(self, namespace):
        self.namespace = namespace

    def run(self, *args):
        cmd = ["kubectl", *args, "-n", self.namespace, "--request-timeout=30s"]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=KUBECTL_TIMEOUT_S,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return res.stdout if res.returncode == 0 else None

    def json(self, *args):
        out = self.run(*args, "-o", "json")
        if out is None:
            return None
        try:
            return json.loads(out)
        except ValueError:
            return None


def conditions(pod):
    return {
        c.get("type"): parse_ts(c.get("lastTransitionTime"))
        for c in pod.get("status", {}).get("conditions", [])
        if c.get("status") == "True"
    }


def container_status(pod):
    statuses = pod.get("status", {}).get("containerStatuses") or []
    return statuses[0] if statuses else {}


def container_name(pod):
    containers = pod.get("spec", {}).get("containers") or [{}]
    return containers[0].get("name", "")


def event_ts(event):
    for value in (event.get("firstTimestamp"), event.get("eventTime"), event.get("lastTimestamp")):
        ts = parse_ts(value)
        if ts:
            return ts
    return parse_ts(event.get("metadata", {}).get("creationTimestamp")) or now()


def log_values(log_text, engine=""):
    """Returns the timestamps parsed from one container's `kubectl logs --timestamps` output.

    Keys ending in `_ts` are datetimes used by compute_cold_metrics / compute_restore_metrics;
    `engine` is 'vllm' or 'sglang' once a line identifies it; `restored` is set when the log
    belongs to a pod that was restored from the snapshot. A restored pod's log starts at the
    checkpoint line, so a checkpoint line without cold-start markers before it means a restore.
    """
    values = {"engine": engine}
    hooks = 0
    cold = False

    for raw in log_text.split("\n"):
        stamp, _, text = raw.partition(" ")
        ts = parse_ts(stamp)
        if ts is None or '"GET /' in text:
            continue

        # --- Phase 1 & 2: launcher, engine hooked, subprocesses ---
        if m := HOOK_RE.search(text):
            values["engine"] = "sglang"
            hooks += 1
            cold = True
            if hooks == 1:
                values.update(hook_s=float(m.group(1)), hook_ts=ts)
            else:
                values["workers_ts"] = ts
        elif HOOKING_RE.search(text):
            values["engine"] = "sglang"
            values.setdefault("launcher_ts", ts)
        elif VLLM_HOOKING_RE.search(text):
            values["engine"] = "vllm"
            values.setdefault("launcher_ts", ts)
        elif m := VLLM_HOOK_RE.search(text):
            values["engine"] = "vllm"
            hooks += 1
            cold = True
            if hooks == 1:
                values.setdefault("launcher_ts", ts)
                values.update(hook_s=float(m.group(2)), hook_ts=ts)
            else:
                values["workers_ts"] = ts
        elif VLLM_APISERVER_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values.setdefault("hook_ts", ts)
        elif VLLM_PATCHED_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            if "workers_ts" not in values and "weights_begin_ts" not in values:
                values.setdefault("hook_ts", ts)
        elif VLLM_ENGINECORE_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values.setdefault("workers_ts", ts)

        # --- Phase 3: model weights ---
        elif WEIGHTS_BEGIN_RE.search(text):
            values["engine"] = "sglang"
            cold = True
            values["weights_begin_ts"] = ts
        elif VLLM_WEIGHTS_BEGIN_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values.setdefault("weights_begin_ts", ts)
        elif WEIGHTS_END_RE.search(text):
            values["weights_end_ts"] = ts
        elif VLLM_WEIGHTS_END_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values["weights_end_ts"] = ts

        # --- Phase 4: KV cache, CUDA graphs, warmup ---
        elif KV_RE.search(text):
            values["kv_ts"] = ts
        elif VLLM_KV_GIB_RE.search(text) or VLLM_KV_TOKENS_RE.search(text):
            values["engine"] = "vllm"
            values["kv_ts"] = ts
        elif VLLM_COMPILE_RE.search(text):
            values["engine"] = "vllm"
            values["compile_ts"] = ts
        elif GRAPH_END_RE.search(text):
            values["graph_end_ts"] = ts
        elif VLLM_GRAPH_END_RE.search(text):
            values["engine"] = "vllm"
            values["graph_end_ts"] = ts
        elif WARMUP_DONE_RE.search(text):
            values["warmup_done_ts"] = ts
        elif VLLM_WARMUP_DONE_RE.search(text):
            values["engine"] = "vllm"
            values["warmup_done_ts"] = ts

        # --- Phase 5: snapshot (and restore) ---
        elif HOLD_RE.search(text):
            cold = True
            values.setdefault("warmup_done_ts", ts)
        elif GC_RE.search(text):
            values["gc_ts"] = ts
        elif IDLE_RE.search(text):
            values["idle_ts"] = ts
        elif RELEASE_REQ_RE.search(text):
            values["engine"] = "sglang"
            cold = True
            values["release_req_ts"] = ts
        elif RELEASE_HTTP_OK_RE.search(text):
            values["engine"] = "sglang"
            values.setdefault("released_ts", ts)
        elif VLLM_SLEEP_REQ_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values["sleep_start_ts"] = ts
        elif RELEASED_RE.search(text):
            values["released_ts"] = ts
        elif VLLM_SLEEP_DONE_RE.search(text):
            values["engine"] = "vllm"
            values["released_ts"] = ts
        elif VLLM_TRIGGER_RE.search(text):
            values.setdefault("trigger_ts", ts)
            values.setdefault("released_ts", ts)
        elif PURGED_RE.search(text):
            values["purged_ts"] = ts
        elif CHECKPOINT_RE.search(text):
            if cold:
                values["checkpoint_ts"] = ts
            else:
                values.update(restored=True, restore_ts=ts)
        elif CHECKPOINT_OK_RE.search(text):
            if cold:
                values.setdefault("checkpoint_ts", ts)
            elif "restore_ts" not in values:
                values.update(restored=True, restore_ts=ts)
        elif VLLM_RESTORED_RE.search(text):
            values["engine"] = "vllm"
            if not cold and "restore_ts" not in values:
                values.update(restored=True, restore_ts=ts)
        elif VLLM_WAKE_REQ_RE.search(text):
            values["engine"] = "vllm"
            values["wake_req_ts"] = ts
        elif RESUME_REQ_RE.search(text):
            values["engine"] = "sglang"
            values["resume_req_ts"] = ts
        elif RESUME_HTTP_OK_RE.search(text) or RESUME_OK_RE.search(text):
            values["engine"] = "sglang"
            values.setdefault("resumed_ts", ts)
        elif RESUMED_RE.search(text):
            values["resumed_ts"] = ts
        elif VLLM_WAKE_DONE_RE.search(text):
            values["engine"] = "vllm"
            values["resumed_ts"] = ts
        elif UP_RE.search(text):
            values["up_ts"] = ts
        elif STATUS_RESTORED_RE.search(text):
            # "Restored server_status to ServerStatus.Up." is the last line of the SGLang
            # wrapper's sleep/snapshot/wake cycle (docker/scripts/snapshot/sglang/wrapper.py),
            # so it always wins over an earlier "up" candidate.
            values["up_ts"] = ts
        elif VLLM_APP_UP_RE.search(text):
            # uvicorn logs this for both engines, so it must not set the engine. vLLM takes
            # the snapshot inside its patched lifespan, so the line follows the wake; SGLang
            # logs it when the HTTP server starts, before the warmup hook and the snapshot.
            # It therefore only counts as "up" once a snapshot or restore marker was seen.
            if "resumed_ts" not in values and "wake_req_ts" in values:
                values["resumed_ts"] = ts
            if any(k in values for k in ("checkpoint_ts", "restore_ts", "resumed_ts", "trigger_ts")):
                values.setdefault("up_ts", ts)
    return values


def pod_times(kubectl, pod, events, engine=""):
    meta = pod.get("metadata", {})
    name = meta.get("name", "")
    eng = infer_engine(pod) or engine
    conds = conditions(pod)
    running = (container_status(pod).get("state") or {}).get("running") or {}
    times = {
        "name": name,
        "engine": eng,
        "created": parse_ts(meta.get("creationTimestamp")),
        "scheduled": conds.get("PodScheduled"),
        "sandbox": conds.get("PodReadyToStartContainers"),
        "restored": conds.get("PodRestored"),
        "ready": conds.get("Ready"),
        "started": parse_ts(running.get("startedAt")),
        "pulling_ts": None,
        "pulled_ts": None,
    }
    # Events expire after about an hour; without them, the image pull is merged into the next row.
    for event in sorted(events, key=event_ts):
        if event.get("involvedObject", {}).get("name") != name:
            continue
        if event.get("reason") == "Pulling" and times["pulling_ts"] is None:
            times["pulling_ts"] = event_ts(event)
        elif event.get("reason") == "Pulled" and times["pulled_ts"] is None:
            times["pulled_ts"] = event_ts(event)
    log = kubectl.run("logs", name, "-c", container_name(pod), "--timestamps") or ""
    times["log"] = log_values(log, engine=eng)
    if not times["engine"] and times["log"].get("engine"):
        times["engine"] = times["log"]["engine"]
    return times


def snapshot_ready(snaps, pod=None):
    ready = []
    for snap in snaps:
        for c in snap.get("status", {}).get("conditions", []):
            if c.get("type") == "Ready" and c.get("status") == "True":
                ts = parse_ts(c.get("lastTransitionTime"))
                if ts:
                    ready.append(ts)
    if not ready:
        return None
    if pod and pod.get("created") and pod.get("ready"):
        in_window = [
            ts for ts in ready
            if pod["created"] <= ts <= pod["ready"] + dt.timedelta(seconds=30)
        ]
        return max(in_window) if in_window else None
    return max(ready)


def is_restored(pod):
    return bool(pod["restored"] or pod["log"].get("restored"))


def kind_text(pod):
    if is_restored(pod):
        return "restore from the snapshot"
    if pod["log"].get("released_ts") or pod["log"].get("checkpoint_ts"):
        return "cold start + take the snapshot"
    return "cold start"


# --- Metrics: the unified 1..4f hierarchy ---------------------------------------------------------
#
# 1  Kubernetes: schedule the pod and start the container   1a..1d
# 2  LLM engine cold start                                   2a..2g   (Pod 1 only)
# 3  Wait for the readiness probe
# 4  Take the snapshot                                       4a..4f   (Pod 1; Pod 2 has 4e, 4f)
#
# The numbered steps add up to pod created -> Ready and the lettered sub-steps add up to their
# step, so each boundary is clamped to be no earlier than the previous one.


def compute_k8s_metrics(pod):
    """Computes contiguous Kubernetes sub-step timestamps and durations (1, 1a..1d)."""
    t0 = pod["created"]
    t_1a = max(t0, pod.get("scheduled") or t0)
    t_1b = max(t_1a, pod.get("sandbox") or pod.get("pulling_ts") or t_1a)
    t_1c = max(t_1b, pod.get("pulled_ts") or t_1b)
    t_1d = max(t_1c, pod.get("started") or t_1c)
    return t_1d, {
        "1": (t_1d - t0).total_seconds(),
        "1a": (t_1a - t0).total_seconds(),
        "1b": (t_1b - t_1a).total_seconds(),
        "1c": (t_1c - t_1b).total_seconds(),
        "1d": (t_1d - t_1c).total_seconds(),
        "t_1a": t_1a,
        "t_1b": t_1b,
        "t_1c": t_1c,
        "t_1d": t_1d,
    }


def compute_cold_metrics(pod):
    """Computes contiguous ColdStart metrics (1, 1a..1d, 2, 2a..2g, 3, 4, 4a..4f) for Pod 1."""
    t_1d, k8s = compute_k8s_metrics(pod)
    log = pod["log"]
    weights_begin = log.get("weights_begin_ts")
    launcher = log.get("launcher_ts")
    if weights_begin and launcher and launcher > weights_begin:
        launcher = None
    hook = log.get("hook_ts")
    if weights_begin and hook and hook > weights_begin:
        hook = None
    if launcher is None and hook is not None and log.get("hook_s") is not None:
        launcher = max(t_1d, hook - dt.timedelta(seconds=float(log["hook_s"])))

    t_end_2 = (
        log.get("sleep_start_ts")
        or log.get("warmup_done_ts")
        or log.get("gc_ts")
        or log.get("idle_ts")
        or log.get("release_req_ts")
        or log.get("released_ts")
        or t_1d
    )
    t_2a = max(t_1d, launcher or t_1d)
    t_2b = max(t_2a, hook or t_2a)
    t_2c = max(t_2b, weights_begin or log.get("workers_ts") or t_2b)
    t_2d = max(t_2c, log.get("weights_end_ts") or t_2c)
    t_2e = max(t_2d, log.get("kv_ts") or t_2d)
    t_2f = max(t_2e, log.get("graph_end_ts") or log.get("compile_ts") or t_2e)
    t_2g = max(t_2f, t_end_2)

    t_4a = max(t_2g, log.get("gc_ts") or t_2g)
    t_4b = max(t_4a, log.get("idle_ts") or t_4a)
    t_4c = max(t_4b, log.get("released_ts") or t_4b)
    t_4d = max(t_4c, log.get("purged_ts") or t_4c)
    t_4e = max(t_4d, log.get("checkpoint_ts") or t_4d)
    up = log.get("up_ts") or log.get("resumed_ts")
    t_4f = max(t_4e, up or t_4e)
    t_ready = max(t_4f, pod.get("ready") or t_4f)

    m = dict(k8s)
    m.update({
        "2": (t_2g - t_1d).total_seconds(),
        "2a": (t_2a - t_1d).total_seconds(),
        "2b": (t_2b - t_2a).total_seconds(),
        "2c": (t_2c - t_2b).total_seconds(),
        "2d": (t_2d - t_2c).total_seconds(),
        "2e": (t_2e - t_2d).total_seconds(),
        "2f": (t_2f - t_2e).total_seconds(),
        "2g": (t_2g - t_2f).total_seconds(),
        "3": (t_ready - t_4f).total_seconds(),
        "4": (t_4f - t_2g).total_seconds(),
        "4a": (t_4a - t_2g).total_seconds(),
        "4b": (t_4b - t_4a).total_seconds(),
        "4c": (t_4c - t_4b).total_seconds(),
        "4d": (t_4d - t_4c).total_seconds(),
        "4e": (t_4e - t_4d).total_seconds(),
        "4f": (t_4f - t_4e).total_seconds(),
        "t_2a": t_2a,
        "t_2b": t_2b,
        "t_2c": t_2c,
        "t_2d": t_2d,
        "t_2e": t_2e,
        "t_2f": t_2f,
        "t_2g": t_2g,
        "t_4a": t_4a,
        "t_4b": t_4b,
        "t_4c": t_4c,
        "t_4d": t_4d,
        "t_4e": t_4e,
        "t_4f": t_4f,
        "t_ready": t_ready,
    })
    return m


def compute_restore_metrics(pod):
    """Computes contiguous Restore metrics (1, 1a..1d, 4e, 4f, 3) for Pod 2."""
    t_1d, k8s = compute_k8s_metrics(pod)
    log = pod["log"]
    t_4e = max(t_1d, log.get("restore_ts") or pod.get("restored") or t_1d)
    up = log.get("up_ts") or log.get("resumed_ts")
    t_4f = max(t_4e, up or t_4e)
    t_ready = max(t_4f, pod.get("ready") or t_4f)
    m = dict(k8s)
    m.update({
        "4e": (t_4e - t_1d).total_seconds(),
        "4f": (t_4f - t_4e).total_seconds(),
        "3": (t_ready - t_4f).total_seconds(),
        "t_4e": t_4e,
        "t_4f": t_4f,
        "t_ready": t_ready,
    })
    return m


def pod_metrics(pod):
    """The pod's metrics for the way it started: restore metrics for a restored pod, else cold."""
    return compute_restore_metrics(pod) if is_restored(pod) else compute_cold_metrics(pod)


# Labels of the 1..4f hierarchy, shared by the timing table and the CSV (same names as the
# demo's results.csv). Each entry is (metric key, label).
K8S_STEPS = (
    ("1a", "Wait for a GPU node"),
    ("1b", "Set up the pod sandbox (gvisor)"),
    ("1c", "Pull the image (image streaming)"),
    ("1d", "Start the container"),
)
ENGINE_STEPS = (
    ("2a", "Start Python"),
    ("2b", "Import engine and dependencies (vLLM / SGLang, PyTorch)"),
    ("2c", "Initialize LLM engine and background workers/processes"),
    ("2d", "Download and load model weights"),
    ("2e", "Allocate KV cache memory"),
    ("2f", "Capture CUDA graphs / compile execution graph"),
    ("2g", "Start API server / frontend and run warm-up requests"),
)
SNAPSHOT_STEPS = (
    ("4a", "Freeze Python garbage collection"),
    ("4b", "Wait for an idle scheduler"),
    ("4c", "Release GPU memory (weights + KV cache) to CPU RAM"),
    ("4d", "Purge the local weight cache"),
    ("4e", "gVisor checkpoint + upload to GCS"),
    ("4f", "Resume GPU memory (weights + KV cache)"),
)
LABEL_1 = "Kubernetes: schedule the pod and start the container"
LABEL_2 = "LLM Engine cold start (vLLM / SGLang)"
LABEL_3 = "Wait for the readiness probe"
LABEL_4 = "Take the snapshot (phase 5)"


def composite_metrics(cold_m, rest_m):
    """The composite metrics used for alerting (check_thresholds.py applies the thresholds).

    Returns a list of (mode, name, computation, seconds).
    """
    out = []
    if cold_m is not None:
        out.append(("ColdStart", "E2E Cold Start (including node provisioning)", "Rows 1+2+3",
                    cold_m["1"] + cold_m["2"] + cold_m["3"]))
    if rest_m is not None:
        out.append(("Restore", "Warm Start (from restoration)", "1b+1c+4e+4f+3",
                    rest_m["1b"] + rest_m["1c"] + rest_m["4e"] + rest_m["4f"] + rest_m["3"]))
    if cold_m is not None:
        out.append(("ColdStart", "Cold Start (excluding node provisioning)", "1b-d+2",
                    cold_m["1b"] + cold_m["1c"] + cold_m["1d"] + cold_m["2"]))
        out.append(("ColdStart", "Application Cold Start", "Row 2 (2a-g)", cold_m["2"]))
        out.append(("ColdStart", "Total Snapshot Overhead Time", "Row 4 (4a-f)", cold_m["4"]))
    return out


def csv_rows(cold_m, rest_m, p1, snap_ready, engine):
    """Returns (ModelServer, Mode, MetricName, MetricValue) tuples: the 1..4f hierarchy of each
    pod, the composite metrics, and Pod 1's total and PodSnapshot Ready time."""
    model_server = engine_display(engine)
    out = []

    def add(mode, name, val):
        if val is not None:
            out.append((model_server, mode, name, float(val)))

    if cold_m is not None:
        for key, label in (("1", LABEL_1), *K8S_STEPS, ("2", LABEL_2), *ENGINE_STEPS,
                           ("3", LABEL_3), ("4", LABEL_4), *SNAPSHOT_STEPS):
            add("ColdStart", f"{key}. {label}", cold_m[key])
    if rest_m is not None:
        for key, label in (("1", LABEL_1), *K8S_STEPS, ("3", LABEL_3), *SNAPSHOT_STEPS[-2:]):
            add("Restore", f"{key}. {label}", rest_m[key])
    for mode, name, _, secs in composite_metrics(cold_m, rest_m):
        add(mode, name, secs)
    if cold_m is not None:
        add("ColdStart", "Total: pod created -> Ready", delta(p1["created"], p1["ready"]))
        if snap_ready and snap_ready >= p1["created"]:
            add("ColdStart", "PodSnapshot Ready", delta(p1["created"], snap_ready))
    return out


def write_csv(rows, csv_path=""):
    """Formats rows as the 5-column results.csv and writes it to csv_path if given.

    Seq is always 1: a nightly run is a single iteration (the demo appends one per iteration).
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Seq", "ModelServer", "Mode", "MetricName", "MetricValue"])
    for model_server, mode, name, val in rows:
        writer.writerow([1, model_server, mode, name, f"{val:.2f}"])
    text = buf.getvalue()
    if csv_path:
        try:
            path = pathlib.Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"WARNING: could not write the CSV to {csv_path}: {exc}", file=sys.stderr)
    return text


# --- Timing table --------------------------------------------------------------------------------

LABEL_W = 68
COL_W = 12
WIDTH = 2 + LABEL_W + 2 * COL_W
LETTERS = "abcdefghijklmnopqrstuvwxyz"


class Row(NamedTuple):
    """One line of a pod's timing table: a step with its sub-steps, or a sub-step."""

    label: str
    seconds: float
    end: dt.datetime
    precise: bool  # True when both ends come from log timestamps (shown with one decimal)
    subs: tuple = ()


def pod_rows(pod, m):
    """The steps from pod creation to Ready, as Rows, from the pod's metrics `m`."""
    k8s = Row(LABEL_1, m["1"], m["t_1d"], False,
              tuple(Row(label, m[k], m["t_" + k], False) for k, label in K8S_STEPS))
    if is_restored(pod):
        return [
            k8s,
            Row("Restore the snapshot (4e. gVisor checkpoint / GCS transfer)", m["4e"], m["t_4e"], True),
            Row("Resume GPU memory (weights + KV cache) (4f)", m["4f"], m["t_4f"], True),
            Row(LABEL_3 + " (3)", m["3"], m["t_ready"], False),
        ]
    return [
        k8s,
        Row(LABEL_2, m["2"], m["t_2g"], True,
            tuple(Row(label, m[k], m["t_" + k], True) for k, label in ENGINE_STEPS)),
        Row(LABEL_3, m["3"], m["t_ready"], False),
        Row(LABEL_4, m["4"], m["t_4f"], True,
            tuple(Row(label, m[k], m["t_" + k], True) for k, label in SNAPSHOT_STEPS)),
    ]


def fmt_cum(seconds):
    """The time since the pod was created, in whole seconds: +MM:SS."""
    total = int(round(max(0.0, seconds or 0.0)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"+{hours}:{minutes:02d}:{secs:02d}" if hours else f"+{minutes:02d}:{secs:02d}"


def fmt_row(row):
    return fmt_dur(row.seconds, 1 if row.precise else 0)


def fmt_total(seconds):
    return fmt_dur(seconds, 0) if seconds is not None else "not Ready yet"


def clock(ts):
    return ts.astimezone().strftime("%H:%M:%S")


def table_line(label, duration="", cumulative="", indent=0):
    width = LABEL_W - indent
    print(f"  {' ' * indent}{clean(label, width):<{width}}{duration:>{COL_W}}{cumulative:>{COL_W}}".rstrip())


def table_header(title, *columns):
    cols = "".join(f"{column:>{COL_W}}" for column in columns)
    print("-" * WIDTH)
    print(f"{clean(title, WIDTH - len(cols)):<{WIDTH - len(cols)}}{cols}")


def print_pod(number, pod, rows, snap_ready):
    created = pod["created"]
    eng_disp = engine_display(pod.get("engine") or pod["log"].get("engine"))
    eng_note = f" ({eng_disp})" if eng_disp != "Model Server" else ""
    table_header(f"POD {number}{eng_note}: {kind_text(pod)}", "Duration", "Cumulative")
    print(f"  {clean(pod['name'], WIDTH - 30)}, created at {clock(created)}")
    print("-" * WIDTH)
    for n, row in enumerate(rows, 1):
        table_line(f"{n}. {row.label}", fmt_row(row), fmt_cum(delta(created, row.end)))
        for letter, sub in zip(LETTERS, row.subs):
            table_line(f"{letter}. {sub.label}", fmt_row(sub), fmt_cum(delta(created, sub.end)), indent=3)
    print("  " + "-" * (WIDTH - 2))
    total = delta(created, pod["ready"])
    if total is None:
        table_line("Not Ready yet")
    else:
        table_line("Total: pod created -> Ready", fmt_total(total), fmt_cum(total))
    if snap_ready and not is_restored(pod) and snap_ready >= created:
        print(f"  The PodSnapshot was Ready at {fmt_cum(delta(created, snap_ready))} ({clock(snap_ready)}).")


def print_report(p1, p2, snap_ready, engine="", csv_path=""):
    pods = [(number, pod) for number, pod in ((1, p1), (2, p2)) if pod and pod["created"]]
    totals = {number: delta(pod["created"], pod["ready"]) for number, pod in pods}
    metrics = {number: pod_metrics(pod) for number, pod in pods}
    cold_m = metrics[1] if 1 in metrics and not is_restored(p1) else None
    rest_m = metrics[2] if 2 in metrics and is_restored(p2) else None
    t1, t2 = totals.get(1), totals.get(2)
    speedup = t1 / t2 if cold_m is not None and rest_m is not None and t1 and t2 else None
    eng_disp = engine_display(engine)
    hdr_suffix = f" ({eng_disp})" if eng_disp != "Model Server" else ""

    print("")
    print("=" * WIDTH)
    print(f"  TIMING SUMMARY{hdr_suffix}: pod created -> Ready")
    print("=" * WIDTH)
    if not pods:
        print("  No model server pods found.")
    for number, pod in pods:
        print(f"  {f'Pod {number}, {kind_text(pod)}:':<44}{fmt_total(totals[number]):>13}")
    if speedup is not None:
        print(f"  Pod 2 was Ready {speedup:.1f}x faster than Pod 1.")
    if pods:
        print("")
        print("  The numbered steps add up to the total, and the lettered sub-steps add up to their step.")
        print("  Cumulative: the time since the pod was created, at the end of the step.")
    for number, pod in pods:
        print_pod(number, pod, pod_rows(pod, metrics[number]), snap_ready if number == 1 else None)

    composites = composite_metrics(cold_m, rest_m)
    if composites:
        table_header(f"COMPOSITE METRICS USED FOR ALERTING{hdr_suffix}", "Computation", "Seconds")
        print("-" * WIDTH)
        for _, name, computation, secs in composites:
            table_line(name, computation, f"{secs:.2f}s")

    rows = csv_rows(cold_m, rest_m, p1, snap_ready, engine)
    if rows:
        csv_text = write_csv(rows, csv_path=csv_path)
        saved_note = f" (saved to {csv_path})" if csv_path else ""
        table_header(f"CSV RESULTS{hdr_suffix}{saved_note}")
        print("-" * WIDTH)
        print(csv_text.rstrip("\n"))
    print("=" * WIDTH)
    if pods:
        recap = ", ".join(f"Pod {number} {fmt_total(totals[number])}" for number, _ in pods)
        more = f" (Pod 2 was Ready {speedup:.1f}x faster)" if speedup else ""
        print(f"  Pod created -> Ready{hdr_suffix}: {recap}{more}.")
        print("=" * WIDTH)


def load_pods(kubectl, selector, engine=""):
    """Reads the cluster once: returns (pod1, pod2, podsnapshot_ready_ts).

    Pod 1 is the oldest pod matching `selector` that is not being deleted (cold start + snapshot)
    and Pod 2 the newest (restore from the snapshot); each is a pod_times dict, or None when
    there are fewer than one / two pods. Raises RuntimeError when the pods cannot be listed.
    This is the single pod-selection path shared by `report` and check_thresholds.py.
    """
    pods = kubectl.json("get", "pods", "-l", selector)
    if pods is None:
        raise RuntimeError(f"could not list the pods with kubectl get pods -l {selector} -n {kubectl.namespace}")
    items = sorted(
        (p for p in pods.get("items", []) if not p.get("metadata", {}).get("deletionTimestamp")),
        key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""),
    )
    events = (kubectl.json("get", "events", "--field-selector", "involvedObject.kind=Pod") or {}).get("items", [])
    snaps = (kubectl.json("get", "podsnapshots") or {}).get("items", [])
    p1 = pod_times(kubectl, items[0], events, engine=engine) if items else None
    p2 = pod_times(kubectl, items[-1], events, engine=engine) if len(items) >= 2 else None
    return p1, p2, snapshot_ready(snaps, pod=p1)


def detected_engine(p1, p2, engine=""):
    """The engine given, or the one detected from the pods' labels / names / logs, or ''."""
    if engine:
        return str(engine)
    pod = p1 or p2
    return str(pod.get("engine") or pod["log"].get("engine") or "") if pod else ""


def cmd_report(args):
    eng = str(args.engine or infer_engine(selector=args.selector) or "")
    try:
        p1, p2, snap_ready = load_pods(Kubectl(args.namespace), args.selector, engine=eng)
    except RuntimeError as exc:
        print(f"ERROR: {exc}.", file=sys.stderr)
        return 1
    print_report(p1, p2, snap_ready, engine=detected_engine(p1, p2, eng), csv_path=args.csv)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="print the timing table")
    report.add_argument("--namespace", required=True)
    report.add_argument("--selector", required=True)
    report.add_argument("--engine", default="", help="engine type: vllm or sglang")
    report.add_argument("--csv", default="", help="optional path to write the metrics CSV")
    args = parser.parse_args(argv)
    try:
        return cmd_report(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
