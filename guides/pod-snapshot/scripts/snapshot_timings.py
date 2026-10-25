#!/usr/bin/env python3
"""Startup, snapshot and restore timings for GKE Pod Snapshots (vLLM and SGLang).

Trimmed for the nightly CI from the demo's demo_phases.py (pod-snapshot-sglang-demo branch,
docker/scripts/snapshot/sglang/demo/demo_phases.py), which also has a live `watch` mode, terminal
colors, a cold-start cache and run_demo.sh step/latency reporting.

Subcommand:
  report  Prints the timing table: the time from pod creation to Ready of each pod and the
          speedup, then for each pod (Pod 1 cold starts and takes the snapshot, Pod 2 restores
          from it) the steps from pod creation to Ready, with lettered sub-steps and the cumulative
          time, and the composite metrics. With --csv, also writes the metrics as CSV (and R).

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
import os
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

KUBECTL_TIMEOUT_S = 60

P1_SGLANG = "Phase 1/5 SGLang start"
P1_VLLM = "Phase 1/5 vLLM start"
P2_SGLANG = "Phase 2/5 Subprocesses"
P2_VLLM = "Phase 2/5 EngineCore"
P3 = "Phase 3/5 Model weights"
P4 = "Phase 4/5 KV cache, CUDA graphs, warmup"
P5 = "Phase 5/5 Snapshot"
RESTORE = "Restore"
SGLANG = "SGLang"
VLLM = "vLLM"


NUM = r"([0-9]+(?:\.[0-9]+)?)"
# SGLang regexes
HOOK_RE = re.compile(r"Hooked SGLang _wait_and_warmup in " + NUM + "s")
HOOKING_RE = re.compile(r"Hooking SGLang HTTP server startup")
DOWNLOAD_RE = re.compile(r"(?:has no files matching .*will attempt download|Fetching [0-9]+ files)")
WEIGHTS_BEGIN_RE = re.compile(r"Load weight begin\. avail mem=" + NUM + " GB")
WEIGHTS_END_RE = re.compile(r"Load weight end\. elapsed=" + NUM + " s")
KV_RE = re.compile(r"KV Cache is allocated\..*#tokens: ([0-9]+)")
GRAPH_BEGIN_RE = re.compile(r"Capture (?:target )?(?:(\w+) )?cuda graph begin", re.IGNORECASE)
GRAPH_END_RE = re.compile(
    r"Capture (?:target )?(?:(\w+) )?cuda graph end\. (?:elapsed=|Time elapsed: )" + NUM + " ?s",
    re.IGNORECASE,
)
KV_ALLOC_RE = re.compile(r"Engine startup timings \(s\):.*\bkv_cache_allocation=" + NUM)
WARMUP_START_RE = re.compile(r"Starting SGLang server warmup")
WARMUP_DONE_RE = re.compile(
    r"Server warmup completed in " + NUM + r"s(?: \(cold-start elapsed_since_start=" + NUM + r"s\))?"
)
HOLD_RE = re.compile(r"server_status = ServerStatus\.Starting|Holding server_status at Starting")
GC_RE = re.compile(r"Froze Python garbage collection in " + NUM + "s")
IDLE_RE = re.compile(r"The SGLang scheduler is idle \(waited " + NUM + r"s\)")
FLUSH_RE = re.compile(r"Executing POST /flush_cache")
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
ENTER_RE = re.compile(r"Entering patched _wait_and_warmup .*elapsed_since_start=" + NUM + "s")
WEIGHTS_READY_RE = re.compile(r"Model weights are ready in GPUs \(waited " + NUM + r"s\)")
LAUNCH_CB_RE = re.compile(r"launch_callback completed in " + NUM + "s")

# vLLM regexes
VLLM_HOOKING_RE = re.compile(r"Hooking vLLM HTTP server startup")
VLLM_HOOK_RE = re.compile(r"Hooked vLLM build_app \(([^)]+)\) in " + NUM + "s")
VLLM_PATCHED_RE = re.compile(r"Successfully patched vLLM FastAPI lifespan context")
VLLM_APISERVER_RE = re.compile(r"vLLM API server version (\S+)")
VLLM_ENGINECORE_RE = re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\)|Initializing an? LLM engine")
VLLM_WEIGHTS_BEGIN_RE = re.compile(r"(?:Starting to load model (\S+)\.\.\.|Loading model from scratch\.\.\.)")
VLLM_SAFETENSORS_RE = re.compile(r"Loading weights took " + NUM + r" seconds")
VLLM_WEIGHTS_END_RE = re.compile(r"Model loading took " + NUM + r" Gi?B(?: memory)? and " + NUM + r" seconds")
VLLM_PROFILE_RE = re.compile(r"Memory profiling takes " + NUM + r" seconds")
VLLM_KV_GIB_RE = re.compile(r"Available KV cache memory: " + NUM + r" GiB")
VLLM_KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
VLLM_COMPILE_RE = re.compile(r"torch\.compile takes " + NUM + r" s in total")
VLLM_GRAPH_BEGIN_RE = re.compile(r"Capturing (?:CUDA graphs|cudagraphs)", re.IGNORECASE)
VLLM_GRAPH_END_RE = re.compile(r"Graph captur(?:e|ing) finished in " + NUM + r" secs, took " + NUM + r" GiB")
VLLM_WARMUP_DONE_RE = re.compile(
    r"init engine \(profile, create kv cache, warmup model\) took " + NUM + r" seconds"
)
VLLM_SLEEP_REQ_RE = re.compile(r"Executing engine\.sleep\(level=1\) to release physical VRAM")
VLLM_SLEEP_FREED_RE = re.compile(
    r"(?:CuMemAllocator: sleep freed "
    + NUM
    + r" GiB memory in total, of which "
    + NUM
    + r" GiB is backed up in CPU|Sleep mode freed "
    + NUM
    + r" GiB memory)"
)
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

PROBLEM_RES = (
    (re.compile(r"Server warmup failed"), "ERROR"),
    (re.compile(r"Snapshot checkpointing failed"), "ERROR"),
    (re.compile(r"Sleep/wake cycle failed"), "ERROR"),
    (re.compile(r"The SGLang scheduler did not become idle"), "ERROR"),
    (re.compile(r"SGLang server is not importable"), "ERROR"),
    (re.compile(r"vLLM API server is not importable"), "ERROR"),
    (re.compile(r"vLLM engine does not support (?:sleep|wake_up)"), "ERROR"),
    (re.compile(r"Pod snapshot trigger not available"), "WARNING"),
)
# Python exceptions start at the beginning of the line; SGLang's own log lines start with "[date]".
EXCEPTION_RE = re.compile(r"^(?:[a-z_][A-Za-z0-9_]*\.)*[A-Z][A-Za-z0-9_]*(?:Error|Exception)(?::|$)")
PULLED_RE = re.compile(r"Successfully pulled image .* in ((?:[0-9]+h)?(?:[0-9]+m)?[0-9.]+m?s)")
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)$")
# Terminal escape sequences and other control characters, which logs and events must not inject.
CONTROL_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]|[\x00-\x1f\x7f-\x9f]")


class Milestone(NamedTuple):
    ts: dt.datetime
    key: str
    tag: str
    text: str
    level: str = ""


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


def pull_text(message):
    m = PULLED_RE.search(message)
    if m:
        return m.group(1)
    if "already present" in message:
        return "already on the node"
    return None


def log_milestones(log_text, attempt=0, engine=""):
    """Returns (milestones, values) for one container attempt of `kubectl logs --timestamps`.

    A restored pod's log starts at the checkpoint line, so a log without cold-start markers before
    that line belongs to a pod that was restored from the snapshot.
    """
    milestones = []
    values = {"graphs": [], "engine": engine}
    seen = set()
    hooks = 0
    cold = False
    suffix = f"#{attempt}" if attempt else ""

    def prob_tag():
        return VLLM if values.get("engine") == "vllm" else SGLANG

    def add(ts, key, tag, text, level=""):
        if key not in seen:
            seen.add(key)
            milestones.append(Milestone(ts, key + suffix, tag, text, level))

    for raw in log_text.split("\n"):
        stamp, _, text = raw.partition(" ")
        ts = parse_ts(stamp)
        if ts is None or '"GET /' in text:
            continue
        tag = P5 if cold else RESTORE

        # --- SGLang Phase 1 & 2 milestones ---
        if m := HOOK_RE.search(text):
            values["engine"] = "sglang"
            hooks += 1
            cold = True
            secs = float(m.group(1))
            if hooks == 1:
                values.update(hook_s=secs, hook_ts=ts)
                add(ts, "hook", P1_SGLANG, f"SGLang loaded and hooked in {secs:.2f}s")
            else:
                values.update(workers_s=max(values.get("workers_s", 0.0), secs), workers_ts=ts)
                add(ts, "workers", P2_SGLANG, f"SGLang subprocesses loaded and hooked in {secs:.2f}s")
        elif HOOKING_RE.search(text):
            values["engine"] = "sglang"
            values.setdefault("launcher_ts", ts)

        # --- vLLM Phase 1 & 2 milestones ---
        elif VLLM_HOOKING_RE.search(text):
            values["engine"] = "vllm"
            values.setdefault("launcher_ts", ts)
        elif m := VLLM_HOOK_RE.search(text):
            values["engine"] = "vllm"
            hooks += 1
            cold = True
            secs = float(m.group(2))
            if hooks == 1:
                values.setdefault("launcher_ts", ts)
                values.update(hook_s=secs, hook_ts=ts)
                add(ts, "hook", P1_VLLM, f"vLLM API server loaded and hooked ({m.group(1)}) in {secs:.2f}s")
            else:
                values.update(workers_s=max(values.get("workers_s", 0.0), secs), workers_ts=ts)
                add(ts, "workers", P2_VLLM, f"vLLM subprocess loaded and hooked in {secs:.2f}s")
        elif m := VLLM_APISERVER_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values.setdefault("hook_ts", ts)
            add(ts, "hook", P1_VLLM, f"vLLM API server started (v{m.group(1)})")
        elif VLLM_PATCHED_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values["patched_ts"] = ts
            if "workers_ts" not in values and "weights_begin_ts" not in values:
                values.setdefault("hook_ts", ts)
            add(ts, "patched", P1_VLLM, "Patched vLLM FastAPI lifespan context for GKE snapshotting")
        elif m := VLLM_ENGINECORE_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values.setdefault("workers_ts", ts)
            ver = f" (v{m.group(1)})" if m.group(1) else ""
            add(ts, "workers", P2_VLLM, f"Initializing vLLM EngineCore{ver}")

        # --- Phase 3 (Model weights) ---
        elif DOWNLOAD_RE.search(text):
            values["download"] = True
            add(ts, "download", P3, "No local copy: downloading the weights from Hugging Face")
        elif m := WEIGHTS_BEGIN_RE.search(text):
            values["engine"] = "sglang"
            cold = True
            values["weights_begin_ts"] = ts
            add(ts, "weights_begin", P3, f"Loading the model weights ({m.group(1)} GB of GPU memory free)")
        elif VLLM_WEIGHTS_BEGIN_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values.setdefault("weights_begin_ts", ts)
            add(ts, "weights_begin", P3, "Loading the model weights")
        elif m := VLLM_SAFETENSORS_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            secs = float(m.group(1))
            values.update(safetensors_s=secs, safetensors_ts=ts)
            add(ts, "safetensors", P3, f"Read safetensors into GPU memory in {secs:.2f}s")
        elif m := WEIGHTS_END_RE.search(text):
            secs = float(m.group(1))
            values.update(weights_s=secs, weights_end_ts=ts)
            how = "Downloaded and loaded" if values.get("download") else "Loaded"
            add(ts, "weights_end", P3, f"{how} the model weights in {secs:.2f}s")
        elif m := VLLM_WEIGHTS_END_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            gib = float(m.group(1))
            secs = float(m.group(2))
            values.update(weights_gib=gib, weights_s=secs, weights_end_ts=ts)
            how = "Downloaded and loaded" if values.get("download") else "Loaded"
            add(ts, "weights_end", P3, f"{how} the model weights ({gib:.2f} GiB) in {secs:.2f}s")

        # --- Phase 4 (KV cache, CUDA graphs, warmup) ---
        elif m := KV_RE.search(text):
            values.update(kv_tokens=int(m.group(1)), kv_ts=ts)
            add(ts, "kv", P4, f"KV cache allocated ({m.group(1)} tokens)")
        elif m := VLLM_PROFILE_RE.search(text):
            values["engine"] = "vllm"
            values.update(profile_s=float(m.group(1)), profile_ts=ts)
        elif m := VLLM_KV_GIB_RE.search(text):
            values["engine"] = "vllm"
            values.update(kv_gib=float(m.group(1)), kv_ts=ts)
        elif m := VLLM_KV_TOKENS_RE.search(text):
            values["engine"] = "vllm"
            tok_str = m.group(1)
            values.update(kv_tokens=int(tok_str.replace(",", "")), kv_ts=ts)
            gib_note = f", {values['kv_gib']:.2f} GiB" if "kv_gib" in values else ""
            add(ts, "kv", P4, f"KV cache allocated ({tok_str} tokens{gib_note})")
        elif m := VLLM_COMPILE_RE.search(text):
            values["engine"] = "vllm"
            secs = float(m.group(1))
            values.update(compile_s=secs, compile_ts=ts)
            add(ts, "compile", P4, f"torch.compile finished in {secs:.2f}s")
        elif GRAPH_BEGIN_RE.search(text) or VLLM_GRAPH_BEGIN_RE.search(text):
            add(ts, "graph_begin", P4, "Capturing CUDA graphs")
        elif m := GRAPH_END_RE.search(text):
            kind = (m.group(1) or "").lower()
            secs = float(m.group(2))
            values["graphs"].append((kind, secs))
            values["graph_end_ts"] = ts
            name = f"{kind.capitalize()} CUDA graphs" if kind else "CUDA graphs"
            add(ts, f"graph_end:{kind or len(values['graphs'])}", P4, f"{name} captured in {secs:.2f}s")
        elif m := VLLM_GRAPH_END_RE.search(text):
            values["engine"] = "vllm"
            secs = float(m.group(1))
            gib = float(m.group(2))
            values["graphs"].append(("", secs))
            values["graph_end_ts"] = ts
            add(ts, "graph_end:vllm", P4, f"CUDA graphs captured in {secs:.2f}s ({gib:.2f} GiB)")
        elif m := KV_ALLOC_RE.search(text):
            values["kv_alloc_s"] = float(m.group(1))
        elif WARMUP_START_RE.search(text):
            add(ts, "warmup_start", P4, "Warming up the server")
        elif m := WARMUP_DONE_RE.search(text):
            secs = float(m.group(1))
            values.update(warmup_s=secs, warmup_done_ts=ts)
            if m.group(2):
                values["cold_start_s"] = float(m.group(2))
            add(ts, "warmup_done", P4, f"Warmup done in {secs:.2f}s")
        elif m := VLLM_WARMUP_DONE_RE.search(text):
            values["engine"] = "vllm"
            secs = float(m.group(1))
            values.update(warmup_s=secs, warmup_done_ts=ts)
            add(ts, "warmup_done", P4, f"Engine init and warmup done in {secs:.2f}s")

        # --- Phase 5 (Snapshot) & Restore ---
        elif HOLD_RE.search(text):
            cold = True
            values.setdefault("warmup_done_ts", ts)
            add(ts, "hold", P5, "/health returns 503 (Starting) until the snapshot is done")
        elif m := GC_RE.search(text):
            values.update(gc_s=float(m.group(1)), gc_ts=ts)
        elif m := IDLE_RE.search(text):
            secs = float(m.group(1))
            values.update(idle_s=secs, idle_ts=ts)
            add(ts, "idle", P5, f"Scheduler idle after {secs:.2f}s; releasing GPU memory to CPU RAM")
        elif FLUSH_RE.search(text):
            add(ts, "flush", P5, "Waiting for an idle scheduler (POST /flush_cache)")
        elif RELEASE_REQ_RE.search(text):
            values["engine"] = "sglang"
            cold = True
            values["release_req_ts"] = ts
            add(ts, "release_req", P5, "Releasing GPU memory (POST /release_memory_occupation)")
        elif RELEASE_HTTP_OK_RE.search(text):
            values["engine"] = "sglang"
            if "released_ts" not in values:
                secs = delta(values.get("release_req_ts"), ts)
                values["released_ts"] = ts
                if secs is not None:
                    values["released_s"] = secs
                    add(ts, "released", P5, f"GPU memory released in {secs:.2f}s; taking the gVisor checkpoint")
                else:
                    add(ts, "released", P5, "GPU memory released; taking the gVisor checkpoint")
        elif VLLM_SLEEP_REQ_RE.search(text):
            values["engine"] = "vllm"
            cold = True
            values["sleep_start_ts"] = ts
            add(ts, "sleep_req", P5, "Releasing GPU memory (engine.sleep(level=1)) to CPU RAM")
        elif m := VLLM_SLEEP_FREED_RE.search(text):
            values["engine"] = "vllm"
            if m.group(1) and m.group(2):
                values.update(sleep_freed_gib=float(m.group(1)), sleep_cpu_gib=float(m.group(2)))
            elif m.group(3):
                values.update(sleep_freed_gib=float(m.group(3)))
        elif m := RELEASED_RE.search(text):
            secs = float(m.group(1))
            values.update(released_s=secs, released_ts=ts)
            add(ts, "released", P5, f"GPU memory released in {secs:.2f}s; taking the gVisor checkpoint")
        elif m := VLLM_SLEEP_DONE_RE.search(text):
            values["engine"] = "vllm"
            secs = float(m.group(1) or m.group(2))
            values.update(released_s=secs, released_ts=ts)
            cpu_note = f" ({values['sleep_cpu_gib']:.2f} GiB backed up in CPU RAM)" if "sleep_cpu_gib" in values else ""
            add(ts, "released", P5, f"GPU memory released{cpu_note} in {secs:.2f}s; taking the gVisor checkpoint")
        elif VLLM_TRIGGER_RE.search(text):
            values.setdefault("trigger_ts", ts)
            if "released_ts" not in values:
                start_t = values.get("sleep_start_ts") or values.get("release_req_ts")
                secs = delta(start_t, ts)
                values["released_ts"] = ts
                if secs is not None:
                    values["released_s"] = secs
        elif m := PURGED_RE.search(text):
            values.update(purged_s=float(m.group(1)), purged_ts=ts)
        elif m := CHECKPOINT_RE.search(text):
            secs = float(m.group(1))
            if cold:
                values.update(checkpoint_s=secs, checkpoint_ts=ts)
                add(ts, "checkpoint", P5, f"Checkpoint done in {secs:.2f}s (includes the upload to GCS)")
            else:
                values.update(restored=True, restore_ts=ts)
                add(ts, "checkpoint", RESTORE, "Restored from the snapshot; resuming GPU memory")
        elif CHECKPOINT_OK_RE.search(text):
            if cold:
                if "checkpoint_ts" not in values:
                    start_t = values.get("trigger_ts") or values.get("released_ts")
                    secs = delta(start_t, ts)
                    values["checkpoint_ts"] = ts
                    if secs is not None:
                        values["checkpoint_s"] = secs
                        add(ts, "checkpoint", P5, f"Checkpoint done in {secs:.2f}s (includes the upload to GCS)")
                add(ts, "checkpoint_ok", tag, "Snapshot checkpoint done")
            else:
                if "restore_ts" not in values:
                    values.update(restored=True, restore_ts=ts)
                add(ts, "checkpoint", RESTORE, "Restored from the snapshot; resuming GPU memory")
        elif VLLM_RESTORED_RE.search(text):
            values["engine"] = "vllm"
            if not cold:
                if "restore_ts" not in values:
                    values.update(restored=True, restore_ts=ts)
                add(ts, "checkpoint", RESTORE, "Restored from the snapshot; resuming GPU memory")
        elif VLLM_WAKE_REQ_RE.search(text):
            values["engine"] = "vllm"
            values["wake_req_ts"] = ts
            add(ts, "wake_req", tag, "Resuming GPU memory (engine.wake_up())")
        elif RESUME_REQ_RE.search(text):
            values["engine"] = "sglang"
            values["resume_req_ts"] = ts
            add(ts, "resume_req", tag, "Resuming GPU memory (POST /resume_memory_occupation)")
        elif RESUME_HTTP_OK_RE.search(text):
            values["engine"] = "sglang"
            if "resumed_ts" not in values:
                secs = delta(values.get("resume_req_ts"), ts)
                values["resumed_ts"] = ts
                if secs is not None:
                    values["resumed_s"] = secs
                    add(ts, "resumed", tag, f"GPU memory resumed in {secs:.2f}s")
        elif m := RESUMED_RE.search(text):
            secs = float(m.group(1))
            values.update(resumed_s=secs, resumed_ts=ts)
            add(ts, "resumed", tag, f"GPU memory resumed in {secs:.2f}s")
        elif m := VLLM_WAKE_DONE_RE.search(text):
            values["engine"] = "vllm"
            secs = float(m.group(1) or m.group(2))
            total_wake = values.get("resumed_s", 0.0) + secs
            values.update(resumed_s=total_wake, resumed_ts=ts)
            add(ts, "resumed", tag, f"GPU memory resumed (engine.wake_up()) in {total_wake:.2f}s")
        elif RESUME_OK_RE.search(text):
            values["engine"] = "sglang"
            if "resumed_ts" not in values:
                secs = delta(values.get("resume_req_ts"), ts)
                values["resumed_ts"] = ts
                if secs is not None:
                    values["resumed_s"] = secs
                    add(ts, "resumed", tag, f"GPU memory resumed in {secs:.2f}s")
                else:
                    add(ts, "resume_ok", tag, "GPU memory resumed")
            else:
                add(ts, "resume_ok", tag, "GPU memory resumed")
        elif m := UP_RE.search(text):
            values.update(wake_to_up_s=float(m.group(1)), up_ts=ts)
            add(ts, "up", tag, "Server is Up; waiting for the readiness probe")
        elif STATUS_RESTORED_RE.search(text):
            values.setdefault("up_ts", ts)
            add(ts, "up", tag, "Server status restored; waiting for the readiness probe")
        elif VLLM_APP_UP_RE.search(text):
            values["engine"] = "vllm"
            if "resumed_ts" not in values and "wake_req_ts" in values:
                secs = delta(values.get("wake_req_ts"), ts)
                values["resumed_ts"] = ts
                if secs is not None:
                    values["resumed_s"] = secs
                    add(ts, "resumed", tag, f"GPU memory resumed (engine.wake_up()) in {secs:.2f}s")
            values.setdefault("up_ts", ts)
            add(ts, "up", tag, "Application startup complete; waiting for the readiness probe")
        elif m := ENTER_RE.search(text):
            values["enter_s"] = float(m.group(1))
        elif m := WEIGHTS_READY_RE.search(text):
            values["weights_ready_s"] = float(m.group(1))
        elif m := LAUNCH_CB_RE.search(text):
            values["launch_cb_s"] = float(m.group(1))
        else:
            for regex, level in PROBLEM_RES:
                if regex.search(text):
                    add(ts, "problem:" + regex.pattern, prob_tag(), clean(text), level)
                    break
            else:
                if EXCEPTION_RE.match(text.strip()):
                    add(ts, "exception", prob_tag(), clean(text), "ERROR")
    return milestones, values


def pod_times(kubectl, pod, events, engine=""):
    meta = pod.get("metadata", {})
    name = meta.get("name", "")
    eng = infer_engine(pod) or engine
    conds = conditions(pod)
    status = container_status(pod)
    running = (status.get("state") or {}).get("running") or {}
    times = {
        "name": name,
        "engine": eng,
        "created": parse_ts(meta.get("creationTimestamp")),
        "scheduled": conds.get("PodScheduled"),
        "sandbox": conds.get("PodReadyToStartContainers"),
        "restored": conds.get("PodRestored"),
        "ready": conds.get("Ready"),
        "started": parse_ts(running.get("startedAt")),
        "restarts": int(status.get("restartCount") or 0),
        "runtime": clean(pod.get("spec", {}).get("runtimeClassName") or "", 40),
        "streaming": "ImageStreaming" in conds,
        "pull": None,
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
            times["pull"] = pull_text(event.get("message") or "")
    log = kubectl.run("logs", name, "-c", container_name(pod), "--timestamps") or ""
    times["log"] = log_milestones(log, engine=eng)[1]
    if not times["engine"] and times["log"].get("engine"):
        times["engine"] = times["log"]["engine"]
    return times


def cmd_report(args):
    eng = str(args.engine or infer_engine(selector=args.selector) or "")
    kubectl = Kubectl(args.namespace)
    pods = kubectl.json("get", "pods", "-l", args.selector)
    if pods is None:
        print(f"ERROR: could not list the pods with kubectl get pods -l {args.selector}.", file=sys.stderr)
        return 1
    items = sorted(
        (p for p in pods.get("items", []) if not p.get("metadata", {}).get("deletionTimestamp")),
        key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""),
    )
    events = (kubectl.json("get", "events", "--field-selector", "involvedObject.kind=Pod") or {}).get("items", [])
    snaps = (kubectl.json("get", "podsnapshots") or {}).get("items", [])
    p1 = pod_times(kubectl, items[0], events, engine=eng) if items else None
    p2 = pod_times(kubectl, items[-1], events, engine=eng) if len(items) >= 2 else None

    snap_ts = snapshot_ready(snaps, pod=p1)

    if not eng:
        active_pod = p1 or p2
        eng = str(active_pod.get("engine", "")) if isinstance(active_pod, dict) else ""
    csv_path = getattr(args, "csv", "") or os.environ.get("CSV_FILE", "")
    print_report(p1, p2, snap_ts, engine=eng, csv_path=csv_path)
    return 0


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


# Layout of the timing table: a label column and two value columns.
LABEL_W = 68
COL_W = 12
WIDTH = 2 + LABEL_W + 2 * COL_W
LETTERS = "abcdefghijklmnopqrstuvwxyz"


class Row(NamedTuple):
    """One line of a pod's timing table: a step with its sub-steps, or a sub-step."""

    label: str
    seconds: float
    end: dt.datetime
    precise: bool
    subs: tuple = ()


def is_restored(pod):
    return bool(pod["restored"] or pod["log"].get("restored"))


def kind_text(pod):
    if is_restored(pod):
        return "restore from the snapshot"
    if pod["log"].get("released_ts") or pod["log"].get("checkpoint_ts"):
        return "cold start + take the snapshot"
    return "cold start"


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


def build_csv_rows(
    p1,
    p2,
    snap_ready=None,
    orig_rows=None,
    totals=None,
    speedup=None,
    engine="",
):
    """Returns a list of (ModelServer, Mode, MetricName, MetricValue) tuples combining:
    1. The Unified Metric Hierarchy (1..4f)
    2. The 4 Composite Metrics Used for Alerting
    3. All metrics from the original report (totals, PodSnapshot Ready, speedups, original
       engine-specific breakdown rows, CUDA graph / safetensors details).
    """
    active_pod = p1 or p2
    eng = engine or (active_pod.get("engine") or active_pod["log"].get("engine") if active_pod else "")
    model_server = engine_display(eng)
    out = []
    seen = set()

    def add_row(mode, name, val):
        if val is None:
            return
        key = (mode, name)
        if key not in seen:
            seen.add(key)
            out.append((model_server, mode, name, float(val)))

    cold_m = compute_cold_metrics(p1) if (p1 and p1.get("created") and not is_restored(p1)) else None
    rest_m = compute_restore_metrics(p2) if (p2 and p2.get("created") and is_restored(p2)) else None

    # 1. Unified Metric Hierarchy (1..4f)
    if cold_m is not None:
        cold_items = [
            ("1. Kubernetes: schedule the pod and start the container", cold_m["1"]),
            ("1a. Wait for a GPU node", cold_m["1a"]),
            ("1b. Set up the pod sandbox (gvisor)", cold_m["1b"]),
            ("1c. Pull the image (image streaming)", cold_m["1c"]),
            ("1d. Start the container", cold_m["1d"]),
            ("2. LLM Engine cold start (vLLM / SGLang)", cold_m["2"]),
            ("2a. Start Python", cold_m["2a"]),
            ("2b. Import engine and dependencies (vLLM / SGLang, PyTorch)", cold_m["2b"]),
            ("2c. Initialize LLM engine and background workers/processes", cold_m["2c"]),
            ("2d. Download and load model weights", cold_m["2d"]),
            ("2e. Allocate KV cache memory", cold_m["2e"]),
            ("2f. Capture CUDA graphs / compile execution graph", cold_m["2f"]),
            ("2g. Start API server / frontend and run warm-up requests", cold_m["2g"]),
            ("3. Wait for the readiness probe", cold_m["3"]),
            ("4. Take the snapshot (phase 5)", cold_m["4"]),
            ("4a. Freeze Python garbage collection", cold_m["4a"]),
            ("4b. Wait for an idle scheduler", cold_m["4b"]),
            ("4c. Release GPU memory (weights + KV cache) to CPU RAM", cold_m["4c"]),
            ("4d. Purge the local weight cache", cold_m["4d"]),
            ("4e. gVisor checkpoint + upload to GCS", cold_m["4e"]),
            ("4f. Resume GPU memory (weights + KV cache)", cold_m["4f"]),
        ]
        for name, val in cold_items:
            add_row("ColdStart", name, val)

    if rest_m is not None:
        rest_items = [
            ("1. Kubernetes: schedule the pod and start the container", rest_m["1"]),
            ("1a. Wait for a GPU node", rest_m["1a"]),
            ("1b. Set up the pod sandbox (gvisor)", rest_m["1b"]),
            ("1c. Pull the image (image streaming)", rest_m["1c"]),
            ("1d. Start the container", rest_m["1d"]),
            ("3. Wait for the readiness probe", rest_m["3"]),
            ("4e. gVisor checkpoint + upload to GCS", rest_m["4e"]),
            ("4f. Resume GPU memory (weights + KV cache)", rest_m["4f"]),
        ]
        for name, val in rest_items:
            add_row("Restore", name, val)

    # 2. Composite Metrics Used for Alerting
    if cold_m is not None:
        add_row(
            "ColdStart",
            "E2E Cold Start (including node provisioning)",
            cold_m["1"] + cold_m["2"] + cold_m["3"],
        )
    if rest_m is not None:
        add_row(
            "Restore",
            "Warm Start (from restoration)",
            rest_m["1b"] + rest_m["1c"] + rest_m["4e"] + rest_m["4f"] + rest_m["3"],
        )
    if cold_m is not None:
        add_row(
            "ColdStart",
            "Cold Start (excluding node provisioning)",
            cold_m["1b"] + cold_m["1c"] + cold_m["1d"] + cold_m["2"],
        )
        add_row("ColdStart", "Application Cold Start", cold_m["2"])
        add_row("ColdStart", "Total Snapshot Overhead Time", cold_m["4"])

    # 3. Additional metrics from the original report not already covered by 1..4f or Composites:
    #    - Total: pod created -> Ready (ColdStart includes taking the snapshot: 1+2+3+4)
    #    - PodSnapshot Ready (pod created -> PodSnapshot Ready)
    #    - Speedup ratios
    totals = totals or {}
    orig_rows = orig_rows or {}
    t1, t2 = totals.get(1), totals.get(2)
    snap_row = next((r for r in orig_rows.get(1, []) if r.label.startswith("Take the snapshot")), None)
    if t1 is not None:
        add_row("ColdStart", "Total: pod created -> Ready", t1)
    if snap_ready and p1 and p1.get("created") and snap_ready >= p1["created"]:
        add_row("ColdStart", "PodSnapshot Ready", delta(p1["created"], snap_ready))
    if speedup is not None and t1 and t2:
        add_row("Restore", "Speedup vs Pod 1 (cold start + snapshot)", speedup)
        if snap_row and t1 > snap_row.seconds:
            add_row("Restore", "Speedup vs Pod 1 (without snapshot)", (t1 - snap_row.seconds) / t2)

    return out, cold_m, rest_m


FINE_GRAINED_METRICS = frozenset({"TTFT", "ITL / TPOT"})


def _fmt_csv_val(name, val):
    return f"{val:.4f}" if name in FINE_GRAINED_METRICS else f"{val:.2f}"


def format_r_dataframe(csv_text):
    """Formats CSV text into R syntax creating a `data` vector and `df` data.frame."""
    lines = [line for line in csv_text.splitlines() if line]
    escaped = [line.replace("\\", "\\\\").replace('"', '\\"') for line in lines]
    body = ",\n".join(f'  "{line}"' for line in escaped)
    return f"data <- c(\n{body}\n)\n\ndf <- read.csv(text = data, stringsAsFactors = FALSE)\n"


def _r_path_for(csv_path):
    r_env = os.environ.get("R_FILE", "").strip()
    if r_env:
        return pathlib.Path(r_env)
    return pathlib.Path(csv_path).with_suffix(".R")


def write_and_format_csv(engine_rows, csv_path=""):
    """Appends engine_rows into csv_path (if given) and its companion .R file with Seq = iteration number."""
    existing_rows = []
    if csv_path:
        p = pathlib.Path(csv_path)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ms = row.get("ModelServer", "")
                        if ms:
                            existing_rows.append((
                                int(row.get("Seq", 1)),
                                ms,
                                row.get("Mode", ""),
                                row.get("MetricName", ""),
                                float(row.get("MetricValue", 0.0)),
                            ))
            except (OSError, ValueError):
                pass

    current_ms = engine_rows[0][0] if engine_rows else ""
    env_seq = os.environ.get("SEQ", "").strip()
    if env_seq.isdigit() and int(env_seq) > 0:
        next_seq = int(env_seq)
        existing_rows = [r for r in existing_rows if not (r[0] == next_seq and r[1] == current_ms)]
    else:
        next_seq = max((r[0] for r in existing_rows if r[1] == current_ms), default=0) + 1
    new_sequenced = [(next_seq, ms, mode, name, val) for ms, mode, name, val in engine_rows]
    all_rows = existing_rows + new_sequenced

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Seq", "ModelServer", "Mode", "MetricName", "MetricValue"])
    for seq, ms, mode, name, val in all_rows:
        writer.writerow([seq, ms, mode, name, _fmt_csv_val(name, val)])
    csv_text = buf.getvalue()

    if csv_path and all_rows:
        try:
            pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(csv_path).write_text(csv_text, encoding="utf-8")
            r_path = _r_path_for(csv_path)
            r_path.parent.mkdir(parents=True, exist_ok=True)
            r_path.write_text(format_r_dataframe(csv_text), encoding="utf-8")
        except OSError as exc:
            print(f"WARNING: could not write CSV/R output to {csv_path}: {exc}", file=sys.stderr)

    eng_buf = io.StringIO()
    eng_writer = csv.writer(eng_buf, lineterminator="\n")
    eng_writer.writerow(["Seq", "ModelServer", "Mode", "MetricName", "MetricValue"])
    for seq, ms, mode, name, val in new_sequenced:
        eng_writer.writerow([seq, ms, mode, name, _fmt_csv_val(name, val)])
    return eng_buf.getvalue()


def pod_steps(pod):
    """The steps from pod creation to Ready, using the unified 1..4 metric hierarchy."""
    if is_restored(pod):
        m = compute_restore_metrics(pod)
        return [
            Row(
                "Kubernetes: schedule the pod and start the container",
                m["1"],
                m["t_1d"],
                False,
                (
                    Row("Wait for a GPU node", m["1a"], m["t_1a"], False),
                    Row("Set up the pod sandbox (gvisor)", m["1b"], m["t_1b"], False),
                    Row("Pull the image (image streaming)", m["1c"], m["t_1c"], False),
                    Row("Start the container", m["1d"], m["t_1d"], False),
                ),
            ),
            Row("Restore the snapshot (4e. gVisor checkpoint / GCS transfer)", m["4e"], m["t_4e"], True),
            Row("Resume GPU memory (weights + KV cache) (4f)", m["4f"], m["t_4f"], True),
            Row("Wait for the readiness probe (3)", m["3"], m["t_ready"], False),
        ]

    m = compute_cold_metrics(pod)
    return [
        Row(
            "Kubernetes: schedule the pod and start the container",
            m["1"],
            m["t_1d"],
            False,
            (
                Row("Wait for a GPU node", m["1a"], m["t_1a"], False),
                Row("Set up the pod sandbox (gvisor)", m["1b"], m["t_1b"], False),
                Row("Pull the image (image streaming)", m["1c"], m["t_1c"], False),
                Row("Start the container", m["1d"], m["t_1d"], False),
            ),
        ),
        Row(
            "LLM Engine cold start (vLLM / SGLang)",
            m["2"],
            m["t_2g"],
            True,
            (
                Row("Start Python", m["2a"], m["t_2a"], True),
                Row("Import engine and dependencies (vLLM / SGLang, PyTorch)", m["2b"], m["t_2b"], True),
                Row("Initialize LLM engine and background workers/processes", m["2c"], m["t_2c"], True),
                Row("Download and load model weights", m["2d"], m["t_2d"], True),
                Row("Allocate KV cache memory", m["2e"], m["t_2e"], True),
                Row("Capture CUDA graphs / compile execution graph", m["2f"], m["t_2f"], True),
                Row("Start API server / frontend and run warm-up requests", m["2g"], m["t_2g"], True),
            ),
        ),
        Row("Wait for the readiness probe", m["3"], m["t_ready"], False),
        Row(
            "Take the snapshot (phase 5)",
            m["4"],
            m["t_4f"],
            True,
            (
                Row("Freeze Python garbage collection", m["4a"], m["t_4a"], True),
                Row("Wait for an idle scheduler", m["4b"], m["t_4b"], True),
                Row("Release GPU memory (weights + KV cache) to CPU RAM", m["4c"], m["t_4c"], True),
                Row("Purge the local weight cache", m["4d"], m["t_4d"], True),
                Row("gVisor checkpoint + upload to GCS", m["4e"], m["t_4e"], True),
                Row("Resume GPU memory (weights + KV cache)", m["4f"], m["t_4f"], True),
            ),
        ),
    ]


def build_rows(start, steps):
    """Compatibility wrapper: if steps are already Row objects from pod_steps(), return them."""
    if steps and isinstance(steps[0], Row):
        return list(steps)
    rows = []
    prev, prev_precise = start, False
    for label, points in steps:
        present, pending = [], []
        for point in points:
            if point.ts is None:
                pending.append(point.label)
            else:
                present.append(point._replace(label=" + ".join(pending + [point.label])))
                pending = []
        step_start, step_precise = prev, prev_precise
        subs = []
        for point in sorted(present, key=lambda p: p.ts):
            if point.ts > prev:
                end, end_precise = point.ts, point.precise
            else:
                end, end_precise = prev, prev_precise
            subs.append(Row(point.label, (end - prev).total_seconds(), end, prev_precise and end_precise))
            prev, prev_precise = end, end_precise
        if subs and len(points) == 1:
            rows.append(subs[0])
        elif subs:
            rows.append(Row(label, (prev - step_start).total_seconds(), prev, step_precise and prev_precise,
                            tuple(subs)))
    return rows


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


def comparison(p1, p2, rows, totals):
    """Returns (lines, speedup) comparing the cold start of Pod 1 with the restore of Pod 2."""
    t1, t2 = totals.get(1), totals.get(2)
    if not (p1 and p2 and t1 and t2) or is_restored(p1) or not is_restored(p2):
        return [], None
    lines = [f"Pod 2 was Ready {t1 / t2:.1f}x faster than Pod 1."]
    snap = next((row for row in rows[1] if row.label.startswith("Take the snapshot")), None)
    if snap and t1 > snap.seconds:
        plain = t1 - snap.seconds
        lines += [
            f"Pod 1 spent {fmt_row(snap)} taking the snapshot, which only the first pod does. Without it,",
            f"Pod 1 would have been Ready in about {fmt_dur(plain, 0)}: the restore is {plain / t2:.1f}x faster than that.",
        ]
    w1 = delta(p1["created"], p1["scheduled"]) or 0.0
    w2 = delta(p2["created"], p2["scheduled"]) or 0.0
    if max(w1, w2) >= 5 and t1 > w1 and t2 > w2:
        lines.append(f"Not counting the wait for a GPU node (Pod 1 {fmt_dur(w1, 0)}, Pod 2 {fmt_dur(w2, 0)}), "
                     f"Pod 2 was Ready {(t1 - w1) / (t2 - w2):.1f}x faster.")
    return lines, t1 / t2


def print_pod(number, pod, rows, snap_ready, title_prefix=""):
    created = pod["created"]
    eng_disp = engine_display(pod.get("engine") or pod["log"].get("engine"))
    base = f"POD {number} ({eng_disp}): {kind_text(pod)}" if eng_disp != "Model Server" else f"POD {number}: {kind_text(pod)}"
    title = f"{title_prefix}{base}" if title_prefix else base
    table_header(title, "Duration", "Cumulative")
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
    rows = {number: build_rows(pod["created"], pod_steps(pod)) for number, pod in pods}
    totals = {number: delta(pod["created"], pod["ready"]) for number, pod in pods}
    lines, speedup = comparison(p1, p2, rows, totals)
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
    for text in lines:
        print(f"  {text}")
    if pods:
        print("")
        print("  The numbered steps add up to the total, and the lettered sub-steps add up to their step.")
        print("  Cumulative: the time since the pod was created, at the end of the step.")
    for number, pod in pods:
        print_pod(number, pod, rows[number], snap_ready if number == 1 else None)

    csv_rows, cold_m, rest_m = build_csv_rows(
        p1,
        p2,
        snap_ready=snap_ready,
        orig_rows=rows,
        totals=totals,
        speedup=speedup,
        engine=engine,
    )
    if cold_m is not None or rest_m is not None:
        table_header(f"COMPOSITE METRICS USED FOR ALERTING{hdr_suffix}", "Computation", "Seconds")
        print("-" * WIDTH)
        if cold_m is not None:
            e2e_cold = cold_m["1"] + cold_m["2"] + cold_m["3"]
            table_line("E2E Cold Start (including node provisioning)", "Rows 1+2+3", f"{e2e_cold:.2f}s")
        if rest_m is not None:
            warm_rest = rest_m["1b"] + rest_m["1c"] + rest_m["4e"] + rest_m["4f"] + rest_m["3"]
            table_line("Warm Start (from restoration)", "1b+1c+4e+4f+3", f"{warm_rest:.2f}s")
        if cold_m is not None:
            cold_excl = cold_m["1b"] + cold_m["1c"] + cold_m["1d"] + cold_m["2"]
            table_line("Cold Start (excluding node provisioning)", "1b-d+2", f"{cold_excl:.2f}s")
            table_line("Application Cold Start", "Row 2 (2a-g)", f"{cold_m['2']:.2f}s")
            table_line("Total Snapshot Overhead Time", "Row 4 (4a-f)", f"{cold_m['4']:.2f}s")

    if csv_rows:
        csv_text = write_and_format_csv(csv_rows, csv_path=csv_path)
        saved_note = f" (saved to {csv_path} and {_r_path_for(csv_path)})" if csv_path else ""
        table_header(f"CSV RESULTS{hdr_suffix}{saved_note}")
        print("-" * WIDTH)
        print(csv_text.rstrip("\n"))
    print("=" * WIDTH)
    if pods:
        recap = ", ".join(f"Pod {number} {fmt_total(totals[number])}" for number, _ in pods)
        more = f" (Pod 2 was Ready {speedup:.1f}x faster)" if speedup else ""
        print(f"  Pod created -> Ready{hdr_suffix}: {recap}{more}.")
        print("=" * WIDTH)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="print the timing table")
    report.add_argument("--namespace", required=True)
    report.add_argument("--selector", required=True)
    report.add_argument("--engine", default="", help="engine type: vllm or sglang")
    report.add_argument("--csv", default="", help="optional path to write CSV metrics")
    args = parser.parse_args(argv)
    try:
        return cmd_report(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
