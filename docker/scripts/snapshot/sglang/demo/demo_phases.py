#!/usr/bin/env python3
"""Startup phases and timing table for the SGLang Pod Snapshots demo (run_demo.sh).

Subcommands:
  watch   Runs `kubectl rollout status` on the model server Deployment. While it waits, prints each
          startup phase of the new pods as soon as it shows up in the pod status, the pod events or
          the pod log. Exits with the exit code of `kubectl rollout status`.
  report  Prints the timing table of the demo: the time from pod creation to Ready of each pod and
          the speedup, then for each pod (Pod 1 cold starts and takes the snapshot, Pod 2 restores
          from it) the steps from pod creation to Ready, with lettered sub-steps and the cumulative
          time. Then the wall-clock time of each run_demo.sh step and the test request latencies.

The phases of a cold start are:
  1. Container start -> SGLang hooked
  2. Subprocesses hooked
  3. Model weights
  4. KV cache, CUDA graph capture, warmup
  5. Snapshot (release, checkpoint, resume)

The helper only reads from the cluster: it runs kubectl get, kubectl logs and kubectl rollout status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import NamedTuple

POLL_S = 5
HEARTBEAT_S = 60
KUBECTL_TIMEOUT_S = 60

K8S = "Kubernetes"
P1 = "Phase 1/5 SGLang start"
P2 = "Phase 2/5 Subprocesses"
P3 = "Phase 3/5 Model weights"
P4 = "Phase 4/5 KV cache, CUDA graphs, warmup"
P5 = "Phase 5/5 Snapshot"
RESTORE = "Restore"
SGLANG = "SGLang"

# Milestones printed in bold pink, so that the demo audience notices them. A restored pod's
# "checkpoint" milestone ("Restored from the snapshot") is highlighted as well.
PINK = "\033[1;38;5;205m"
RESET = "\033[0m"
HIGHLIGHT_KEYS = {"pulling", "weights_begin", "kv", "graph_begin", "idle", "released"}

NUM = r"([0-9]+(?:\.[0-9]+)?)"
HOOK_RE = re.compile(r"Hooked SGLang _wait_and_warmup in " + NUM + "s")
HOOKING_RE = re.compile(r"Hooking SGLang HTTP server startup")
DOWNLOAD_RE = re.compile(r"has no files matching .*will attempt download")
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
RELEASED_RE = re.compile(r"Released GPU memory occupation .* in " + NUM + "s")
PURGED_RE = re.compile(r"Purged [0-9]+ cache entry/entries .* in " + NUM + "s")
CHECKPOINT_RE = re.compile(r"gVisor checkpoint completed successfully \(barrier unblocked in " + NUM + "s")
CHECKPOINT_OK_RE = re.compile(r"Snapshot checkpoint created successfully")
RESUMED_RE = re.compile(r"Resumed GPU memory occupation .* in " + NUM + "s")
RESUME_OK_RE = re.compile(r"SGLang memory occupation resumed successfully")
UP_RE = re.compile(r"server_status = ServerStatus\.Up \(wake-to-ready=" + NUM + r"s\)")
STATUS_RESTORED_RE = re.compile(r"Restored server_status to ")
ENTER_RE = re.compile(r"Entering patched _wait_and_warmup .*elapsed_since_start=" + NUM + "s")
WEIGHTS_READY_RE = re.compile(r"Model weights are ready in GPUs \(waited " + NUM + r"s\)")
LAUNCH_CB_RE = re.compile(r"launch_callback completed in " + NUM + "s")
PROBLEM_RES = (
    (re.compile(r"Server warmup failed"), "ERROR"),
    (re.compile(r"Snapshot checkpointing failed"), "ERROR"),
    (re.compile(r"Sleep/wake cycle failed"), "ERROR"),
    (re.compile(r"The SGLang scheduler did not become idle"), "ERROR"),
    (re.compile(r"SGLang server is not importable"), "ERROR"),
    (re.compile(r"Pod snapshot trigger not available"), "WARNING"),
)
# Python exceptions start at the beginning of the line; SGLang's own log lines start with "[date]".
EXCEPTION_RE = re.compile(r"^(?:[a-z_][A-Za-z0-9_]*\.)*[A-Z][A-Za-z0-9_]*(?:Error|Exception)(?::|$)")
PULLED_RE = re.compile(r"Successfully pulled image .* in ((?:[0-9]+h)?(?:[0-9]+m)?[0-9.]+m?s)")
PULLING_RE = re.compile(r'Pulling image "([^"]+)"')
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)$")
# Terminal escape sequences and other control characters, which logs and events must not inject.
CONTROL_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]|[\x00-\x1f\x7f-\x9f]")

# What a pod is busy with after its last milestone, for the "still ..." line during long waits.
ACTIVITY = {
    "created": "waiting for a node",
    "failed_scheduling": "waiting for a node",
    "scale_up": "waiting for the new node",
    "scheduled": "waiting for the image",
    "pulling": "pulling the image",
    "pulled": "starting the container",
    "started": "loading Python, CUDA and SGLang",
    "hook": "starting the SGLang subprocesses",
    "workers": "starting to load the model",
    "weights_begin": "loading the model weights",
    "download": "downloading the model weights from Hugging Face",
    "weights_end": "allocating the KV cache",
    "kv": "capturing CUDA graphs",
    "graph_begin": "capturing CUDA graphs",
    "graph_end": "capturing CUDA graphs",
    "warmup_start": "warming up the server",
    "warmup_done": "waiting for an idle scheduler",
    "hold": "waiting for an idle scheduler",
    "flush": "waiting for an idle scheduler",
    "idle": "releasing GPU memory",
    "release_req": "releasing GPU memory",
    "released": "taking the checkpoint and uploading it to GCS",
    "checkpoint": "resuming GPU memory",
    "checkpoint_ok": "resuming GPU memory",
    "gke": "resuming GPU memory",
    "resumed": "waiting for the readiness probe",
    "resume_ok": "waiting for the readiness probe",
    "up": "waiting for the readiness probe",
}


class Milestone(NamedTuple):
    ts: dt.datetime
    key: str
    tag: str
    text: str
    level: str = ""


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def use_color():
    """run_demo.sh sets DEMO_COLOR; otherwise color only a terminal, unless NO_COLOR is set."""
    setting = os.environ.get("DEMO_COLOR")
    if setting in ("0", "1"):
        return setting == "1"
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def highlighted(milestone):
    key = milestone.key.split("#")[0]
    return key in HIGHLIGHT_KEYS or (key == "checkpoint" and milestone.tag == RESTORE)


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


def fmt_elapsed(seconds):
    total = int(max(0.0, seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"+{hours}:{minutes:02d}:{secs:02d}" if hours else f"+{minutes:02d}:{secs:02d}"


def delta(start, end):
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


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


def log_milestones(log_text, attempt=0):
    """Returns (milestones, values) for one container attempt of `kubectl logs --timestamps`.

    A restored pod's log starts at the checkpoint line, so a log without cold-start markers before
    that line belongs to a pod that was restored from the snapshot.
    """
    milestones = []
    values = {"graphs": []}
    seen = set()
    hooks = 0
    cold = False
    suffix = f"#{attempt}" if attempt else ""

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
        if m := HOOK_RE.search(text):
            hooks += 1
            cold = True
            secs = float(m.group(1))
            if hooks == 1:
                values.update(hook_s=secs, hook_ts=ts)
                add(ts, "hook", P1, f"SGLang loaded and hooked in {secs:.2f}s")
            else:
                values.update(workers_s=max(values.get("workers_s", 0.0), secs), workers_ts=ts)
                add(ts, "workers", P2, f"SGLang subprocesses loaded and hooked in {secs:.2f}s")
        elif HOOKING_RE.search(text):
            # The launcher's first line: Python is running and starts to import SGLang.
            values.setdefault("launcher_ts", ts)
        elif DOWNLOAD_RE.search(text):
            values["download"] = True
            add(ts, "download", P3, "No local copy: downloading the weights from Hugging Face")
        elif m := WEIGHTS_BEGIN_RE.search(text):
            cold = True
            values["weights_begin_ts"] = ts
            add(ts, "weights_begin", P3, f"Loading the model weights ({m.group(1)} GB of GPU memory free)")
        elif m := WEIGHTS_END_RE.search(text):
            secs = float(m.group(1))
            values.update(weights_s=secs, weights_end_ts=ts)
            how = "Downloaded and loaded" if values.get("download") else "Loaded"
            add(ts, "weights_end", P3, f"{how} the model weights in {secs:.2f}s")
        elif m := KV_RE.search(text):
            values.update(kv_tokens=int(m.group(1)), kv_ts=ts)
            add(ts, "kv", P4, f"KV cache allocated ({m.group(1)} tokens)")
        elif GRAPH_BEGIN_RE.search(text):
            add(ts, "graph_begin", P4, "Capturing CUDA graphs")
        elif m := GRAPH_END_RE.search(text):
            kind = (m.group(1) or "").lower()
            secs = float(m.group(2))
            values["graphs"].append((kind, secs))
            values["graph_end_ts"] = ts
            name = f"{kind.capitalize()} CUDA graphs" if kind else "CUDA graphs"
            add(ts, f"graph_end:{kind or len(values['graphs'])}", P4, f"{name} captured in {secs:.2f}s")
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
        elif HOLD_RE.search(text):
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
            add(ts, "release_req", P5, "Releasing GPU memory (POST /release_memory_occupation)")
        elif m := RELEASED_RE.search(text):
            secs = float(m.group(1))
            values.update(released_s=secs, released_ts=ts)
            add(ts, "released", P5, f"GPU memory released in {secs:.2f}s; taking the gVisor checkpoint")
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
            add(ts, "checkpoint_ok", tag, "Snapshot checkpoint done (or restored from it)")
        elif m := RESUMED_RE.search(text):
            secs = float(m.group(1))
            values.update(resumed_s=secs, resumed_ts=ts)
            add(ts, "resumed", tag, f"GPU memory resumed in {secs:.2f}s")
        elif RESUME_OK_RE.search(text):
            add(ts, "resume_ok", tag, "GPU memory resumed")
        elif m := UP_RE.search(text):
            values.update(wake_to_up_s=float(m.group(1)), up_ts=ts)
            add(ts, "up", tag, "Server is Up; waiting for the readiness probe")
        elif STATUS_RESTORED_RE.search(text):
            values.setdefault("up_ts", ts)
            add(ts, "up", tag, "Server status restored; waiting for the readiness probe")
        elif m := ENTER_RE.search(text):
            values["enter_s"] = float(m.group(1))
        elif m := WEIGHTS_READY_RE.search(text):
            values["weights_ready_s"] = float(m.group(1))
        elif m := LAUNCH_CB_RE.search(text):
            values["launch_cb_s"] = float(m.group(1))
        else:
            for regex, level in PROBLEM_RES:
                if regex.search(text):
                    add(ts, "problem:" + regex.pattern, SGLANG, clean(text), level)
                    break
            else:
                if EXCEPTION_RE.match(text.strip()):
                    add(ts, "exception", SGLANG, clean(text), "ERROR")
    return milestones, values


def event_milestones(events):
    """Milestones from the events of one pod. Probe failures are left out: they are expected."""
    milestones = []
    seen = set()

    def add(ts, key, tag, text, level=""):
        if key not in seen:
            seen.add(key)
            milestones.append(Milestone(ts, key, tag, text, level))

    for event in sorted(events, key=event_ts):
        reason = event.get("reason") or ""
        message = event.get("message") or ""
        ts = event_ts(event)
        if reason == "FailedScheduling":
            add(ts, "failed_scheduling", K8S, f"Waiting for a node: {clean(message, 160)}")
        elif reason == "TriggeredScaleUp":
            add(ts, "scale_up", K8S, "The cluster autoscaler is adding a node")
        elif reason == "NotTriggerScaleUp":
            add(ts, "no_scale_up", K8S, f"The cluster autoscaler did not add a node: {clean(message, 160)}", "WARNING")
        elif reason == "Pulling":
            m = PULLING_RE.search(message)
            add(ts, "pulling", K8S, f"Pulling image {clean(m.group(1))}" if m else "Pulling the image")
        elif reason == "Pulled":
            pulled = pull_text(message)
            add(ts, "pulled", K8S, f"Image pulled in {pulled}" if pulled and pulled[0].isdigit()
                else "Image already on the node" if pulled else "Image pulled")
        elif reason == "GKEPodSnapshotting":
            restore = "restore" in message.lower()
            level = "WARNING" if "fail" in message.lower() else ""
            add(ts, "gke:" + ("restore" if restore else "checkpoint"), RESTORE if restore else P5,
                f"GKE: {clean(message)}", level)
        elif reason in ("Failed", "BackOff", "FailedMount", "FailedCreatePodSandBox", "Evicted",
                        "Preempting", "Preempted", "OOMKilling", "Killing"):
            add(ts, "event:" + reason, K8S, f"{reason}: {clean(message)}", "WARNING")
    return milestones


class Watcher:
    """Prints the milestones of the pods that start while `kubectl rollout status` waits."""

    def __init__(self, kubectl, selector, label):
        self.k = kubectl
        self.selector = selector
        self.label = label
        self.first_poll = True
        self.skip = set()
        self.labels = {}
        self.created = {}
        self.printed = set()
        self.finished = set()
        self.restarts = {}
        self.previous = {}
        self.last_key = {}
        self.last_tag = {}
        self.last_output = time.monotonic()
        self.color = use_color()

    def poll(self):
        pods = self.k.json("get", "pods", "-l", self.selector)
        if pods is None:
            raise RuntimeError(f"kubectl get pods -l {self.selector} failed")
        items = sorted(pods.get("items", []), key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""))
        if self.first_poll:
            self.first_poll = False
            self.skip = {
                p["metadata"]["name"]
                for p in items
                if "Ready" in conditions(p) or p["metadata"].get("deletionTimestamp")
            }
        watched = [p for p in items if p["metadata"]["name"] not in self.skip]
        if not watched:
            return
        events = self.k.json("get", "events", "--field-selector", "involvedObject.kind=Pod") or {}
        by_pod = {}
        for event in events.get("items", []):
            by_pod.setdefault(event.get("involvedObject", {}).get("name"), []).append(event)
        new = []
        for pod in watched:
            name = pod["metadata"]["name"]
            if name not in self.labels:
                self.labels[name] = self.label if not self.labels else f"{self.label} ({name.rsplit('-', 1)[-1]})"
                self.created[name] = parse_ts(pod["metadata"].get("creationTimestamp"))
            for m in self.pod_milestones(pod, by_pod.get(name, [])):
                if (name, m.key) not in self.printed:
                    self.printed.add((name, m.key))
                    new.append((name, m))
        new.sort(key=lambda item: item[1].ts)
        for name, m in new:
            self.emit(name, m)
            if m.key == "ready":
                self.finished.add(name)
        self.heartbeat()

    def pod_milestones(self, pod, events):
        meta = pod["metadata"]
        name = meta["name"]
        conds = conditions(pod)
        created = parse_ts(meta.get("creationTimestamp")) or now()
        ms = [Milestone(created, "created", K8S, "Pod created")]
        if "PodScheduled" in conds:
            node = clean(pod.get("spec", {}).get("nodeName") or "?")
            ms.append(Milestone(conds["PodScheduled"], "scheduled", K8S, f"Scheduled on node {node}"))
        ms += event_milestones(events)
        cs = container_status(pod)
        state = cs.get("state") or {}
        restarts = int(cs.get("restartCount") or 0)
        running = state.get("running") or {}
        if running.get("startedAt"):
            key = f"started#{restarts}" if restarts else "started"
            text = f"Container restarted (restart {restarts})" if restarts else "Container started"
            ms.append(Milestone(parse_ts(running["startedAt"]) or now(), key, K8S, text))
        waiting = state.get("waiting") or {}
        reason = waiting.get("reason") or ""
        if reason and reason not in ("ContainerCreating", "PodInitializing"):
            text = f"{reason}: {clean(waiting.get('message') or '')}".rstrip(": ")
            ms.append(Milestone(now(), f"waiting:{reason}#{restarts}", K8S, text, "WARNING"))
        if restarts > self.restarts.get(name, 0):
            self.restarts[name] = restarts
            term = (cs.get("lastState") or {}).get("terminated") or {}
            ms.append(Milestone(
                parse_ts(term.get("finishedAt")) or now(), f"exited#{restarts}", K8S,
                f"The container exited (exit code {term.get('exitCode', '?')}, "
                f"{clean(term.get('reason') or 'no reason')}); see: kubectl logs --previous -n "
                f"{self.k.namespace} {name}", "WARNING"))
            previous = self.k.run("logs", name, "-c", container_name(pod), "--previous", "--timestamps")
            self.previous[name] = log_milestones(previous or "", attempt=restarts - 1)[0]
        ms += self.previous.get(name, [])
        if name not in self.finished and (running or state.get("terminated")):
            text = self.k.run("logs", name, "-c", container_name(pod), "--timestamps")
            if text:
                ms += log_milestones(text, attempt=restarts)[0]
        if "Ready" in conds:
            took = fmt_dur(delta(conds.get("PodScheduled") or created, conds["Ready"]), 0)
            ms.append(Milestone(conds["Ready"], "ready", K8S, f"Ready: PodScheduled -> Ready in {took}"))
        if meta.get("deletionTimestamp"):
            ms.append(Milestone(now(), "deleting", K8S, "The pod is being deleted", "WARNING"))
        return ms

    def emit(self, name, m, ts=None):
        ts = ts or m.ts
        created = self.created.get(name)
        elapsed = fmt_elapsed(delta(created, ts)) if created else "+--:--"
        level = f"{m.level}: " if m.level else ""
        local = ts.astimezone().strftime("%H:%M:%S")
        line = f"  {local}  {self.labels[name]}  {elapsed}  [{m.tag}] {level}{m.text}"
        if self.color and highlighted(m):
            line = f"{PINK}{line}{RESET}"
        print(line, flush=True)
        self.last_output = time.monotonic()
        if not m.key.startswith("heartbeat"):
            self.last_key[name] = m.key.split("#")[0].split(":")[0]
            self.last_tag[name] = m.tag

    def heartbeat(self):
        if time.monotonic() - self.last_output < HEARTBEAT_S:
            return
        for name in self.labels:
            if name in self.finished:
                continue
            activity = ACTIVITY.get(self.last_key.get(name, "created"), "starting")
            self.emit(name, Milestone(now(), "heartbeat", self.last_tag.get(name, K8S), f"Still {activity}..."))


def cmd_watch(args):
    kubectl = Kubectl(args.namespace)
    watcher = Watcher(kubectl, args.selector, args.label)
    cmd = ["kubectl", "rollout", "status", f"deployment/{args.deployment}", "-n", args.namespace,
           f"--timeout={args.timeout}s"]
    warned = False
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            while True:
                done = proc.poll() is not None
                try:
                    watcher.poll()
                except Exception as exc:  # pylint: disable=broad-except
                    # The phase log is informational: never stop waiting for the rollout because of it.
                    if not warned:
                        warned = True
                        print(f"  WARNING: could not read the pod phases ({clean(exc)}); still waiting.", flush=True)
                if done:
                    break
                try:
                    proc.wait(timeout=POLL_S)
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 130
        out.seek(0)
        output = out.read().decode("utf-8", errors="replace")
    if proc.returncode != 0:
        sys.stderr.write(output)
        print(f"ERROR: kubectl rollout status exited with code {proc.returncode}.", file=sys.stderr)
        return proc.returncode
    if not watcher.labels:
        print(f"  No new pods to watch: the {args.deployment} pods were already Ready.", flush=True)
    return 0


def pod_times(kubectl, pod, events):
    meta = pod.get("metadata", {})
    name = meta.get("name", "")
    conds = conditions(pod)
    status = container_status(pod)
    running = (status.get("state") or {}).get("running") or {}
    times = {
        "name": name,
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
    times["log"] = log_milestones(log)[1]
    return times


def parse_pairs(pairs):
    result = []
    for pair in pairs or []:
        label, sep, value = pair.rpartition("=")
        try:
            result.append((label, float(value)))
        except ValueError:
            sep = ""
        if not sep or not label:
            raise SystemExit(f"ERROR: expected LABEL=SECONDS, got {pair!r}")
    return result


def cmd_report(args):
    steps = parse_pairs(args.step)
    latencies = parse_pairs(args.latency)
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
    p1 = pod_times(kubectl, items[0], events) if items else None
    p2 = pod_times(kubectl, items[-1], events) if len(items) >= 2 else None
    print_report(p1, p2, snapshot_ready(snaps), steps, latencies)
    return 0


def snapshot_ready(snaps):
    ready = []
    for snap in snaps:
        for c in snap.get("status", {}).get("conditions", []):
            if c.get("type") == "Ready" and c.get("status") == "True":
                ts = parse_ts(c.get("lastTransitionTime"))
                if ts:
                    ready.append(ts)
    return max(ready) if ready else None


# Layout of the timing table: a label column and two value columns.
LABEL_W = 68
COL_W = 12
WIDTH = 2 + LABEL_W + 2 * COL_W
LETTERS = "abcdefghijklmnopqrstuvwxyz"


class Point(NamedTuple):
    """The end of one sub-step of a pod's startup: what happened since the previous point, and when
    it ended. Kubernetes timestamps are whole seconds (precise=False); pod log timestamps are not."""

    label: str
    ts: dt.datetime | None
    precise: bool = True


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


def pod_steps(pod):
    """The steps from pod creation to Ready, as (label, points); each point ends one sub-step."""
    log = pod["log"]
    if pod["pull"] == "already on the node":
        pull = "The image is already on the node"
    else:
        pull = "Pull the image" + (" (image streaming)" if pod["streaming"] else "")
    start = "Start the container" + (f" (restarts: {pod['restarts']})" if pod["restarts"] else "")
    up = log.get("up_ts") or log.get("resumed_ts")
    steps = [("Kubernetes: schedule the pod and start the container", [
        Point("Wait for a GPU node", pod["scheduled"], False),
        Point("Set up the pod sandbox" + (f" ({pod['runtime']})" if pod["runtime"] else ""),
              pod["sandbox"] or pod["pulling_ts"], False),
        Point(pull, pod["pulled_ts"], False),
        Point(start, pod["started"], False),
    ])]
    if is_restored(pod):
        restored = log.get("restore_ts")
        steps += [
            ("Restore the snapshot", [Point("Restore the snapshot", restored or pod["restored"], bool(restored))]),
            ("Resume GPU memory (weights + KV cache)", [Point("Resume GPU memory (weights + KV cache)", up)]),
        ]
    else:
        graphs = ", ".join(f"{kind} {fmt_dur(secs, 1)}" for kind, secs in log["graphs"] if kind)
        weights = "Download and load" if log.get("download") else "Load"
        steps += [
            ("SGLang cold start (phases 1-4)", [
                Point("Start Python", log.get("launcher_ts")),
                Point("Import SGLang and PyTorch", log.get("hook_ts")),
                Point("Start the scheduler + detokenizer; they import SGLang again", log.get("workers_ts")),
                Point("Initialize the scheduler", log.get("weights_begin_ts")),
                Point(f"{weights} the model weights", log.get("weights_end_ts")),
                Point("Allocate the KV cache", log.get("kv_ts")),
                Point(f"Capture CUDA graphs ({graphs})" if graphs else "Capture CUDA graphs",
                      log.get("graph_end_ts")),
                Point("Start the HTTP server and warm up", log.get("warmup_done_ts")),
            ]),
            ("Take the snapshot (phase 5)", [
                Point("Freeze Python garbage collection", log.get("gc_ts")),
                Point("Wait for an idle scheduler", log.get("idle_ts")),
                Point("Release GPU memory (weights + KV cache) to CPU RAM", log.get("released_ts")),
                Point("Purge the local weight cache", log.get("purged_ts")),
                Point("gVisor checkpoint + upload to GCS", log.get("checkpoint_ts")),
                Point("Resume GPU memory (weights + KV cache)", up),
            ]),
        ]
    steps.append(("Wait for the readiness probe", [Point("Wait for the readiness probe", pod["ready"], False)]))
    return steps


def build_rows(start, steps):
    """Turns pod_steps() into rows. Each point ends the interval that began at the previous point,
    so the sub-steps add up to their step and the steps add up to the total. A point without a
    timestamp is merged into the next point of its step; a step without timestamps is left out."""
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
                # Whole-second Kubernetes timestamps can be a little earlier than log timestamps.
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


def print_pod(number, pod, rows, snap_ready):
    created = pod["created"]
    table_header(f"POD {number}: {kind_text(pod)}", "Duration", "Cumulative")
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


def print_report(p1, p2, snap_ready, steps, latencies):
    pods = [(number, pod) for number, pod in ((1, p1), (2, p2)) if pod and pod["created"]]
    rows = {number: build_rows(pod["created"], pod_steps(pod)) for number, pod in pods}
    totals = {number: delta(pod["created"], pod["ready"]) for number, pod in pods}
    lines, speedup = comparison(p1, p2, rows, totals)

    print("")
    print("=" * WIDTH)
    print("  TIMING SUMMARY: pod created -> Ready")
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

    if steps:
        table_header("RUN_DEMO.SH STEPS (wall clock; pauses between steps not counted)", "Duration", "Cumulative")
        print("-" * WIDTH)
        cumulative = 0.0
        for n, (label, secs) in enumerate(steps, 1):
            cumulative += secs
            table_line(f"{n}. {label}", fmt_dur(secs, 0), fmt_cum(cumulative))
        print("  " + "-" * (WIDTH - 2))
        table_line("Total", fmt_dur(cumulative, 0), fmt_cum(cumulative))
    if latencies:
        table_header("TEST REQUESTS (run_demo.sh verify)", "Latency")
        print("-" * WIDTH)
        for label, secs in latencies:
            text = clean(f"Completion request to {label}", WIDTH - COL_W - 2)
            print(f"  {text:<{WIDTH - COL_W - 2}}{fmt_dur(secs, 2):>{COL_W}}")
    print("=" * WIDTH)
    if pods:
        recap = ", ".join(f"Pod {number} {fmt_total(totals[number])}" for number, _ in pods)
        more = f" (Pod 2 was Ready {speedup:.1f}x faster)" if speedup else ""
        print(f"  Pod created -> Ready: {recap}{more}.")
        print("=" * WIDTH)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watch", help="wait for the rollout and print the startup phases of new pods")
    watch.add_argument("--namespace", required=True)
    watch.add_argument("--selector", required=True)
    watch.add_argument("--deployment", required=True)
    watch.add_argument("--timeout", type=int, required=True, help="rollout timeout in seconds")
    watch.add_argument("--label", required=True, help='name of the new pod in the output, e.g. "Pod 1"')
    report = sub.add_parser("report", help="print the timing table")
    report.add_argument("--namespace", required=True)
    report.add_argument("--selector", required=True)
    report.add_argument("--step", action="append", metavar="LABEL=SECONDS", help="wall-clock time of a step")
    report.add_argument("--latency", action="append", metavar="POD=SECONDS", help="test request latency")
    args = parser.parse_args(argv)
    try:
        return cmd_watch(args) if args.command == "watch" else cmd_report(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
