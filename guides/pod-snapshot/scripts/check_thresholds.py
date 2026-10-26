#!/usr/bin/env python3
"""Computes the GKE Pod Snapshots alerting metrics and compares them against thresholds.

Called by .github/scripts/e2e/e2e-validate-pod-snapshot.sh (the nightly's
`--extra-validate pod-snapshot` step) after guides/pod-snapshot/scripts/nightly-deploy-gke.sh
has brought up Pod 1 (cold start + snapshot) and Pod 2 (restore from the snapshot).

Reads both pods with kubectl once (snapshot_timings.load_pods), prints the informational timing
table and writes the per-phase CSV (snapshot_timings.print_report; a failure there is logged
and does not skip the checks), then checks the 5 composite metrics from the GKE Pod Snapshots
CI/CD Plan. Every check is evaluated and printed as a PASS/FAIL table (verify_helpers, shared
with the other nightly verifiers); the exit code is 1 if any check fails and 0 if all pass.

Usage:
  check_thresholds.py --namespace NS [--selector llm-d.ai/guide=pod-snapshot] [--engine vllm|sglang]
                      [--csv results.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
VERIFY_DIR = REPO_ROOT / ".github" / "scripts" / "nightly-e2e-verification"
for p in (str(HERE), str(VERIFY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import snapshot_timings as dp  # noqa: E402
import verify_helpers as v  # noqa: E402

# Thresholds for the composite metrics of the GKE Pod Snapshots CI/CD Plan ("Composite Metrics
# Used for Alerting"). E2E Cold Start, Cold Start and the restore speedup use the design doc's
# initial thresholds (from the 94-run baseline). Total Snapshot Overhead uses the p99 of 20 runs
# per engine on the nightly cluster (llm-d-scale / llm-d-e2e-pod-snapshot-us-central1,
# 2026-10-24), rounded up to a multiple of 5 and never below the worst observed run.
THRESHOLDS = {
    "vllm": {
        "max_e2e_cold_start_s": 1800.0,    # ECS > 1800 -> fail
        "max_cold_start_s": 350.0,         # CS > 350 -> fail
        "max_snapshot_overhead_s": 65.0,   # TSOT == 0 or > 65 -> fail (p99 60.7s, median 56.6s)
        "min_restore_speedup": 11.0,       # ECS / WSR < 11 -> fail
    },
    "sglang": {
        "max_e2e_cold_start_s": 1800.0,    # ECS > 1800 -> fail
        "max_cold_start_s": 350.0,         # CS > 350 -> fail
        "max_snapshot_overhead_s": 85.0,   # TSOT == 0 or > 85 -> fail (p99 79.0s, max 80.8s, median 51.6s)
        "min_restore_speedup": 11.0,       # ECS / WSR < 11 -> fail
    },
}
MAX_POD2_APP_COLD_START_S = 0.0    # ACS on Pod 2 > 0 -> fail (Pod 2 cold started instead of restoring)


def evaluate(p1, p2, engine: str) -> list[v.Check]:
    """Returns one verify_helpers.Check per alerting metric (plus pod presence checks),
    using the thresholds of `engine` (vllm or sglang)."""
    checks: list[v.Check] = []
    t = THRESHOLDS.get(engine)
    checks.append(v.Check("Engine has thresholds", t is not None,
                          f"engine={engine or '?'}" if t else
                          f"engine={engine or '?'} (expected one of {sorted(THRESHOLDS)})"))
    if t is None:
        return checks

    p1_cold = bool(p1 and p1.get("created")) and not dp.is_restored(p1)
    p2_ok = bool(p2 and p2.get("created"))
    checks.append(v.Check("Pod 1 cold started", p1_cold,
                          "found, cold start" if p1_cold else "missing, or restored from an old snapshot"))
    checks.append(v.Check("Pod 2 present", p2_ok, "found" if p2_ok else "not found"))

    ecs = None
    if p1_cold:
        c = dp.compute_cold_metrics(p1)
        ecs = c["1"] + c["2"] + c["3"]
        checks.append(v.Check("E2E Cold Start (Rows 1+2+3)", ecs <= t["max_e2e_cold_start_s"],
                              f"{ecs:.2f}s <= {t['max_e2e_cold_start_s']:.0f}s"))
        cs = c["1b"] + c["1c"] + c["1d"] + c["2"]
        checks.append(v.Check("Cold Start, no node provisioning (1b+1c+1d+2)", cs <= t["max_cold_start_s"],
                              f"{cs:.2f}s <= {t['max_cold_start_s']:.0f}s"))
        tsot = c["4"]
        checks.append(v.Check("Total Snapshot Overhead Time (Row 4)", 0.0 < tsot <= t["max_snapshot_overhead_s"],
                              f"0 < {tsot:.2f}s <= {t['max_snapshot_overhead_s']:.0f}s"))

    if p2_ok and dp.is_restored(p2) and ecs is not None:
        r = dp.compute_restore_metrics(p2)
        wsr = r["1b"] + r["1c"] + r["4e"] + r["4f"] + r["3"]
        speedup = ecs / wsr if wsr > 0 else float("inf")
        checks.append(v.Check("Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)", speedup >= t["min_restore_speedup"],
                              f"{speedup:.2f}x >= {t['min_restore_speedup']:.0f}x (WSR {wsr:.2f}s)"))

    if p2_ok:
        acs2 = 0.0 if dp.is_restored(p2) else dp.compute_cold_metrics(p2)["2"]
        checks.append(v.Check("Application Cold Start on Pod 2 (Row 2)", acs2 <= MAX_POD2_APP_COLD_START_S,
                              f"{acs2:.2f}s <= {MAX_POD2_APP_COLD_START_S:.0f}s"))
    return checks


def write_step_summary(engine: str, checks: list[v.Check]) -> None:
    """Appends a markdown table to $GITHUB_STEP_SUMMARY when running in GitHub Actions."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [f"### Pod Snapshot alerting thresholds ({engine or 'model server'})", "",
             "| Status | Check | Detail |", "| --- | --- | --- |"]
    lines += [f"| {'PASS' if c.passed else 'FAIL'} | {c.name} | {c.detail} |" for c in checks]
    with open(summary_path, "a") as fh:
        fh.write("\n".join(lines) + "\n\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--selector", default="llm-d.ai/guide=pod-snapshot")
    parser.add_argument("--engine", default="", help="vllm or sglang (inferred from the pods if omitted)")
    parser.add_argument("--csv", default="", help="optional path to write the per-phase metrics CSV")
    args = parser.parse_args(argv)

    engine = args.engine or str(dp.infer_engine(selector=args.selector) or "")
    try:
        p1, p2, snap_ready = dp.load_pods(dp.Kubectl(args.namespace), args.selector, engine)
    except RuntimeError as e:
        v.error(f"ERROR: {e}")
        return 1
    engine = dp.detected_engine(p1, p2, engine)

    # Informational: the full timing table for on-call, plus the per-phase CSV. It must never
    # skip the threshold checks below, which decide pass/fail.
    print()
    print("=== Pod snapshot per-phase timing report ===")
    try:
        dp.print_report(p1, p2, snap_ready, engine=engine, csv_path=args.csv)
    except Exception as e:  # noqa: BLE001 - any report bug is logged, not fatal
        v.error(f"Warning: the timing report failed ({type(e).__name__}: {e}); continuing to the threshold checks.")

    checks = evaluate(p1, p2, engine)
    print()
    print(f"=== Pod Snapshot alerting thresholds — {engine or 'model server'} / {args.namespace} ===")
    print()
    v.print_checks_table(checks)
    print()
    write_step_summary(engine, checks)

    failed = [c for c in checks if not c.passed]
    if failed:
        v.error(f"FAIL: {len(failed)}/{len(checks)} check(s) failed")
        return 1
    print(f"PASS: {len(checks)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
