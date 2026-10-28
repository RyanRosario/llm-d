#!/usr/bin/env python3
"""Computes the GKE Pod Snapshots alerting metrics and compares them against thresholds.

Called by .github/scripts/e2e/e2e-validate-pod-snapshot.sh (the nightly's
`--extra-validate pod-snapshot` step) after guides/pod-snapshot/scripts/nightly-deploy-gke.sh
has brought up Pod 1 (cold start + snapshot) and Pod 2 (restore from the snapshot).

Reads both pods with kubectl once (snapshot_timings.load_pods), prints and writes the per-phase
CSV (snapshot_timings.print_report; a failure there is logged and does not skip the checks), then
checks pod state, phase-marker completeness, and the 5 composite metrics from the GKE Pod Snapshots
CI/CD Plan. Every check is evaluated and printed as a PASS/FAIL table (verify_helpers, shared with
the other nightly verifiers); the exit code is 1 if any check fails and 0 if all pass.

Usage:
  check_thresholds.py --namespace NS --engine vllm|sglang [--selector llm-d.ai/guide=pod-snapshot]
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

import snapshot_timings as st  # noqa: E402
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
    """Returns one verify_helpers.Check per alerting metric (plus pod presence and phase-marker checks),
    using the thresholds of `engine` (vllm or sglang)."""
    t = THRESHOLDS[engine]
    checks: list[v.Check] = []

    p1_cold = bool(p1 and p1.get("created")) and not st.is_restored(p1)
    p2_ok = bool(p2 and p2.get("created"))
    checks.append(v.Check("Pod 1 cold started", p1_cold,
                          "found, cold start" if p1_cold else "missing, or restored from an old snapshot"))
    checks.append(v.Check("Pod 2 present", p2_ok, "found" if p2_ok else "not found"))

    ecs = None
    if p1_cold:
        missing1 = [k for k in st.COLD_MARKERS if k not in p1["log"]]
        checks.append(v.Check("Pod 1 phase markers", not missing1,
                              f"all {len(st.COLD_MARKERS)} present" if not missing1 else f"missing: {', '.join(missing1)}"))
        c = st.compute_cold_metrics(p1)
        ecs, cs, tsot = c["ecs"], c["cs"], c["4"]
        checks.append(v.Check("E2E Cold Start (Rows 1+2+3)", ecs <= t["max_e2e_cold_start_s"],
                              f"{ecs:.2f}s <= {t['max_e2e_cold_start_s']:.0f}s"))
        checks.append(v.Check("Cold Start, no node provisioning (1b+1c+1d+2)", cs <= t["max_cold_start_s"],
                              f"{cs:.2f}s <= {t['max_cold_start_s']:.0f}s"))
        checks.append(v.Check("Total Snapshot Overhead Time (Row 4)", 0.0 < tsot <= t["max_snapshot_overhead_s"],
                              f"0 < {tsot:.2f}s <= {t['max_snapshot_overhead_s']:.0f}s"))

    if p2_ok and st.is_restored(p2):
        missing2 = [k for k in st.RESTORE_MARKERS if k not in p2["log"]]
        checks.append(v.Check("Pod 2 phase markers", not missing2,
                              f"all {len(st.RESTORE_MARKERS)} present" if not missing2 else f"missing: {', '.join(missing2)}"))
        if ecs is not None:
            wsr = st.compute_restore_metrics(p2)["wsr"]
            speedup = ecs / wsr if wsr > 0 else float("inf")
            checks.append(v.Check("Restore speedup ECS / WSR (WSR = 1b+1c+4e+4f+3)", speedup >= t["min_restore_speedup"],
                                  f"{speedup:.2f}x >= {t['min_restore_speedup']:.0f}x (WSR {wsr:.2f}s)"))

    if p2_ok:
        restored = st.is_restored(p2)
        acs2 = 0.0 if restored else st.compute_cold_metrics(p2)["2"]
        detail = (
            f"{acs2:.2f}s <= {MAX_POD2_APP_COLD_START_S:.0f}s"
            if restored or acs2 > 0
            else f"{acs2:.2f}s <= {MAX_POD2_APP_COLD_START_S:.0f}s (Pod 2 did not restore)"
        )
        checks.append(v.Check("Application Cold Start on Pod 2 (Row 2)",
                              restored and acs2 <= MAX_POD2_APP_COLD_START_S,
                              detail))
    return checks


def write_step_summary(engine: str, checks: list[v.Check]) -> None:
    """Appends a markdown table to $GITHUB_STEP_SUMMARY when running in GitHub Actions."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [f"### Pod Snapshot alerting thresholds ({engine})", "",
             "| Status | Check | Detail |", "| --- | --- | --- |"]
    lines += [f"| {'PASS' if c.passed else 'FAIL'} | {c.name} | {c.detail} |" for c in checks]
    with open(summary_path, "a") as fh:
        fh.write("\n".join(lines) + "\n\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--selector", default="llm-d.ai/guide=pod-snapshot")
    parser.add_argument("--engine", required=True, choices=tuple(THRESHOLDS), help="vllm or sglang")
    parser.add_argument("--csv", default="", help="optional path to write the per-phase metrics CSV")
    args = parser.parse_args(argv)

    try:
        p1, p2, snap_ready = st.load_pods(st.Kubectl(args.namespace), args.selector)
    except RuntimeError as e:
        v.error(f"ERROR: {e}")
        return 1

    # Informational: the per-phase CSV report for on-call and artifact upload. It must never
    # skip the threshold checks below, which decide pass/fail.
    print()
    print("=== Pod snapshot per-phase timing report ===")
    try:
        st.print_report(p1, p2, snap_ready, engine=args.engine, csv_path=args.csv)
    except Exception as e:  # noqa: BLE001 - any report bug is logged, not fatal
        v.error(f"Warning: the timing report failed ({type(e).__name__}: {e}); continuing to the threshold checks.")

    checks = evaluate(p1, p2, args.engine)
    print()
    print(f"=== Pod Snapshot alerting thresholds — {args.engine} / {args.namespace} ===")
    print()
    v.print_checks_table(checks)
    print()
    write_step_summary(args.engine, checks)

    failed = [c for c in checks if not c.passed]
    if failed:
        v.error(f"FAIL: {len(failed)}/{len(checks)} check(s) failed")
        return 1
    print(f"PASS: {len(checks)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
