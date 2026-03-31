#!/usr/bin/env python3
"""Build CertOS artifacts (Paper 6).

Produces:
- lifecycle_events.jsonl
- invariant_tests.log
- latency_report.csv
- recovery_report.md
- audit_completeness.json

Runs run_certos_lifecycle and augments outputs.
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CertOS artifacts")
    parser.add_argument("--out", default="reports/certos", help="Output directory")
    args = parser.parse_args()

    out = _resolve_repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Run lifecycle
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "scripts/run_certos_lifecycle.py", "--steps", "96", "--out", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1

    csv_path = out / "certos_lifecycle.csv"
    summary_path = out / "certos_summary.json"
    if not csv_path.exists():
        return 1

    rows = list(csv_mod.DictReader(open(csv_path)))
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    audit_path = out / "audit_ops.jsonl"
    raw_audit = [json.loads(line) for line in audit_path.read_text().splitlines()] if audit_path.exists() else []
    step_audit = {}
    for entry in raw_audit:
        step_audit.setdefault(int(entry["step"]), []).append(entry)

    # lifecycle_events.jsonl
    with open(out / "lifecycle_events.jsonl", "w") as f:
        for row in rows:
            step = int(row["step"])
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "status": row["status"],
                        "lifecycle_op": row.get("lifecycle_op", ""),
                        "fallback": row.get("fallback", "False") == "True",
                        "validation_passed": row.get("validation_passed", "False") == "True",
                        "hash_chain_ok": row.get("hash_chain_ok", "False") == "True",
                        "audit_ops": [entry["op"] for entry in step_audit.get(step, [])],
                        "cert_hash": row.get("cert_hash") or None,
                        "prev_hash": row.get("prev_hash") or None,
                    }
                )
                + "\n"
            )

    # invariant_tests.log
    inv_ok = all(row.get("invariant_violations", "none") == "none" for row in rows)
    inv1_ok = all("INV-1" not in row.get("invariant_violations", "") for row in rows)
    inv2_ok = summary.get("hash_chain_ok", False) and all(
        "INV-2" not in row.get("invariant_violations", "") for row in rows
    )
    inv3_ok = all("INV-3" not in row.get("invariant_violations", "") for row in rows)
    (out / "invariant_tests.log").write_text(
        f"CertOS invariant tests\n"
        f"INV-1 (no dispatch without cert): {'PASS' if inv1_ok else 'FAIL'}\n"
        f"INV-2 (hash chain): {'PASS' if inv2_ok else 'FAIL'}\n"
        f"INV-3 (fallback iff H<=0): {'PASS' if inv3_ok else 'FAIL'}\n"
        f"All steps invariant_violations=none: {'PASS' if inv_ok else 'FAIL'}\n"
        f"VALIDATE events observed: {summary.get('validation_events', 0)}\n"
        f"EXPIRE events observed: {summary.get('expire_events', 0)}\n"
    )

    # latency_report.csv
    steps_per_sec = len(rows) / elapsed if elapsed > 0 else 0
    step_ms = 1000 * elapsed / len(rows) if rows else 0
    pd_latency = f"metric,value\nstep_latency_ms,{step_ms:.2f}\nthroughput_steps_per_sec,{steps_per_sec:.2f}\n"
    (out / "latency_report.csv").write_text(pd_latency)

    # recovery_report.md
    n_fallback = sum(1 for row in rows if row.get("fallback", "False") == "True")
    n_valid = sum(1 for row in rows if row["status"] == "valid")
    recovery_transitions = sum(
        1
        for prev, curr in zip(rows, rows[1:])
        if prev["status"] == "fallback" and curr["status"] == "valid"
    )
    (out / "recovery_report.md").write_text(
        f"# CertOS Recovery Report\n\n"
        f"- Total steps: {len(rows)}\n"
        f"- Valid: {n_valid}\n"
        f"- Fallback: {n_fallback}\n"
        f"- Recovery transitions observed: {recovery_transitions}\n"
        f"- Recovery observed: {'yes' if recovery_transitions > 0 else 'no'}\n"
    )

    # audit_completeness.json
    audit_entries = summary.get("audit_entries", len(rows))
    raw_audit_entries = summary.get("raw_audit_entries", len(raw_audit))
    step_coverage_pct = 100.0 * len(step_audit) / max(len(rows), 1)
    (out / "audit_completeness.json").write_text(
        json.dumps(
            {
                "audit_entries": audit_entries,
                "raw_audit_entries": raw_audit_entries,
                "total_steps": len(rows),
                "completeness_pct": 100.0 * audit_entries / max(len(rows), 1),
                "step_coverage_pct": step_coverage_pct,
                "validate_events": summary.get("validation_events", 0),
                "expire_events": summary.get("expire_events", 0),
                "hash_chain_ok": summary.get("hash_chain_ok", False),
            },
            indent=2,
        )
    )

    # Only write the claim linkage file when running inside the repo
    try:
        csv_rel = csv_path.relative_to(REPO)
        summary_rel = summary_path.relative_to(REPO)
        audit_rel = audit_path.relative_to(REPO)

        linkage_path = REPO / "reports" / "paper6_claim_linkage.md"
        linkage_path.write_text(
            "# Paper 6 Claim Linkage\n\n"
            f"- Lifecycle source: `{csv_rel}`\n"
            f"- Summary source: `{summary_rel}`\n"
            f"- Audit source: `{audit_rel}`\n"
            f"- VALIDATE lifecycle handling: {'PASS' if summary.get('validation_events', 0) > 0 else 'FAIL'} "
            f"({summary.get('validation_events', 0)} events, {summary.get('validation_failures', 0)} failures)\n"
            f"- Audited EXPIRE events: {'PASS' if summary.get('expire_events', 0) > 0 else 'FAIL'} "
            f"({summary.get('expire_events', 0)} events)\n"
            f"- INV-2 hash-chain verification: {'PASS' if summary.get('hash_chain_ok', False) else 'FAIL'}\n"
            f"- Artifact generation from runtime truth: {'PASS' if inv_ok and audit_path.exists() else 'FAIL'}\n"
            f"- Recovery transitions from runtime truth: {recovery_transitions}\n"
        )
    except ValueError:
        pass  # Output outside repo (e.g. test fixture); skip linkage write

    print(f"CertOS artifacts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
